"""Generate image captions for Frames wiki pages and save to JSONL.

This script scans the Frames wiki dataset produced by build_frames_wiki_dataset.py,
generates captions for each referenced image URL, and writes a resumable JSONL
file that can be used as text evidence during retrieval.

The actual captioning model is left as a TODO hook (ImageCaptioner); plug in a
visual captioning model when ready. The placeholder implementation emits a
deterministic string so downstream plumbing can be tested without a vision
checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests
from PIL import Image
import torch
from transformers import Blip2ForConditionalGeneration, Blip2Processor, BlipForConditionalGeneration, BlipProcessor


DATA_DIR = Path("data/frames_wiki_dataset")
PAGES_DIR = DATA_DIR / "pages"
DEFAULT_OUTPUT = DATA_DIR / "image_captions.jsonl"


def _hash_id(value: str) -> str:
    """Stable short id for an image URL."""
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def load_existing(path: Path) -> Dict[str, Dict]:
    """Load existing captions for resume functionality."""
    existing: Dict[str, Dict] = {}
    if not path.exists():
        return existing
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_id = record.get("image_id")
            if not image_id:
                continue
            existing[image_id] = record
    return existing


@dataclass
class CaptionConfig:
    caption_type: str = "default"
    model_name: str = "Salesforce/blip-image-captioning-large"
    model_type: str = "blip"  # {"blip", "blip2"}
    cache_dir: Optional[Path] = None
    device: Optional[str] = None


class ImageCaptioner:
    """BLIP/BLIP-2 captioner wrapper."""

    def __init__(self, config: Optional[CaptionConfig] = None):
        self.config = config or CaptionConfig()
        if self.config.cache_dir:
            self.config.cache_dir = Path(self.config.cache_dir)
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self) -> None:
        cache_kwargs = {}
        if self.config.cache_dir:
            cache_kwargs["cache_dir"] = str(self.config.cache_dir)

        model_type = (self.config.model_type or "blip").lower()
        if model_type == "blip2":
            self.processor = Blip2Processor.from_pretrained(self.config.model_name, **cache_kwargs)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                **cache_kwargs,
            ).to(self.device)
        else:
            self.processor = BlipProcessor.from_pretrained(self.config.model_name, **cache_kwargs)
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                **cache_kwargs,
            ).to(self.device)
        self.model.eval()

    def _load_image(self, image_url: str) -> Optional[Image.Image]:
        try:
            if image_url.startswith("http"):
                resp = requests.get(image_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            else:
                img = Image.open(image_url).convert("RGB")
            return img
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Failed to load image {image_url}: {exc}")
            return None

    def generate_caption(self, image_url: str, page_id: str) -> str:
        img = self._load_image(image_url)
        if img is None:
            return f"(failed to load image for page {page_id})"

        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            if isinstance(self.model, Blip2ForConditionalGeneration):
                generated_ids = self.model.generate(**inputs, max_new_tokens=60)
            else:
                generated_ids = self.model.generate(**inputs, max_new_tokens=60)
        decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        caption = decoded[0] if decoded else ""
        return caption.strip()


def iter_page_images(page_dir: Path) -> Iterable[str]:
    """Yield image URLs listed under a page directory."""
    images_path = page_dir / "images.txt"
    if not images_path.exists():
        return []
    with images_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_page_meta(page_dir: Path) -> Dict:
    meta_path = page_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def build_captions(output: Path, caption_type: str, limit: Optional[int], model_name: str, model_type: str, cache_dir: Optional[str]) -> None:
    captioner = ImageCaptioner(CaptionConfig(caption_type=caption_type, model_name=model_name, model_type=model_type, cache_dir=cache_dir))
    existing = load_existing(output)
    seen_ids: Set[str] = set(existing.keys())

    output.parent.mkdir(parents=True, exist_ok=True)
    out_file = output.open("a", encoding="utf-8")

    page_dirs = sorted([p for p in PAGES_DIR.iterdir() if p.is_dir()])
    total_pages = len(page_dirs)
    created = 0

    for idx, page_dir in enumerate(page_dirs, start=1):
        if limit is not None and created >= limit:
            break

        page_id = page_dir.name
        meta = load_page_meta(page_dir)
        page_url = meta.get("url")
        images = iter_page_images(page_dir)
        if not images:
            continue

        for image_url in images:
            if limit is not None and created >= limit:
                break
            image_id = _hash_id(image_url)
            if image_id in seen_ids:
                continue
            caption = captioner.generate_caption(image_url, page_id)
            record = {
                "image_id": image_id,
                "page_id": page_id,
                "image_url": image_url,
                "page_url": page_url,
                "caption": caption,
                "caption_type": caption_type,
                "caption_model": captioner.config.model_name,
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            seen_ids.add(image_id)
            created += 1
        print(f"[{idx}/{total_pages}] processed page {page_id} (new captions: {created})")

    out_file.close()
    print(f"Finished. Captions written to {output.resolve()} (total new: {created}, existing skipped: {len(existing)})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate image captions for Frames wiki pages.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Path to output JSONL file.")
    parser.add_argument("--caption_type", type=str, default="default", help="Tag for ablations (e.g., short/long).")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of new captions to write.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Salesforce/blip-image-captioning-large",
        help="Hugging Face model name for captioning (e.g., Salesforce/blip2-opt-2.7b).",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="blip",
        choices=["blip", "blip2"],
        help="Use BLIP or BLIP-2 pipeline.",
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="Optional HF cache dir.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_captions(
        Path(args.output),
        caption_type=args.caption_type,
        limit=args.limit,
        model_name=args.model_name,
        model_type=args.model_type,
        cache_dir=args.cache_dir,
    )
