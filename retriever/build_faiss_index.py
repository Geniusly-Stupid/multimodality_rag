"""Build a FAISS retrieval index from Frames Wikipedia pages."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

try:
    import faiss
except ImportError as exc:  # pragma: no cover - explicit runtime dependency guard
    raise SystemExit("Missing dependency 'faiss'. Install faiss-cpu/ faiss-gpu before running.") from exc

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
FRAMES_DATASET_DIR = Path("data/frames_wiki_dataset")
if not FRAMES_DATASET_DIR.exists():
    FRAMES_DATASET_DIR = ROOT_DIR / "frames_wiki_dataset"
PAGES_DIR = FRAMES_DATASET_DIR / "pages"
OUTPUT_DIR = BASE_DIR / "faiss_index"

CHUNK_MIN_WORDS = 200
CHUNK_MAX_WORDS = 350
CHUNK_OVERLAP = 50
MIN_PAGE_WORDS = 120

DEFAULT_MODEL_NAME = "BAAI/bge-base-en-v1.5"
DEFAULT_POOLING = "cls"
DEFAULT_MAX_LENGTH = 512
DEFAULT_NORMALIZE = True
ENCODING_BATCH_SIZE = 16

FAISS_INDEX_TYPE = "flat"  # or "ivf"
FAISS_METRIC = "ip"  # "ip" or "l2"
FAISS_NLIST = 100

RUN_DEMO_QUERY = None


@dataclass
class EncoderConfig:
    model_name: str = DEFAULT_MODEL_NAME
    pooling: str = DEFAULT_POOLING
    max_length: int = DEFAULT_MAX_LENGTH
    normalize: bool = DEFAULT_NORMALIZE


class TextEncoder:
    """Thin wrapper around a Hugging Face encoder with configurable pooling."""

    def __init__(self, config: EncoderConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name).to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size

    def encode(self, texts: Sequence[str], batch_size: int = ENCODING_BATCH_SIZE) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype="float32")
        all_embeddings: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**encoded)
                token_embeddings = outputs.last_hidden_state
                pooled = self._pool(token_embeddings, encoded["attention_mask"])
                if self.config.normalize:
                    pooled = F.normalize(pooled, p=2, dim=1)
                batch_arr = pooled.cpu().numpy().astype("float32")
                all_embeddings.append(batch_arr)
        return np.vstack(all_embeddings)

    def _pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooling = self.config.pooling.lower()
        if pooling == "cls":
            return token_embeddings[:, 0]
        if pooling == "mean":
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            summed = torch.sum(token_embeddings * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-6)
            return summed / counts
        raise ValueError(f"Unsupported pooling strategy: {self.config.pooling}")


def load_pages(pages_dir: Path, min_words: int = MIN_PAGE_WORDS) -> List[Dict[str, Any]]:
    """Load text + metadata for each Wikipedia page."""
    pages: List[Dict[str, Any]] = []
    if not pages_dir.exists():
        raise SystemExit(f"Missing pages directory: {pages_dir}")

    for page_dir in sorted(pages_dir.iterdir()):
        text_path = page_dir / "text.txt"
        meta_path = page_dir / "meta.json"
        if not text_path.exists():
            continue
        text = text_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        if len(text.split()) < min_words:
            continue
        meta: Dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        pages.append(
            {
                "page_id": page_dir.name,
                "text": text,
                "url": meta.get("url"),
            }
        )
    print(f"Loaded {len(pages)} pages with >= {min_words} words.")
    if not pages:
        raise SystemExit("No valid pages found. Did you run build_frames_wiki_dataset.py?")
    return pages


def chunk_text(text: str, min_words: int, max_words: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]

    step = max(max_words - overlap, 1)
    chunks: List[List[str]] = []
    for start in range(0, len(words), step):
        end = min(len(words), start + max_words)
        chunk_words = words[start:end]
        if len(chunk_words) < min_words and chunks:
            chunks[-1].extend(chunk_words)
        else:
            chunks.append(chunk_words)

    if chunks and len(chunks) >= 2 and len(chunks[-1]) < min_words:
        chunks[-2].extend(chunks[-1])
        chunks.pop()

    return [" ".join(chunk) for chunk in chunks]


def build_chunks(pages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    chunk_uid = 0
    for page in pages:
        chunk_texts = chunk_text(page["text"], CHUNK_MIN_WORDS, CHUNK_MAX_WORDS, CHUNK_OVERLAP)
        for local_id, ctext in enumerate(chunk_texts):
            record = {
                "id": chunk_uid,
                "page_id": page["page_id"],
                "chunk_id": local_id,
                "text": ctext,
                "source_url": page.get("url"),
            }
            records.append(record)
            chunk_uid += 1
    print(f"Created {len(records)} text chunks.")
    return records


def build_faiss_index(
    embeddings: np.ndarray,
    index_type: str = FAISS_INDEX_TYPE,
    metric: str = FAISS_METRIC,
    nlist: int = FAISS_NLIST,
) -> faiss.Index:
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")
    num_rows, dim = embeddings.shape
    if num_rows == 0:
        raise ValueError("No embeddings provided for FAISS index.")
    xb = np.ascontiguousarray(embeddings.astype("float32"))

    use_ip = metric.lower() == "ip"
    if index_type.lower() == "flat":
        index = faiss.IndexFlatIP(dim) if use_ip else faiss.IndexFlatL2(dim)
    elif index_type.lower() == "ivf":
        quantizer = faiss.IndexFlatIP(dim) if use_ip else faiss.IndexFlatL2(dim)
        nlist = min(max(1, nlist), num_rows)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT if use_ip else faiss.METRIC_L2)
        if not index.is_trained:
            print(f"Training IVF index with nlist={nlist} on {num_rows} vectors...")
            index.train(xb)
    else:
        raise ValueError(f"Unsupported FAISS index type: {index_type}")

    index.add(xb)
    return index


def save_chunks(path: Path, chunks: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def write_metadata(path: Path, *, num_chunks: int, embedding_dim: int, encoder_name: str, faiss_index_type: str) -> None:
    payload = {
        "num_chunks": num_chunks,
        "embedding_dim": embedding_dim,
        "encoder_name": encoder_name,
        "faiss_index_type": faiss_index_type,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def search_demo(query: str, encoder: TextEncoder, index: faiss.Index, chunks: Sequence[Dict[str, Any]], top_k: int = 5) -> None:
    if not query:
        return
    if not chunks:
        print("No chunks available for search demo.")
        return
    print(f"\n[demo] Searching for: {query!r}")
    query_emb = encoder.encode([query])
    distances, indices = index.search(query_emb, top_k)
    for rank in range(min(top_k, len(chunks))):
        chunk_index = int(indices[0][rank])
        if chunk_index < 0 or chunk_index >= len(chunks):
            continue
        record = chunks[chunk_index]
        score = float(distances[0][rank])
        preview = record["text"][:160].replace("\n", " ") + "..."
        print(f"  #{rank + 1} | score={score:.4f} | page={record['page_id']} | preview={preview}")


def main() -> None:
    start_time = time.time()
    pages = load_pages(PAGES_DIR, MIN_PAGE_WORDS)
    chunks = build_chunks(pages)
    if not chunks:
        raise SystemExit("No chunks generated. Check chunking parameters.")

    chunk_texts = [record["text"] for record in chunks]

    encoder_config = EncoderConfig()
    encoder = TextEncoder(encoder_config)
    print(f"Encoding {len(chunk_texts)} chunks with {encoder_config.model_name}...")
    encode_start = time.time()
    embeddings = encoder.encode(chunk_texts, batch_size=ENCODING_BATCH_SIZE)
    encode_time = time.time() - encode_start
    print(f"Encoded {embeddings.shape[0]} chunks in {encode_time:.2f}s. Dim={embeddings.shape[1]}")

    index = build_faiss_index(embeddings, index_type=FAISS_INDEX_TYPE, metric=FAISS_METRIC, nlist=FAISS_NLIST)
    print(f"FAISS index built ({index.__class__.__name__}) with {index.ntotal} vectors.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)
    faiss.write_index(index, str(OUTPUT_DIR / "index.faiss"))
    save_chunks(OUTPUT_DIR / "chunks.jsonl", chunks)
    write_metadata(
        OUTPUT_DIR / "metadata.json",
        num_chunks=len(chunks),
        embedding_dim=embeddings.shape[1],
        encoder_name=encoder_config.model_name,
        faiss_index_type=index.__class__.__name__,
    )

    elapsed = time.time() - start_time
    print(
        f"Saved FAISS resources to {OUTPUT_DIR.resolve()} "
        f"(pages={len(pages)}, chunks={len(chunks)}, elapsed={elapsed:.2f}s)."
    )

    if RUN_DEMO_QUERY:
        search_demo(RUN_DEMO_QUERY, encoder, index, chunks)


if __name__ == "__main__":
    main()
