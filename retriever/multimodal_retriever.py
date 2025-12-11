"""Multimodal retriever supporting both text and image retrieval."""

from __future__ import annotations

import os

# Fix OpenMP library conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import faiss
import numpy as np

from .image_encoder import CLIPTextEncoder, ImageEncoderConfig
from .rag_retriever import DEFAULT_INDEX_DIRS, EncoderConfig, RAGRetriever
from .reranker import CrossEncoderReranker, RerankerConfig

BASE_DIR = Path(__file__).resolve().parent
IMAGE_INDEX_DIR = BASE_DIR / "faiss_index" / "images"
IMAGE_INDEX_PATH = "index.faiss"
IMAGE_METADATA_PATH = "images.jsonl"
IMAGE_METADATA_JSON = "metadata.json"
CAPTION_INDEX_FILENAME = "frames_captions.index"
CAPTION_CHUNKS_FILENAME = "frames_captions.jsonl"
CAPTION_METADATA_FILENAME = "frames_captions_meta.json"


def load_image_metadata(path: Path) -> List[Dict]:
    """Load image metadata from JSONL."""
    images: List[Dict] = []
    if not path.exists():
        return images
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            images.append(json.loads(line))
    return images


@dataclass
class MultimodalRetrieverConfig:
    """Configuration for multimodal retriever."""

    text_encoder_config: Optional[EncoderConfig] = None
    image_encoder_config: Optional[ImageEncoderConfig] = None
    use_text_retrieval: bool = True
    use_image_retrieval: bool = True
    use_caption_retrieval: bool = False
    text_weight: float = 0.5  # Weight for text results when merging
    image_weight: float = 0.5  # Weight for image results when merging
    caption_weight: float = 0.5  # Weight for caption results when merging
    default_retrieval_mode: str = "text_clip"  # {"text", "text_clip", "text_caption", "caption_only"}
    use_reranker: bool = False
    reranker_config: Optional[RerankerConfig] = None


class MultimodalRetriever:
    """Retriever that supports both text and image retrieval."""

    def __init__(
        self,
        config: Optional[MultimodalRetrieverConfig] = None,
        text_retriever: Optional[RAGRetriever] = None,
    ):
        self.config = config or MultimodalRetrieverConfig()
        self.text_retriever = text_retriever or RAGRetriever(
            encoder_config=self.config.text_encoder_config or EncoderConfig()
        )
        self.text_encoder = self.text_retriever.encoder

        # Optional caption retriever (shares text encoder)
        self.caption_retriever: Optional[RAGRetriever] = None
        if self.config.use_caption_retrieval:
            try:
                self.caption_retriever = RAGRetriever(
                    encoder=self.text_encoder,
                    encoder_config=self.config.text_encoder_config or EncoderConfig(),
                    index_dirs=DEFAULT_INDEX_DIRS,
                    index_filename=CAPTION_INDEX_FILENAME,
                    chunks_filename=CAPTION_CHUNKS_FILENAME,
                    metadata_filename=CAPTION_METADATA_FILENAME,
                    result_type="caption",
                    text_field="text",
                )
            except FileNotFoundError as exc:  # pragma: no cover - runtime guard
                print(f"Warning: Caption index not found: {exc}")
                print("Run 'python retriever/build_faiss_index.py --mode caption' to create it.")

        # Initialize image retrieval components
        self.image_index = None
        self.image_metadata = []
        self.image_encoder = None

        if self.config.use_image_retrieval:
            self._load_image_index()

        self.reranker = CrossEncoderReranker(self.config.reranker_config or RerankerConfig()) if self.config.use_reranker else None

    def _load_image_index(self) -> None:
        """Load image FAISS index and metadata."""
        image_index_path = IMAGE_INDEX_DIR / IMAGE_INDEX_PATH
        image_metadata_path = IMAGE_INDEX_DIR / IMAGE_METADATA_PATH

        if not image_index_path.exists():
            print(f"Warning: Image index not found at {image_index_path}")
            print("Run 'python retriever/build_image_index.py' first to build the image index.")
            return

        self.image_index = faiss.read_index(str(image_index_path))
        self.image_metadata = load_image_metadata(image_metadata_path)

        # Initialize CLIP text encoder for query encoding
        self.image_encoder = CLIPTextEncoder(config=self.config.image_encoder_config or ImageEncoderConfig())

        # Validate dimensions
        if self.image_index.d != self.image_encoder.embedding_dim:
            print(
                f"Warning: Image index dimension {self.image_index.d} != "
                f"encoder dimension {self.image_encoder.embedding_dim}"
            )

        print(f"Loaded image index with {len(self.image_metadata)} images.")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        retrieval_mode: Optional[str] = None,
        return_modality: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve text/image/caption results for a query with flexible modes.

        Args:
            query: Text query.
            top_k: Number of results to return after fusion.
            retrieval_mode: One of {"text", "text_clip", "text_caption", "caption_only"}.
            return_modality: Legacy override to force text/image only.
        """
        if not query:
            return []

        mode = retrieval_mode or self.config.default_retrieval_mode
        if return_modality == "text":
            mode = "text"
        elif return_modality == "image":
            mode = "image_only"

        use_text = self.config.use_text_retrieval and mode in {"text", "text_clip", "text_caption"}
        use_caption = self.config.use_caption_retrieval and mode in {"text_caption", "caption_only"}
        use_image = self.config.use_image_retrieval and mode in {"text_clip", "image_only"}

        if mode == "caption_only":
            use_text = False
            use_image = False

        results: List[Dict] = []

        if use_text:
            text_results = self.text_retriever.retrieve(query, top_k=top_k, retrieval_mode=mode)
            for result in text_results:
                result["modality"] = "text"
                result["weighted_score"] = result.get("score", 0.0) * self.config.text_weight
            results.extend(text_results)

        if use_caption:
            if self.caption_retriever is None:
                print("Warning: caption retrieval requested but caption index is not loaded.")
            else:
                cap_results = self.caption_retriever.retrieve(query, top_k=top_k, retrieval_mode=mode)
                for result in cap_results:
                    result["modality"] = "caption"
                    result["weighted_score"] = result.get("score", 0.0) * self.config.caption_weight
                results.extend(cap_results)

        if use_image and self.image_index is not None:
            image_results = self._retrieve_images(query, top_k=top_k)
            for result in image_results:
                result["modality"] = "image"
                result["weighted_score"] = result["score"] * self.config.image_weight
            results.extend(image_results)

        if not results:
            return results

        if self.reranker:
            rerank_texts: List[str] = []
            for item in results:
                modality = item.get("modality", "text")
                base_text = item.get("text") or item.get("caption") or ""
                if modality == "caption":
                    rerank_texts.append(f"[CAPTION] {base_text}")
                elif modality == "image":
                    rerank_texts.append(f"[IMAGE] {base_text}")
                else:
                    rerank_texts.append(f"[TEXT] {base_text}")
            scores = self.reranker.score(query, rerank_texts)
            for item, score in zip(results, scores):
                item["rerank_score"] = float(score)
            results.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        else:
            results.sort(key=lambda x: x.get("weighted_score", x.get("score", 0)), reverse=True)

        return results[:top_k]

    def _retrieve_images(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve images using CLIP text encoder."""
        if not query or self.image_index is None or not self.image_metadata:
            return []

        # Encode query using CLIP text encoder
        query_vec = self.image_encoder.encode([query])
        distances, indices = self.image_index.search(query_vec, top_k)

        results: List[Dict] = []
        for rank in range(min(top_k, len(self.image_metadata))):
            idx = int(indices[0][rank])
            if idx < 0 or idx >= len(self.image_metadata):
                continue

            image_meta = self.image_metadata[idx]
            results.append(
                {
                    "image_id": image_meta.get("image_id"),
                    "caption": image_meta.get("caption", ""),
                    "text": image_meta.get("caption", ""),
                    "captions": image_meta.get("captions", []),
                    "score": float(distances[0][rank]),
                    "modality": "image",
                }
            )

        return results

