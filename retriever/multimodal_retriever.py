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
from .rag_retriever import EncoderConfig, RAGRetriever

BASE_DIR = Path(__file__).resolve().parent
IMAGE_INDEX_DIR = BASE_DIR / "faiss_index" / "images"
IMAGE_INDEX_PATH = "index.faiss"
IMAGE_METADATA_PATH = "images.jsonl"
IMAGE_METADATA_JSON = "metadata.json"


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
    text_weight: float = 0.5  # Weight for text results when merging
    image_weight: float = 0.5  # Weight for image results when merging


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

        # Initialize image retrieval components
        self.image_index = None
        self.image_metadata = []
        self.image_encoder = None

        if self.config.use_image_retrieval:
            self._load_image_index()

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
        self, query: str, top_k: int = 5, return_modality: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve both text and image results for a query.

        Args:
            query: Text query
            top_k: Number of results to return per modality
            return_modality: If 'text', return only text. If 'image', return only images.
                            If None, return merged results.

        Returns:
            List of retrieved items (text chunks or images)
        """
        results: List[Dict] = []

        # Text retrieval
        if self.config.use_text_retrieval and return_modality != "image":
            text_results = self.text_retriever.retrieve(query, top_k=top_k)
            for result in text_results:
                result["modality"] = "text"
                result["weighted_score"] = result["score"] * self.config.text_weight
            results.extend(text_results)

        # Image retrieval
        if self.config.use_image_retrieval and self.image_index is not None and return_modality != "text":
            image_results = self._retrieve_images(query, top_k=top_k)
            for result in image_results:
                result["modality"] = "image"
                result["weighted_score"] = result["score"] * self.config.image_weight
            results.extend(image_results)

        # Sort by weighted score if both modalities are used
        if return_modality is None and len(results) > 0:
            results.sort(key=lambda x: x.get("weighted_score", x.get("score", 0)), reverse=True)
            results = results[:top_k]

        return results

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
            # Handle different metadata formats (from processed data vs flickr8k)
            # Processed data format: img_path, image_caption, page_idx, source_file
            # Flickr8k format: image_id, caption, captions
            img_path = image_meta.get("img_path") or image_meta.get("image_path", "")
            caption = image_meta.get("caption") or image_meta.get("image_caption", "")
            # image_caption might be a list, convert to string if needed
            if isinstance(caption, list):
                caption = caption[0] if caption else ""
            captions = image_meta.get("captions", [])
            if not captions and image_meta.get("image_caption"):
                captions = image_meta.get("image_caption")
                if not isinstance(captions, list):
                    captions = [captions] if captions else []
            
            results.append(
                {
                    "image_id": image_meta.get("image_id", ""),
                    "img_path": img_path,
                    "caption": caption,
                    "captions": captions,
                    "source_file": image_meta.get("source_file", ""),
                    "page_id": image_meta.get("page_id") or image_meta.get("page_idx", ""),
                    "source_url": image_meta.get("source_url", ""),
                    "score": float(distances[0][rank]),
                    "modality": "image",
                }
            )

        return results

