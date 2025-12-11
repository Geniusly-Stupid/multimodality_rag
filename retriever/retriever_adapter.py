"""Adapter that builds a simple vector store from RAGAnything chunks (caption-only for images)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import faiss
import numpy as np


def _to_numpy(vector: Any) -> np.ndarray:
    arr = np.asarray(vector, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Vector must be 1D or 2D, got shape {arr.shape}")
    return np.ascontiguousarray(arr.astype("float32"))


@dataclass
class FaissVectorStore:
    """Thin FAISS wrapper with in-memory metadata."""

    metric: str = "ip"  # {"ip", "l2"}
    dim: Optional[int] = None

    def __post_init__(self) -> None:
        self.index: Optional[faiss.Index] = None
        self.doc_ids: List[Any] = []
        self.metadata: List[Dict[str, Any]] = []
        self._vectors: List[np.ndarray] = []

    def _ensure_index(self, vector: np.ndarray) -> None:
        if self.index is None:
            self.dim = vector.shape[1]
            if self.metric.lower() == "l2":
                self.index = faiss.IndexFlatL2(self.dim)
            else:
                self.index = faiss.IndexFlatIP(self.dim)
        elif self.dim != vector.shape[1]:
            raise ValueError(f"Vector dimension mismatch: expected {self.dim}, got {vector.shape[1]}")

    def add(self, *, doc_id: Any, vector: Any, metadata: Dict[str, Any]) -> None:
        vec = _to_numpy(vector)
        self._ensure_index(vec)
        self.index.add(vec)
        self.doc_ids.append(doc_id)
        self.metadata.append(metadata)
        self._vectors.append(vec)

    def search(self, query_vector: Any, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q_vec = _to_numpy(query_vector)
        scores, indices = self.index.search(q_vec, top_k)

        results: List[Dict[str, Any]] = []
        max_rank = min(top_k, len(self.metadata))
        for rank in range(max_rank):
            idx = int(indices[0][rank])
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            result: Dict[str, Any] = {
                "doc_id": self.doc_ids[idx],
                "score": float(scores[0][rank]),
                "metadata": meta.get("metadata", meta),
            }
            result.update({k: v for k, v in meta.items() if k not in {"metadata"}})
            results.append(result)
        return results


class RetrieverAdapter:
    """
    Build a FAISS index from RAGAnything chunks.

    Image chunks are represented by their captions/text only (no image encoder).
    """

    def __init__(self, text_encoder, vector_store: FaissVectorStore):
        self.text_encoder = text_encoder
        self.vector_store = vector_store

    def _encode_text(self, text: str) -> Optional[np.ndarray]:
        if not text:
            return None
        if hasattr(self.text_encoder, "encode"):
            vecs = self.text_encoder.encode([text])
        elif hasattr(self.text_encoder, "encode_text"):
            vecs = self.text_encoder.encode_text([text])
        else:
            raise AttributeError("Text encoder must expose encode() or encode_text()")
        if isinstance(vecs, np.ndarray) and vecs.ndim == 2:
            return vecs[0]
        return np.asarray(vecs, dtype="float32")

    def _text_for_chunk(self, chunk: Dict[str, Any]) -> str:
        if not isinstance(chunk, dict):
            return ""
        if chunk.get("text"):
            return str(chunk["text"])
        meta = chunk.get("metadata") or {}
        if isinstance(meta, dict):
            if meta.get("caption"):
                return str(meta["caption"])
            if meta.get("text"):
                return str(meta["text"])
        return ""

    def build_index(self, chunks: Sequence[Dict[str, Any]]) -> None:
        for c in chunks:
            text = self._text_for_chunk(c)
            if not text:
                continue
            emb = self._encode_text(text)
            if emb is None:
                continue
            metadata = dict(c)
            inner_meta = metadata.get("metadata") or {}
            if not metadata.get("text"):
                metadata["text"] = text
            if inner_meta.get("caption") and not metadata.get("caption"):
                metadata["caption"] = inner_meta.get("caption")
            if metadata.get("image") and not metadata.get("modality"):
                metadata["modality"] = "caption" if metadata.get("caption") else "text"
            self.vector_store.add(doc_id=c.get("id"), vector=emb, metadata=metadata)

    def search(self, query: str, top_k: int = 5, **_: Any) -> List[Dict[str, Any]]:
        q_vec = self._encode_text(query)
        if q_vec is None:
            return []
        return self.vector_store.search(q_vec, top_k)

    def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> List[Dict[str, Any]]:
        return self.search(query, top_k=top_k, **kwargs)
