"""FAISS-based retriever for Frames Benchmark chunks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# Default locations (new preferred under retriever/, fallback to legacy root)
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DEFAULT_INDEX_DIRS = [
    BASE_DIR / "faiss_index",
    ROOT_DIR / "faiss_index",
]
DEFAULT_INDEX_PATH = "index.faiss"
DEFAULT_CHUNKS_PATH = "chunks.jsonl"
DEFAULT_EMBEDDINGS_PATH = "embeddings.npy"
DEFAULT_METADATA_PATH = "metadata.json"

DEFAULT_TEXT_MODEL = "BAAI/bge-base-en-v1.5"


@dataclass
class EncoderConfig:
    """Configuration for the text encoder."""

    model_name: str = DEFAULT_TEXT_MODEL
    pooling: str = "cls"  # {"cls", "mean"}
    max_length: int = 512
    normalize: bool = True


class TextEncoder:
    """Wrapper around Hugging Face encoder for text embedding."""

    def __init__(self, config: Optional[EncoderConfig] = None):
        self.config = config or EncoderConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("TextEncoder device:", self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModel.from_pretrained(self.config.model_name).to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size

    def encode(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        """Encode a list of texts into float32 numpy vectors."""
        # print(len(texts))
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype="float32")
        vectors: List[np.ndarray] = []

        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
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

            vectors.append(pooled.cpu().numpy().astype("float32"))

        return np.vstack(vectors)

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


def _find_existing(path_candidates: Iterable[Path]) -> Optional[Path]:
    for candidate in path_candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_path(filename: str, base_dirs: Sequence[Path]) -> Path:
    candidates = [base / filename for base in base_dirs]
    found = _find_existing(candidates)
    return found or candidates[0]


def load_chunks(path: Path) -> List[Dict]:
    """Load chunk metadata from JSONL."""
    chunks: List[Dict] = []
    if not path.exists():
        raise FileNotFoundError(f"Missing chunks file: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    if not chunks:
        raise ValueError("Chunks file is empty.")
    return chunks


class RAGRetriever:
    """Encode queries and retrieve from a FAISS index."""

    def __init__(
        self,
        encoder_config: Optional[EncoderConfig] = None,
        encoder: Optional[TextEncoder] = None,
        index_dirs: Sequence[Path] = DEFAULT_INDEX_DIRS,
        index_filename: str = DEFAULT_INDEX_PATH,
        chunks_filename: str = DEFAULT_CHUNKS_PATH,
        metadata_filename: str = DEFAULT_METADATA_PATH,
        result_type: str = "text",
        text_field: str = "text",
    ):
        self.encoder = encoder or TextEncoder(encoder_config or EncoderConfig())
        self.encoder_config = encoder_config or self.encoder.config
        self.result_type = result_type
        self.text_field = text_field
        self.index_path = _resolve_path(index_filename, index_dirs)
        self.chunks_path = _resolve_path(chunks_filename, index_dirs)
        self.metadata_path = _resolve_path(metadata_filename, index_dirs)

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        self.chunks = load_chunks(self.chunks_path)
        self.metadata = {}
        if self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self._validate()

    def _validate(self) -> None:
        if self.index.d != self.encoder.embedding_dim:
            raise ValueError(
                f"Index dimension {self.index.d} does not match encoder dimension {self.encoder.embedding_dim}"
            )

    def retrieve(self, query: str, top_k: int = 5, retrieval_mode: Optional[str] = None) -> List[Dict]:
        """Return scored chunks for a query."""
        if not query:
            return []
        query_vec = self.encoder.encode([query])
        distances, indices = self.index.search(query_vec, top_k)
        results: List[Dict] = []
        for rank in range(min(top_k, len(self.chunks))):
            idx = int(indices[0][rank])
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            text_value = chunk.get(self.text_field) or chunk.get("text") or ""
            results.append(
                {
                    "text": text_value,
                    "caption": chunk.get("caption", text_value),
                    "page_id": chunk.get("page_id"),
                    "source_url": chunk.get("source_url") or chunk.get("page_url"),
                    "image_id": chunk.get("image_id"),
                    "image_url": chunk.get("image_url"),
                    "caption_type": chunk.get("caption_type"),
                    "caption_model": chunk.get("caption_model"),
                    "score": float(distances[0][rank]),
                    "modality": self.result_type or chunk.get("modality", "text"),
                }
            )
        return results
