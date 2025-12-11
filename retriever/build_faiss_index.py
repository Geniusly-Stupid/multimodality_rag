"""Build a FAISS index from RAGAnything-parsed chunks (text/caption only)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

try:
    import faiss
except ImportError as exc:  # pragma: no cover - explicit runtime dependency guard
    raise SystemExit("Missing dependency 'faiss'. Install faiss-cpu/ faiss-gpu before running.") from exc

from parser.raganything_parser import RAGAnythingParser

# Fix OpenMP library conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DEFAULT_MODEL_NAME = "BAAI/bge-base-en-v1.5"
DEFAULT_POOLING = "cls"
DEFAULT_MAX_LENGTH = 512
DEFAULT_NORMALIZE = True
ENCODING_BATCH_SIZE = 16

FAISS_INDEX_TYPE = "flat"  # or "ivf"
FAISS_METRIC = "ip"  # "ip" or "l2"
FAISS_NLIST = 100


def _text_from_chunk(chunk: Dict[str, Any]) -> str:
    meta = chunk.get("metadata") or {}
    return (
        chunk.get("enriched_description")
        or chunk.get("text")
        or meta.get("caption")
        or meta.get("text")
        or ""
    )


class TextEncoder:
    """Thin wrapper around a Hugging Face encoder with configurable pooling."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        pooling: str = DEFAULT_POOLING,
        max_length: int = DEFAULT_MAX_LENGTH,
        normalize: bool = DEFAULT_NORMALIZE,
    ):
        self.model_name = model_name
        self.pooling = pooling
        self.max_length = max_length
        self.normalize = normalize
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
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
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**encoded)
                token_embeddings = outputs.last_hidden_state
                pooled = self._pool(token_embeddings, encoded["attention_mask"])
                if self.normalize:
                    pooled = F.normalize(pooled, p=2, dim=1)
                batch_arr = pooled.cpu().numpy().astype("float32")
                all_embeddings.append(batch_arr)
        return np.vstack(all_embeddings)

    def _pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooling = self.pooling.lower()
        if pooling == "cls":
            return token_embeddings[:, 0]
        if pooling == "mean":
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            summed = torch.sum(token_embeddings * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-6)
            return summed / counts
        raise ValueError(f"Unsupported pooling strategy: {self.pooling}")


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


def write_metadata(
    path: Path,
    *,
    num_chunks: int,
    embedding_dim: int,
    encoder_name: str,
    faiss_index_type: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "num_chunks": num_chunks,
        "embedding_dim": embedding_dim,
        "encoder_name": encoder_name,
        "faiss_index_type": faiss_index_type,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


async def build_index(
    doc_path: Path,
    output_dir: Path,
    encoder: TextEncoder,
    index_type: str,
    metric: str,
    nlist: int,
) -> None:
    parser = RAGAnythingParser()
    chunks = await parser.parse_and_enrich(doc_path)
    if not chunks:
        raise SystemExit("No chunks produced by RAGAnything.")

    texts = [_text_from_chunk(c) for c in chunks]
    print(f"Encoding {len(texts)} chunks with {encoder.model_name}...")
    embeddings = encoder.encode(texts, batch_size=ENCODING_BATCH_SIZE)

    index = build_faiss_index(embeddings, index_type=index_type, metric=metric, nlist=nlist)
    print(f"FAISS index built ({index.__class__.__name__}) with {index.ntotal} vectors.")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings)
    faiss.write_index(index, str(output_dir / "index.faiss"))
    save_chunks(output_dir / "chunks.jsonl", chunks)
    write_metadata(
        output_dir / "metadata.json",
        num_chunks=len(chunks),
        embedding_dim=embeddings.shape[1],
        encoder_name=encoder.model_name,
        faiss_index_type=index.__class__.__name__,
        extra={"source": str(doc_path)},
    )
    print(f"Saved FAISS resources to {output_dir.resolve()} (chunks={len(chunks)}).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index from a document parsed by RAGAnything.")
    parser.add_argument("--doc_path", type=str, required=True, help="Path to the document to index.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for saving FAISS assets.")
    parser.add_argument("--index_type", choices=["flat", "ivf"], default=FAISS_INDEX_TYPE, help="FAISS index type.")
    parser.add_argument("--metric", choices=["ip", "l2"], default=FAISS_METRIC, help="Similarity metric.")
    parser.add_argument("--nlist", type=int, default=FAISS_NLIST, help="Number of IVF clusters (when using IVF).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    doc_path = Path(args.doc_path)
    output_dir = Path(args.output_dir)
    encoder = TextEncoder()

    async def _run() -> None:
        await build_index(
            doc_path=doc_path,
            output_dir=output_dir,
            encoder=encoder,
            index_type=args.index_type,
            metric=args.metric,
            nlist=args.nlist,
        )

    asyncio.run(_run())


if __name__ == "__main__":
    main()

