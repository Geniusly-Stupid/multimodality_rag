"""SciFact text retriever backed by the shared TextEncoder."""

from __future__ import annotations

import json
import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import faiss
import numpy as np
from datasets import load_dataset

from .rag_retriever import EncoderConfig, TextEncoder

DATA_DIR = Path("data/scifact")
CORPUS_DIR = DATA_DIR / "scifact_corpus_source" / "train"
DEFAULT_TOP_K = 5


@dataclass
class CorpusEntry:
    doc_id: str
    text: str
    url: Optional[str] = None


def _load_parquet_dir(directory: Path):
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {directory}")
    dataset = load_dataset("parquet", data_files={"data": [str(fp) for fp in files]})
    return dataset["data"]


def load_scifact_corpus(directory: Path = CORPUS_DIR) -> List[CorpusEntry]:
    """Load SciFact corpus from local parquet files (bigbio)."""
    dataset = _load_parquet_dir(directory)
    entries: List[CorpusEntry] = []
    for row in dataset:
        doc_id = row.get("doc_id") or row.get("document_id") or row.get("id")
        if doc_id is None:
            continue
        title = row.get("title") or ""
        abstract = row.get("abstract") or []
        if isinstance(abstract, str):
            abstract_text = abstract
        elif isinstance(abstract, (list, tuple)):
            abstract_text = " ".join(str(part) for part in abstract)
        else:
            abstract_text = str(abstract)
        text = (title + ". " + abstract_text).strip()
        entries.append(CorpusEntry(doc_id=str(doc_id), text=text or title, url=row.get("url")))
    if not entries:
        raise ValueError("SciFact corpus is empty.")
    return entries


class SciFactRetriever:
    """Encode SciFact abstracts into a FAISS index and retrieve claims."""

    def __init__(self, encoder_config: Optional[EncoderConfig] = None, corpus_dir: Path = CORPUS_DIR):
        self.encoder = TextEncoder(encoder_config or EncoderConfig())
        self.corpus = load_scifact_corpus(corpus_dir)
        texts = [entry.text for entry in self.corpus]
        embeddings = self.encoder.encode(texts, batch_size=32)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, claim: str, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
        """Retrieve top-k documents for a claim."""
        if not claim:
            return []
        query_vec = self.encoder.encode([claim])
        distances, indices = self.index.search(query_vec, top_k)
        results: List[Dict] = []
        for rank in range(top_k):
            idx = int(indices[0][rank])
            if idx < 0 or idx >= len(self.corpus):
                continue
            entry = self.corpus[idx]
            results.append(
                {
                    "doc_id": entry.doc_id,
                    "text": entry.text,
                    "url": entry.url,
                    "score": float(distances[0][rank]),
                }
            )
        return results
