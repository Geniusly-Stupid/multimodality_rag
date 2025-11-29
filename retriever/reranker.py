"""Cross-encoder reranker for text and caption evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass
class RerankerConfig:
    model_name: str = DEFAULT_RERANKER_MODEL
    max_length: int = 512


class CrossEncoderReranker:
    """Thin wrapper around a cross-encoder reranker (e.g., BGE reranker)."""

    def __init__(self, config: Optional[RerankerConfig] = None):
        self.config = config or RerankerConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score(self, query: str, docs: Sequence[str], batch_size: int = 8) -> List[float]:
        if not docs:
            return []
        scores: List[float] = []
        for start in range(0, len(docs), batch_size):
            batch_docs = docs[start : start + batch_size]
            pairs: List[Tuple[str, str]] = [(query, doc) for doc in batch_docs]
            encoded = self.tokenizer(
                pairs,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**encoded)
            logits = outputs.logits.view(-1)
            scores.extend(logits.detach().cpu().tolist())
        return scores

    def rerank(self, query: str, docs: Sequence[str], top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        scores = self.score(query, docs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked
