"""
Evaluate BGE text embeddings, BM25, and a random baseline on the locally cached
SciFact dataset. Relevant documents per claim are taken from the BigBio claims
parquet (`cited_doc_ids` + positive evidence labels).
"""

from __future__ import annotations

import argparse
import collections
import math
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


DATA_DIR = Path(__file__).parent / "data"
TOKEN_PATTERN = re.compile(r"\b\w+\b")
TEXT_MODEL_NAME = "BAAI/bge-base-en-v1.5"
POSITIVE_LABELS = {"SUPPORT", "SUPPORTS", "REFUTE", "REFUTES", "CONTRADICT"}


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


class SimpleBM25:
    def __init__(self, documents: Dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = len(documents)
        self.doc_term_freqs = {}
        self.doc_lengths = {}
        self.doc_freqs = collections.Counter()
        total_length = 0
        for doc_id, text in documents.items():
            tokens = tokenize(text)
            term_freqs = collections.Counter(tokens)
            self.doc_term_freqs[doc_id] = term_freqs
            length = len(tokens)
            self.doc_lengths[doc_id] = length
            total_length += length
            self.doc_freqs.update(term_freqs.keys())
        self.avgdl = total_length / self.N if self.N else 0.0
        self.idf = {
            term: math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
            for term, df in self.doc_freqs.items()
        }

    def score(self, query: str) -> Dict[str, float]:
        tokens = tokenize(query)
        scores = collections.defaultdict(float)
        for term in tokens:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for doc_id, tf_map in self.doc_term_freqs.items():
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * (self.doc_lengths[doc_id] / self.avgdl)
                )
                scores[doc_id] += idf * numerator / denominator
        return dict(scores)


class TextEncoder:
    """Lightweight wrapper around the BGE text encoder."""

    def __init__(self, model_name: str = TEXT_MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("Text encoder loaded.")

    def encode(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        embeddings = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch, padding=True, truncation=True, return_tensors="pt"
                ).to(self.device)
                outputs = self.model(**encoded)
                sentence_embeddings = outputs[0][:, 0]
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                embeddings.append(sentence_embeddings.cpu())
        return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0)


def dcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    return sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:k], start=1)
        if doc_id in relevant_set
    )


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_list = list(relevant)
    if not relevant_list:
        return 0.0
    best_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_list), k)))
    if best_dcg == 0.0:
        return 0.0
    return dcg_at_k(ranked, relevant_list, k) / best_dcg


def average_precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for idx, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant_set:
            hits += 1
            precision_sum += hits / idx
    return precision_sum / min(len(relevant_set), k)


def evaluate_rankings(
    rankings: List[List[str]],
    queries: Sequence[Dict[str, Sequence[str]]],
    k: int,
) -> Dict[str, float]:
    ndcgs = []
    maps = []
    for ranking, query in zip(rankings, queries):
        relevant = query["relevant_doc_ids"]
        ndcgs.append(ndcg_at_k(ranking, relevant, k))
        maps.append(average_precision_at_k(ranking, relevant, k))
    return {
        "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "map": float(np.mean(maps)) if maps else 0.0,
    }


def load_parquet_dir(directory: Path):
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {directory}")
    dataset = load_dataset("parquet", data_files={"data": [str(fp) for fp in files]})
    return dataset["data"]


def _first_present(record: dict, keys: Sequence[str]):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def build_corpus() -> Dict[str, str]:
    corpus_dir = DATA_DIR / "scifact_corpus_source" / "train"
    dataset = load_parquet_dir(corpus_dir)
    docs = {}
    for rec in dataset:
        record = dict(rec)
        doc_id = _first_present(record, ("doc_id", "document_id", "id"))
        if doc_id is None:
            continue
        doc_id = str(doc_id)
        title = record.get("title") or ""
        abstract = record.get("abstract") or []
        if isinstance(abstract, str):
            abstract_text = abstract
        elif isinstance(abstract, (list, tuple)):
            abstract_text = " ".join(str(part) for part in abstract)
        else:
            abstract_text = str(abstract)
        text = (title + ". " + abstract_text).strip()
        docs[doc_id] = text if text else title
    return docs


def _extract_relevance(
    record: dict, valid_doc_ids: set[str], label_counter: collections.Counter
) -> List[str]:
    relevant_docs: set[str] = set()

    cited_doc_ids = record.get("cited_doc_ids") or []
    if not isinstance(cited_doc_ids, list):
        cited_doc_ids = list(cited_doc_ids)
    for doc_id in cited_doc_ids:
        doc_id_str = str(doc_id)
        if doc_id_str in valid_doc_ids:
            relevant_docs.add(doc_id_str)

    evidences = record.get("evidences") or []
    for evidence in evidences:
        if not isinstance(evidence, dict):
            continue
        label = str(evidence.get("label", "")).upper()
        if label:
            label_counter[label] += 1
        doc_id = evidence.get("doc_id")
        if doc_id is None:
            continue
        doc_id_str = str(doc_id)
        if doc_id_str not in valid_doc_ids:
            continue
        if not label or label in POSITIVE_LABELS:
            relevant_docs.add(doc_id_str)

    return sorted(relevant_docs)


def build_queries(
    split: str, valid_doc_ids: set[str]
) -> Tuple[List[Dict[str, Sequence[str]]], collections.Counter]:
    claims_dir = DATA_DIR / "scifact_claims_source" / split
    dataset = load_parquet_dir(claims_dir)
    label_counter: collections.Counter = collections.Counter()
    queries: List[Dict[str, Sequence[str]]] = []
    for rec in dataset:
        record = dict(rec)
        claim_id = _first_present(record, ("claim_id", "id"))
        text = _first_present(record, ("claim", "text"))
        if claim_id is None or text is None:
            continue
        claim_id = str(claim_id)
        text = str(text)

        relevant_docs = _extract_relevance(record, valid_doc_ids, label_counter)
        queries.append(
            {
                "query_id": claim_id,
                "text": text,
                "relevant_doc_ids": relevant_docs,
            }
        )
    return queries, label_counter


def rank_with_encoder(
    encoder: TextEncoder,
    doc_ids: List[str],
    doc_texts: List[str],
    queries: Sequence[Dict[str, str]],
    batch_size: int,
) -> List[List[str]]:
    doc_embeddings = encoder.encode(doc_texts, batch_size=batch_size)
    rankings: List[List[str]] = []
    for query in queries:
        q_emb = encoder.encode([query["text"]])
        if q_emb.numel() == 0:
            rankings.append(doc_ids.copy())
            continue
        scores = torch.matmul(q_emb, doc_embeddings.T).numpy().flatten()
        sorted_idx = np.argsort(-scores)
        rankings.append([doc_ids[i] for i in sorted_idx])
    return rankings


def rank_with_bm25(
    bm25: SimpleBM25,
    doc_ids: List[str],
    queries: Sequence[Dict[str, str]],
) -> List[List[str]]:
    rankings: List[List[str]] = []
    for query in queries:
        scores = bm25.score(query["text"])
        sorted_ids = sorted(doc_ids, key=lambda doc_id: scores.get(doc_id, 0.0), reverse=True)
        rankings.append(sorted_ids)
    return rankings


def rank_with_random(doc_ids: List[str], num_queries: int, seed: int) -> List[List[str]]:
    rng = random.Random(seed)
    rankings: List[List[str]] = []
    for _ in range(num_queries):
        shuffled = doc_ids.copy()
        rng.shuffle(shuffled)
        rankings.append(shuffled)
    return rankings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SciFact retrieval with text-only BGE embeddings."
    )
    parser.add_argument("--split", type=str, default="validation", help="Query split to evaluate.")
    parser.add_argument("--top-k", type=int, default=10, help="Cutoff for NDCG@K / MAP@K.")
    parser.add_argument("--seed", type=int, default=42, help="Random baseline seed.")
    parser.add_argument(
        "--min-relevant",
        type=int,
        default=0,
        help="Minimum number of relevant documents required per query.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for encoding documents with the text encoder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    corpus = build_corpus()
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[doc_id] for doc_id in doc_ids]
    valid_doc_ids = set(doc_ids)

    queries, label_counts = build_queries(args.split, valid_doc_ids)
    if label_counts:
        print(
            "Evidence label counts: "
            + ", ".join(f"{label}={count}" for label, count in label_counts.items())
        )

    zero_rel = sum(1 for q in queries if len(q["relevant_doc_ids"]) == 0)
    if zero_rel:
        print(f"{zero_rel} queries have zero relevant documents (after filtering to corpus docs).")

    if args.min_relevant > 0:
        before = len(queries)
        queries = [
            q for q in queries if len(q["relevant_doc_ids"]) >= args.min_relevant
        ]
        removed = before - len(queries)
        if removed:
            print(f"Filtered out {removed} queries with < {args.min_relevant} relevant docs.")

    print(f"Evaluating on split '{args.split}' with {len(doc_ids)} documents and {len(queries)} queries.")
    if not queries:
        print("No queries left after filtering; nothing to evaluate.")
        print(
            "Try lowering --min-relevant or switching to a different split (e.g., '--split train')."
        )
        return

    bm25 = SimpleBM25({doc_id: text for doc_id, text in zip(doc_ids, doc_texts)})
    random_rankings = rank_with_random(doc_ids, len(queries), args.seed)
    random_metrics = evaluate_rankings(random_rankings, queries, args.top_k)

    bm25_rankings = rank_with_bm25(bm25, doc_ids, queries)
    bm25_metrics = evaluate_rankings(bm25_rankings, queries, args.top_k)

    encoder = TextEncoder()
    encoder_rankings = rank_with_encoder(
        encoder,
        doc_ids,
        doc_texts,
        queries,
        batch_size=args.batch_size,
    )
    encoder_metrics = evaluate_rankings(encoder_rankings, queries, args.top_k)

    print(f"\nMetrics @ {args.top_k}:")
    print(f"Random    -> NDCG: {random_metrics['ndcg']:.4f}, MAP: {random_metrics['map']:.4f}")
    print(f"BM25      -> NDCG: {bm25_metrics['ndcg']:.4f}, MAP: {bm25_metrics['map']:.4f}")
    print(f"BGE Text  -> NDCG: {encoder_metrics['ndcg']:.4f}, MAP: {encoder_metrics['map']:.4f}")

    for idx, query in enumerate(queries[:5]):
        print(f"\nQuery {query['query_id']}: {query['text']}")
        print("  Relevant:", ", ".join(query["relevant_doc_ids"]) or "None")
        print("  Top 5 BM25:", ", ".join(bm25_rankings[idx][:5]))
        print("  Top 5 BGE :", ", ".join(encoder_rankings[idx][:5]))


if __name__ == "__main__":
    main()
