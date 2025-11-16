"""Evaluate text retrieval on the SciFact dataset."""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from datasets import load_dataset

# Ensure project root is on sys.path for local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from retriever.rag_retriever import EncoderConfig
from retriever.scifact_retriever import SciFactRetriever

OUTPUT_DIR = Path("evaluation_results")
DATA_DIR = Path("data/scifact")
CLAIMS_DIR = DATA_DIR / "scifact_claims_source"
POSITIVE_LABELS = {"SUPPORT", "SUPPORTS", "REFUTE", "REFUTES", "CONTRADICT"}


def ndcg_at_k(labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    dcg = 0.0
    for i, rel in enumerate(labels):
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    ideal_len = sum(labels)
    ideal_labels = [1] * min(ideal_len, len(labels))
    idcg = 0.0
    for i, rel in enumerate(ideal_labels):
        idcg += (2 ** rel - 1) / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    precisions = []
    hits = 0
    for i, rel in enumerate(labels, start=1):
        if rel:
            hits += 1
            precisions.append(hits / i)
    total_rel = sum(labels)
    if total_rel == 0:
        return 0.0
    return sum(precisions) / total_rel


def recall_at_k(labels: Sequence[int], total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    return sum(labels) / total_relevant


def relevance_labels(retrieved: Sequence[Dict], gold_ids: Sequence[str], top_k: int) -> List[int]:
    gold_set = set(str(g) for g in gold_ids if g is not None)
    labels: List[int] = []
    for idx in range(min(top_k, len(retrieved))):
        doc_id = str(retrieved[idx].get("doc_id"))
        labels.append(1 if doc_id in gold_set else 0)
    return labels


def _load_parquet_dir(directory: Path):
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


def load_scifact_claims(split: str = "validation") -> List[Dict]:
    """Load SciFact claims + gold doc ids from local parquet source."""
    directory = CLAIMS_DIR / split
    dataset = _load_parquet_dir(directory)
    claims: List[Dict] = []
    for row in dataset:
        claim_id = _first_present(row, ("claim_id", "id"))
        text = _first_present(row, ("claim", "text"))
        if claim_id is None or text is None:
            continue
        gold_docs: List[str] = []

        cited = row.get("cited_doc_ids") or []
        if not isinstance(cited, list):
            cited = list(cited)
        gold_docs.extend(str(doc_id) for doc_id in cited)

        evidences = row.get("evidences") or []
        for ev in evidences:
            if not isinstance(ev, dict):
                continue
            label = str(ev.get("label", "")).upper()
            doc_id = ev.get("doc_id")
            if doc_id is None:
                continue
            if not label or label in POSITIVE_LABELS:
                gold_docs.append(str(doc_id))

        gold_unique = sorted({doc for doc in gold_docs if doc is not None})
        claims.append({"id": str(claim_id), "claim": str(text), "gold_docs": gold_unique})
    if not claims:
        raise ValueError(f"No claims loaded from {directory}")
    return claims


def evaluate(top_k: int, limit: int | None) -> Dict:
    claims = load_scifact_claims()
    total = len(claims) if limit is None else min(limit, len(claims))
    print(f"Evaluating {total} SciFact claims (top_k={top_k})...")

    retriever = SciFactRetriever(encoder_config=EncoderConfig())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "scifact_results.jsonl"
    results_file = results_path.open("w", encoding="utf-8")

    agg = {"ndcg": 0.0, "map": 0.0, "recall": 0.0}

    for idx, sample in enumerate(claims):
        if idx >= total:
            break
        claim = sample["claim"]
        gold_docs = sample["gold_docs"]
        retrieved = retriever.retrieve(claim, top_k=top_k)
        labels = relevance_labels(retrieved, gold_docs, top_k)

        ndcg = ndcg_at_k(labels)
        ap = map_at_k(labels)
        rec = recall_at_k(labels, max(len(set(gold_docs)), 1))

        agg["ndcg"] += ndcg
        agg["map"] += ap
        agg["recall"] += rec

        record = {
            "id": sample["id"],
            "claim": claim,
            "gold_docs": gold_docs,
            "retrieved": retrieved,
            "labels": labels,
            "metrics": {"ndcg": ndcg, "map": ap, "recall": rec},
        }
        results_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{idx + 1}/{total}] NDCG={ndcg:.2f} MAP={ap:.2f} Recall={rec:.2f}")

    results_file.close()

    num = total or 1
    metrics = {
        "total": total,
        "ndcg": agg["ndcg"] / num,
        "map": agg["map"] / num,
        "recall@k": agg["recall"] / num,
        "top_k": top_k,
    }
    (OUTPUT_DIR / "scifact_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nEvaluation complete:")
    print(json.dumps(metrics, indent=2))
    print(f"Results written to {results_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on SciFact.")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args.top_k, args.limit)


if __name__ == "__main__":
    main()
