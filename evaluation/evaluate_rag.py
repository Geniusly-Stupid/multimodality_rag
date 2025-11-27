"""Evaluate RAG on google/frames-benchmark."""

from __future__ import annotations

import os

# Fix OpenMP library conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

from datasets import load_dataset

# Ensure project root is on sys.path for local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from generator.rag_generator import GeneratorConfig, RAGGenerator
from rag_pipeline import RAGPipeline
from retriever.rag_retriever import EncoderConfig, RAGRetriever

OUTPUT_DIR = Path("evaluation_results")


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split()) if text else ""


def exact_match(pred: str, ref: str) -> float:
    return float(normalize_text(pred) == normalize_text(ref))


def f1_score(pred: str, ref: str) -> float:
    pred_tokens = normalize_text(pred).split()
    ref_tokens = normalize_text(ref).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    ref_counts = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    common = 0
    for t in pred_tokens:
        if ref_counts.get(t, 0) > 0:
            common += 1
            ref_counts[t] -= 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def parse_links(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, (list, tuple)):
                return [str(v) for v in parsed]
        except Exception:
            pass
        stripped = stripped.strip("[]")
        return [v.strip().strip("'\"") for v in stripped.split(",") if v.strip()]
    return []


def relevance_labels(retrieved: Sequence[Dict], gold_links: Sequence[str], top_k: int) -> List[int]:
    """Binary relevance: 1 if chunk source_url contains any gold link substring."""
    gold_norm = [link.rstrip("/") for link in gold_links if link]
    labels: List[int] = []
    for idx in range(min(top_k, len(retrieved))):
        chunk = retrieved[idx]
        url = (chunk.get("source_url") or "").rstrip("/")
        is_rel = 0
        for gold in gold_norm:
            if gold and gold in url:
                is_rel = 1
                break
        labels.append(is_rel)
    return labels


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


def retrieval_recall(labels: Sequence[int]) -> float:
    return 1.0 if any(labels) else 0.0


def average_generation_length(predictions: Sequence[str]) -> float:
    lengths = [len(p.split()) for p in predictions if p]
    return sum(lengths) / len(lengths) if lengths else 0.0


def evaluate(top_k: int, limit: int | None) -> Dict:
    dataset = load_dataset("google/frames-benchmark")["test"]
    total = len(dataset) if limit is None else min(limit, len(dataset))
    print(f"Evaluating {total} samples (top_k={top_k})...")

    retriever = RAGRetriever(encoder_config=EncoderConfig())
    generator = RAGGenerator(GeneratorConfig())
    pipeline = RAGPipeline(retriever, generator)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "frames_rag_results.jsonl"
    results_file = results_path.open("w", encoding="utf-8")

    agg = {
        "em": 0.0,
        "f1": 0.0,
        "ndcg": 0.0,
        "map": 0.0,
        "retrieval_recall": 0.0,
    }
    preds: List[str] = []

    for idx, example in enumerate(dataset):
        if idx >= total:
            break
        prompt = example.get("prompt") or example.get("Prompt") or ""
        answer = example.get("answer") or example.get("Answer") or ""
        gold_links = parse_links(example.get("wiki_links"))

        output = pipeline.answer(prompt, top_k=top_k)
        labels = relevance_labels(output.retrieved_chunks, gold_links, top_k)

        em = exact_match(output.generated_answer, answer)
        f1 = f1_score(output.generated_answer, answer)
        ndcg = ndcg_at_k(labels)
        ap = map_at_k(labels)
        rec = retrieval_recall(labels)

        agg["em"] += em
        agg["f1"] += f1
        agg["ndcg"] += ndcg
        agg["map"] += ap
        agg["retrieval_recall"] += rec
        preds.append(output.generated_answer)

        record = {
            "id": idx,
            "prompt": prompt,
            "answer": answer,
            "prediction": output.generated_answer,
            "retrieved_chunks": output.retrieved_chunks,
            "labels": labels,
            "metrics": {"em": em, "f1": f1, "ndcg": ndcg, "map": ap, "recall": rec},
        }
        results_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{idx + 1}/{total}] EM={em:.1f} F1={f1:.2f} NDCG={ndcg:.2f} MAP={ap:.2f} Recall={rec:.1f}")

    results_file.close()

    num = total or 1
    metrics = {
        "total": total,
        "em": agg["em"] / num,
        "f1": agg["f1"] / num,
        "ndcg": agg["ndcg"] / num,
        "map": agg["map"] / num,
        "retrieval_recall@k": agg["retrieval_recall"] / num,
        "avg_generation_length": average_generation_length(preds),
        "top_k": top_k,
    }
    (OUTPUT_DIR / "frames_rag_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nEvaluation complete:")
    print(json.dumps(metrics, indent=2))
    print(f"Results written to {results_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG on google/frames-benchmark.")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of samples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args.top_k, args.limit)


if __name__ == "__main__":
    main()
