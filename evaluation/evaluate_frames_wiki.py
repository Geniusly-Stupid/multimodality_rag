"""Evaluate RAG on google/frames-benchmark."""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence
from urllib.parse import urlparse, unquote
import re

from datasets import load_dataset

# Ensure project root is on sys.path for local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from generator.rag_generator import GeneratorConfig, RAGGenerator
from rag_pipeline import RAGPipeline
from retriever.build_faiss_index import load_chunks_from_folder
from retriever.retriever_adapter import FaissVectorStore, RetrieverAdapter
from retriever.rag_retriever import EncoderConfig, TextEncoder

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


def link_tail(link: str) -> str:
    try:
        parsed = urlparse(str(link))
        path = parsed.path or ""
        tail = path.rstrip("/").split("/")[-1]
        return unquote(tail).lower()
    except Exception:
        return str(link).strip().lower()


def chunk_source_tail(chunk: Dict) -> str:
    meta = chunk.get("metadata") or {}
    for candidate in (
        meta.get("source_name"),
        meta.get("page_id"),
        chunk.get("source_url"),
        meta.get("source_url"),
        chunk.get("id"),
    ):
        if candidate:
            return link_tail(candidate)
    return ""


def strip_hash(pid: str) -> str:
    """
    Remove trailing _[0-9a-f]{8} hash from page_id or source_name.
    Example: 'punxsutawney_phil_40efb9a9' -> 'punxsutawney_phil'
    """
    if not pid:
        return ""
    pid = str(pid).lower()
    return re.sub(r"_[0-9a-f]{8}$", "", pid)

def relevance_labels(retrieved: Sequence[Dict], gold_links: Sequence[str], top_k: int) -> List[int]:
    """
    Binary relevance: 1 if retrieved chunk matches expected page ids or wiki link tails.

    Primary alignment uses page_ids from samples.jsonl (stem like 'punxsutawney_phil_40efb9a9'),
    which should match txt filename stems stored in metadata['source_name'].
    Fallback uses wiki link tails if page_ids are absent.
    """
    gold_tails = {link_tail(link) for link in gold_links if link}
    labels: List[int] = []
    for idx in range(min(top_k, len(retrieved))):
        chunk = retrieved[idx]
        source_id = (chunk.get("metadata") or {}).get("source_name") or chunk.get("page_id")

        # Standardize (remove trailing hash)
        stripped_sid = strip_hash(source_id)

        print("source:", source_id, "→", stripped_sid)
        print("gold:", gold_tails)
        
        labels.append(1 if stripped_sid and stripped_sid in gold_tails else 0)
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


def evaluate(args) -> Dict:
    dataset = load_dataset("google/frames-benchmark")["test"]
    total = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    print(f"Evaluating {total} samples (top_k={args.top_k})...")

    chunks = load_chunks_from_folder(args.doc_path)
    print("Loaded chunks:", len(chunks))

    text_encoder = TextEncoder(EncoderConfig())
    vector_store = FaissVectorStore(metric="ip")
    retriever = RetrieverAdapter(text_encoder=text_encoder, vector_store=vector_store)
    retriever.build_index(chunks)
    print("Index built successfully.")
    print("Vector store size:", retriever.vector_store.index)

    gen_config = GeneratorConfig(
        model_name=args.generator_model or GeneratorConfig().model_name,
        max_new_tokens=args.max_new_tokens,
        cache_dir=args.cache_dir,
        use_remote=args.use_remote,
        remote_model_name=args.remote_model,
        remote_api_base=args.remote_api_base,
        remote_api_key_env=args.remote_api_key_env,
        remote_api_key=args.remote_api_key,
        use_stream=args.remote_stream,
    )
    generator = RAGGenerator(gen_config)
    print("Generator initialized successfully.")

    pipeline = RAGPipeline(retriever, generator)
    print("Pipeline initialized successfully.")

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
        # print(example)
        prompt = example.get("prompt") or example.get("Prompt") or ""
        answer = example.get("answer") or example.get("Answer") or ""
        gold_links = parse_links(example.get("wiki_links"))

        output = pipeline.answer(prompt, top_k=args.top_k)
        labels = relevance_labels(output.retrieved_chunks, gold_links, args.top_k)
        print("labels:", labels)

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
        "top_k": args.top_k,
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
    parser.add_argument("--doc_path", type=str, required=True, help="Folder containing parsed JSON chunks or raw txt files for retrieval.")
    parser.add_argument("--use_remote", action="store_true", help="Use remote LLM API instead of local HF model.")
    parser.add_argument("--remote_model", type=str, default="", help="Remote model name for API calls.")
    parser.add_argument("--remote_api_base", type=str, default="", help="Remote API base URL.")
    parser.add_argument("--remote_api_key_env", type=str, default="", help="Env var name that holds the remote API key.")
    parser.add_argument("--remote_api_key", type=str, default="", help="Direct remote API key (overrides env lookup).")
    parser.add_argument("--remote_stream", action="store_true", help="Enable streaming responses from remote LLM (remote mode only).")
    parser.add_argument("--generator_model", type=str, default=None, help="Generator model name.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens for generation.")
    parser.add_argument("--cache_dir", type=str, default="D:/huggingface_cache", help="Directory for caching models/tokenizers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
