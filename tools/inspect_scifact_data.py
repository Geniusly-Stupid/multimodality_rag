"""
Quick inspection utility for the locally cached SciFact (BigBio) parquet files.

Example:
    python inspect_scifact_data.py --split validation --samples 3
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from datasets import load_dataset


DATA_DIR = Path(__file__).parent / "data"
POSITIVE_LABELS = {"SUPPORT", "SUPPORTS", "REFUTE", "REFUTES", "CONTRADICT"}


def load_parquet_dir(directory: Path):
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {directory}")
    dataset = load_dataset("parquet", data_files={"data": [str(fp) for fp in files]})
    return dataset["data"]


def first_present(record: dict, keys: Sequence[str]):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def load_corpus() -> Dict[str, str]:
    directory = DATA_DIR / "scifact_corpus_source" / "train"
    dataset = load_parquet_dir(directory)
    corpus = {}
    for row in dataset:
        doc_id = first_present(row, ("doc_id", "document_id", "id"))
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
        corpus[str(doc_id)] = text if text else title
    return corpus


def _extract_relevance(row, valid_doc_ids: set[str], label_counts: collections.Counter):
    cited = row.get("cited_doc_ids") or []
    if not isinstance(cited, list):
        cited = list(cited)
    relevant = {str(doc_id) for doc_id in cited if str(doc_id) in valid_doc_ids}

    evidences = row.get("evidences") or []
    for ev in evidences:
        if not isinstance(ev, dict):
            continue
        label = str(ev.get("label", "")).upper()
        if label:
            label_counts[label] += 1
        doc_id = ev.get("doc_id")
        if doc_id is None:
            continue
        doc_id_str = str(doc_id)
        if doc_id_str not in valid_doc_ids:
            continue
        if not label or label in POSITIVE_LABELS:
            relevant.add(doc_id_str)
    return sorted(relevant)


def load_claims(split: str, corpus: Dict[str, str]):
    directory = DATA_DIR / "scifact_claims_source" / split
    dataset = load_parquet_dir(directory)
    valid_doc_ids = set(corpus.keys())
    label_counts = collections.Counter()
    claims = []
    for row in dataset:
        claim_id = first_present(row, ("claim_id", "id"))
        text = first_present(row, ("claim", "text"))
        if claim_id is None or text is None:
            continue
        relevant = _extract_relevance(row, valid_doc_ids, label_counts)
        claims.append(
            {
                "claim_id": str(claim_id),
                "text": str(text),
                "relevant": relevant,
                "evidences": row.get("evidences") or [],
                "cited_doc_ids": row.get("cited_doc_ids") or [],
            }
        )
    return claims, label_counts


def describe_relevance(relevant: Dict[str, List[str]]):
    lengths = [len(docs) for docs in relevant.values()]
    if not lengths:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / len(lengths),
    }


def sample_items(items: Iterable, n: int):
    items = list(items)
    return items[:n]


def main():
    parser = argparse.ArgumentParser(description="Inspect SciFact parquet data.")
    parser.add_argument("--split", type=str, default="validation", help="Split to inspect.")
    parser.add_argument("--samples", type=int, default=3, help="Samples to print for sanity check.")
    args = parser.parse_args()

    print(f"Inspecting SciFact split '{args.split}'...")
    corpus = load_corpus()
    claims, label_counts = load_claims(args.split, corpus)

    print(f"Documents: {len(corpus)}")
    print(f"Claims   : {len(claims)}")
    print("Evidence label counts:", dict(label_counts))

    rel_stats = describe_relevance({c["claim_id"]: c["relevant"] for c in claims})
    zero_rel_claims = sum(1 for claim in claims if not claim["relevant"])
    print(
        f"Claims with positive labels: {rel_stats['count']} "
        f"(min={rel_stats['min']}, max={rel_stats['max']}, mean={rel_stats['mean']:.2f})"
    )
    print(f"Claims without positives: {zero_rel_claims}")

    print("\nSample claims:")
    for claim in sample_items(claims, args.samples):
        print(f"- Claim {claim['claim_id']}: {claim['text']}")
        print(f"  Cited doc IDs   : {', '.join(map(str, claim['cited_doc_ids'])) or 'None'}")
        print(f"  Relevant doc IDs: {', '.join(claim['relevant']) or 'None'}")
        if claim["evidences"]:
            evid_snippets = [
                f"(doc {ev.get('doc_id')}, label={ev.get('label')})"
                for ev in claim["evidences"][:3]
            ]
            more = " ..." if len(claim["evidences"]) > 3 else ""
            print("  Evidences       :", ", ".join(evid_snippets) + more)

    print("\nSample documents:")
    for doc_id, text in sample_items(corpus.items(), args.samples):
        snippet = text[:200].replace("\n", " ")
        ellipsis = "..." if len(text) > 200 else ""
        print(f"- Doc {doc_id}: {snippet}{ellipsis}")


if __name__ == "__main__":
    main()
