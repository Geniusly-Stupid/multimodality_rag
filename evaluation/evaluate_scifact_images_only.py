"""Evaluate image-only retrieval on the SciFact dataset (no text merging)."""

from __future__ import annotations

import os

# Fix OpenMP library conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
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

from retriever.multimodal_retriever import MultimodalRetriever, MultimodalRetrieverConfig
from retriever.image_encoder import ImageEncoderConfig

OUTPUT_DIR = Path("evaluation_results")
DATA_DIR = Path("data/scifact")
CLAIMS_DIR = DATA_DIR / "scifact_claims_source"
POSITIVE_LABELS = {"SUPPORT", "SUPPORTS", "REFUTE", "REFUTES", "CONTRADICT"}

# Check image index path
IMAGE_INDEX_DIR = Path("retriever/faiss_index/images")
IMAGE_INDEX_PATH = IMAGE_INDEX_DIR / "index.faiss"
IMAGE_METADATA_PATH = IMAGE_INDEX_DIR / "images.jsonl"


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


def evaluate_images_only(top_k: int, limit: int | None) -> Dict:
    """Evaluate image-only retrieval (no text merging)."""
    claims = load_scifact_claims()
    total = len(claims) if limit is None else min(limit, len(claims))
    print(f"Evaluating {total} SciFact claims with IMAGE-ONLY retrieval (top_k={top_k})...")
    
    # Check if image index files exist
    print(f"\nChecking image index...")
    print(f"  Index file: {IMAGE_INDEX_PATH}")
    print(f"  Exists: {IMAGE_INDEX_PATH.exists()}")
    print(f"  Metadata file: {IMAGE_METADATA_PATH}")
    print(f"  Exists: {IMAGE_METADATA_PATH.exists()}")
    if not IMAGE_INDEX_PATH.exists():
        print(f"\n⚠ ERROR: Image index file not found at {IMAGE_INDEX_PATH}")
        print(f"  Please run: python retriever/build_image_index.py")
        return {}
    
    # Initialize image-only retriever
    print(f"\nInitializing image-only retriever...")
    config = MultimodalRetrieverConfig(
        use_text_retrieval=False,  # Disable text retrieval
        use_image_retrieval=True,  # Only image retrieval
        text_weight=0.0,
        image_weight=1.0,
    )
    retriever = MultimodalRetriever(config=config)
    
    # Check if image index was loaded successfully
    if retriever.image_index is None:
        print("⚠ WARNING: Image index is not loaded!")
        print("  This means image retrieval will return 0 results.")
        print("  To fix this, run: python retriever/build_image_index.py")
        return {}
    elif len(retriever.image_metadata) == 0:
        print("⚠ WARNING: Image metadata is empty!")
        print("  Image index exists but has no images.")
        return {}
    else:
        print(f"✓ Image index loaded successfully: {len(retriever.image_metadata)} images available")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "scifact_images_only_results.jsonl"
    results_file = results_path.open("w", encoding="utf-8")
    
    # Statistics
    total_images_retrieved = 0
    queries_with_images = 0
    
    print(f"\nStarting evaluation...")
    print("=" * 80)
    
    for idx, sample in enumerate(claims):
        if idx >= total:
            break
        claim = sample["claim"]
        gold_docs = sample["gold_docs"]
        
        # Retrieve images only (no text merging)
        retrieved = retriever.retrieve(claim, top_k=top_k, return_modality="image")
        
        # All results should be images
        image_results = [r for r in retrieved if r.get("modality") == "image"]
        total_images_retrieved += len(image_results)
        if len(image_results) > 0:
            queries_with_images += 1
        
        # Print first few queries with details
        if idx < 5:
            print(f"\nQuery {idx + 1}: {claim[:80]}...")
            print(f"  Retrieved {len(image_results)} images:")
            for i, img in enumerate(image_results[:3], 1):
                print(f"    {i}. Image ID: {img.get('image_id', 'N/A')}")
                print(f"       Caption: {img.get('caption', 'N/A')[:60]}...")
                print(f"       Score: {img.get('score', 0):.6f}")
        
        # Save detailed results
        record = {
            "id": sample["id"],
            "claim": claim,
            "gold_docs": gold_docs,
            "retrieved_images": image_results,  # All image results with scores
            "num_images": len(image_results),
            "top_scores": [img.get("score", 0) for img in image_results[:5]],  # Top 5 scores
        }
        results_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        if (idx + 1) % 50 == 0:
            print(f"[{idx + 1}/{total}] Processed...")
    
    results_file.close()
    
    # Summary statistics
    metrics = {
        "total_queries": total,
        "top_k": top_k,
        "total_images_retrieved": total_images_retrieved,
        "queries_with_images": queries_with_images,
        "queries_without_images": total - queries_with_images,
        "avg_images_per_query": total_images_retrieved / total if total > 0 else 0.0,
        "pct_queries_with_images": (queries_with_images / total * 100) if total > 0 else 0.0,
    }
    
    (OUTPUT_DIR / "scifact_images_only_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print(json.dumps(metrics, indent=2))
    print(f"\nResults written to: {results_path}")
    print(f"Metrics written to: {OUTPUT_DIR / 'scifact_images_only_metrics.json'}")
    
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate image-only retrieval on SciFact (no text merging)."
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of images to retrieve per query")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries to evaluate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_images_only(args.top_k, args.limit)


if __name__ == "__main__":
    main()

