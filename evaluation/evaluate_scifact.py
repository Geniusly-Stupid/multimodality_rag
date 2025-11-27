"""Evaluate text retrieval on the SciFact dataset."""

from __future__ import annotations

import os

# Fix OpenMP library conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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
from retriever.multimodal_retriever import MultimodalRetriever, MultimodalRetrieverConfig
from retriever.image_encoder import ImageEncoderConfig

# Check image index path
IMAGE_INDEX_DIR = Path("retriever/faiss_index/images")
IMAGE_INDEX_PATH = IMAGE_INDEX_DIR / "index.faiss"
IMAGE_METADATA_PATH = IMAGE_INDEX_DIR / "images.jsonl"

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
    """Compute relevance labels for retrieved items.
    
    For text items, checks doc_id against gold_ids.
    For image items, always returns 0 (images are not in SciFact gold standard).
    """
    gold_set = set(str(g) for g in gold_ids if g is not None)
    labels: List[int] = []
    for idx in range(min(top_k, len(retrieved))):
        item = retrieved[idx]
        modality = item.get("modality", "text")
        
        # For text items, check doc_id
        if modality == "text":
            doc_id = str(item.get("doc_id", ""))
            labels.append(1 if doc_id in gold_set else 0)
        else:
            # For images, they're not in SciFact gold standard, so always 0
            # (or you could implement image-based relevance if needed)
            labels.append(0)
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


def evaluate(top_k: int, limit: int | None, use_multimodal: bool = True, text_weight: float = 0.6, image_weight: float = 0.4) -> Dict:
    claims = load_scifact_claims()
    total = len(claims) if limit is None else min(limit, len(claims))
    print(f"Evaluating {total} SciFact claims (top_k={top_k})...")
    print(f"Multimodal retrieval: {use_multimodal}")
    if use_multimodal:
        print(f"Text weight: {text_weight}, Image weight: {image_weight}")
        # Check if image index files exist
        print(f"\nChecking image index...")
        print(f"  Index file: {IMAGE_INDEX_PATH}")
        print(f"  Exists: {IMAGE_INDEX_PATH.exists()}")
        print(f"  Metadata file: {IMAGE_METADATA_PATH}")
        print(f"  Exists: {IMAGE_METADATA_PATH.exists()}")
        if not IMAGE_INDEX_PATH.exists():
            print(f"\n⚠ ERROR: Image index file not found at {IMAGE_INDEX_PATH}")
            print(f"  Please run: python retriever/build_image_index.py")
            print(f"  This will create the image index needed for multimodal retrieval.")

    # Create a wrapper class to make SciFactRetriever compatible with MultimodalRetriever
    class SciFactRetrieverWrapper:
        """Wrapper to make SciFactRetriever compatible with MultimodalRetriever interface."""
        def __init__(self, scifact_retriever: SciFactRetriever):
            self.scifact_retriever = scifact_retriever
        
        def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
            """Retrieve using SciFactRetriever and ensure results have correct format."""
            results = self.scifact_retriever.retrieve(query, top_k=top_k)
            # Ensure all results have modality field
            for result in results:
                if "modality" not in result:
                    result["modality"] = "text"
            return results
    
    # Initialize retriever
    if use_multimodal:
        # Create SciFact retriever for text
        scifact_retriever = SciFactRetriever(encoder_config=EncoderConfig())
        text_retriever_wrapper = SciFactRetrieverWrapper(scifact_retriever)
        
        # Create multimodal retriever
        config = MultimodalRetrieverConfig(
            use_text_retrieval=True,
            use_image_retrieval=True,
            text_weight=text_weight,
            image_weight=image_weight,
        )
        retriever = MultimodalRetriever(
            config=config,
            text_retriever=text_retriever_wrapper
        )
        
        # Check if image index was loaded successfully
        if retriever.image_index is None:
            print("⚠ WARNING: Image index is not loaded!")
            print("  This means image retrieval will return 0 results.")
            print("  To fix this, run: python retriever/build_image_index.py")
        elif len(retriever.image_metadata) == 0:
            print("⚠ WARNING: Image metadata is empty!")
            print("  Image index exists but has no images.")
        else:
            print(f"✓ Image index loaded successfully: {len(retriever.image_metadata)} images available")
    else:
        # Use only text retrieval (original behavior)
        retriever = SciFactRetriever(encoder_config=EncoderConfig())
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "scifact_results.jsonl"
    results_file = results_path.open("w", encoding="utf-8")

    agg = {"ndcg": 0.0, "map": 0.0, "recall": 0.0}
    agg_text = {"ndcg": 0.0, "map": 0.0, "recall": 0.0}  # Text-only metrics
    agg_image = {"count": 0}  # Count of image results

    for idx, sample in enumerate(claims):
        if idx >= total:
            break
        claim = sample["claim"]
        gold_docs = sample["gold_docs"]
        retrieved = retriever.retrieve(claim, top_k=top_k)
        
        # Also get raw image results (before merging) to see all image scores
        raw_image_results_all = []
        if use_multimodal and hasattr(retriever, '_retrieve_images') and retriever.image_index is not None:
            try:
                raw_image_results_all = retriever._retrieve_images(claim, top_k=top_k)
            except:
                pass
        
        labels = relevance_labels(retrieved, gold_docs, top_k)

        # Separate text and image results for analysis
        text_results = [r for r in retrieved if r.get("modality") == "text"]
        image_results = [r for r in retrieved if r.get("modality") == "image"]
        
        # Debug: Print first query's retrieval details
        if idx == 0 and use_multimodal:
            print(f"\n[DEBUG] First query retrieval details:")
            print(f"  Query: {claim[:100]}...")
            print(f"  Total retrieved: {len(retrieved)}")
            print(f"  Text results: {len(text_results)}")
            print(f"  Image results: {len(image_results)}")
            if hasattr(retriever, 'image_index') and retriever.image_index is not None:
                print(f"  Image index status: Loaded ({retriever.image_index.ntotal} vectors)")
            else:
                print(f"  Image index status: NOT LOADED")
            
            # Check what happened during retrieval - test image retrieval directly
            if hasattr(retriever, '_retrieve_images'):
                try:
                    raw_image_results = retriever._retrieve_images(claim, top_k=top_k)
                    print(f"  [DEBUG] Raw image retrieval returned: {len(raw_image_results)} images")
                    if raw_image_results:
                        print(f"  [DEBUG] Top image scores (before weighting):")
                        for i, img in enumerate(raw_image_results[:3]):
                            print(f"    Image {i+1}: score={img.get('score', 0):.4f}, weighted={img.get('score', 0) * image_weight:.4f}")
                    else:
                        print(f"  [DEBUG] No images retrieved - checking conditions...")
                        print(f"    - image_index is None: {retriever.image_index is None}")
                        print(f"    - image_metadata empty: {len(retriever.image_metadata) == 0}")
                except Exception as e:
                    print(f"  [DEBUG] Error testing image retrieval: {e}")
            
            # Show text vs image scores after weighting
            if text_results and image_results:
                print(f"  [DEBUG] Score comparison (after weighting):")
                print(f"    Top text score: {text_results[0].get('weighted_score', text_results[0].get('score', 0)):.4f}")
                print(f"    Top image score: {image_results[0].get('weighted_score', image_results[0].get('score', 0)):.4f}")
            elif text_results:
                print(f"  [DEBUG] Only text results found. Top text score: {text_results[0].get('weighted_score', text_results[0].get('score', 0)):.4f}")
            elif image_results:
                print(f"  [DEBUG] Only image results found. Top image score: {image_results[0].get('weighted_score', image_results[0].get('score', 0)):.4f}")
            
            # Show all retrieved items with scores
            print(f"  [DEBUG] All retrieved items (sorted by weighted_score):")
            for i, item in enumerate(retrieved[:10]):
                modality = item.get('modality', 'unknown')
                score = item.get('score', 0)
                weighted = item.get('weighted_score', score)
                print(f"    {i+1}. {modality}: score={score:.4f}, weighted={weighted:.4f}")
        
        # Compute metrics for all results
        ndcg = ndcg_at_k(labels)
        ap = map_at_k(labels)
        rec = recall_at_k(labels, max(len(set(gold_docs)), 1))

        agg["ndcg"] += ndcg
        agg["map"] += ap
        agg["recall"] += rec

        # Compute text-only metrics (for comparison)
        if text_results:
            text_labels = relevance_labels(text_results, gold_docs, len(text_results))
            text_ndcg = ndcg_at_k(text_labels)
            text_ap = map_at_k(text_labels)
            text_rec = recall_at_k(text_labels, max(len(set(gold_docs)), 1))
            agg_text["ndcg"] += text_ndcg
            agg_text["map"] += text_ap
            agg_text["recall"] += text_rec
        else:
            text_ndcg = text_ap = text_rec = 0.0

        agg_image["count"] += len(image_results)

        record = {
            "id": sample["id"],
            "claim": claim,
            "gold_docs": gold_docs,
            "retrieved": retrieved,  # Final merged results (top_k)
            "raw_image_results": raw_image_results_all,  # All image results before merging
            "labels": labels,
            "metrics": {"ndcg": ndcg, "map": ap, "recall": rec},
            "text_metrics": {"ndcg": text_ndcg, "map": text_ap, "recall": text_rec},
            "num_text_results": len(text_results),
            "num_image_results": len(image_results),
            "num_raw_image_results": len(raw_image_results_all),  # Total images retrieved (before merging)
        }
        results_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{idx + 1}/{total}] NDCG={ndcg:.2f} MAP={ap:.2f} Recall={rec:.2f} | "
              f"Text: {len(text_results)}, Images: {len(image_results)}")

    results_file.close()

    num = total or 1
    metrics = {
        "total": total,
        "ndcg": agg["ndcg"] / num,
        "map": agg["map"] / num,
        "recall@k": agg["recall"] / num,
        "top_k": top_k,
        "multimodal": use_multimodal,
    }
    
    if use_multimodal:
        metrics["text_weight"] = text_weight
        metrics["image_weight"] = image_weight
        metrics["text_only_metrics"] = {
            "ndcg": agg_text["ndcg"] / num,
            "map": agg_text["map"] / num,
            "recall@k": agg_text["recall"] / num,
        }
        metrics["total_image_results"] = agg_image["count"]
        metrics["avg_images_per_query"] = agg_image["count"] / num
    
    (OUTPUT_DIR / "scifact_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nEvaluation complete:")
    print(json.dumps(metrics, indent=2))
    print(f"Results written to {results_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on SciFact.")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--text-only", action="store_true", default=False,
                       help="Use text-only retrieval (disable multimodal). Default: multimodal enabled")
    parser.add_argument("--text-weight", type=float, default=0.6,
                       help="Weight for text results when merging with images. Default: 0.6")
    parser.add_argument("--image-weight", type=float, default=0.4,
                       help="Weight for image results when merging with text. Default: 0.4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_multimodal = not args.text_only  # Multimodal is default, unless --text-only is specified
    evaluate(
        top_k=args.top_k,
        limit=args.limit,
        use_multimodal=use_multimodal,
        text_weight=args.text_weight,
        image_weight=args.image_weight,
    )


if __name__ == "__main__":
    main()
