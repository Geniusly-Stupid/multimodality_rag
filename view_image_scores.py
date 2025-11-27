"""View image retrieval scores from evaluation results."""

import json
from pathlib import Path

RESULTS_FILE = Path("evaluation_results/scifact_results.jsonl")


def view_image_scores(limit: int = 5):
    """View image scores from evaluation results."""
    
    if not RESULTS_FILE.exists():
        print(f"Results file not found: {RESULTS_FILE}")
        print("Please run evaluation first: python evaluation/evaluate_scifact.py --top_k 5")
        return
    
    print(f"Reading results from: {RESULTS_FILE}\n")
    print("=" * 80)
    
    with RESULTS_FILE.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            
            record = json.loads(line.strip())
            claim = record.get("claim", "")
            retrieved = record.get("retrieved", [])
            
            # Separate text and image results
            text_results = [r for r in retrieved if r.get("modality") == "text"]
            image_results = [r for r in retrieved if r.get("modality") == "image"]
            
            print(f"\nQuery {idx + 1}: {claim[:100]}...")
            print("-" * 80)
            
            # Show image results with scores
            if image_results:
                print(f"Image Results ({len(image_results)}):")
                for i, img in enumerate(image_results, 1):
                    print(f"  {i}. Image ID: {img.get('image_id', 'N/A')}")
                    print(f"     Caption: {img.get('caption', 'N/A')[:70]}...")
                    print(f"     Raw Score: {img.get('score', 0):.6f}")
                    print(f"     Weighted Score: {img.get('weighted_score', img.get('score', 0)):.6f}")
            else:
                print("Image Results: 0 (no images in top results)")
            
            # Show text results for comparison
            if text_results:
                print(f"\nText Results ({len(text_results)}):")
                for i, text in enumerate(text_results[:3], 1):  # Show top 3
                    print(f"  {i}. Doc ID: {text.get('doc_id', 'N/A')}")
                    print(f"     Text: {text.get('text', 'N/A')[:70]}...")
                    print(f"     Raw Score: {text.get('score', 0):.6f}")
                    print(f"     Weighted Score: {text.get('weighted_score', text.get('score', 0)):.6f}")
            
            # Show all results sorted
            print(f"\nAll Results (sorted by weighted_score):")
            for i, item in enumerate(retrieved[:5], 1):
                modality = item.get('modality', 'unknown')
                raw_score = item.get('score', 0)
                weighted_score = item.get('weighted_score', raw_score)
                print(f"  {i}. [{modality.upper()}] Raw={raw_score:.6f}, Weighted={weighted_score:.6f}")
            
            print()


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    view_image_scores(limit=limit)

