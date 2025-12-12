"""Quick test to verify img_path is now included in image retrieval results."""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import gc
from pathlib import Path
from retriever.multimodal_retriever import MultimodalRetriever, MultimodalRetrieverConfig

def test_img_path():
    print("Testing img_path in image retrieval results...")
    print("=" * 80)
    
    gc.collect()
    
    config = MultimodalRetrieverConfig(
        use_text_retrieval=False,
        use_image_retrieval=True,
    )
    retriever = MultimodalRetriever(config=config)
    
    gc.collect()
    
    # Test query
    query = "diagram chart"
    print(f"\nQuery: {query}")
    print("-" * 80)
    
    results = retriever.retrieve(query, top_k=3)
    
    print(f"\nRetrieved {len(results)} images:\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}]")
        print(f"  img_path: {r.get('img_path', '(empty)')}")
        print(f"  image_id: {r.get('image_id', '(empty)')}")
        print(f"  caption: {r.get('caption', '(empty)')[:60]}...")
        print(f"  source_file: {r.get('source_file', '(empty)')}")
        print(f"  page_id: {r.get('page_id', '(empty)')}")
        print(f"  score: {r.get('score', 0):.4f}")
        print()
    
    # Check if img_path is populated
    has_img_path = any(r.get('img_path') for r in results)
    if has_img_path:
        print("✓ SUCCESS: img_path is now populated!")
    else:
        print("✗ WARNING: img_path is still empty")
    
    return results

if __name__ == "__main__":
    try:
        results = test_img_path()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
