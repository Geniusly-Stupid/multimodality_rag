"""Safe test script - exactly following successful test patterns."""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_text_step_by_step():
    """Test text retrieval step by step, exactly like successful tests."""
    print("=" * 80)
    print("Testing Text Retrieval (Step by Step)")
    print("=" * 80)
    
    gc.collect()
    
    # Step 1: Import
    print("\nStep 1: Importing modules...")
    from retriever.build_faiss_index import TextEncoder, EncoderConfig
    from retriever.rag_retriever import RAGRetriever
    import faiss
    import numpy as np
    print("   ✓ OK")
    
    gc.collect()
    
    # Step 2: Create encoder (like successful test)
    print("\nStep 2: Creating TextEncoder...")
    encoder = TextEncoder(EncoderConfig())
    print("   ✓ OK")
    
    gc.collect()
    
    # Step 3: Test encoding (like successful test)
    print("\nStep 3: Testing encode()...")
    query_vec = encoder.encode(['information retrieval'])
    query_vec = np.ascontiguousarray(query_vec.astype("float32"))
    print(f"   ✓ OK - shape: {query_vec.shape}")
    
    gc.collect()
    
    # Step 4: Load index (like successful test)
    print("\nStep 4: Loading FAISS index...")
    index_path = PROJECT_ROOT / "retriever/faiss_index/index.faiss"
    index = faiss.read_index(str(index_path))
    print(f"   ✓ OK - {index.ntotal} vectors, dim={index.d}")
    
    gc.collect()
    
    # Step 5: Search (like successful test)
    print("\nStep 5: Searching index...")
    distances, indices = index.search(query_vec, 3)
    print(f"   ✓ OK - Found {len(indices[0])} results")
    print(f"   Distances: {distances[0]}")
    print(f"   Indices: {indices[0]}")
    
    gc.collect()
    
    # Step 6: Load chunks
    print("\nStep 6: Loading chunks...")
    import json
    chunks_path = PROJECT_ROOT / "retriever/faiss_index/chunks.jsonl"
    chunks = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    print(f"   ✓ OK - Loaded {len(chunks)} chunks")
    
    gc.collect()
    
    # Step 7: Build results
    print("\nStep 7: Building results...")
    results = []
    for rank in range(min(3, len(chunks))):
        idx = int(indices[0][rank])
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        results.append({
            "text": chunk["text"],
            "page_id": chunk.get("page_id"),
            "source_file": chunk.get("source_file"),
            "score": float(distances[0][rank]),
        })
    print(f"   ✓ OK - {len(results)} results")
    
    print("\nTop results:")
    for i, r in enumerate(results[:3], 1):
        text = r.get("text", "")[:80].replace("\n", " ")
        score = r.get("score", 0.0)
        print(f"   {i}. (score={score:.4f}) {text}...")
    
    print("\n✓ Text retrieval test passed!")
    return True


def test_image_step_by_step():
    """Test image retrieval step by step."""
    print("\n" + "=" * 80)
    print("Testing Image Retrieval (Step by Step)")
    print("=" * 80)
    
    gc.collect()
    
    # Step 1: Import
    print("\nStep 1: Importing modules...")
    from retriever.image_encoder import CLIPTextEncoder, ImageEncoderConfig
    import faiss
    import numpy as np
    print("   ✓ OK")
    
    gc.collect()
    
    # Step 2: Load index
    print("\nStep 2: Loading image index...")
    index_path = PROJECT_ROOT / "retriever/faiss_index/images/index.faiss"
    index = faiss.read_index(str(index_path))
    print(f"   ✓ OK - {index.ntotal} vectors, dim={index.d}")
    
    gc.collect()
    
    # Step 3: Load metadata
    print("\nStep 3: Loading image metadata...")
    import json
    metadata_path = PROJECT_ROOT / "retriever/faiss_index/images/images.jsonl"
    image_metadata = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                image_metadata.append(json.loads(line))
    print(f"   ✓ OK - Loaded {len(image_metadata)} images")
    
    gc.collect()
    
    # Step 4: Create CLIP encoder (like successful test)
    print("\nStep 4: Creating CLIP text encoder (this may take a moment)...")
    encoder = CLIPTextEncoder(config=ImageEncoderConfig())
    print("   ✓ OK")
    
    gc.collect()
    
    # Step 5: Encode query
    print("\nStep 5: Encoding query...")
    query_vec = encoder.encode(['diagram chart'])
    query_vec = np.ascontiguousarray(query_vec.astype("float32"))
    print(f"   ✓ OK - shape: {query_vec.shape}")
    
    gc.collect()
    
    # Step 6: Search
    print("\nStep 6: Searching index...")
    distances, indices = index.search(query_vec, 3)
    print(f"   ✓ OK - Found {len(indices[0])} results")
    
    # Step 7: Build results
    print("\nStep 7: Building results...")
    results = []
    for rank in range(min(3, len(image_metadata))):
        idx = int(indices[0][rank])
        if idx < 0 or idx >= len(image_metadata):
            continue
        image_meta = image_metadata[idx]
        results.append({
            "img_path": image_meta.get("img_path", ""),
            "image_id": image_meta.get("image_id", ""),
            "caption": image_meta.get("caption", ""),
            "source_file": image_meta.get("source_file", ""),
            "score": float(distances[0][rank]),
        })
    print(f"   ✓ OK - {len(results)} results")
    
    print("\nTop results:")
    for i, r in enumerate(results[:3], 1):
        img_path = r.get("img_path", r.get("image_id", "unknown"))
        score = r.get("score", 0.0)
        print(f"   {i}. (score={score:.4f}) {img_path}")
    
    print("\n✓ Image retrieval test passed!")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Multimodal Retrieval Test (Step by Step)")
    print("=" * 80)
    
    results = []
    
    # Test text retrieval
    try:
        results.append(("Text Retrieval", test_text_step_by_step()))
    except Exception as e:
        print(f"\n✗ Text retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Text Retrieval", False))
    
    # Test image retrieval
    try:
        results.append(("Image Retrieval", test_image_step_by_step()))
    except Exception as e:
        print(f"\n✗ Image retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Image Retrieval", False))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n⚠ Some tests failed")
    
    print("=" * 80)
