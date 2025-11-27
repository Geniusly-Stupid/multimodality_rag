"""Test script to view image retrieval scores directly."""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retriever.multimodal_retriever import MultimodalRetriever, MultimodalRetrieverConfig
from retriever.scifact_retriever import SciFactRetriever
from retriever.rag_retriever import EncoderConfig


def test_image_retrieval_scores():
    """Test image retrieval and show raw scores."""
    
    # Create a wrapper for SciFactRetriever
    class SciFactRetrieverWrapper:
        def __init__(self, scifact_retriever):
            self.scifact_retriever = scifact_retriever
        
        def retrieve(self, query: str, top_k: int = 5):
            results = self.scifact_retriever.retrieve(query, top_k=top_k)
            for result in results:
                if "modality" not in result:
                    result["modality"] = "text"
            return results
    
    # Initialize multimodal retriever
    print("Initializing multimodal retriever...")
    scifact_retriever = SciFactRetriever(encoder_config=EncoderConfig())
    text_retriever_wrapper = SciFactRetrieverWrapper(scifact_retriever)
    
    config = MultimodalRetrieverConfig(
        use_text_retrieval=True,
        use_image_retrieval=True,
        text_weight=0.6,
        image_weight=0.4,
    )
    retriever = MultimodalRetriever(
        config=config,
        text_retriever=text_retriever_wrapper
    )
    
    # Check image index status
    if retriever.image_index is None:
        print("❌ Image index not loaded!")
        return
    print(f"✓ Image index loaded: {retriever.image_index.ntotal} images")
    print(f"✓ Image metadata: {len(retriever.image_metadata)} entries\n")
    
    # Test queries
    test_queries = [
        "COVID-19 vaccine effectiveness",
        "machine learning algorithms",
        "climate change impact",
        "a dog playing in the park",  # More image-like query
        "scientific research methods",
    ]
    
    for query in test_queries:
        print("=" * 80)
        print(f"Query: {query}")
        print("-" * 80)
        
        # 1. Get raw image retrieval results (before weighting)
        print("\n1. Raw Image Retrieval Results (before weighting):")
        raw_image_results = retriever._retrieve_images(query, top_k=5)
        if raw_image_results:
            for i, img in enumerate(raw_image_results, 1):
                print(f"   {i}. Image ID: {img.get('image_id', 'N/A')}")
                print(f"      Caption: {img.get('caption', 'N/A')[:80]}...")
                print(f"      Raw Score: {img.get('score', 0):.6f}")
                print(f"      Weighted Score (×{config.image_weight}): {img.get('score', 0) * config.image_weight:.6f}")
        else:
            print("   No images retrieved")
        
        # 2. Get text retrieval results
        print("\n2. Text Retrieval Results (before weighting):")
        text_results = text_retriever_wrapper.retrieve(query, top_k=5)
        if text_results:
            for i, text in enumerate(text_results[:3], 1):  # Show top 3
                print(f"   {i}. Doc ID: {text.get('doc_id', 'N/A')}")
                print(f"      Text preview: {text.get('text', 'N/A')[:80]}...")
                print(f"      Raw Score: {text.get('score', 0):.6f}")
                print(f"      Weighted Score (×{config.text_weight}): {text.get('score', 0) * config.text_weight:.6f}")
        else:
            print("   No text results")
        
        # 3. Get merged results (after weighting and sorting)
        print("\n3. Merged Results (after weighting and sorting):")
        merged_results = retriever.retrieve(query, top_k=5)
        if merged_results:
            for i, item in enumerate(merged_results, 1):
                modality = item.get('modality', 'unknown')
                raw_score = item.get('score', 0)
                weighted_score = item.get('weighted_score', raw_score)
                print(f"   {i}. [{modality.upper()}]")
                if modality == 'text':
                    print(f"      Doc ID: {item.get('doc_id', 'N/A')}")
                    print(f"      Text: {item.get('text', 'N/A')[:60]}...")
                else:
                    print(f"      Image ID: {item.get('image_id', 'N/A')}")
                    print(f"      Caption: {item.get('caption', 'N/A')[:60]}...")
                print(f"      Raw Score: {raw_score:.6f}")
                print(f"      Weighted Score: {weighted_score:.6f}")
        else:
            print("   No results")
        
        print()


if __name__ == "__main__":
    test_image_retrieval_scores()

