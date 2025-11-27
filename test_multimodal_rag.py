"""Test script for multimodal RAG pipeline with images."""

from __future__ import annotations

import os

# Fix OpenMP library conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from generator.rag_generator import GeneratorConfig, RAGGenerator
from rag_pipeline import RAGPipeline
from retriever.multimodal_retriever import MultimodalRetriever, MultimodalRetrieverConfig


def test_multimodal_rag():
    """Test the multimodal RAG pipeline."""
    print("=" * 80)
    print("Testing Multimodal RAG Pipeline")
    print("=" * 80)

    # Initialize multimodal retriever
    print("\n1. Initializing Multimodal Retriever...")
    multimodal_config = MultimodalRetrieverConfig(
        use_text_retrieval=True,
        use_image_retrieval=True,
        text_weight=0.6,
        image_weight=0.4,
    )
    retriever = MultimodalRetriever(config=multimodal_config)

    # Initialize generator
    print("\n2. Initializing Generator...")
    gen_config = GeneratorConfig()
    generator = RAGGenerator(gen_config)

    # Create pipeline
    print("\n3. Creating RAG Pipeline...")
    pipeline = RAGPipeline(retriever, generator)

    # Test queries
    test_queries = [
        "a dog playing in the park",
        "people eating food",
        "a beautiful sunset",
    ]

    print("\n4. Testing queries...")
    print("=" * 80)

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 80)
        output = pipeline.answer(query, top_k=5)

        print(f"\nRetrieved {len(output.retrieved_chunks)} items:")
        text_count = sum(1 for item in output.retrieved_chunks if item.get("modality") == "text")
        image_count = sum(1 for item in output.retrieved_chunks if item.get("modality") == "image")
        print(f"  - Text chunks: {text_count}")
        print(f"  - Images: {image_count}")

        if output.retrieved_chunks:
            print("\nTop results:")
            for idx, item in enumerate(output.retrieved_chunks[:3], start=1):
                modality = item.get("modality", "unknown")
                score = item.get("score", 0.0)
                if modality == "text":
                    preview = item.get("text", "")[:100]
                    print(f"  {idx}. [TEXT] score={score:.4f}: {preview}...")
                elif modality == "image":
                    caption = item.get("caption", "")
                    image_id = item.get("image_id", "")
                    print(f"  {idx}. [IMAGE] score={score:.4f}: {image_id} - {caption}")

        print(f"\nGenerated Answer:\n{output.generated_answer}")
        print("=" * 80)


if __name__ == "__main__":
    test_multimodal_rag()

