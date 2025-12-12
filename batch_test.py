"""Batch test script using rag_pipeline logic - process queries one by one with different modes."""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import gc
import argparse
from pathlib import Path
from typing import Dict, List

from generator.rag_generator import GeneratorConfig, RAGGenerator
from retriever.multimodal_retriever import MultimodalRetriever, MultimodalRetrieverConfig
from rag_pipeline import RAGPipeline, format_output


def load_test_queries(test_file: Path) -> List[Dict[str, str]]:
    """Load test queries from test.txt file."""
    queries = []
    if not test_file.exists():
        return queries
    
    with open(test_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Parse pairs: query (odd lines) and answer (even lines)
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            queries.append({
                "query": lines[i],
                "ground_truth": lines[i + 1],
                "query_id": len(queries) + 1
            })
        else:
            queries.append({
                "query": lines[i],
                "ground_truth": "",
                "query_id": len(queries) + 1
            })
    
    return queries


def process_queries_with_mode(queries: List[Dict], mode: str, top_k: int = 5, output_file: Path = None):
    """Process queries with a specific retrieval mode.
    
    Args:
        queries: List of query dictionaries
        mode: "text_only", "image_only", or "both"
        top_k: Number of items to retrieve
        output_file: Output file path
    """
    print(f"\n{'=' * 80}")
    print(f"Mode: {mode.upper().replace('_', ' ')}")
    print("=" * 80)
    
    # Initialize retriever based on mode
    gc.collect()
    
    if mode == "text_only":
        multimodal_config = MultimodalRetrieverConfig(
            use_text_retrieval=True,
            use_image_retrieval=False,
        )
    elif mode == "image_only":
        multimodal_config = MultimodalRetrieverConfig(
            use_text_retrieval=False,
            use_image_retrieval=True,
        )
    else:  # both
        multimodal_config = MultimodalRetrieverConfig(
            use_text_retrieval=True,
            use_image_retrieval=True,
            text_weight=0.6,
            image_weight=0.4,
        )
    
    retriever = MultimodalRetriever(config=multimodal_config)
    print(f"✓ Retriever initialized ({mode})")
    
    gc.collect()
    
    gen_config = GeneratorConfig()
    generator = RAGGenerator(gen_config)
    print("✓ Generator initialized")
    
    pipeline = RAGPipeline(retriever, generator)
    print("✓ Pipeline initialized")
    
    # Process queries one by one
    print("\n" + "=" * 80)
    print("Processing Queries")
    print("=" * 80)
    
    results = []
    for idx, query_data in enumerate(queries, 1):
        query = query_data["query"]
        ground_truth = query_data.get("ground_truth", "")
        query_id = query_data.get("query_id", idx)
        
        print(f"\n[{idx}/{len(queries)}] Query {query_id} ({mode})")
        print("-" * 80)
        print(f"Query: {query}")
        print("-" * 80)
        
        try:
            gc.collect()
            
            # Process query using pipeline
            output = pipeline.answer(query, top_k=top_k)
            
            # Count modalities
            text_count = sum(1 for chunk in output.retrieved_chunks if chunk.get("modality") == "text")
            image_count = sum(1 for chunk in output.retrieved_chunks if chunk.get("modality") == "image")
            
            print(f"\nRetrieved: {len(output.retrieved_chunks)} items (Text: {text_count}, Images: {image_count})")
            
            # Show top results by modality
            if output.retrieved_chunks:
                print("\nTop results:")
                for i, chunk in enumerate(output.retrieved_chunks[:3], 1):
                    modality = chunk.get("modality", "text")
                    score = chunk.get("score", 0.0)
                    if modality == "text":
                        text_preview = chunk.get("text", "")[:80].replace("\n", " ")
                        print(f"  {i}. [TEXT] score={score:.4f}: {text_preview}...")
                    else:
                        img_path = chunk.get("img_path", chunk.get("image_id", ""))
                        caption = chunk.get("caption", "")
                        print(f"  {i}. [IMAGE] score={score:.4f}: {img_path}")
                        if caption:
                            print(f"      Caption: {caption[:60]}...")
            
            print(f"\nGenerated Answer: {output.generated_answer[:200]}...")
            
            # Store result
            result = {
                "query_id": query_id,
                "query": query,
                "ground_truth": ground_truth,
                "mode": mode,
                "retrieved_count": len(output.retrieved_chunks),
                "text_count": text_count,
                "image_count": image_count,
                "retrieved_chunks": [
                    {
                        "modality": chunk.get("modality", "text"),
                        "score": chunk.get("score", 0.0),
                        "text": chunk.get("text", "")[:500] if chunk.get("modality") == "text" else None,
                        "image_id": chunk.get("image_id", ""),
                        "img_path": chunk.get("img_path", ""),
                        "caption": chunk.get("caption", ""),
                        "source_file": chunk.get("source_file", ""),
                        "page_id": chunk.get("page_id", ""),
                        "source_url": chunk.get("source_url", ""),
                    }
                    for chunk in output.retrieved_chunks
                ],
                "generated_answer": output.generated_answer,
            }
            results.append(result)
            
            # Save after each query
            if output_file:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"✓ Saved results (query {query_id}/{len(queries)})")
            
            gc.collect()
            
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user. Saving current results...")
            if output_file:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"✓ Results saved to {output_file}")
            break
        except Exception as e:
            print(f"\n✗ Error processing query {query_id}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "query_id": query_id,
                "query": query,
                "ground_truth": ground_truth,
                "mode": mode,
                "error": str(e),
            })
            # Save even on error
            if output_file:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            gc.collect()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch test with different retrieval modes")
    parser.add_argument("--test_file", type=str, default="data/test.txt", help="Test queries file")
    parser.add_argument("--output_file", type=str, default="rag_results/batch_test_results.json", help="Output file")
    parser.add_argument("--top_k", type=int, default=5, help="Number of items to retrieve")
    parser.add_argument("--mode", type=str, choices=["text_only", "image_only", "both", "all"], 
                       default="all", help="Retrieval mode: text_only, image_only, both, or all (test all modes)")
    args = parser.parse_args()
    
    # Configuration
    test_file = Path(args.test_file)
    output_file = Path(args.output_file)
    top_k = args.top_k
    
    # Load queries
    print("=" * 80)
    print("Loading Test Queries")
    print("=" * 80)
    queries = load_test_queries(test_file)
    if not queries:
        print(f"Error: No queries found in {test_file}")
        return
    
    print(f"Loaded {len(queries)} queries from {test_file}")
    
    # Process based on mode
    all_results = []
    
    if args.mode == "all":
        # Test all three modes
        modes = ["text_only", "image_only", "both"]
        for mode in modes:
            mode_output_file = output_file.parent / f"{output_file.stem}_{mode}{output_file.suffix}"
            results = process_queries_with_mode(queries, mode, top_k, mode_output_file)
            all_results.extend(results)
        
        # Also save combined results
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Combined results saved to {output_file}")
    else:
        # Test single mode
        results = process_queries_with_mode(queries, args.mode, top_k, output_file)
        all_results = results
    
    # Final summary
    print("\n" + "=" * 80)
    print("Final Summary")
    print("=" * 80)
    print(f"Total queries processed: {len(queries)}")
    print(f"Total results: {len(all_results)}")
    print(f"Successful: {sum(1 for r in all_results if 'error' not in r)}")
    print(f"Failed: {sum(1 for r in all_results if 'error' in r)}")
    
    # Summary by mode
    if args.mode == "all":
        print("\nResults by mode:")
        for mode in ["text_only", "image_only", "both"]:
            mode_results = [r for r in all_results if r.get("mode") == mode]
            if mode_results:
                avg_text = sum(r.get("text_count", 0) for r in mode_results) / len(mode_results)
                avg_image = sum(r.get("image_count", 0) for r in mode_results) / len(mode_results)
                print(f"  {mode}: {len(mode_results)} queries, avg text={avg_text:.1f}, avg image={avg_image:.1f}")
    
    print(f"\n✓ Results saved to {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
