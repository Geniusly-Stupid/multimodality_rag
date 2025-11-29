"""High-level RAG pipeline wiring retriever and generator modules."""

from __future__ import annotations

import os

# Fix OpenMP library conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

from generator.rag_generator import GeneratorConfig, RAGGenerator
from retriever.multimodal_retriever import MultimodalRetriever, MultimodalRetrieverConfig
from retriever.rag_retriever import EncoderConfig, RAGRetriever


@dataclass
class RAGOutput:
    query: str
    retrieved_chunks: List[Dict]
    generated_answer: str


class RAGPipeline:
    """Compose retriever and generator into a single interface."""

    def __init__(self, retriever: RAGRetriever | MultimodalRetriever, generator: RAGGenerator):
        self.retriever = retriever
        self.generator = generator

    def answer(self, query: str, top_k: int = 5, retrieval_mode: Optional[str] = None) -> RAGOutput:
        retrieve_kwargs = {"top_k": top_k}
        if retrieval_mode is not None:
            retrieve_kwargs["retrieval_mode"] = retrieval_mode
        retrieved = self.retriever.retrieve(query, **retrieve_kwargs)
        generated_answer = self.generator.generate(query, retrieved)
        return RAGOutput(query=query, retrieved_chunks=retrieved, generated_answer=generated_answer)


def format_output(rag_output: RAGOutput) -> str:
    lines = [f"Query: {rag_output.query}", "", "Retrieved Items:"]
    if not rag_output.retrieved_chunks:
        lines.append("  (none)")
    else:
        for idx, chunk in enumerate(rag_output.retrieved_chunks, start=1):
            modality = chunk.get("modality", "text")
            score = chunk.get("score", 0.0)
            
            if modality == "text":
                preview = chunk.get("text", "")[:200].replace("\n", " ")
                lines.append(
                    f"  [{idx}] [TEXT] score={score:.4f} page_id={chunk.get('page_id')} url={chunk.get('source_url')}"
                )
                lines.append(f"      {preview}...")
            elif modality == "caption":
                caption = chunk.get("caption", "")
                image_id = chunk.get("image_id", "")
                page_id = chunk.get("page_id", "")
                lines.append(
                    f"  [{idx}] [CAPTION] score={score:.4f} image_id={image_id} page_id={page_id} url={chunk.get('source_url')}"
                )
                lines.append(f"      {caption}")
            elif modality == "image":
                caption = chunk.get("caption", "")
                image_id = chunk.get("image_id", "")
                lines.append(
                    f"  [{idx}] [IMAGE] score={score:.4f} image_id={image_id}"
                )
                lines.append(f"      Caption: {caption}")
    lines.append("")
    lines.append("Answer:")
    lines.append(rag_output.generated_answer or "(empty)")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RAG pipeline for Frames Benchmark.")
    parser.add_argument("--query", type=str, required=True, help="User question to answer.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        default="text_clip",
        choices=["text", "text_clip", "text_caption", "caption_only"],
        help="Retrieval strategy to use.",
    )
    parser.add_argument(
        "--generator_model",
        type=str,
        default=None,
        help="Generator model name.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="D:/huggingface_cache",
        help="Directory for caching models/tokenizers.",
    )
    parser.add_argument(
        "--use_reranker",
        action="store_true",
        help="Enable cross-encoder reranking (bge-reranker-v2-m3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Use multimodal retriever (supports both text and images)
    multimodal_config = MultimodalRetrieverConfig(
        use_text_retrieval=args.retrieval_mode in {"text", "text_clip", "text_caption"},
        use_image_retrieval=args.retrieval_mode == "text_clip",
        use_caption_retrieval=args.retrieval_mode in {"text_caption", "caption_only"},
        default_retrieval_mode=args.retrieval_mode,
        text_weight=0.6,  # Slightly favor text
        image_weight=0.4,
        caption_weight=0.5,
        use_reranker=args.use_reranker,
    )
    retriever = MultimodalRetriever(config=multimodal_config)
    
    gen_config = GeneratorConfig(
        model_name=args.generator_model or GeneratorConfig().model_name,
        max_new_tokens=args.max_new_tokens,
        cache_dir=args.cache_dir,
    )
    generator = RAGGenerator(gen_config)
    pipeline = RAGPipeline(retriever, generator)
    output = pipeline.answer(args.query, top_k=args.top_k, retrieval_mode=args.retrieval_mode)
    print(format_output(output))


if __name__ == "__main__":
    main()
