"""High-level RAG pipeline wiring retriever and generator modules."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

from generator.rag_generator import GeneratorConfig, RAGGenerator
from retriever.rag_retriever import EncoderConfig, RAGRetriever


@dataclass
class RAGOutput:
    query: str
    retrieved_chunks: List[Dict]
    generated_answer: str


class RAGPipeline:
    """Compose retriever and generator into a single interface."""

    def __init__(self, retriever: RAGRetriever, generator: RAGGenerator):
        self.retriever = retriever
        self.generator = generator

    def answer(self, query: str, top_k: int = 5) -> RAGOutput:
        retrieved = self.retriever.retrieve(query, top_k=top_k)
        generated_answer = self.generator.generate(query, retrieved)
        return RAGOutput(query=query, retrieved_chunks=retrieved, generated_answer=generated_answer)


def format_output(rag_output: RAGOutput) -> str:
    lines = [f"Query: {rag_output.query}", "", "Retrieved Chunks:"]
    if not rag_output.retrieved_chunks:
        lines.append("  (none)")
    else:
        for idx, chunk in enumerate(rag_output.retrieved_chunks, start=1):
            preview = chunk["text"][:200].replace("\n", " ")
            lines.append(
                f"  [{idx}] score={chunk['score']:.4f} page_id={chunk.get('page_id')} url={chunk.get('source_url')}"
            )
            lines.append(f"      {preview}...")
    lines.append("")
    lines.append("Answer:")
    lines.append(rag_output.generated_answer or "(empty)")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RAG pipeline for Frames Benchmark.")
    parser.add_argument("--query", type=str, required=True, help="User question to answer.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = RAGRetriever(encoder_config=EncoderConfig())
    gen_config = GeneratorConfig(
        model_name=args.generator_model or GeneratorConfig().model_name,
        max_new_tokens=args.max_new_tokens,
        cache_dir=args.cache_dir,
    )
    generator = RAGGenerator(gen_config)
    pipeline = RAGPipeline(retriever, generator)
    output = pipeline.answer(args.query, top_k=args.top_k)
    print(format_output(output))


if __name__ == "__main__":
    main()
