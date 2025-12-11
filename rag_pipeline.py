"""High-level RAG pipeline wiring retriever and generator modules."""

from __future__ import annotations

import os

# Fix OpenMP library conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional

from generator.rag_generator import GeneratorConfig, RAGGenerator
from parser.raganything_parser import RAGAnythingParser
from retriever.retriever_adapter import FaissVectorStore, RetrieverAdapter
from retriever.rag_retriever import EncoderConfig, TextEncoder


@dataclass
class RAGOutput:
    query: str
    retrieved_chunks: List[Dict]
    generated_answer: str


class RAGPipeline:
    """Compose retriever and generator into a single interface."""

    def __init__(self, retriever, generator: RAGGenerator):
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
            meta = chunk.get("metadata") or {}
            modality = chunk.get("modality") or meta.get("modality") or "text"
            score = chunk.get("score", 0.0)

            text_val = (chunk.get("text") or meta.get("text") or meta.get("caption") or "")[:200].replace("\n", " ")
            caption_val = (chunk.get("caption") or meta.get("caption") or "")[:200].replace("\n", " ")
            display_text = caption_val or text_val
            lines.append(f"  [{idx}] [{modality.upper()}] score={score:.4f}")
            if display_text:
                lines.append(f"      {display_text}...")
    lines.append("")
    lines.append("Answer:")
    lines.append(rag_output.generated_answer or "(empty)")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RAG pipeline for Frames Benchmark.")
    parser.add_argument("--query", type=str, required=True, help="User question to answer.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--doc_path",
        type=str,
        required=True,
        help="Path to the document to parse and index with RAGAnything.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    async def _run() -> None:
        # Parse and index document via RAGAnything (caption/text only)
        parser = RAGAnythingParser()
        chunks = await parser.parse_and_enrich(args.doc_path)

        text_encoder = TextEncoder(EncoderConfig())
        vector_store = FaissVectorStore(metric="ip")
        retriever = RetrieverAdapter(text_encoder=text_encoder, vector_store=vector_store)
        retriever.build_index(chunks)

        gen_config = GeneratorConfig(
            model_name=args.generator_model or GeneratorConfig().model_name,
            max_new_tokens=args.max_new_tokens,
            cache_dir=args.cache_dir,
        )
        generator = RAGGenerator(gen_config)
        pipeline = RAGPipeline(retriever, generator)
        output = pipeline.answer(args.query, top_k=args.top_k)
        print(format_output(output))

    asyncio.run(_run())


if __name__ == "__main__":
    main()
