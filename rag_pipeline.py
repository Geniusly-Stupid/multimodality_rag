"""High-level RAG pipeline wiring retriever and generator modules."""

from __future__ import annotations

import os
import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from generator.rag_generator import GeneratorConfig, RAGGenerator
from retriever.build_faiss_index import load_chunks_from_folder
from retriever.retriever_adapter import FaissVectorStore, RetrieverAdapter
from retriever.rag_retriever import EncoderConfig, TextEncoder

# Fix OpenMP library conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


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

            text_val = (
                chunk.get("text")
                or chunk.get("content")
                or meta.get("text")
                or meta.get("caption")
                or ""
            )[:400].replace("\n", " ")
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
        help="Folder containing parsed JSON chunks or raw txt files to index.",
    )
    parser.add_argument(
        "--generator_model",
        type=str,
        default=None,
        help="Generator model name.",
    )
    parser.add_argument("--use_remote", action="store_true", help="Use remote LLM API instead of local HF model.")
    parser.add_argument(
        "--remote_model",
        type=str,
        default="",
        help="Remote model name for API calls (e.g., qwen/qwen3-next-80b-a3b-instruct).",
    )
    parser.add_argument(
        "--remote_api_base",
        type=str,
        default="",
        help="Remote API base URL (e.g., https://integrate.api.nvidia.com/v1).",
    )
    parser.add_argument(
        "--remote_api_key_env",
        type=str,
        default="",
        help="Env var name that holds the remote API key (e.g., NVIDIA_API_KEY).",
    )
    parser.add_argument(
        "--remote_api_key",
        type=str,
        default="",
        help="Direct remote API key (overrides env lookup).",
    )
    parser.add_argument(
        "--remote_stream",
        action="store_true",
        help="Enable streaming responses from remote LLM (remote mode only).",
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

    chunks = load_chunks_from_folder(args.doc_path)
    print("Loaded chunks:", len(chunks))

    text_encoder = TextEncoder(EncoderConfig())
    vector_store = FaissVectorStore(metric="ip")
    print(text_encoder)
    retriever = RetrieverAdapter(text_encoder=text_encoder, vector_store=vector_store)
    print(retriever)
    retriever.build_index(chunks)
    print("Index built successfully.")
    print("Vector store size:", retriever.vector_store.index)
    
    gen_config = GeneratorConfig(
        model_name=args.generator_model or GeneratorConfig().model_name,
        max_new_tokens=args.max_new_tokens,
        cache_dir=args.cache_dir,
        use_remote=args.use_remote,
        remote_model_name=args.remote_model,
        remote_api_base=args.remote_api_base,
        remote_api_key_env=args.remote_api_key_env,
        remote_api_key=args.remote_api_key,
        use_stream=args.remote_stream,
    )
    generator = RAGGenerator(gen_config)
    print("Generator initialized successfully.")
    pipeline = RAGPipeline(retriever, generator)
    print("Pipeline initialized successfully.")
    output = pipeline.answer(args.query, top_k=args.top_k)
    print("Answer generated successfully.")
    print(format_output(output))
    print("Output formatted successfully.")


if __name__ == "__main__":
    main()
