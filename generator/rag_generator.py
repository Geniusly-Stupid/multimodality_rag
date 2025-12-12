"""Autoregressive generator used for RAG answer synthesis (local HF or remote API)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

DEFAULT_GENERATOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_CACHE_DIR = None


@dataclass
class GeneratorConfig:
    model_name: str = DEFAULT_GENERATOR_MODEL
    max_new_tokens: int = 256
    cache_dir: Optional[Path] = DEFAULT_CACHE_DIR
    use_remote: bool = False
    remote_model_name: str = ""
    remote_api_base: str = ""
    remote_api_key_env: str = ""
    use_stream: bool = False
    remote_api_key: str = ""

    def __post_init__(self) -> None:
        if self.cache_dir:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)


class RAGGenerator:
    """Wraps a causal LM for conditional generation from retrieved evidence."""

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.is_remote = bool(self.config.use_remote)

        if self.is_remote:
            api_key_env = self.config.remote_api_key_env or "NVIDIA_API_KEY"
            api_key = self.config.remote_api_key or os.getenv(api_key_env)
            if not api_key:
                raise ValueError(f"Missing API key; provide via remote_api_key or env var: {api_key_env}")
            self.client = OpenAI(
                base_url=self.config.remote_api_base or "https://integrate.api.nvidia.com/v1",
                api_key=api_key,
            )
            self.remote_model = self.config.remote_model_name or "qwen/qwen3-next-80b-a3b-instruct"
            self.tokenizer = None
            self.model = None
            self.device = None
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer_kwargs = {}
            model_kwargs = {}
            if self.config.cache_dir:
                tokenizer_kwargs["cache_dir"] = str(self.config.cache_dir)
                model_kwargs["cache_dir"] = str(self.config.cache_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, **tokenizer_kwargs)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name, **model_kwargs).to(self.device)
            self.model.eval()
            self.client = None
            self.remote_model = None

    def build_prompt(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """Format the prompt with evidence from normalized RAGAnything chunks."""
        text_evidence = []
        caption_evidence = []

        for idx, chunk in enumerate(retrieved_chunks, start=1):
            meta = chunk.get("metadata") or {}
            modality = chunk.get("modality") or meta.get("modality") or "text"
            text_val = (chunk.get("text") or meta.get("text") or meta.get("caption") or "").strip()
            caption_val = (chunk.get("caption") or meta.get("caption") or "").strip()

            if modality == "caption" or caption_val:
                caption_evidence.append(f"[{idx}] {caption_val or text_val}")
            else:
                if text_val:
                    text_evidence.append(f"[{idx}] {text_val}")

        prompt_parts = [
            "You are a helpful assistant that answers questions using the supplied evidence.\n",
            f"User question:\n{query}\n\n",
        ]

        if text_evidence:
            prompt_parts.append("Text Evidence:\n")
            prompt_parts.append("\n".join(text_evidence) + "\n\n")

        if caption_evidence:
            prompt_parts.append("Caption Evidence:\n")
            prompt_parts.append("\n".join(caption_evidence) + "\n\n")

        prompt_parts.append("Answer:\n")

        return "".join(prompt_parts)

    @torch.inference_mode()
    def generate(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """Generate an answer for a query given retrieved evidence."""
        prompt = self.build_prompt(query, retrieved_chunks)

        if self.is_remote:
            stream = bool(self.config.use_stream)
            if stream:
                completion = self.client.chat.completions.create(
                    model=self.remote_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    top_p=0.7,
                    max_tokens=self.config.max_new_tokens,
                    stream=True,
                )
                collected = []
                for chunk in completion:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        collected.append(delta.content)
                return "".join(collected).strip()
            else:
                completion = self.client.chat.completions.create(
                    model=self.remote_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    top_p=0.7,
                    max_tokens=self.config.max_new_tokens,
                )
                return completion.choices[0].message.content.strip()

        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        output_ids = self.model.generate(**encoded, generation_config=gen_config)
        generated = output_ids[0, encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
