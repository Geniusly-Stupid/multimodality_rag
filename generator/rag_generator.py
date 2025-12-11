"""Autoregressive generator used for RAG answer synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

DEFAULT_GENERATOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_CACHE_DIR = None 


@dataclass
class GeneratorConfig:
    model_name: str = DEFAULT_GENERATOR_MODEL
    max_new_tokens: int = 256
    cache_dir: Optional[Path] = DEFAULT_CACHE_DIR

    def __post_init__(self) -> None:
        if self.cache_dir:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)


class RAGGenerator:
    """Wraps a causal LM for conditional generation from retrieved evidence."""

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
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

    def build_prompt(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """Format the prompt with evidence passages (text, captions, images)."""
        text_evidence = []
        caption_evidence = []
        image_info = []

        for idx, chunk in enumerate(retrieved_chunks, start=1):
            modality = chunk.get("modality", "text")
            if modality == "caption":
                caption = (chunk.get("caption") or chunk.get("text") or "").strip()
                if caption:
                    caption_evidence.append(f"[{idx}] {caption}")
            elif modality == "image":
                caption = (chunk.get("caption") or chunk.get("text") or "").strip()
                image_id = chunk.get("image_id", "")
                if caption:
                    caption_evidence.append(f"[{idx}] (Image {image_id}) {caption}")
                else:
                    image_info.append(f"[Image {idx}] ID: {image_id}")
            else:
                text = (chunk.get("text") or "").strip()
                if text:
                    text_evidence.append(f"[{idx}] {text}")

        prompt_parts = [
            "You are a helpful assistant that answers questions using the supplied evidence.\n",
            f"User question:\n{query}\n\n",
        ]

        if text_evidence:
            prompt_parts.append("Wiki Text Evidence:\n")
            prompt_parts.append("\n".join(text_evidence) + "\n\n")

        if caption_evidence:
            prompt_parts.append("Image Caption Evidence:\n")
            prompt_parts.append("\n".join(caption_evidence) + "\n\n")

        if image_info:
            prompt_parts.append("Additional Image Metadata:\n")
            prompt_parts.append("\n".join(image_info) + "\n\n")

        prompt_parts.append("Answer:\n")

        return "".join(prompt_parts)

    @torch.inference_mode()
    def generate(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """Generate an answer for a query given retrieved evidence."""
        prompt = self.build_prompt(query, retrieved_chunks)
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
