"""Autoregressive generator used for RAG answer synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

DEFAULT_GENERATOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_CACHE_DIR = Path("D:/huggingface_cache")


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
        """Format the prompt with evidence passages (text and/or images)."""
        evidence_lines = []
        image_info = []
        
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            modality = chunk.get("modality", "text")
            
            if modality == "text":
                text = chunk.get("text", "").strip()
                if text:
                    evidence_lines.append(f"[{idx}] {text}")
            elif modality == "image":
                caption = chunk.get("caption", "").strip()
                image_id = chunk.get("image_id", "")
                if caption:
                    image_info.append(f"[Image {idx}] ID: {image_id}, Caption: {caption}")
        
        evidence_block = "\n".join(evidence_lines) if evidence_lines else "None provided."
        image_block = "\n".join(image_info) if image_info else ""
        
        prompt_parts = [
            "You are a helpful assistant that answers questions using the supplied evidence.\n",
            f"User question:\n{query}\n\n",
        ]
        
        if evidence_block != "None provided.":
            prompt_parts.append(f"Relevant text evidence:\n{evidence_block}\n\n")
        
        if image_block:
            prompt_parts.append(f"Relevant images found:\n{image_block}\n\n")
        
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
