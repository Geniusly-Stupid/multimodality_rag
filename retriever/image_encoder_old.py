"""CLIP-based image encoder for multimodal RAG."""

from __future__ import annotations

import os

# Fix OpenMP library conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"  # Same as your working implementation


@dataclass
class ImageEncoderConfig:
    """Configuration for the CLIP image encoder."""

    model_name: str = DEFAULT_CLIP_MODEL
    normalize: bool = True


class ImageEncoder:
    """Wrapper around CLIP model for image embedding."""

    def __init__(self, config: Optional[ImageEncoderConfig] = None):
        self.config = config or ImageEncoderConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model: {self.config.model_name} on {self.device}")
        self.processor = CLIPProcessor.from_pretrained(self.config.model_name)
        self.model = CLIPModel.from_pretrained(self.config.model_name).to(self.device)
        self.model.eval()
        print("Image encoder loaded.")
        # CLIP image embedding dimension - use projection dimension
        # For CLIP models, the final embedding dim is typically 512 for base models
        # We'll set it based on the model config
        if "base" in self.config.model_name.lower():
            self.embedding_dim = 512
        elif "large" in self.config.model_name.lower():
            self.embedding_dim = 768
        else:
            # Fallback: use vision config hidden size (may need adjustment)
            self.embedding_dim = getattr(self.model.config.vision_config, "projection_dim", 512)

    def encode(self, images: Sequence[Image.Image], batch_size: int = 16) -> np.ndarray:
        """Encode a list of PIL Images into float32 numpy vectors."""
        if not images:
            return np.zeros((0, self.embedding_dim), dtype="float32")

        # Use the same approach as your working code - return torch tensor first
        all_embeddings_tensor = self.encode_tensor(images, batch_size)
        # Convert to numpy at the end
        return all_embeddings_tensor.cpu().numpy().astype("float32")
    
    def encode_tensor(self, images: Sequence[Image.Image], batch_size: int = 16) -> torch.Tensor:
        """Encode images and return as torch.Tensor (like your working implementation)."""
        if not images:
            return torch.empty(0, device=self.device)
        
        all_embeddings = []
        # Process in batches - exactly like your working implementation
        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]
            # Process images - exactly as your working code
            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)
            
            # Compute image embeddings - exactly as your working code
            with torch.no_grad():
                image_embeddings = self.model.get_image_features(**inputs)
            
            # Normalize embeddings - exactly as your working code
            if self.config.normalize:
                image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
            
            all_embeddings.append(image_embeddings)
        
        # Concatenate all batches
        return torch.cat(all_embeddings, dim=0)

    def encode_from_paths(self, image_paths: Sequence[str], batch_size: int = 16) -> np.ndarray:
        """Encode images from file paths."""
        images: list[Image.Image] = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Warning: Failed to load image {path}: {e}")
                # Create a blank image as placeholder
                images.append(Image.new("RGB", (224, 224), color="black"))

        return self.encode(images, batch_size=batch_size)


class CLIPTextEncoder:
    """CLIP text encoder for query encoding (to match image embeddings)."""

    def __init__(self, config: Optional[ImageEncoderConfig] = None):
        self.config = config or ImageEncoderConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP text encoder: {self.config.model_name} on {self.device}")
        self.processor = CLIPProcessor.from_pretrained(self.config.model_name)
        self.model = CLIPModel.from_pretrained(self.config.model_name).to(self.device)
        self.model.eval()
        print("CLIP text encoder loaded.")
        # CLIP text embedding dimension - use projection dimension
        # For CLIP models, the final embedding dim matches image embedding dim
        if "base" in self.config.model_name.lower():
            self.embedding_dim = 512
        elif "large" in self.config.model_name.lower():
            self.embedding_dim = 768
        else:
            # Fallback: use text config projection dim
            self.embedding_dim = getattr(self.model.config.text_config, "projection_dim", 512)

    def encode(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        """Encode a list of texts into float32 numpy vectors."""
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype="float32")

        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            # Process text - same as your working implementation
            inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(
                self.device
            )

            with torch.no_grad():
                # Use get_text_features - same as your working implementation
                text_features = self.model.get_text_features(**inputs)
                # Normalize embeddings - same as your working implementation
                if self.config.normalize:
                    text_features = F.normalize(text_features, p=2, dim=1)
                batch_arr = text_features.cpu().numpy().astype("float32")
                all_embeddings.append(batch_arr)

        return np.vstack(all_embeddings)

