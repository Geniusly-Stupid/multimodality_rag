"""CLIP-based image encoder for multimodal RAG - using exact structure from working code."""

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

DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"


@dataclass
class ImageEncoderConfig:
    """Configuration for the CLIP image encoder."""

    model_name: str = DEFAULT_CLIP_MODEL
    normalize: bool = True


class ImageEncoder:
    """Wrapper around CLIP model for image embedding - using exact structure from working code."""

    def __init__(self, config: Optional[ImageEncoderConfig] = None):
        self.config = config or ImageEncoderConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # macOS compatibility: set torch to use single thread to avoid segfaults
        if self.device == "cpu" and hasattr(torch, 'set_num_threads'):
            try:
                torch.set_num_threads(1)
            except:
                pass
        
        # Load Image Encoder (CLIP) - exactly as your working code
        print(f"Loading CLIP model: {self.config.model_name}")
        try:
            self.processor = CLIPProcessor.from_pretrained(
                self.config.model_name,
                use_fast=False  # Use slow processor to avoid macOS issues
            )
        except Exception as e:
            print(f"Warning: Failed to load processor with use_fast=False: {e}")
            self.processor = CLIPProcessor.from_pretrained(self.config.model_name)
        
        self.model = CLIPModel.from_pretrained(self.config.model_name).to(self.device)
        self.model.eval()  # Set to evaluation mode
        print("Image encoder loaded.")
        
        # Get embedding dimension
        if "base" in self.config.model_name.lower():
            self.embedding_dim = 512
        elif "large" in self.config.model_name.lower():
            self.embedding_dim = 768
        else:
            self.embedding_dim = 512

    def encode(self, images: Sequence[Image.Image], batch_size: int = 16) -> np.ndarray:
        """
        Encodes a list of images into embeddings - exactly as your working code.
        
        Args:
            images: A list of PIL Image objects to encode.
            
        Returns:
            numpy.ndarray: A numpy array of shape (len(images), embedding_dim)
        """
        if not images:
            return np.zeros((0, self.embedding_dim), dtype="float32")
        
        # Get tensor embeddings first (like your working code)
        embeddings_tensor = self.encode_tensor(images, batch_size)
        # Convert to numpy
        return embeddings_tensor.cpu().numpy().astype("float32")
    
    def encode_tensor(self, images: Sequence[Image.Image], batch_size: int = 16) -> torch.Tensor:
        """
        Encodes a list of images into embeddings - exactly as your working code.
        Processes in batches to avoid memory issues.
        
        Args:
            images: A list of PIL Image objects to encode.
            batch_size: Number of images to process at once.
            
        Returns:
            torch.Tensor: A tensor of shape (len(images), embedding_dim)
        """
        if not images:
            return torch.empty(0, device=self.device)
        
        # Ensure all images are RGB and properly formatted (fix for macOS segmentation fault)
        processed_images = []
        for img in images:
            if not isinstance(img, Image.Image):
                raise ValueError(f"Expected PIL Image, got {type(img)}")
            # Ensure image is RGB mode to avoid issues
            if img.mode != "RGB":
                img = img.convert("RGB")
            # Create a fresh copy to avoid file handle issues (macOS fix)
            processed_images.append(img.copy() if hasattr(img, 'copy') else img)
        
        # If batch_size is 1 or images fit in one batch, process directly
        if len(processed_images) <= batch_size:
            # Process images - ensure we pass a list and handle potential macOS issues
            # On macOS, sometimes processing needs to be done more carefully
            try:
                # Use images parameter explicitly as a list
                inputs = self.processor(images=list(processed_images), return_tensors="pt")
                # Move to device after processing (safer on macOS)
                if self.device != "cpu":
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in inputs.items()}
                else:
                    # Ensure tensors are on CPU explicitly
                    inputs = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                             for k, v in inputs.items()}
            except (RuntimeError, ValueError, SystemError) as e:
                # Fallback: try processing one at a time (common macOS workaround)
                if len(processed_images) == 1:
                    inputs = self.processor(images=[processed_images[0]], return_tensors="pt")
                    if self.device != "cpu":
                        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                 for k, v in inputs.items()}
                    else:
                        inputs = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                 for k, v in inputs.items()}
                else:
                    raise e
            
            # Compute image embeddings - exactly as your working code
            with torch.no_grad():
                image_embeddings = self.model.get_image_features(**inputs)
                
            # Normalize embeddings - exactly as your working code
            if self.config.normalize:
                image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
            
            return image_embeddings
        
        # Process in batches
        all_embeddings = []
        for start in range(0, len(processed_images), batch_size):
            batch_images = processed_images[start : start + batch_size]
            # Process images - ensure we pass a list
            try:
                inputs = self.processor(images=list(batch_images), return_tensors="pt")
                # Move to device after processing (safer on macOS)
                if self.device != "cpu":
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in inputs.items()}
                else:
                    inputs = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                             for k, v in inputs.items()}
            except (RuntimeError, ValueError, SystemError) as e:
                # Fallback: process one at a time (macOS workaround)
                batch_embeddings = []
                for single_img in batch_images:
                    single_inputs = self.processor(images=[single_img], return_tensors="pt")
                    if self.device != "cpu":
                        single_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                        for k, v in single_inputs.items()}
                    else:
                        single_inputs = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                        for k, v in single_inputs.items()}
                    with torch.no_grad():
                        single_emb = self.model.get_image_features(**single_inputs)
                    if self.config.normalize:
                        single_emb = F.normalize(single_emb, p=2, dim=1)
                    batch_embeddings.append(single_emb)
                image_embeddings = torch.cat(batch_embeddings, dim=0)
                all_embeddings.append(image_embeddings)
                continue
            
            # Compute image embeddings - exactly as your working code
            with torch.no_grad():
                image_embeddings = self.model.get_image_features(**inputs)
                
            # Normalize embeddings - exactly as your working code
            if self.config.normalize:
                image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
            
            all_embeddings.append(image_embeddings)
        
        # Concatenate all batches
        return torch.cat(all_embeddings, dim=0)


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
        
        # Get embedding dimension
        if "base" in self.config.model_name.lower():
            self.embedding_dim = 512
        elif "large" in self.config.model_name.lower():
            self.embedding_dim = 768
        else:
            self.embedding_dim = 512

    def encode(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        """Encode a list of texts into float32 numpy vectors."""
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype="float32")

        # Process text - exactly as your working code
        inputs = self.processor(text=list(texts), return_tensors="pt", padding=True, truncation=True).to(
            self.device
        )

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            if self.config.normalize:
                text_features = F.normalize(text_features, p=2, dim=1)
        
        return text_features.cpu().numpy().astype("float32")

