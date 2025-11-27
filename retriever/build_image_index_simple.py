"""Simplified version of build_image_index that processes images one at a time."""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "faiss_index" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_IMAGES = 10
LOCAL_IMAGE_DIR = "data/flickr8k/Images"
CAPTIONS_FILE = "data/flickr8k/captions.txt"
MODEL_NAME = "openai/clip-vit-large-patch14"


def load_flickr8k_images(num_images: int) -> List[Dict[str, Any]]:
    """Load images from Flickr8K dataset."""
    import csv
    
    image_path = Path(LOCAL_IMAGE_DIR)
    captions_path = Path(CAPTIONS_FILE)
    
    # Load captions
    captions_dict: Dict[str, List[str]] = {}
    if captions_path.exists():
        with captions_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_name = row.get("image", "").strip()
                caption = row.get("caption", "").strip()
                if image_name and caption:
                    if image_name not in captions_dict:
                        captions_dict[image_name] = []
                    captions_dict[image_name].append(caption)
    
    # Load images
    image_files = sorted(image_path.glob("*.jpg"))[:num_images]
    records = []
    
    for idx, img_path in enumerate(image_files):
        try:
            img = Image.open(img_path).convert("RGB")
            image_name = img_path.name
            captions = captions_dict.get(image_name, [f"Image {image_name}"])
            
            records.append({
                "id": idx,
                "image_path": str(img_path),
                "image": img,  # Keep image in memory
                "image_id": img_path.stem,
                "captions": captions,
                "caption": captions[0] if captions else f"Image {image_name}",
            })
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
    
    return records


def main():
    print("=" * 80)
    print("Building Image Index (Simplified Version)")
    print("=" * 80)
    
    # Load images
    print(f"\nLoading {NUM_IMAGES} images...")
    image_records = load_flickr8k_images(NUM_IMAGES)
    print(f"Loaded {len(image_records)} images")
    
    if not image_records:
        print("No images loaded. Exiting.")
        return
    
    # Initialize CLIP - exactly like your working code
    print(f"\nLoading CLIP model: {MODEL_NAME}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print("CLIP model loaded.")
    
    # Encode images one by one
    print(f"\nEncoding {len(image_records)} images (one by one)...")
    embeddings_list = []
    
    for idx, record in enumerate(image_records):
        img = record["image"]
        print(f"  [{idx+1}/{len(image_records)}] Encoding {record['image_id']}...")
        
        try:
            # Process image - exactly like your working code
            inputs = processor(images=[img], return_tensors="pt").to(device)
            
            # Compute embeddings - exactly like your working code
            with torch.no_grad():
                image_embeddings = model.get_image_features(**inputs)
                image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
            
            # Convert to numpy
            emb_np = image_embeddings.cpu().numpy().astype("float32")[0]  # Get first (and only) embedding
            embeddings_list.append(emb_np)
            
        except Exception as e:
            print(f"    Error encoding image {idx}: {e}")
            # Add zero embedding as fallback
            embeddings_list.append(np.zeros(768, dtype="float32"))
    
    embeddings = np.vstack(embeddings_list)
    print(f"\n✓ Encoded {embeddings.shape[0]} images. Shape: {embeddings.shape}")
    
    # Build FAISS index
    print("\nBuilding FAISS index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    print(f"✓ FAISS index built with {index.ntotal} vectors")
    
    # Save files
    print("\nSaving files...")
    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)
    faiss.write_index(index, str(OUTPUT_DIR / "index.faiss"))
    
    # Save metadata
    with (OUTPUT_DIR / "images.jsonl").open("w", encoding="utf-8") as f:
        for record in image_records:
            meta = {
                "id": record["id"],
                "image_id": record["image_id"],
                "caption": record["caption"],
                "captions": record["captions"],
                "modality": "image",
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    
    metadata = {
        "num_images": len(image_records),
        "embedding_dim": embeddings.shape[1],
        "encoder_name": MODEL_NAME,
        "faiss_index_type": "IndexFlatIP",
        "dataset": "flickr8k",
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    
    print(f"\n✓ Saved all files to {OUTPUT_DIR}")
    print("=" * 80)
    print("Done!")


if __name__ == "__main__":
    main()

