"""Step-by-step test for building image index."""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from retriever.build_image_index import load_flickr30k_images, build_faiss_index, save_image_metadata
from retriever.image_encoder import ImageEncoder, ImageEncoderConfig
import numpy as np
import faiss
import json

print("=" * 80)
print("Step 1: Loading images...")
print("=" * 80)
NUM_IMAGES = 10  # Start with 10 images
image_records = load_flickr30k_images(NUM_IMAGES)
print(f"✓ Loaded {len(image_records)} images\n")

print("=" * 80)
print("Step 2: Loading CLIP model...")
print("=" * 80)
try:
    encoder_config = ImageEncoderConfig()
    encoder = ImageEncoder(encoder_config)
    print(f"✓ CLIP model loaded (embedding dim: {encoder.embedding_dim})\n")
except Exception as e:
    print(f"✗ Failed to load CLIP model: {e}")
    sys.exit(1)

print("=" * 80)
print("Step 3: Encoding images...")
print("=" * 80)
try:
    images = [record["image"] for record in image_records]
    print(f"Encoding {len(images)} images...")
    embeddings = encoder.encode(images, batch_size=4)  # Smaller batch
    print(f"✓ Encoded {embeddings.shape[0]} images. Shape: {embeddings.shape}\n")
except Exception as e:
    print(f"✗ Failed to encode images: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 80)
print("Step 4: Building FAISS index...")
print("=" * 80)
try:
    index = build_faiss_index(embeddings)
    print(f"✓ FAISS index built with {index.ntotal} vectors\n")
except Exception as e:
    print(f"✗ Failed to build index: {e}")
    sys.exit(1)

print("=" * 80)
print("All steps completed successfully!")
print("=" * 80)
print(f"Summary:")
print(f"  - Images loaded: {len(image_records)}")
print(f"  - Embeddings shape: {embeddings.shape}")
print(f"  - Index vectors: {index.ntotal}")

