"""Build FAISS index for Flickr30K images using CLIP."""

from __future__ import annotations

import os

# Fix OpenMP library conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from datasets import load_dataset
from PIL import Image

# Handle imports for both module and direct execution
import sys
from pathlib import Path

# Add parent directory to path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retriever.image_encoder import ImageEncoder, ImageEncoderConfig

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "faiss_index" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_IMAGES = 10
FAISS_INDEX_TYPE = "flat"
FAISS_METRIC = "ip"
BATCH_SIZE = 1
USE_DUMMY_IMAGES = False  # Set to False to load from local directory
LOCAL_IMAGE_DIR = "data/flickr8k/Images"  # Path to Flickr8K images directory
CAPTIONS_FILE = "data/flickr8k/captions.txt"  # Path to captions file


def load_flickr30k_images(num_images: int = NUM_IMAGES) -> List[Dict[str, Any]]:
    """Load images from local directory or create dummy images for testing."""
    print(f"Loading {num_images} images...")
    
    # Try loading from local directory first (Flickr8K)
    if LOCAL_IMAGE_DIR and Path(LOCAL_IMAGE_DIR).exists():
        print(f"Loading images from local directory: {LOCAL_IMAGE_DIR}")
        captions_file = Path(CAPTIONS_FILE) if CAPTIONS_FILE else None
        return _load_flickr8k_images(LOCAL_IMAGE_DIR, num_images, captions_file)
    
    # Fallback to dummy images for testing
    if USE_DUMMY_IMAGES:
        print("Note: Creating dummy images for testing.")
        print("To use real images:")
        print("  1. Set LOCAL_IMAGE_DIR to a directory containing images")
        print("  2. Set CAPTIONS_FILE to the captions file path (optional)")
        print()
        return _create_dummy_images(num_images)
    
    raise ValueError("No image source specified. Set LOCAL_IMAGE_DIR or USE_DUMMY_IMAGES=True")


def _load_flickr8k_images(image_dir: str, num_images: int, captions_file: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load images from Flickr8K dataset directory with captions."""
    from PIL import Image
    import csv
    
    image_path = Path(image_dir)
    if not image_path.exists():
        raise ValueError(f"Image directory does not exist: {image_dir}")
    
    # Load captions if available
    captions_dict: Dict[str, List[str]] = {}
    if captions_file and captions_file.exists():
        print(f"Loading captions from {captions_file}...")
        with captions_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_name = row.get("image", "").strip()
                caption = row.get("caption", "").strip()
                if image_name and caption:
                    if image_name not in captions_dict:
                        captions_dict[image_name] = []
                    captions_dict[image_name].append(caption)
        print(f"Loaded captions for {len(captions_dict)} images")
    
    # Find all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_path.glob(f"*{ext}"))
        image_files.extend(image_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        raise ValueError(f"No image files found in {image_dir}")
    
    # Sort and limit
    image_files = sorted(image_files)[:num_images]
    print(f"Found {len(image_files)} images in directory (loading {num_images})")
    
    image_records: List[Dict[str, Any]] = []
    for idx, img_path in enumerate(image_files):
        try:
            image = Image.open(img_path).convert("RGB")
            image_name = img_path.name
            
            # Get captions for this image
            captions = captions_dict.get(image_name, [])
            if not captions:
                captions = [f"Image {image_name}"]
            
            record = {
                "id": idx,
                "image": image,
                "image_id": img_path.stem,
                "captions": captions,
                "caption": captions[0] if captions else f"Image {image_name}",
            }
            image_records.append(record)
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            continue
    
    print(f"Loaded {len(image_records)} images from directory.")
    return image_records


def _create_dummy_images(num_images: int) -> List[Dict[str, Any]]:
    """Create dummy images for testing when dataset is unavailable."""
    from PIL import Image
    import random
    
    image_records: List[Dict[str, Any]] = []
    test_captions = [
        "a dog playing in the park",
        "people eating food at a restaurant",
        "a beautiful sunset over the ocean",
        "children playing soccer",
        "a cat sitting on a windowsill",
        "a busy city street",
        "mountains covered in snow",
        "a person reading a book",
        "flowers in a garden",
        "a car driving on a highway",
    ]
    
    for idx in range(num_images):
        # Create a simple colored image
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        image = Image.new("RGB", (224, 224), color=color)
        caption = test_captions[idx % len(test_captions)]
        
        record = {
            "id": idx,
            "image": image,
            "image_id": f"dummy_{idx}",
            "captions": [caption],
            "caption": caption,
        }
        image_records.append(record)
    
    print(f"Created {len(image_records)} dummy images for testing.")
    return image_records


def build_faiss_index(embeddings: np.ndarray, index_type: str = FAISS_INDEX_TYPE, metric: str = FAISS_METRIC) -> faiss.Index:
    """Build FAISS index from embeddings."""
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")
    num_rows, dim = embeddings.shape
    if num_rows == 0:
        raise ValueError("No embeddings provided for FAISS index.")

    xb = np.ascontiguousarray(embeddings.astype("float32"))
    use_ip = metric.lower() == "ip"

    if index_type.lower() == "flat":
        index = faiss.IndexFlatIP(dim) if use_ip else faiss.IndexFlatL2(dim)
    else:
        raise ValueError(f"Unsupported FAISS index type: {index_type}")

    index.add(xb)
    return index


def save_image_metadata(path: Path, records: List[Dict[str, Any]]) -> None:
    """Save image metadata to JSONL."""
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            # Convert PIL Image to None (can't serialize)
            metadata = {
                "id": record["id"],
                "image_id": record["image_id"],
                "caption": record["caption"],
                "captions": record["captions"],
                "modality": "image",
            }
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")


def main() -> None:
    start_time = time.time()

    # Load images
    image_records = load_flickr30k_images(NUM_IMAGES)
    if not image_records:
        raise SystemExit("No images loaded.")

    # Extract PIL Images and ensure they're properly formatted
    # Reload images to avoid any potential issues with file handles
    from PIL import Image as PILImage
    images = []
    for record in image_records:
        img = record["image"]
        # Ensure image is RGB and create a fresh copy
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Create a copy to avoid file handle issues
        img_copy = img.copy()
        images.append(img_copy)

    # Encode images
    print(f"\nInitializing CLIP encoder...")
    encoder_config = ImageEncoderConfig()
    
    # Force garbage collection before loading model
    import gc
    gc.collect()
    
    try:
        print("Creating ImageEncoder instance...")
        encoder = ImageEncoder(encoder_config)
        print(f"✓ CLIP model loaded successfully (embedding dim: {encoder.embedding_dim})")
        
        # Test encoding a single dummy image first (skip on macOS if it causes segfault)
        # Note: On macOS, the first forward pass can sometimes cause segfaults
        # If this fails, we'll try with real images instead
        SKIP_TEST_ENCODING = False  # Set to True to skip test encoding on macOS
        if not SKIP_TEST_ENCODING:
            print("Testing encoding with a dummy image...")
            from PIL import Image as PILImage
            # Create a properly formatted test image
            dummy_img = PILImage.new("RGB", (224, 224), color="black")
            # Ensure it's a fresh copy to avoid any reference issues
            dummy_img = dummy_img.copy()
            try:
                # Use the first real image instead of dummy if available (more reliable)
                if images and len(images) > 0:
                    test_img = images[0].copy() if hasattr(images[0], 'copy') else images[0]
                    print("Using first real image for test encoding...")
                    test_emb = encoder.encode([test_img], batch_size=1)
                else:
                    test_emb = encoder.encode([dummy_img], batch_size=1)
                print(f"✓ Test encoding successful, shape: {test_emb.shape}")
            except (SystemError, RuntimeError, ValueError) as e:
                print(f"⚠ Warning: Test encoding failed: {e}")
                print("This might be a macOS compatibility issue. Continuing with real images...")
                import traceback
                traceback.print_exc()
            except Exception as e:
                print(f"⚠ Warning: Test encoding failed with unexpected error: {e}")
                print("Continuing anyway...")
                import traceback
                traceback.print_exc()
        else:
            print("Skipping test encoding (macOS compatibility mode)")
        
    except Exception as e:
        print(f"✗ Failed to load CLIP model: {e}")
        import traceback
        traceback.print_exc()
        raise
    except SystemExit as e:
        print(f"✗ System exit during CLIP model loading: {e}")
        raise
    except BaseException as e:
        print(f"✗ Unexpected error during CLIP model loading: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"\nEncoding {len(images)} images with {encoder_config.model_name}...")
    print(f"Using batch size: {BATCH_SIZE}")
    encode_start = time.time()
    try:
        # Encode images one by one to avoid segmentation fault
        print("Starting encoding (one by one to avoid issues)...")
        all_embeddings = []
        for idx, img in enumerate(images):
            if (idx + 1) % 10 == 0:
                print(f"  Encoded {idx + 1}/{len(images)} images...")
            try:
                # Encode single image
                emb = encoder.encode([img], batch_size=1)
                all_embeddings.append(emb[0])  # Get first (and only) embedding
            except Exception as e:
                print(f"  Warning: Failed to encode image {idx}: {e}")
                # Create zero embedding as fallback
                all_embeddings.append(np.zeros(encoder.embedding_dim, dtype="float32"))
        
        embeddings = np.vstack(all_embeddings)
        encode_time = time.time() - encode_start
        print(f"✓ Encoded {embeddings.shape[0]} images in {encode_time:.2f}s. Dim={embeddings.shape[1]}")
    except Exception as e:
        print(f"✗ Failed to encode images: {e}")
        import traceback
        traceback.print_exc()
        raise
    except BaseException as e:
        print(f"✗ Unexpected error during encoding: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Build FAISS index
    index = build_faiss_index(embeddings, index_type=FAISS_INDEX_TYPE, metric=FAISS_METRIC)
    print(f"FAISS index built ({index.__class__.__name__}) with {index.ntotal} vectors.")

    # Save files
    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)
    faiss.write_index(index, str(OUTPUT_DIR / "index.faiss"))
    save_image_metadata(OUTPUT_DIR / "images.jsonl", image_records)

    metadata = {
        "num_images": len(image_records),
        "embedding_dim": embeddings.shape[1],
        "encoder_name": encoder_config.model_name,
        "faiss_index_type": index.__class__.__name__,
        "dataset": "flickr30k",
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    elapsed = time.time() - start_time
    print(f"\nSaved image index to {OUTPUT_DIR.resolve()}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Images: {len(image_records)}, Embeddings: {embeddings.shape}")


if __name__ == "__main__":
    main()

