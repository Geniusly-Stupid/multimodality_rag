"""Test script to verify image loading works."""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from retriever.build_image_index import load_flickr30k_images

# Test loading 10 images
print("Testing image loading...")
records = load_flickr30k_images(10)
print(f"\nSuccessfully loaded {len(records)} images")
print("\nFirst 3 images:")
for i, r in enumerate(records[:3], 1):
    print(f"{i}. ID: {r['image_id']}")
    print(f"   Caption: {r['caption']}")
    print(f"   All captions: {len(r['captions'])} captions")
    print()

