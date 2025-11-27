"""Check if any Wikipedia pages have images (num_images > 0)."""

from __future__ import annotations

import json
from pathlib import Path

# Path to the pages directory
PAGES_DIR = Path("data/frames_wiki_dataset/pages")
if not PAGES_DIR.exists():
    PAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "frames_wiki_dataset" / "pages"


def check_images_in_pages() -> None:
    """Check all pages for num_images > 0 and also check images.txt files."""
    if not PAGES_DIR.exists():
        print(f"Error: Pages directory not found at {PAGES_DIR}")
        return

    pages_with_images: list[dict] = []
    pages_with_images_txt: list[dict] = []
    total_pages = 0
    pages_with_meta = 0

    print(f"Scanning pages in {PAGES_DIR}...")
    print("-" * 80)

    for page_dir in sorted(PAGES_DIR.iterdir()):
        if not page_dir.is_dir():
            continue

        total_pages += 1
        meta_path = page_dir / "meta.json"
        images_txt_path = page_dir / "images.txt"

        # Check meta.json
        if meta_path.exists():
            pages_with_meta += 1
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)

                num_images = meta.get("num_images", 0)
                if num_images > 0:
                    pages_with_images.append(
                        {
                            "page_id": page_dir.name,
                            "title": meta.get("title", "Unknown"),
                            "url": meta.get("url", ""),
                            "num_images": num_images,
                        }
                    )
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to parse {meta_path}: {e}")

        # Check images.txt
        if images_txt_path.exists():
            try:
                with images_txt_path.open("r", encoding="utf-8") as f:
                    image_urls = [line.strip() for line in f if line.strip()]
                if image_urls:
                    title = "Unknown"
                    url = ""
                    if meta_path.exists():
                        try:
                            with meta_path.open("r", encoding="utf-8") as f:
                                meta = json.load(f)
                                title = meta.get("title", "Unknown")
                                url = meta.get("url", "")
                        except (json.JSONDecodeError, KeyError):
                            pass
                    pages_with_images_txt.append(
                        {
                            "page_id": page_dir.name,
                            "title": title,
                            "url": url,
                            "num_images_in_txt": len(image_urls),
                        }
                    )
            except Exception as e:
                print(f"Warning: Failed to read {images_txt_path}: {e}")

    # Print results
    print(f"\nTotal page directories: {total_pages}")
    print(f"Pages with meta.json: {pages_with_meta}")
    print(f"Pages with images (num_images > 0 in meta.json): {len(pages_with_images)}")
    print(f"Pages with images (images.txt has URLs): {len(pages_with_images_txt)}")
    print("-" * 80)

    if pages_with_images:
        print("\nPages with num_images > 0 in meta.json:")
        print("=" * 80)
        for idx, page in enumerate(pages_with_images, start=1):
            print(f"{idx}. {page['title']}")
            print(f"   Page ID: {page['page_id']}")
            print(f"   URL: {page['url']}")
            print(f"   Number of images: {page['num_images']}")
            print()
    else:
        print("\nNo pages found with num_images > 0 in meta.json.")

    if pages_with_images_txt:
        print("\nPages with image URLs in images.txt:")
        print("=" * 80)
        for idx, page in enumerate(pages_with_images_txt[:20], start=1):  # Show first 20
            print(f"{idx}. {page['title']}")
            print(f"   Page ID: {page['page_id']}")
            print(f"   URL: {page['url']}")
            print(f"   Number of image URLs: {page['num_images_in_txt']}")
            print()
        if len(pages_with_images_txt) > 20:
            print(f"... and {len(pages_with_images_txt) - 20} more pages with images.")
    else:
        print("\nNo pages found with image URLs in images.txt files.")


if __name__ == "__main__":
    check_images_in_pages()

