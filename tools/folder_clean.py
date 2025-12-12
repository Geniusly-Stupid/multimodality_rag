import shutil
from pathlib import Path

def flatten_pages_texts(
    pages_dir: Path = Path("data/frames_wiki_dataset/pages"),
    output_dir: Path = Path("data/frames_wiki_dataset/merged_texts"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pages_dir.exists() or not pages_dir.is_dir():
        raise FileNotFoundError(f"Pages directory not found: {pages_dir}")

    for subdir in sorted(pages_dir.iterdir()):
        if not subdir.is_dir():
            continue
        text_file = subdir / "text.txt"
        if not text_file.exists():
            continue

        dest = output_dir / f"{subdir.name}.txt"
        shutil.move(str(text_file), dest)
        # Remove the now-empty subfolder (and any remaining contents, if any)
        shutil.rmtree(subdir)

    print(f"Done. Merged texts are in: {output_dir.resolve()}")

if __name__ == "__main__":
    flatten_pages_texts()
