from raganything_parser import MineruParser
from multimodal_parser import enrich_chunks_sync  # 你写的 MultimodalParser 包含 BLIP captioner
import os
import json
from pathlib import Path

def parse_and_enrich(file_path):
    # Step 1: parse document
    parser = MineruParser()
    parsed_chunks = parser.parse_document(file_path)
    # print("Parsed chunks: ", parsed_chunks)
    print("parse finished.")

    # # Step 2: multimodal enrichment
    # print("start enriching...")
    # enriched = enrich_chunks_sync(parsed_chunks, file_path=file_path)
    # print("enriching finished.")
    return parsed_chunks


def process_folder(folder_path, output_dir=None):
    folder_path = Path(folder_path)
    assert folder_path.exists(), f"Folder not found: {folder_path}"

    # Create output folder
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for file in folder_path.iterdir():
        if not file.is_file():
            continue

        # Skip non-supported formats if needed
        if file.suffix.lower() not in [".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg"]:
            print(f"Skipping unsupported file: {file.name}")
            continue

        enriched = parse_and_enrich(str(file))

        # Save result
        if output_dir:
            out_path = output_dir / f"{file.stem}_enriched.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(enriched, f, indent=2, ensure_ascii=False)
            print(f"  💾 Saved to {out_path}")


if __name__ == "__main__":
    input_folder = r"D:\Desktop\SI650\project\multimodality_rag\data\SI650\pdf"
    output_folder = r"D:\Desktop\SI650\project\multimodality_rag\data\SI650\processed"

    process_folder(input_folder, output_folder)
    print("\n🎉 All files processed!")
