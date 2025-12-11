from raganything_parser import MineruParser
from multimodal_parser import enrich_chunks_sync  # 你写的 MultimodalParser 包含 BLIP captioner

def parse_and_enrich(file_path):
    # Step 1: parse document
    parser = MineruParser()
    parsed_chunks = parser.parse_document(file_path)
    # print("Parsed chunks: ", parsed_chunks)
    print("parse finished. start enriching...")

    # Step 2: multimodal enrichment
    enriched = enrich_chunks_sync(parsed_chunks, file_path=file_path)
    print("enriching finished.")
    return enriched


if __name__ == "__main__":
    enriched_chunks = parse_and_enrich(r"D:\Desktop\SI650\project\multimodality_rag\data\SI650-Week-01-Course.pdf")
    print(enriched_chunks)   # preview
