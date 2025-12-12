"""
Multimodal parser using ONLY raganything.modalprocessors,
following exactly the same calling structure you provided.

- Uses LightRAG (no API key required)
- Replaces OpenAI caption function with local BLIP captioner
- Calls ImageModalProcessor.process_multimodal_content
- Calls TableModalProcessor.process_multimodal_content
"""

import asyncio
import base64
from pathlib import Path

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from raganything.modalprocessors import ImageModalProcessor, TableModalProcessor

# Your local BLIP captioner
from image_captioner import ImageCaptioner


class MultimodalParser:
    def __init__(self):
        # ------------------------------------------
        # Initialize LightRAG (no LLM; no API usage)
        # ------------------------------------------
        self.rag = LightRAG(
            working_dir="./rag_storage",
            llm_model_func=lambda *args, **kwargs: "(no llm in this pipeline)",
            embedding_func=EmbeddingFunc(
                embedding_dim=512,
                max_token_size=8192,
                func=lambda texts: [[0] * 512 for _ in texts],  # dummy embedding
            ),
        )

        # Local BLIP captioner
        self.captioner = ImageCaptioner()

        # ------------------------------------------
        # Async caption function expected by modalprocessors
        # ------------------------------------------
        async def blip_caption_func(prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs):
            if image_data:
                import base64, io
                from PIL import Image

                img_bytes = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                caption = self.captioner.generate_caption_from_pil(img)
                return caption

            return "(no-image)"

        # Create processors
        self.image_processor = ImageModalProcessor(
            lightrag=self.rag,
            modal_caption_func=blip_caption_func,
        )

        async def table_caption_func(prompt, **kwargs):
            return f"(table summary) {prompt[:100]}"

        self.table_processor = TableModalProcessor(
            lightrag=self.rag,
            modal_caption_func=table_caption_func,
        )

    # ----------------------------------------------------
    # Main multimodal enrichment pipeline
    # ----------------------------------------------------
    async def enrich_chunks(self, chunks, file_path):
        enriched = []

        for chunk in chunks:
            out = dict(chunk)  # shallow copy
            out["enriched_description"] = None
            out["entities"] = None

            # ----------------------
            # IMAGE CHUNK
            # ----------------------
            if chunk.get("type") == "image":
                img_path = chunk["img_path"]

                # base64 encode image bytes
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()

                modal_content = {
                    "img_path": img_path,
                    "image_caption": chunk.get("image_caption", []),
                    "image_footnote": chunk.get("image_footnote", []),
                }

                description, entity = await self.image_processor.process_multimodal_content(
                    modal_content=modal_content,
                    content_type="image",
                    file_path=file_path,
                    entity_name="image",
                )

                out["enriched_description"] = description
                out["entities"] = entity

            # ----------------------
            # TABLE CHUNK
            # ----------------------
            elif chunk.get("metadata", {}).get("table_body"):
                modal_content = {
                    "table_body": chunk["metadata"]["table_body"],
                    "table_caption": chunk["metadata"].get("caption", []),
                }

                description, entity = await self.table_processor.process_multimodal_content(
                    modal_content=modal_content,
                    content_type="table",
                    file_path=file_path,
                    entity_name="table",
                )

                out["enriched_description"] = description
                out["entities"] = entity

            enriched.append(out)

        return enriched


# Convenience sync wrapper
def enrich_chunks_sync(chunks, file_path):
    parser = MultimodalParser()
    return asyncio.run(parser.enrich_chunks(chunks, file_path))
