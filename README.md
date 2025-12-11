# Multimodal RAG with RAGAnything

This repository now centers on a minimal, caption-first Retrieval-Augmented Generation pipeline powered by **RAGAnything**. Documents are parsed via the unified `RAGAnythingParser`, chunked semantically, and indexed through a lightweight FAISS-backed `RetrieverAdapter` that embeds text and captions only (no image encoder required).

## Architecture
- **Parsing & Chunking**: `parser/raganything_parser.py` uses `MultiModalRAG` to load, parse, and chunk documents, returning normalized records with text, optional image path, and metadata.
- **Retrieval**: `retriever/retriever_adapter.py` wraps a simple FAISS store (`FaissVectorStore`). Chunks are embedded using the text encoder only; image content is represented by its caption/text.
- **Generation**: `generator/rag_generator.py` formats retrieved evidence (text + caption) into a prompt for the causal LM.
- **Pipeline**: `rag_pipeline.py` wires parser → adapter → generator for end-to-end QA.

## Workflow
1) **Parse & index a document**
```bash
python rag_pipeline.py --doc_path /path/to/document --query "Your question"
```
This:
- loads/parses/chunks the document with RAGAnything,
- builds an in-memory FAISS index over text/caption fields,
- runs retrieval with the caption/text encoder only,
- generates an answer with the configured LM.

2) **Programmatic usage**
```python
from parser.raganything_parser import RAGAnythingParser
from retriever.retriever_adapter import FaissVectorStore, RetrieverAdapter
from retriever.rag_retriever import TextEncoder, EncoderConfig
from generator.rag_generator import RAGGenerator, GeneratorConfig

parser = RAGAnythingParser()
chunks = parser.parse("data/sample.pdf")

encoder = TextEncoder(EncoderConfig())
store = FaissVectorStore(metric="ip")
retriever = RetrieverAdapter(text_encoder=encoder, vector_store=store)
retriever.build_index(chunks)

generator = RAGGenerator(GeneratorConfig())
results = retriever.retrieve("What happened in 1984?", top_k=5)
answer = generator.generate("What happened in 1984?", results)
```

## Directory Snapshot
- `parser/raganything_parser.py` — unified parser/chunker
- `retriever/retriever_adapter.py` — caption/text-only adapter + FAISS store
- `retriever/rag_retriever.py` — text encoder wrapper (BGE) reused by the adapter
- `generator/rag_generator.py` — prompt builder + generation
- `rag_pipeline.py` — CLI pipeline entrypoint
- `tools/` & `evaluation/` — legacy utilities (kept for reference; not used by the new flow)

## Requirements
- Python 3.9+
- PyTorch + transformers
- faiss-cpu
- raganything

Install base deps:
```bash
pip install -r requirements.txt
pip install faiss-cpu raganything
```

## Differences vs. Old Version
- Replaced bespoke chunking/FAISS builders with RAGAnything’s unified parser.
- Retrieval now indexes only text/captions; image encoder paths and CLIP-based image search are removed from the default pipeline.
- Pipeline builds indexes on the fly per document (no prebuilt wiki/caption indices required).
- README rewritten to match the new architecture; legacy scripts remain but are no longer primary entrypoints.

## TODO / Future Work
- Prune or port remaining legacy utilities to the RAGAnything flow.
- Add persistence for FAISS indexes built from RAGAnything chunks.
- Extend prompt formatting for richer multimodal metadata when available.

