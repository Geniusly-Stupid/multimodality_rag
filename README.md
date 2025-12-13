# Multimodal RAG

This repository implements a **minimal, caption-first Multimodal Retrieval-Augmented Generation (RAG) pipeline**, built on top of **RAG-Anything**.
The pipeline focuses on **textual representations only**, where visual content (e.g., images or tables) is incorporated via **captions or textual summaries**, without requiring an image encoder.

Documents are parsed, semantically chunked, embedded, and indexed using a lightweight **FAISS** vector store. The system then performs retrieval and uses a large language model (LLM) to generate answers based on the retrieved evidence.

For more implementation details and design rationale, please refer to the project report:
👉 [https://ai.feishu.cn/wiki/H8ugwqrqGiGw6Ok21eYc0yhinjf?from=from_copylink](https://ai.feishu.cn/wiki/H8ugwqrqGiGw6Ok21eYc0yhinjf?from=from_copylink)


## Architecture Overview

The pipeline consists of four main components:

### 1. Parsing & Chunking

* Implemented in: `parser/parser.py`
* Responsibilities:

  * Load documents (PDFs or text files)
  * Parse content using RAG-Anything
  * Perform semantic chunking
  * Output **normalized records** containing:

    * text content
    * optional image path
    * metadata (document ID, section, etc.)

### 2. Retrieval

* Implemented in: `retriever/retriever_adapter.py`
* Uses a lightweight FAISS-based vector store (`FaissVectorStore`)
* Key characteristics:

  * Only **text encoders** are used
  * Image content is represented **indirectly via captions**
  * Supports fast nearest-neighbor search over embedded chunks

### 3. Generation

* Implemented in: `generator/rag_generator.py`
* Responsibilities:

  * Format retrieved evidence (text + captions)
  * Construct prompts for the causal language model
  * Generate final answers using the LLM

### 4. End-to-End Pipeline

* Implemented in: `rag_pipeline.py`
* Wires together:

  * Vector database construction
  * Retriever
  * Generator
* Enables end-to-end question answering from raw documents


## How to Run the Pipeline

Run the full RAG pipeline with a remote LLM:

```bash
python rag_pipeline.py \
  --doc_path data/frames_wiki_dataset \
  --query "What is ...Baby One More Time?" \
  --use_remote \
  --remote_model "qwen/qwen3-next-80b-a3b-instruct" \
  --remote_api_base "https://integrate.api.nvidia.com/v1" \
  --remote_api_key [APIKEY]
```

This command will:

* Parse and chunk documents under `doc_path`
* Build an **in-memory FAISS index** over text and caption fields
* Retrieve relevant chunks using text embeddings only
* Generate an answer using the specified remote LLM


## How to Evaluate (Google Frames Benchmark)

To evaluate the pipeline on the Google Frames Wiki dataset:

```bash
python evaluation/evaluate_frames_wiki.py \
  --doc_path data/frames_wiki_dataset/page \
  --use_remote \
  --remote_model "qwen/qwen3-next-80b-a3b-instruct" \
  --remote_api_base "https://integrate.api.nvidia.com/v1" \
  --remote_api_key [APIKEY]
```

This script will:

* Build an in-memory FAISS index over text and caption fields
* Run retrieval and generation for all evaluation queries
* Compute evaluation metrics for the dataset


## Reference
1. https://github.com/HKUDS/RAG-Anything