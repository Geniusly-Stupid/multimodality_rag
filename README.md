<<<<<<< HEAD
# Multimodal RAG System for ScienceQA

## Overview

This project is a web-based, multimodal Retrieval-Augmented Generation (RAG) system designed to perform complex searches on the ScienceQA dataset. It features an interactive web interface where users can query the dataset using text, images, or a combination of both.

A key feature of this system is its sophisticated score-fusion mechanism, which includes a dynamic weighting strategy. A fine-tuned BERT regression model analyzes the user's query text to predict the optimal weight to assign to text and image similarity scores, allowing for more intelligent and context-aware retrieval.

## Features

- **Interactive Web UI**: A simple, single-page application for easy querying.
- **Multimodal Queries**: Supports text-only, image-only, and combined text-image searches.
- **Fast Retrieval**: Utilizes pre-computed embeddings for both text and images to ensure quick search results.
- **Dual Fusion Modes**:
  - **Fixed Ratio**: A static, pre-defined 70/30 weight for text/image similarity scores.
  - **Dynamic Ratio**: A trained BERT model intelligently decides the fusion weight based on the query's semantics.

## Architecture

- **Frontend**: A single-page application built with HTML, CSS, and vanilla JavaScript.
- **Backend**: A Flask server (`app.py`) that serves the frontend and exposes the multimodal RAG API.
- **Core Retriever Models**:
  - **Text Embeddings**: `BAAI/bge-base-en-v1.5`
  - **Image Embeddings**: `openai/clip-vit-large-patch14`
- **Dynamic Alpha Model**: A fine-tuned `bert-base-uncased` model for regression, used to predict the image dependence of a query.

## Setup and Usage

Follow these steps to set up and run the application locally.

### 1. Prerequisites

- Python 3.8+
- `pip` (Python package installer)

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Install dependencies
pip install -r requirements.txt
```

### 3. Data Setup

Due to their size, the core dataset and generated model/index files are not included in this repository. You will need to acquire and place them as follows:

- **ScienceQA Dataset**:
  - You can download the dataset from https://drive.google.com/drive/folders/16kuhXdM-MOhYcFIyRj91WvnDnjnF-xHw. For simplicity and efficiency, we currently only use the test split of the dataset.
  - **`problems.json`**: This file contains the questions, choices, and other text data. It should be placed inside the `adaptive-alpha/` directory.
  - **Images**: The image files corresponding to the problems should be placed in a directory structure like `images/test/<problem_id>/image.png`.

- **Analysis Results (for training)**:
  - To train the dynamic alpha model from scratch, you first need to generate the analysis data. Everything about adaptive-alpha folder training details are uploaded to https://drive.google.com/drive/folders/1qNMVsVL24dOC3K0kHSxyRdv0DSHPlxr6?usp=sharing, including dataset, fine-tuned models and training statistics. You can directly utilize the documents there.
  - This project assumes a pre-existing directory `adaptive-alpha/analysis_results/` populated with JSON files. These files are the output of `adaptive-alpha/analyze_questions.py`, which you would need to adapt and run on the ScienceQA dataset to determine the ground-truth `image_dependence` for each question.

### 4. Training and Pre-computation

You must run the following two scripts in order before launching the web application.

- **Train the Dynamic Alpha Model**:
  This step trains the BERT regression model that predicts the optimal fusion weight.
  ```bash
  python adaptive-alpha/train_bert.py
  ```
  This will create the `adaptive-alpha/bert_regression_final/` directory containing the trained model.

- **Pre-compute Embeddings**:
  This step encodes all text and images from the dataset and saves them as tensors for fast retrieval.
  ```bash
  python precompute_embeddings.py
  ```
  This will generate three key files in the root directory: `text_embeddings.pt`, `image_embeddings.pt`, and `index_to_info.json`.

### 5. Running the Web Application

Once the training and pre-computation are complete, you can launch the web server.

```bash
python app.py
```

Navigate to `http://127.0.0.1:5001` in your web browser to use the application.

## Project Structure

```
.
├── adaptive-alpha/         # Contains logic for the dynamic alpha model
│   ├── analysis_results/   # (Required, not in repo) Data for training the alpha model
│   ├── bert_regression_final/ # (Generated) The trained alpha prediction model
│   ├── train_bert.py       # Script to train the alpha model
│   └── ...
├── images/                 # (Required, not in repo) Image data from ScienceQA
│   └── test/
├── static/                 # (Generated) Copied images for web serving
├── templates/
│   └── index.html          # Frontend HTML and JavaScript
├── app.py                  # Main Flask web application
├── precompute_embeddings.py  # Script to generate text and image embeddings
├── retriever.py            # Core class for encoding text and images
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## TODO (Possibly, could be something else)

### 1.Improve UI Design
The current web interface is functional but visually minimal.
Add styled result cards, thumbnails for retrieved images, and progress bars to visualize dynamic α weights.
Consider lightweight UI frameworks (e.g., Bootstrap or Tailwind CSS) for a cleaner, more modern layout.

### 2.Implement Reranking Module
Introduce a cross-encoder reranker on top of the top-K retrieval results.
Example candidates:

Text: bge-reranker-v2-m3 or colbert-x

Image: CLIP cross-encoder scoring
This should refine retrieval precision without major latency overhead.

### 3.Ablation and Evaluation Experiments
Conduct systematic experiments comparing different retrieval configurations:

- Text-only retrieval

- Fixed α (e.g., 0.7/0.3) fusion

- Dynamic α (predicted by the BERT regression model)
Evaluate using Recall@K and visualize performance differences to highlight the effectiveness of adaptive fusion.
=======
# Multimodal Retrieval-Augmented QA

## Project Summary

This project investigates **multimodal retrieval** and **reasoning** by building a unified RAG (Retrieval-Augmented Generation) pipeline across text and image modalities.
The system focuses on:

* Constructing modality-specific vector stores (text, captions, wiki pages)
* Retrieving cross-modal evidence (text ↔ image)
* Integrating retrieved evidence into a generation model
* Evaluating retrieval and generation quality on multiple benchmarks

Core datasets used for evaluation:

* **SciFact** — text-based retrieval quality
* **Flickr30K** — text-to-image grounding
* **google/frames-benchmark** — full RAG pipeline (retrieval + reasoning + generation)

## Why This Matters

Multimodal RAG introduces several real-world challenges:

* **Noisy multimodal evidence** — retrieved wiki pages, captions, or images may not align well.
* **Cross-modal semantic fusion** — text and images require different encoders but must produce comparable embeddings.
* **Faithful generation** — models must rely on retrieved evidence, not hallucinate.
* **Evaluation difficulty** — exact-match scoring is insufficient for long-form generated answers.

This project aims to tackle these through **modality-adaptive retrieval**, **reranking**, and **evidence-aware generation**.

## Datasets

### **SciFact**

* Text-only retrieval benchmark
* Task: *Claim → Supporting/Refuting Documents*

### **Flickr30K**

* Image–caption dataset
* Task: *Text Query → Relevant Images*
* Used to evaluate cross-modal alignment

### **google/frames-benchmark**

* Large-scale reasoning benchmark
* Task: *Question → Multimodal Evidence → Answer*
* Our primary RAG evaluation dataset

## Methods

* Build separate vector DBs for different modalities:

  * Wikipedia pages (text chunks)
  * Flickr30K caption embeddings (Planned)
  * Optional image embeddings using CLIP (Planned)
* First-stage retrieval using FAISS (IP or L2)
* Optional cross-modal or re-ranking module
* Evidence-aware generation using models such as `Qwen2.5`
* Metrics:

  * **Retrieval:** NDCG@k, MAP@k, Recall@k, latency
  * **Generation:** Exact match, F1, semantic similarity metrics (planned)

## Repository Structure

```
project_root/
  retriever/
    build_faiss_index.py      # Build Frames Benchmark FAISS index
    rag_retriever.py          # FAISS-backed RAG retriever
    scifact_retriever.py      # SciFact text retriever
    flickr30k_retriever.py    # Flickr30K caption→image retriever
    faiss_index/              # Generated FAISS indices (output)
  generator/
    rag_generator.py          # Autoregressive generator wrapper
  evaluation/
    evaluate_rag.py           # Google Frames RAG evaluation
    evaluate_scifact.py       # SciFact retrieval evaluation
    evaluate_flickr30k.py     # (Optional) Flickr30K retrieval evaluation
  data/
    frames_wiki_dataset/      # Wikipedia pages extracted from frames
    scifact/                  # SciFact corpus (if stored locally)
    flickr30k/                # Flickr30K captions/images
  rag_pipeline.py             # High-level RAG pipeline CLI interface
  tools/
    build_frames_wiki_dataset.py     # Crawl Wikipedia pages for Frames
    build_frames_image_captions.py   # Generate image captions (BLIP/BLIP-2) for Frames
```


## Running the System

### 0. Download the Data

You have two options:

1. **Direct Download**
   Download all processed data from:
   [https://drive.google.com/drive/folders/1u30jS1L_jNXX04HAEoVsrMO-XLP0B9v4?usp=drive_link](https://drive.google.com/drive/folders/1u30jS1L_jNXX04HAEoVsrMO-XLP0B9v4?usp=drive_link)

2. **Rebuild the Frames Wiki Dataset Manually**
   If you prefer to generate the Frames Wikipedia dataset yourself, run:

   ```
   python tools/build_frames_wiki_dataset.py
   ```

### 1. (Optional) Generate image captions for Frames

Uses BLIP/BLIP-2 to turn each image URL into text; required for caption retrieval modes.

```bash
# Ensure images.txt contains direct image URLs (upload.wikimedia.org/...jpg/png)
python tools/build_frames_image_captions.py \
  --model_type blip \
  --model_name Salesforce/blip-image-captioning-large \
  --caption_type default
```

### 2. Build FAISS indices

```bash
# Text-only chunks
python retriever/build_faiss_index.py --mode text

# Caption-only (needs image_captions.jsonl)
python retriever/build_faiss_index.py --mode caption

# Build both in one run
python retriever/build_faiss_index.py --mode both
```

If you want CLIP image retrieval, also build the image index:

```bash
python retriever/build_image_index.py
```

### 3. Evaluate SciFact retrieval

```bash
python evaluation/evaluate_scifact.py --top_k 5
```

### 4. Evaluate Google Frames RAG (choose retrieval mode)

```bash
# text + caption (recommended for caption baseline)
python evaluation/evaluate_rag.py --retrieval_mode text_caption --top_k 5

# other modes: text | text_clip | caption_only
python evaluation/evaluate_rag.py --retrieval_mode caption_only --top_k 5 --limit 20
```

### 5. Run interactive RAG queries

```bash
python rag_pipeline.py \
  --query "Who is Jane Ballou?" \
  --retrieval_mode text_caption \
  --top_k 5 \
  --use_reranker
```
Supported retrieval modes:
- `text`: text chunks only
- `text_clip`: text chunks + CLIP image retrieval
- `text_caption`: text chunks + caption retrieval
- `caption_only`: captions only


## Notes

* Default text encoder: `BAAI/bge-base-en-v1.5`
* Default generator: `Qwen/Qwen2.5-1.5B-Instruct` (can be changed)
* Reranker (optional): `BAAI/bge-reranker-v2-m3`
* All generated indices and evaluation outputs are saved to:

  * `retriever/faiss_index/`
  * `evaluation_results/`


# Problems and TODOs

## **1. Missing Image Content in Wiki Pages**

* The current Wikipedia extraction stores only **text**, not **images**.
* For multimodal RAG, we need to:

  * Parse and download image URLs
  * Possibly extract CLIP/SigLIP embeddings for each image
  * Store image metadata & embeddings in a separate FAISS index

**TODO**

* Extend `build_frames_wiki_dataset.py` to extract image URLs
* Add `image_encoder.py` for image embeddings
* Fuse text and image retrieval results


## **2. Poor Answer Quality (Especially with `Qwen2.5-1.5B-Instruct`)**

* The 1.5B model often produces vague or incorrect answers.
* The model struggles with:

  * multi-hop reasoning
  * numerical reasoning
  * long contexts containing many wiki chunks

**TODO**

* Replace generator with a larger model (e.g., Qwen2.5-7B or Llama 3.1)
* Add reranker to improve evidence relevance
* Add prompt formatting to highlight key evidence
* Try CoT prompting or self-reflection prompting


## **3. Generation Evaluation Metrics Are Incomplete**

* Exact Match (EM) is too strict for Frames Benchmark
* Most answers do **not** match gold answers verbatim
* Need metrics that capture **semantic correctness** and **reasoning quality**

**TODO**

* Add token-level F1
* Add Rouge-L
* Add BLEU or METEOR
* Add semantic similarity metrics (BERTScore, SentenceBERT cosine)
* Add “faithfulness” metrics comparing retrieved evidence & generated answer
* Add human evaluation guidelines
>>>>>>> origin/img-caption
