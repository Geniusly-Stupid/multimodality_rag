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

### 1. Build the Frames FAISS index

```bash
python retriever/build_faiss_index.py
```

### 2. Evaluate SciFact retrieval

```bash
python evaluation/evaluate_scifact.py --top_k 5
```

### 3. Evaluate Google Frames RAG

```bash
python evaluation/evaluate_rag.py --top_k 5
```

### 4. Run interactive RAG queries

```bash
python rag_pipeline.py --query "Who is Jane Ballou?" --top_k 5
```


## Notes

* Default text encoder: `BAAI/bge-base-en-v1.5`
* Default generator: `Qwen/Qwen2.5-1.5B-Instruct` (can be changed)
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