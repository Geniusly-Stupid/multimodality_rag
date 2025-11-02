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
  - **`problems.json`**: This file contains the questions, choices, and other text data. It should be placed inside the `adaptive-alpha/` directory.
  - **Images**: The image files corresponding to the problems should be placed in a directory structure like `images/test/<problem_id>/image.png`.

- **Analysis Results (for training)**:
  - To train the dynamic alpha model from scratch, you first need to generate the analysis data.
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
