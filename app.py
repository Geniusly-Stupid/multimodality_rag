from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import torch.nn.functional as F
import json
from retriever import Retriever
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__, static_folder='static')

# --- 1. Initialize Retriever and Load Pre-computed Data ---
print("Initializing Retriever...")
retriever = Retriever()
print("Retriever initialized.")

print("Loading pre-computed embeddings and mapping...")
try:
    # Load the pre-computed text embeddings
    CANDIDATE_EMBEDDINGS = torch.load('text_embeddings.pt', map_location=retriever.device)
    print(f"Loaded {CANDIDATE_EMBEDDINGS.shape[0]} candidate text embeddings.")

    # Load the pre-computed image embeddings
    IMAGE_EMBEDDINGS = torch.load('image_embeddings.pt', map_location=retriever.device)
    print(f"Loaded {IMAGE_EMBEDDINGS.shape[0]} candidate image embeddings.")

    # Load the index-to-info mapping
    with open('index_to_info.json', 'r') as f:
        INDEX_TO_INFO = json.load(f)
    print(f"Loaded info for {len(INDEX_TO_INFO)} candidates.")

    # Basic validation
    assert CANDIDATE_EMBEDDINGS.shape[0] == len(INDEX_TO_INFO),         "Mismatch between text embeddings and info mapping count!"
    assert IMAGE_EMBEDDINGS.shape[0] == len(INDEX_TO_INFO),         "Mismatch between image embeddings and info mapping count!"

    # --- 2. Load Alpha Prediction Model ---
    print("Loading Alpha Prediction Model...")
    try:
        ALPHA_MODEL_PATH = 'adaptive-alpha/bert_regression_final'
        alpha_tokenizer = BertTokenizer.from_pretrained(ALPHA_MODEL_PATH)
        alpha_model = BertForSequenceClassification.from_pretrained(ALPHA_MODEL_PATH)
        alpha_model.eval() # Set to evaluation mode
        print("Alpha Prediction Model loaded.")
    except Exception as e:
        print(f"Warning: Could not load Alpha Prediction Model. Dynamic mode will not be available. Error: {e}")
        alpha_model = None
        alpha_tokenizer = None

except FileNotFoundError as e:
    print(f"FATAL: Could not load pre-computed files: {e}")
    print("Please run 'precompute_embeddings.py' first.")
    CANDIDATE_EMBEDDINGS = None
    IMAGE_EMBEDDINGS = None
    INDEX_TO_INFO = []
    alpha_model = None

def predict_alpha(text: str):
    """Predicts the alpha value (text weight) based on the query text."""
    if not alpha_model or not alpha_tokenizer:
        print("Alpha prediction model not available, returning default alpha.")
        return 0.7 # Default alpha

    try:
        inputs = alpha_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = alpha_model(**inputs)
        
        # The raw logit from the regression model is the direct prediction.
        # We should not apply a sigmoid function here.
        image_dependence = outputs.logits.item()
        # Clamp the value between 0 and 1 to be safe from out-of-range predictions
        image_dependence = max(0.0, min(1.0, image_dependence))

        alpha = 1.0 - image_dependence
        print(f"Predicted image dependence: {image_dependence:.4f}, Dynamic alpha: {alpha:.4f}")
        return alpha
    except Exception as e:
        print(f"Error during alpha prediction: {e}. Returning default alpha.")
        return 0.7

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/rag', methods=['POST'])
def rag_api():
    if CANDIDATE_EMBEDDINGS is None or IMAGE_EMBEDDINGS is None:
        return jsonify({'error': 'Server is not ready. Embeddings not loaded.'}), 500

    # --- 1. Parse Request ---
    query_text = request.form.get('query', '')
    query_image_file = request.files.get('image')
    mode = request.form.get('mode', 'fixed') # New: get mode from request

    if not query_text and not query_image_file:
        return jsonify({'error': 'A text query or an image query is required.'}), 400

    text_similarities = torch.zeros(len(INDEX_TO_INFO), device=retriever.device)
    image_similarities = torch.zeros(len(INDEX_TO_INFO), device=retriever.device)

    try:
        # --- 2. Encode Query and Compute Similarities ---
        if query_text:
            query_text_embedding = retriever.encode_text([query_text])
            image_similarities = F.cosine_similarity(query_text_embedding, IMAGE_EMBEDDINGS)
            text_similarities = F.cosine_similarity(query_text_embedding, CANDIDATE_EMBEDDINGS)

        if query_image_file:
            query_image = Image.open(query_image_file.stream)
            query_image_embedding = retriever.encode_image([query_image])
            if not query_text:
                text_similarities = F.cosine_similarity(query_image_embedding, CANDIDATE_EMBEDDINGS)
            image_similarities = F.cosine_similarity(query_image_embedding, IMAGE_EMBEDDINGS)

        # --- 3. Determine Alpha and Fuse Scores ---
        if mode == 'dynamic' and query_text:
            text_weight = predict_alpha(query_text)
            image_weight = 1.0 - text_weight
        else: # Fixed mode, or dynamic mode without text to analyze
            text_weight = 0.7
            image_weight = 0.3
            # In fixed mode, if only one modality is present, give it full weight
            if mode == 'fixed':
                if not query_text: # Image-only query
                    text_weight = 0.0
                    image_weight = 1.0
                elif not query_image_file: # Text-only query
                    text_weight = 1.0
                    image_weight = 0.0

        fused_scores = (text_weight * text_similarities) + (image_weight * image_similarities)

        # --- 4. Get Top-k Results ---
        top_k = 5
        top_k_indices = torch.topk(fused_scores, k=min(top_k, len(INDEX_TO_INFO))).indices.tolist()

        # --- 5. Format Results ---
        results = []
        for idx in top_k_indices:
            candidate_data = INDEX_TO_INFO[idx]
            
            question_text = candidate_data.get("display_text", "")
            choices = candidate_data.get("choices", [])
            formatted_choices = []
            for i, choice in enumerate(choices):
                formatted_choices.append(f"{chr(65 + i)}. {choice}")
            choices_text = "\n".join(formatted_choices)
            formatted_display_text = f"Question: {question_text}\nChoices:\n{choices_text}"

            results.append({
                'candidate': {
                    'id': candidate_data.get('id'),
                    'image_path': candidate_data.get('image_path'),
                    'formatted_text': formatted_display_text
                },
                'fused_score': fused_scores[idx].item(),
                'text_score': text_similarities[idx].item(),
                'image_score': image_similarities[idx].item()
            })
        
        return jsonify({'results': results, 'alpha_used': text_weight})

    except Exception as e:
        print(f"Error during retrieval: {e}")
        return jsonify({'error': 'An error occurred during retrieval.'}), 500


if __name__ == '__main__':
    # use_reloader=False is important to prevent models from loading twice
    app.run(debug=True, port=5001, use_reloader=False)