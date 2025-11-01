
import torch
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
from PIL import Image
import torch.nn.functional as F
import numpy as np

class Retriever:
    """
    A multimodal retriever that encodes and ranks candidate text-image pairs 
    against a text-image query.
    """
    def __init__(self, text_model_name='BAAI/bge-base-en-v1.5', image_model_name='openai/clip-vit-large-patch14'):
        """
        Initializes the retriever by loading pretrained text and image encoders.
        """
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load Text Encoder (BGE)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name).to(self.device)
        self.text_model.eval() # Set to evaluation mode
        print("Text encoder loaded.")

        # Load Image Encoder (CLIP)
        self.image_processor = CLIPProcessor.from_pretrained(image_model_name)
        self.image_model = CLIPModel.from_pretrained(image_model_name).to(self.device)
        self.image_model.eval() # Set to evaluation mode
        print("Image encoder loaded.")

    def encode_text(self, texts: list[str]):
        """
        Encodes a list of texts into embeddings.
        
        Args:
            texts (list[str]): A list of strings to encode.
            
        Returns:
            torch.Tensor: A tensor of shape (len(texts), embedding_dim)
        """
        if not texts:
            return torch.empty(0)
        # Tokenize sentences
        encoded_input = self.text_tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.text_model(**encoded_input)
            # Perform pooling (sentence-transformers recommend CLS pooling for BGE)
            sentence_embeddings = model_output[0][:, 0]
        
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def encode_image(self, images: list):
        """
        Encodes a list of images into embeddings.
        
        Args:
            images (list[PIL.Image]): A list of PIL Image objects to encode.
            
        Returns:
            torch.Tensor: A tensor of shape (len(images), embedding_dim)
        """
        if not images:
            return torch.empty(0)
        # Process images
        inputs = self.image_processor(images=images, return_tensors="pt").to(self.device)
        
        # Compute image embeddings
        with torch.no_grad():
            image_embeddings = self.image_model.get_image_features(**inputs)
            
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        return image_embeddings

    def retrieve(self, query_text: str, query_image: Image, candidates: list[dict], alpha: float = 0.7, top_k: int = 5):
        """
        Retrieves the top-k most relevant candidates for a given multimodal query.
        
        Args:
            query_text (str): The text part of the query.
            query_image (PIL.Image): The image part of the query.
            candidates (list[dict]): A list of candidate dictionaries, each with 'text' and 'image' keys.
            alpha (float): The weight for fusing text and image similarity scores.
            top_k (int): The number of top candidates to return.
            
        Returns:
            list[dict]: A ranked list of the top-k candidates with their scores.
        """
        # --- 1. Encode the Query ---
        query_text_embedding = self.encode_text([query_text]) if query_text else None
        query_image_embedding = self.encode_image([query_image]) if query_image else None

        # --- 2. Encode the Candidates ---
        candidate_texts = [c['text'] for c in candidates]
        candidate_images = [c['image'] for c in candidates]
        
        candidate_text_embeddings = self.encode_text(candidate_texts)
        candidate_image_embeddings = self.encode_image(candidate_images)

        # --- 3. Compute Similarities ---
        # Compute text similarities if query text is provided
        if query_text_embedding is not None:
            text_similarities = F.cosine_similarity(query_text_embedding, candidate_text_embeddings)
        else:
            text_similarities = torch.zeros(len(candidates), device=self.device)

        # Compute image similarities if query image is provided
        if query_image_embedding is not None:
            image_similarities = F.cosine_similarity(query_image_embedding, candidate_image_embeddings)
        else:
            image_similarities = torch.zeros(len(candidates), device=self.device)
            
        # --- 4. Fuse Scores ---
        # Adjust alpha based on query modality
        if query_text_embedding is None and query_image_embedding is not None:
            effective_alpha = 0.0
        elif query_text_embedding is not None and query_image_embedding is None:
            effective_alpha = 1.0
        else:
            effective_alpha = alpha

        fused_scores = effective_alpha * text_similarities + (1 - effective_alpha) * image_similarities
        
        # --- 5. Rank and Return Top-k ---
        # Get the indices of the top-k scores
        top_k_indices = torch.topk(fused_scores, k=min(top_k, len(candidates)), dim=-1).indices.tolist()
        
        # Create a ranked list of candidates
        ranked_candidates = []
        for idx in top_k_indices:
            ranked_candidates.append({
                'candidate': candidates[idx],
                'fused_score': fused_scores[idx].item(),
                'text_similarity': text_similarities[idx].item(),
                'image_similarity': image_similarities[idx].item(),
            })
            
        return ranked_candidates

# --- Example Usage ---
if __name__ == '__main__':
    print("Running Multimodal Retriever Example with REAL images...")

    # --- 1. Initialize Retriever ---
    retriever = Retriever()

    # --- 2. Define Candidate Data with Image Paths ---
    # IMPORTANT: Create a folder named 'images' in the same directory
    # and place your images there.
    # For this example, let's assume you have:
    # images/
    #   ├── dog_park.jpg
    #   ├── ocean_sunset.jpg
    #   ├── spaghetti.jpg
    #   ├── city_night.jpg
    #   └── snowy_mountains.jpg
    
    candidates_with_paths = [
        {'text': 'A photo of a happy dog playing in a green park.', 'image_path': 'images/dog_park.jpg'},
        {'text': 'A beautiful sunset over the blue ocean.', 'image_path': 'images/ocean_sunset.jpg'},
        {'text': 'A delicious plate of spaghetti with red sauce.', 'image_path': 'images/spaghetti.jpg'},
        {'text': 'A modern city skyline at night with bright lights.', 'image_path': 'images/city_night.jpg'},
        {'text': 'A quiet, snowy landscape in the mountains.', 'image_path': 'images/snowy_mountains.jpg'},
    ]

    # --- 3. Load Images from Paths ---
    loaded_candidates = []
    for candidate in candidates_with_paths:
        try:
            # Load the image from the file path using Pillow
            image = Image.open(candidate['image_path'])
            loaded_candidates.append({
                'text': candidate['text'],
                'image': image
            })
        except FileNotFoundError:
            print(f"Warning: Image not found at {candidate['image_path']}. Skipping this candidate.")
    
    # --- 4. Define and Load a Query Image ---
    query_text = "A view of the sea"
    query_image_path = 'images/ocean_sunset.jpg' # Use one of your images as the query
    
    try:
        query_image = Image.open(query_image_path)
        print(f"\n--- Retrieving for Query: '{query_text}' and Image: '{query_image_path}' ---")
        
        # --- 5. Perform Retrieval ---
        # Note: We pass the 'loaded_candidates' which contains the actual image objects
        top_results = retriever.retrieve(query_text, query_image, loaded_candidates, alpha=0.7, top_k=3)

        # --- 6. Display Results ---
        print("Top-3 Results:")
        for i, result in enumerate(top_results):
            print(f"  {i+1}. Text: '{result['candidate']['text']}'")
            print(f"     Fused Score: {result['fused_score']:.4f} (Text Sim: {result['text_similarity']:.4f}, Image Sim: {result['image_similarity']:.4f})")

    except FileNotFoundError:
        print(f"Error: Query image not found at {query_image_path}. Cannot run retrieval.")
        print("Please make sure you have created the 'images' folder and placed your image files inside.")

