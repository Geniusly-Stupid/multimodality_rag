
import torch
import torch.nn.functional as F
from datasets import load_dataset
from retriever import Retriever
from PIL import Image

def main():
    """
    Main function to build an index from ScienceQA and run a test retrieval.
    """
    print("1. Initializing the multimodal retriever...")
    # Initialize the retriever from the other script
    retriever = Retriever()

    print("\n2. Loading and preparing the ScienceQA dataset...")
    # Load the test split of the ScienceQA dataset
    # We stream it to avoid downloading the full dataset at once
    try:
        dataset = load_dataset('derek-thomas/ScienceQA', split='test', streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have an internet connection and necessary permissions.")
        return

    # Take a small subset for demonstration
    dataset_head = dataset.take(200)

    # Filter for samples that have both an image and a text hint
    multimodal_samples = []
    for sample in dataset_head:
        if sample['image'] is not None and sample['hint'] and sample['hint'].strip():
            multimodal_samples.append(sample)
    
    if len(multimodal_samples) < 2:
        print("Not enough multimodal samples found in the first 200 entries to run a retrieval test.")
        return

    # Use the first sample as the query and the rest as the candidate pool
    query_sample = multimodal_samples[0]
    candidate_samples = multimodal_samples[1:]
    print(f"Loaded {len(candidate_samples)} candidate samples and 1 query sample.")

    # --- Prepare Query and Candidates ---
    # Query
    query_text = query_sample['question'] + " " + query_sample['hint']
    query_image = query_sample['image']

    # Candidates
    candidate_texts = [c['question'] + " " + c['hint'] for c in candidate_samples]
    candidate_images = [c['image'] for c in candidate_samples]

    print("\n3. Building the index (encoding candidates)...")
    # This is our "index" - the encoded vectors for all candidates
    candidate_text_embeddings = retriever.encode_text(candidate_texts)
    candidate_image_embeddings = retriever.encode_image(candidate_images)
    print(f"Index built with {candidate_text_embeddings.shape[0]} text and {candidate_image_embeddings.shape[0]} image embeddings.")

    print("\n4. Encoding the query...")
    query_text_embedding = retriever.encode_text([query_text])
    query_image_embedding = retriever.encode_image([query_image])

    print("\n5. Performing retrieval...")
    # --- Compute Similarities ---
    text_similarities = F.cosine_similarity(query_text_embedding, candidate_text_embeddings)
    image_similarities = F.cosine_similarity(query_image_embedding, candidate_image_embeddings)

    # --- Fuse Scores ---
    alpha = 0.7 # Weight for text vs. image
    fused_scores = alpha * text_similarities + (1 - alpha) * image_similarities

    # --- Rank and Get Top-k ---
    top_k = 5
    top_k_indices = torch.topk(fused_scores, k=min(top_k, len(candidate_samples)), dim=-1).indices.tolist()

    print("\n--- Retrieval Results ---")
    print(f"Query: \"{query_sample['question']}\" (Image: {'Yes' if query_image else 'No'})")
    print("-------------------------")
    print(f"Top-{top_k} most similar items from the index:")

    for i, idx in enumerate(top_k_indices):
        retrieved_sample = candidate_samples[idx]
        print(f"  {i+1}. [Score: {fused_scores[idx]:.4f}] Question: {retrieved_sample['question']}")
        # print(f"     Hint: {{retrieved_sample['hint']}}")

if __name__ == "__main__":
    main()
