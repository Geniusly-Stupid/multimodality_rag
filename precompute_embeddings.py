
import json
import os
import torch
from retriever import Retriever
from PIL import Image
import shutil

def precompute_embeddings():
    """
    Pre-computes text embeddings for the ScienceQA dataset and saves them.
    """
    print("1. Initializing Retriever...")
    # This part can be slow if models need to be downloaded
    retriever = Retriever()
    print("Retriever initialized.")

    print("\n2. Loading ScienceQA data...")
    # Define paths
    problems_json_path = 'adaptive-alpha/problems.json'
    local_images_base_path = 'images/test'
    static_images_path = 'static/scienceqa_images'

    # Ensure the static directory for images exists
    if not os.path.exists(static_images_path):
        os.makedirs(static_images_path)

    candidate_info = []
    texts_for_encoding = []
    image_paths_for_encoding = []

    try:
        with open(problems_json_path, 'r') as f:
            problems = json.load(f)
        print(f"Loaded {len(problems)} problems from JSON.")

        # Process all valid multimodal problems
        for problem_id, data in problems.items():
            has_image = data.get('image') is not None

            if has_image:
                original_image_path = os.path.join(local_images_base_path, problem_id, 'image.png')
                
                if os.path.exists(original_image_path):
                    # Prepare info for the mapping file
                    static_image_name = f"{problem_id}.png"
                    web_image_path = os.path.join('scienceqa_images', static_image_name)
                    
                    info = {
                        'id': problem_id,
                        'display_text': data.get('question', ''),
                        'image_path': web_image_path,
                        'choices': data.get('choices', [])
                    }
                    candidate_info.append(info)

                    # Prepare text for encoding
                    choices_text = " ".join(data.get('choices', []))
                    text_for_retrieval = data.get('question', '') + " " + choices_text
                    texts_for_encoding.append(text_for_retrieval)

                    # Collect the image path for later batch processing
                    image_paths_for_encoding.append(original_image_path)

                    # Copy image to static folder for the web app
                    static_image_dest_path = os.path.join(static_images_path, static_image_name)
                    if not os.path.exists(static_image_dest_path):
                        shutil.copy(original_image_path, static_image_dest_path)

        print(f"Found {len(candidate_info)} valid multimodal candidates to process.")

    except FileNotFoundError:
        print(f"Error: '{problems_json_path}' not found. Cannot pre-compute embeddings.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return

    if not texts_for_encoding:
        print("No texts to encode. Aborting.")
        return

    print(f"\n3. Encoding {len(texts_for_encoding)} texts... (This may take a while)")
    # Perform encoding
    # Note: If this step fails, the underlying torch/transformer installation might be unstable.
    try:
        text_embeddings = retriever.encode_text(texts_for_encoding)
    except Exception as e:
        print(f"FATAL: Failed to encode texts. Error: {e}")
        print("This is likely an issue with your PyTorch or transformers library installation.")
        return

    print(f"Successfully encoded {text_embeddings.shape[0]} texts.")

    # This is memory intensive, so we do it in batches.
    print(f"\n3b. Encoding {len(image_paths_for_encoding)} images in batches... (This may also take a while)")
    try:
        all_image_embeddings = []
        batch_size = 32  # Adjust batch size based on available RAM

        for i in range(0, len(image_paths_for_encoding), batch_size):
            batch_paths = image_paths_for_encoding[i:i+batch_size]
            batch_images = []
            # Load images for the current batch
            for path in batch_paths:
                try:
                    batch_images.append(Image.open(path))
                except Exception as e:
                    print(f"ERROR: Could not load image {path} during batch processing. It will be skipped. Error: {e}")
                    # This will cause a mismatch. A more robust solution would be to have a placeholder or remove the corresponding text embedding.
                    # For now, we will append a None to mark it.
                    batch_images.append(None)
            
            # Filter out None images before encoding
            valid_images_in_batch = [img for img in batch_images if img is not None]
            if not valid_images_in_batch:
                print(f"  Skipping batch {i//batch_size + 1} as no valid images were loaded.")
                # We need to add placeholder embeddings for the whole batch
                placeholder_embedding = torch.zeros((len(batch_images), retriever.image_model.config.projection_dim), device=retriever.device)
                all_image_embeddings.append(placeholder_embedding.cpu()) # Move to CPU to free up GPU memory if applicable
                continue

            batch_embeddings = retriever.encode_image(valid_images_in_batch)
            
            # If some images failed to load, we need to insert placeholders in the embeddings tensor
            if len(valid_images_in_batch) < len(batch_images):
                final_batch_embeddings = []
                valid_idx = 0
                for img in batch_images:
                    if img is not None:
                        final_batch_embeddings.append(batch_embeddings[valid_idx])
                        valid_idx += 1
                    else:
                        placeholder = torch.zeros((retriever.image_model.config.projection_dim), device=retriever.device)
                        final_batch_embeddings.append(placeholder)
                batch_embeddings = torch.stack(final_batch_embeddings)

            all_image_embeddings.append(batch_embeddings.cpu()) # Move to CPU to free up GPU memory if applicable
            print(f"  Encoded batch {i//batch_size + 1}/{(len(image_paths_for_encoding) + batch_size - 1)//batch_size}")
        
        image_embeddings = torch.cat(all_image_embeddings, dim=0)

    except Exception as e:
        print(f"FATAL: Failed to encode images. Error: {e}")
        return
    print(f"Successfully encoded {image_embeddings.shape[0]} images.")

    print("\n4. Saving embeddings and mapping file...")
    # Save the text embeddings tensor
    torch.save(text_embeddings, 'text_embeddings.pt')
    print("Text embeddings saved to 'text_embeddings.pt'.")

    # Save the image embeddings tensor
    torch.save(image_embeddings, 'image_embeddings.pt')
    print("Image embeddings saved to 'image_embeddings.pt'.")

    # Save the mapping from index to candidate info
    with open("index_to_info.json", "w") as f:
        json.dump(candidate_info, f)
    print("Mapping saved to 'index_to_info.json'.")

    print("\n--- Pre-computation complete! ---")

if __name__ == "__main__":
    precompute_embeddings()
