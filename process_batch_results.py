#!/usr/bin/env python3
"""
Script to process batch_test_results.json and extract score and img_path for each query.
Outputs results to a Markdown file with clickable image links.
"""

import json
import os


def process_batch_results(input_file, output_file, base_dir=None):
    """
    Process batch test results JSON file and extract score and img_path for each query.
    Outputs to Markdown format with clickable image links.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output Markdown file
        base_dir: Base directory for resolving relative paths (default: current working directory)
    """
    if base_dir is None:
        base_dir = os.getcwd()
    
    # Get the directory of the output file to calculate relative paths
    output_dir = os.path.dirname(os.path.abspath(output_file))
    if not output_dir:
        output_dir = os.getcwd()
    
    # Read JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Open output file for writing
    with open(output_file, 'w', encoding='utf-8') as f:
        # Process each query
        for query_result in data:
            query_id = query_result.get('query_id', 'N/A')
            query = query_result.get('query', 'N/A')
            retrieved_chunks = query_result.get('retrieved_chunks', [])
            
            # Write query header
            f.write(f"## Query {query_id}: {query}\n\n")
            
            # Write each retrieved chunk with score and img_path
            for idx, chunk in enumerate(retrieved_chunks, start=1):
                score = chunk.get('score', 'N/A')
                img_path = chunk.get('img_path', 'N/A')
                
                if img_path and img_path != 'N/A':
                    # Calculate relative path from output file to image
                    if not os.path.isabs(img_path):
                        abs_img_path = os.path.join(base_dir, img_path)
                    else:
                        abs_img_path = img_path
                    
                    # Normalize paths
                    abs_img_path = os.path.normpath(abs_img_path)
                    output_dir_norm = os.path.normpath(output_dir)
                    
                    # Calculate relative path for image display
                    try:
                        rel_img_path = os.path.relpath(abs_img_path, output_dir_norm)
                        # Ensure forward slashes for Markdown compatibility
                        rel_img_path = rel_img_path.replace('\\', '/')
                    except ValueError:
                        # If paths are on different drives (Windows), use absolute path
                        rel_img_path = img_path
                    
                    # Create file:// URL for clickable link
                    file_url = f"file://{abs_img_path}".replace('\\', '/')
                    
                    # Create clickable link and image display in Markdown
                    f.write(f"{idx}. **Score:** {score}  \n")
                    f.write(f"   **Image:** [{img_path}]({file_url})  \n")
                    f.write(f"   ![Image {idx}]({rel_img_path})  \n\n")
                else:
                    f.write(f"{idx}. **Score:** {score}, **Image:** {img_path}\n\n")
            
            # Add separator between queries
            f.write("---\n\n")


if __name__ == "__main__":
    # Set input and output file paths
    input_file = "rag_results/batch_test_results.json"
    output_file = "rag_results/batch_test_results_processed.md"
    base_dir = os.path.dirname(os.path.abspath(input_file)) or os.getcwd()
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        exit(1)
    
    # Process the file
    print(f"Processing {input_file}...")
    process_batch_results(input_file, output_file, base_dir=os.getcwd())
    print(f"Results written to {output_file}")
    print("Note: Open the .md file in a Markdown viewer/editor to use clickable image links!")
