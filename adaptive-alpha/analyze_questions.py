import json
import os
import time
import google.generativeai as genai

def load_dotenv(dotenv_path='.env'):
    """Loads environment variables from a .env file."""
    if not os.path.exists(dotenv_path):
        print(f"Warning: .env file not found at {dotenv_path}")
        return
    with open(dotenv_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"\'')
                os.environ.setdefault(key, value)

def analyze_questions_with_gemini(input_path, output_dir, model_name, training_ids_path):
    # 1. Configure the API key
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        return

    # 2. Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 3. Read the list of training IDs to process
    try:
        with open(training_ids_path, 'r') as f:
            training_ids = set(json.load(f))
        print(f"Loaded {len(training_ids)} IDs for processing from {training_ids_path}")
    except FileNotFoundError:
        print(f"Error: Training IDs file not found at {training_ids_path}")
        return

    # 4. Read the processed problems
    try:
        with open(input_path, 'r') as f:
            problems = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return

    # 5. Set up the generative model
    model = genai.GenerativeModel(model_name)

    # Prompt template
    prompt_template = """You are an expert annotator for multimodal question answering datasets.

Given the TEXT of a multiple-choice question that originally came with an image,
assign an **image-dependence score** between 0.0 and 1.0.

- 0.0 means the question can be fully answered from text alone.
- 1.0 means the image is essential to answer the question.
- Values in between indicate partial dependence on the image.

Your output MUST be a valid JSON object with two keys:
  "image_dependence" (float between 0.0 and 1.0)
  "reason" (a concise explanation for your score)

Here is the text:
---
{text}
---

Now output ONLY the JSON object, nothing else.
"""

    problems_to_process = [p for p in problems if p.get('id') in training_ids]
    total_problems = len(problems_to_process)
    print(f"Found {total_problems} matching problems to process.")

    for i, problem in enumerate(problems_to_process):
        problem_id = problem.get('id')
        output_filepath = os.path.join(output_dir, f"{problem_id}.json")

        # 6. Skip if already processed
        if os.path.exists(output_filepath):
            print(f"Skipping problem {i+1}/{total_problems} (ID: {problem_id}) - already processed.")
            continue

        print(f"Processing problem {i+1}/{total_problems} (ID: {problem_id})...")

        text_to_analyze = problem.get("text", "")
        prompt = prompt_template.format(text=text_to_analyze)

        # Prepare the data to be saved
        result_data = {"id": problem_id, "text": text_to_analyze}
        try:
            # 7. Call the API
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            
            # 8. Parse the response
            analysis_json = json.loads(response.text)
            result_data.update(analysis_json)

        except Exception as e:
            print(f"  Error processing ID {problem_id}: {e}")
            result_data["error"] = str(e)

        # 9. Save the individual result file
        with open(output_filepath, 'w') as outfile:
            json.dump(result_data, outfile, indent=4)

        # 10. Be respectful of API rate limits
        time.sleep(1)
    
    print(f"\nAnalysis complete. Individual results are saved in the '{output_dir}' directory.")

if __name__ == "__main__":
    load_dotenv()
    analyze_questions_with_gemini(
        input_path="processed_problems.json", 
        output_dir="analysis_results", 
        model_name="gemini-1.5-flash-latest",
        training_ids_path="training_ids.json"
    )