
import json
import random

def select_random_ids(input_path, output_path, num_ids=1000):
    """Randomly selects a specified number of IDs from the input file."""
    try:
        with open(input_path, 'r') as f:
            problems = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return

    # Extract all IDs
    all_ids = [problem.get('id') for problem in problems if problem.get('id')]

    if len(all_ids) < num_ids:
        print(f"Warning: Requested {num_ids} IDs, but only {len(all_ids)} are available. Using all available IDs.")
        selected_ids = all_ids
    else:
        # Select 3000 random IDs
        selected_ids = random.sample(all_ids, num_ids)

    # Save the selected IDs
    with open(output_path, 'w') as outfile:
        json.dump(selected_ids, outfile, indent=4)

    print(f"Successfully selected {len(selected_ids)} random IDs and saved them to {output_path}")

if __name__ == "__main__":
    select_random_ids("processed_problems.json", "training_ids.json")
