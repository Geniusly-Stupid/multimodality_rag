import json

def preprocess_problems(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        problems_dict = json.load(infile)
        processed_problems = []
        # Iterate over the items (key-value pairs) of the dictionary
        for problem_id, problem in problems_dict.items():
            if problem.get("image") is not None:
                # Create a new dictionary to avoid modifying the original data in memory
                processed_problem = problem.copy()
                processed_problem['id'] = problem_id # Add the original key as 'id'

                # Format the text field as per the new multi-line requirement
                question_text = problem.get("question", "")
                choices = problem.get("choices", [])
                formatted_choices = []
                for i, choice in enumerate(choices):
                    # chr(65) is 'A'
                    formatted_choices.append(f"{chr(65 + i)}. {choice}")
                
                choices_text = "\n".join(formatted_choices)
                processed_problem["text"] = f"Question: {question_text}\nChoices:\n{choices_text}"

                # Replace answer index with the actual answer text and append the solution
                answer_index = problem.get("answer")
                if answer_index is not None and isinstance(answer_index, int):
                    if 0 <= answer_index < len(choices):
                        answer_text = choices[answer_index]
                        solution_text = problem.get("solution", "")
                        processed_problem["answer"] = f'{answer_text}; {solution_text}'
                
                processed_problems.append(processed_problem)
        
        # In Python's json.dump, newlines in strings are automatically escaped.
        # When the JSON is loaded, they will be interpreted as newline characters.
        json.dump(processed_problems, outfile, indent=4)

if __name__ == "__main__":
    preprocess_problems("problems.json", "processed_problems.json")