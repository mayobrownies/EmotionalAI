import json
import os

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        print(f"Loaded data from {file_path}")
        return data

def compare_annotations(dataset_annotations, llm_data):
    num_matches = 0
    total_annotations = 0

    for app in dataset_annotations:
        if app.get("App Name") == llm_data.get("App Name"):
            dataset_annotation = app.get("Annotation")
            llm_annotation = llm_data.get("Annotation")
            
            for annotation in dataset_annotation:
                if annotation in llm_annotation:
                    num_matches += 1
                total_annotations += 1

    return num_matches, total_annotations

def process_directory(input_dir, dataset_file):
    dataset_annotations = load_data(dataset_file)
    if not dataset_annotations:
        print("No dataset annotations loaded.")
        return

    total_matches = 0
    total_annotations = 0

    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            print(f'Processing file: {file_path}')

            # Load LLM annotations for the specific app
            llm_annotations = load_data(file_path)
            if llm_annotations:
                # Compare and accumulate matches and total counts
                matches, annotations = compare_annotations(dataset_annotations, llm_annotations)
                total_matches += matches
                total_annotations += annotations

    # Calculate the overall percentage of matching annotations
    if total_annotations == 0:
        print("No annotations found across all files.")
        return
    overall_match_percentage = (total_matches / total_annotations) * 100
    print(f"Overall Match Percentage: {overall_match_percentage:.2f}% of LLM annotations match the dataset.")

# Usage
dataset_file = "C:/Emotional AI/Data/CompleteData/SortedData.json"  # Ensure this is a .json file
input_dir = "C:/Emotional AI/Data/ModelData/mistral_latest"
process_directory(input_dir, dataset_file)
