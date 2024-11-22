import os, json, tiktoken
from collections import Counter

MAX_TOKENS = 200 # token count here

# Function to estimate token count (using tiktoken for accurate tokenization)
def estimate_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return len(tokens)

# Function to add reviews to a structured object instead of a global string
def add_data(data, result):
    groups = {}

    for review in data["Reviews"]:
        mrw = review["Most Related Word"]

        if mrw not in groups:
            groups[mrw] = []

        groups[mrw].append(review)

    for words, reviews in groups.items():
        reviews.sort(key=lambda r: (r["Sentiment"]["pos"] - r["Sentiment"]["neg"], r["Similarity"]), reverse=True)

    new_groups = {}
    new_counts = {}

    for review in data["Reviews"]:
        mrw = review["Most Related Word"]

        if mrw not in new_groups:
            new_groups[mrw] = []

        if mrw not in new_counts:
            new_counts[mrw] = 0

        if new_counts[mrw] < 20:
            new_groups[mrw].append(review["Review"])
            new_counts[mrw] += 1

    mrw_counter = Counter([review["Most Related Word"] for review in data["Reviews"]])

    annotation = [word for word, _ in mrw_counter.most_common(3)]

    if annotation == [''] or annotation == []:
        return

    # Prepare data to be added to the result
    app_data = {
        "name": data.get("App Name"),
        "labels": annotation,
        "reviews": [],
        "sentiment": data["Total Sentiment"]["sentiment"]
    }

    reviews_concatenated = ""
    for mrw, reviews in new_groups.items():
        for review in reviews:
            # If adding the review exceeds the token limit, stop adding more reviews
            if estimate_tokens(reviews_concatenated + review) > MAX_TOKENS:
                continue

            reviews_concatenated += review  # Append review to concatenated string

    app_data["reviews"] = reviews_concatenated
    result.append(app_data)  # Add this app's data to the result

# Function to sort data based on Ranking Score
def sort_data(data):
    return sorted(data, key=lambda app: app["Ranking Score"], reverse=True)

# Function to load data from a file
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        print(f"Loaded data from {file_path}")  # Debugging message
        return data

# Function to process the directory and return an object instead of writing to a file
def process_directory(input_dir, output_dir):
    result = []  # Initialize the result object to store processed data
    
    # Iterate over all JSON files in the input directory
    files_processed = False
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            files_processed = True
            file_path = os.path.join(input_dir, filename)
            print(f'Processing file: {file_path}')

            data = load_data(file_path)  # Load the data
            if data:
                add_data(data, result)  # Add processed data to result

    if not files_processed:
        print(f"No JSON files found in {input_dir}")
    
    return result  # Return the processed data object

# Main function
if __name__ == "__main__":
    # Define input and output directories
    input_dir = 'Data/SimilarityData'
    output_dir = 'Data/CompleteData'
    
    result = process_directory(input_dir, output_dir)

    # Print the final object for debugging or further use
    print("Processed data:", result)

    # You can also save the result to a file if needed (optional)
    output_file_path = os.path.join(output_dir, "StringData.json")
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=4)
    print(f'Processed and saved to {output_file_path}')
    
    print(f"Keyword extraction completed.")
