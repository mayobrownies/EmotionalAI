import os, json, tiktoken
from collections import Counter

# set max token count for reviews
MAX_TOKENS = 0

# estimate number of tokens
def estimate_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")  
    tokens = enc.encode(text)
    return len(tokens)

# reduce the size of the input data with the given token limit
# ensure that detail is kept by requiring a certain number of reviews for each related word
def add_to_object(data, result_object):
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

    app_data = {
        "name": data.get("App Name"),
        "developer": data.get("Developer"),
        "rating": data.get("Ratings"),
        "labels": annotation,
        "reviews": [],
        "sentiment": data["Total Sentiment"]["sentiment"]
    }

    reviews_concatenated = ""
    for mrw, reviews in new_groups.items():
        for review in reviews:
            if estimate_tokens(reviews_concatenated + review) > MAX_TOKENS:
                continue

            reviews_concatenated += review

    app_data["reviews"] = reviews_concatenated
    result_object.append(app_data) 

def sort_data(data):
    return sorted(data, key=lambda app: app["Ranking Score"], reverse=True)

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        print(f"Loaded data from {file_path}")
        return data

def process_directory(input_dir):
    result_object = [] 
    
    files_processed = False
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            files_processed = True
            file_path = os.path.join(input_dir, filename)
            print(f'Processing file: {file_path}')

            data = load_data(file_path) 
            if data:
                add_to_object(data, result_object) 

    if not files_processed:
        print(f"No JSON files found in {input_dir}")
    
    return result_object  

if __name__ == "__main__":
    input_dir = 'Data/SimilarityData'
    output_dir = 'Data/CompleteData'
    
    result_object = process_directory(input_dir)

    print("Processed data:", result_object)

    output_file_path = os.path.join(output_dir, "StringData.json")
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(result_object, file, ensure_ascii=False, indent=4)
    print(f'Processed and saved to {output_file_path}')
    
    print(f"Keyword extraction completed.")
