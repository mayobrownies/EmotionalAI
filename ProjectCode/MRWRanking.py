import os, json
from collections import Counter

complete_data = []

# generate an annotation for apps by using the most common related words
# compute a ranking score so the most positively received apps are pushed to the top of the dataset
def sort_by_mrw(data):
    
    groups = {}
    mrw_counts = {}
    
    for review in data["Reviews"]:
        mrw = review["Most Related Word"]

        if mrw not in groups:
            groups[mrw] = []

        if mrw not in mrw_counts:
            mrw_counts[mrw] = 0

        if mrw_counts[mrw] < 3:
            groups[mrw].append(review)
            mrw_counts[mrw] += 1

    for words, reviews in groups.items():
        reviews.sort(key=lambda r: (r["Sentiment"]["pos"] - r["Sentiment"]["neg"], r["Similarity"]), reverse=True)

    mrw_counter = Counter([review["Most Related Word"] for review in data["Reviews"]])
    
    annotation = [word for word, _ in mrw_counter.most_common(3)]

    sentiment_score = data["Total Sentiment"]["pos"] - data["Total Sentiment"]["neg"]
    review_length = len(data["Reviews"])
    reviews_score = review_length ** 0.4
    app_score = round(sentiment_score * reviews_score, 2)

    if not annotation or annotation[0] == None or annotation[0] == "":
        return None

    return {
            "App Name": data.get("App Name", "Unknown App"),
            "Annotation": annotation,
            "Ranking Score": app_score,
            "Developer": data.get("Developer", "Unknown Developer"),
            "Ratings": data.get("Ratings", "No Ratings"),
            "Description": data.get("Description", "No Description"),
            "Description Sentiment": data.get("Description Sentiment", "No Description Sentiment"),
            "Sorted Reviews": groups,
            "Total Sentiment": data.get("Total Sentiment", "No Sentiment")
        }

# sort apps by computed ranking score
def sort_data(data):
    sorted_data = sorted(data, key=lambda app: app["Ranking Score"], reverse=True)
    return sorted_data

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        print(f"Loaded data from {file_path}") 
        return data

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files_processed = False
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            files_processed = True
            file_path = os.path.join(input_dir, filename)
            print(f'Processing file: {file_path}')

            data = load_data(file_path)
            if data:
                ranked_data = sort_by_mrw(data)
                
                if ranked_data:
                    complete_data.append(ranked_data) 

    sorted_data = sort_data(complete_data)
    output_file_path = os.path.join(output_dir, "FinalData")
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(sorted_data, file, indent=4, ensure_ascii=False)
    print(f'Processed {filename} and saved to {output_file_path}')
    
    if not files_processed:
        print(f"No JSON files found in {input_dir}")

if __name__ == "__main__":
    input_dir = 'Data/SimilarityData' 
    output_dir = 'Data/CompleteData'
    
    process_directory(input_dir, output_dir)

    print(f"Keyword extraction completed. Check the '{output_dir}' folder.")