import os, json
from collections import Counter

complete_data = []

from collections import Counter

# Function to sort reviews based on "Most Related Word" (mrw) and calculate ranking score for an app
def sort_by_mrw(data):
    # Dictionary to hold groups of reviews categorized by Most Related Word (mrw)
    groups = {}
    
    # Dictionary to track the count of reviews for each mrw
    mrw_counts = {}
    
    # Loop through each review in the data
    for review in data["Reviews"]:
        # Get the Most Related Word for the current review
        mrw = review["Most Related Word"]
        
        # If the mrw is not already in the groups, initialize an empty list for it
        if mrw not in groups:
            groups[mrw] = []

        # If mrw is not in mrw_counts, initialize the count for that mrw
        if mrw not in mrw_counts:
            mrw_counts[mrw] = 0

        # If the number of reviews with this mrw is less than 3, add the review to the group
        if mrw_counts[mrw] < 3:
            groups[mrw].append(review)
            mrw_counts[mrw] += 1

    # Sort the reviews within each mrw group by sentiment score (positive - negative) and similarity
    for words, reviews in groups.items():
        reviews.sort(key=lambda r: (r["Sentiment"]["pos"] - r["Sentiment"]["neg"], r["Similarity"]), reverse=True)

    # Count the occurrences of each Most Related Word across all reviews
    mrw_counter = Counter([review["Most Related Word"] for review in data["Reviews"]])

    # Get the top 3 most common Most Related Words as annotation
    annotation = [word for word, _ in mrw_counter.most_common(3)]

    # Calculate the sentiment score for the app from the total sentiment (positive - negative)
    sentiment_score = data["Total Sentiment"]["pos"] - data["Total Sentiment"]["neg"]
    
    # Get the number of reviews for the app
    review_length = len(data["Reviews"])
    
    # Calculate a score based on the number of reviews (with a factor of 0.4)
    reviews_score = review_length ** 0.4
    
    # Final app score is the product of sentiment score and reviews score
    app_score = round(sentiment_score * reviews_score, 2)

    # If annotation is empty or the first word is invalid, return None
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

# Function to sort multiple apps by their ranking score
def sort_data(data):
    # Sort the list of apps based on their Ranking Score in descending order
    sorted_data = sorted(data, key=lambda app: app["Ranking Score"], reverse=True)
    return sorted_data

def load_data(file_path):
    # Load JSON data from a specified file path
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        print(f"Loaded data from {file_path}")  # Debugging message
        return data

def process_directory(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files_processed = False
    
    # Iterate over all JSON files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            files_processed = True
            file_path = os.path.join(input_dir, filename)
            print(f'Processing file: {file_path}')

            data = load_data(file_path)  # Load the data
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
    # Define input and output directories
    input_dir = 'Data/SimilarityData' 
    output_dir = 'Data/CompleteData'
    
    process_directory(input_dir, output_dir)

    print(f"Keyword extraction completed. Check the '{output_dir}' folder.")