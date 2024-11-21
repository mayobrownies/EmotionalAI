import os, json

def filter_reviews(data):
    
    reviews_list = []

    for review in data["Reviews"]:
        reviews_list.append(review["Review"])


    return {
        "App Name": data["App Name"],
        "Description": data["Description"],
        "Reviews": reviews_list
    }

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
                review_data = filter_reviews(data)  # Analyze similarity
                output_file_path = os.path.join(output_dir, filename)

                # Save the sentiment analysis results to a new JSON file
                with open(output_file_path, 'w', encoding='utf-8') as file:
                    json.dump(review_data, file, indent=4, ensure_ascii=False)
            print(f'Processed {filename} and saved to {output_file_path}')
    
    if not files_processed:
        print(f"No JSON files found in {input_dir}")

if __name__ == "__main__":
    # Define input and output directories
    input_dir = 'Data/SimilarityData' 
    output_dir = 'Data/ReviewOnlyData' 
    
    # Process the directory
    process_directory(input_dir, output_dir)

    print(f"Keyword extraction completed. Check the '{output_dir}' folder.")