import os
import json
import ollama

# select which model to use for annotations

#model_name = "llama3.1:8b"
#model_name = "gemma2:latest"
model_name = "mistral:latest"

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
                description = data["Description"]
                reviews = data["Reviews"]
                reviews_text = "\n".join(review["Review"] for review in reviews) if reviews else "No reviews available."

                question = f"""
                Given the following app description and reviews, please extract the most related mental health words. 
                Description: {description}
                Reviews: {reviews_text}
                
                The word bank you use for the reviews should only be the extact keywords from the description.
                Use the most related mental health word from each review to create an annotation list for the app. 
                The annotation should only include the top three most frequently occurring wor from the reviews. 
                I only want the list, do not include anything else.
                Once you make an output, compare the it to the description again to make sure it is correct.
                Words like "help", "helped", "angry", "app", etc. are not mental health conditions.
                The only words that should be included are mental health conditions and general descriptions like "therapy" or "meditation".
                Remember to make sure that these words are advertised in the app's description.
                Remove escape characters. 
                The format should be as follows:
                "[word1, word2, word3, ....]"    
                Check that all words in the list are also included in "Description" if any word is not there, remove it.
                """

                response = ollama.chat(model=model_name, messages = [
                    {
                        'role': 'user',
                        'content': question
                    },
                ])
                annotation = response['message']['content']
                result = {
                        "App Name": data.get("App Name", "Unknown App"),
                        "Annotation": annotation
                    }
                
                
                output_file_path = os.path.join(output_dir, filename)
                with open(output_file_path, 'w', encoding='utf-8') as file:
                    json.dump(result, file, indent=4, ensure_ascii=False)


    print(f'Processed {filename} and saved to {output_file_path}')
    
    if not files_processed:
        print(f"No JSON files found in {input_dir}")

if __name__ == "__main__":
    input_dir = 'Data/SimilarityData' 
    output_dir = f'Data/ModelData/{model_name.replace(":", "_")}'
    
    process_directory(input_dir, output_dir)

    print(f"Keyword extraction completed. Check the '{output_dir}' folder.")
