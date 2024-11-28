import json, os
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

def find_sentiment(data):
    # load model and tokenizer
    model = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model)  
    model = AutoModelForSequenceClassification.from_pretrained(model) 

    # encode description, get model output, and get the probabilty of each sentiment
    description_encoded = tokenizer(data["Description"], return_tensors='pt', max_length=512, truncation=True)
    output = model(**description_encoded) 
    description_scores = softmax(output[0][0].detach().numpy())  
    
    # Classify sentiment of the description
    sentiment_description = { 
        "neg": float(description_scores[0]),
        "neu": float(description_scores[1]),
        "pos": float(description_scores[2]),
        "sentiment": 
            "neu" if description_scores[1] > max(description_scores[0], description_scores[2]) else 
            "neg" if description_scores[0] > description_scores[2] else
            "pos"
    }

    # Initialize for the overall sentiment of the app
    sentiment_reviews_list = []
    sentiment_total = {
        "neg": 0,
        "neu": 0,
        "pos": 0,
        "sentiment": ""
    }

    keys_list = ["neg", "neu", "pos"]  # Keys for sentiment scores

    for review in data["Reviews"]:
        # encode review, get model output, and get the probability of each sentiment
        review_encoded = tokenizer(review["Review"], return_tensors='pt', max_length=512, truncation=True) 
        output = model(**review_encoded)
        review_scores = softmax(output[0][0].detach().numpy())
        
        # Display model outputs and overall sentiment for the review
        review_sentiment = {
            "neg": float(review_scores[0]),
            "neu": float(review_scores[1]),
            "pos": float(review_scores[2]),
            "sentiment": 
                "neu" if review_scores[1] > max(review_scores[0], review_scores[2]) else 
                "neg" if review_scores[0] > review_scores[2] else
                "pos"
        }

        sentiment_reviews_list.append({
            "Rating": review["Rating"],
            "Review": review["Review"],
            "Sentiment": review_sentiment
        })

        # add the review sentiment to later find the overall app sentiment
        for key in sentiment_total.keys():
            if key in keys_list:
                sentiment_total[key] += review_sentiment[key]

    # compute the average sentiment across all reviews for the overall app sentiment
    for key in sentiment_total.keys():
        if key in keys_list:
            sentiment_total[key] /= len(data["Reviews"]) if len(data["Reviews"]) != 0 else 1
            sentiment_total[key] = round(sentiment_total[key], 4)

    sentiment_total["sentiment"] = "neu" if sentiment_total["neu"] > max(sentiment_total["neg"], sentiment_total["pos"]) else "neg" if sentiment_total["neg"] > sentiment_total["pos"] else "pos"

    return {
        "App Name": data.get("App Name", "Unknown App"),
        "Developer": data.get("Developer", "Unknown Developer"),
        "Ratings": data.get("Ratings", "No Ratings"),
        "Description": data.get("Description", "No Description"),
        "Description Sentiment": sentiment_description,
        "Reviews": sentiment_reviews_list,
        "Total Sentiment": sentiment_total
    }


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
                sentiment_data = find_sentiment(data) 
                output_file_path = os.path.join(output_dir, filename)

                with open(output_file_path, 'w', encoding='utf-8') as file:
                    json.dump(sentiment_data, file, indent=4, ensure_ascii=False)
            print(f'Processed {filename} and saved to {output_file_path}')
    
    if not files_processed:
        print(f"No JSON files found in {input_dir}")

if __name__ == "__main__":
    input_dir = 'Data/FilteredData' 
    output_dir = 'Data/SentimentData' 
    
    process_directory(input_dir, output_dir)

    print(f"Keyword extraction completed. Check the '{output_dir}' folder.")
