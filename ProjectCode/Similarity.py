import json, os, torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from DataFilter import preprocess_keywords

model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-roberta-large-v1')

keywords = preprocess_keywords("""        
    "depression", "anxiety", "ADHD", "attention deficit hyperactivity disorder", 
    "bipolar disorder", "schizophrenia", "OCD", "obsessive-compulsive disorder", 
    "PTSD", "post-traumatic stress disorder", "panic disorder", "generalized anxiety disorder", 
    "GAD", "social anxiety", "phobias", "eating disorders", "anorexia", "bulimia", 
    "binge eating disorder", "autism", "autism spectrum disorder", "ASD", "borderline personality disorder", 
    "BPD", "dissociative identity disorder", "DID", "postpartum depression", 
    "seasonal affective disorder", "insomnia", "sleep disorders", 
    "substance abuse", "addiction", "alcoholism", "gambling addiction", 
    "self-harm", "suicidal thoughts", "grief counseling", "trauma", 
    "psychosis", "psychotic disorders", "conduct disorder", "oppositional defiant disorder", 
    "ODD", "stress disorder", "compulsive behavior", "emotion regulation", 
    "anger management", "mood swings", "chronic stress", "social isolation", 
    "behavioral health", "mental health assessment", "self-esteem issues", "relationship counseling", 
    "trauma therapy", "neurodevelopmental disorders", "learning disabilities", 
    "dyslexia", "dyscalculia", "developmental disorders", "emotional dysregulation", 
    "trauma recovery", "intrusive thoughts", "mental breakdown", "cognitive disorders"
""")

app_keywords = preprocess_keywords("""
    "mental health app", "mental wellness", "meditation app", "stress relief app", 
    "self-care app", "anxiety relief", "stress management", "depression support", 
    "mindfulness app", "cognitive behavioral therapy", "CBT app", "sleep improvement", 
    "therapy app", "online counseling", "guided meditation", "emotional well-being", 
    "mood tracker", "mental health tracker", "bipolar disorder", "depression treatment", 
    "mindfulness meditation", "stress relief meditation", "guided breathing", 
    "breathing exercises", "sleep meditation", "mood journal", "anxiety support", 
    "mental health support", "relaxation techniques", "guided relaxation", 
    "counseling app", "mental health tools", "therapy on-demand", "wellness app", 
    "meditation for anxiety", "stress management tools", "mindful breathing", 
    "guided sleep", "calm app", "self-help app", "mental fitness", "mindfulness therapy", 
    "trauma recovery", "meditation for sleep", "mental health management", 
    "emotional regulation", "journaling app", "mental resilience", "well-being app", 
    "mood tracking app", "therapy tools", "anxiety tracker", "mood diary", "self-help tools", 
    "therapy chat", "mental health community", "teletherapy", "virtual therapy", 
    "meditation and mindfulness", "relaxation app", "emotional health", "digital therapy", 
    "therapy chatbot", "anxiety meditation", "depression relief", "burnout recovery", 
    "panic attack support", "emotional support app", "self-reflection app", 
    "mental health coaching", "mental health exercises", "wellness tracker", 
    "mindful journaling", "sleep better app", "relaxation music", "mental clarity", 
    "self-improvement app", "coping strategies", "anxiety journaling", "mental health monitor", 
    "therapy scheduling", "sleep therapy", "mindfulness coach", "mental health resources", 
    "guided therapy", "mental health check-in", "calming exercises", "stress relief tips", 
    "meditation for beginners", "online therapy", "mindful exercises", "sleep health", 
    "habit tracking", "daily affirmations", "sleep sounds", "calm meditation", 
    "progressive muscle relaxation", "body scan meditation", "positive psychology app", 
    "mood management", "mindful living", "digital mental health", "self-compassion exercises", 
    "cognitive therapy", "virtual counseling", "self-guided therapy", "daily wellness check", 
    "calming app", "mental balance", "digital detox", "life coaching", "trauma therapy app", 
    "emotional balance", "behavioral therapy", "mental health education", "mindfulness journal", 
    "mindset shift", "emotional resilience", "self-awareness app", "sleep tracking", 
    "guided mindfulness", "therapy sessions", "daily meditation", "therapist finder", 
    "personalized therapy", "gratitude journaling", "daily mindfulness", "mental health evaluation", 
    "stress reduction app", "meditation sessions", "online counseling app", 
    "emotional intelligence tools", "stress tracking", "self-improvement tracker"
""")

def get_embedding(text, model):
    # inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    
    # with torch.no_grad():
    #     outputs = model(**inputs)
    
    # embeddings = mean_pooling(outputs, inputs['attention_mask'])
    # embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # return embeddings

    return model.encode(text, convert_to_tensor=True)


def find_similarity(embedding_one, embedding_two):
    similarity = cosine_similarity(embedding_one.unsqueeze(0), embedding_two.unsqueeze(0))
    return similarity[0][0]

# Function to calculate cosine similarity between two embeddings
def find_similarity(embedding_one, embedding_two):
    # Use cosine similarity to measure how similar two embeddings are
    similarity = cosine_similarity(embedding_one.unsqueeze(0), embedding_two.unsqueeze(0))
    # Return the similarity value from the cosine similarity result
    return similarity[0][0]

# Function to process app data, calculate similarity between reviews and description, and find most related words
def get_similarity_data(data):
    description = data["Description"].lower()  # Convert description to lowercase
    description_keywords = [word for word in keywords if word in description]  # Extract description keywords
    keyword_embeddings = {word: get_embedding(word, model) for word in description_keywords}  # Keyword embeddings
    app_type_keywords = [word for word in app_keywords if word in description]  # Extract app type keywords
    app_type_embeddings = {word: get_embedding(word, model) for word in app_type_keywords}  # App type embeddings
    description_embedding = get_embedding(''.join(description_keywords), model)  # Description embedding

    reviews_list = []  # List to hold processed reviews

    # Process each review
    for review in data["Reviews"]:
        review_embedding = get_embedding(review["Review"], model)
        most_related_word = ""
        max_similarity = float('-inf')

        # Compare review embedding with keyword embeddings
        for keyword, keyword_embedding in keyword_embeddings.items():
            similarity_score = find_similarity(review_embedding, keyword_embedding)
            if similarity_score > max_similarity:
                max_similarity = similarity_score
                most_related_word = keyword

        # Compare with app type embeddings if no match found
        if review["Review"] is not None and most_related_word not in review["Review"]:
            for keyword, app_type_embedding in app_type_embeddings.items():
                similarity_score = find_similarity(review_embedding, app_type_embedding)
                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    most_related_word = keyword

        # Skip review if no valid most related word found
        if review["Review"] is None or most_related_word not in review["Review"]:
            continue

        # Calculate similarity with the description
        similarity_score = find_similarity(review_embedding, description_embedding)
        
        # Add processed review to the list
        reviews_list.append({
            "Rating": review["Rating"],
            "Review": review["Review"],
            "Sentiment": review["Sentiment"],
            "Similarity": float(similarity_score),
            "Most Related Word": most_related_word
        })

    return {
            "App Name": data.get("App Name", "Unknown App"),
            "Developer": data.get("Developer", "Unknown Developer"),
            "Ratings": data.get("Ratings", "No Ratings"),
            "Description": data.get("Description", "No Description"),
            "Description Sentiment": data.get("Description Sentiment", "No Description Sentiment"),
            "Reviews": reviews_list,
            "Total Sentiment": data.get("Total Sentiment", "No Sentiment")
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
                similarity_data = get_similarity_data(data)  # Analyze similarity
                output_file_path = os.path.join(output_dir, filename)

                # Save the sentiment analysis results to a new JSON file
                with open(output_file_path, 'w', encoding='utf-8') as file:
                    json.dump(similarity_data, file, indent=4, ensure_ascii=False)
            print(f'Processed {filename} and saved to {output_file_path}')
    
    if not files_processed:
        print(f"No JSON files found in {input_dir}")

if __name__ == "__main__":
    # Define input and output directories
    input_dir = 'Data/SentimentData' 
    output_dir = 'Data/SimilarityData' 
    
    # Process the directory
    process_directory(input_dir, output_dir)

    print(f"Keyword extraction completed. Check the '{output_dir}' folder.")