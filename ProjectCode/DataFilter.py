import json
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    text = text.replace('\\n', ' ').replace('\\', '')
    text = text.replace('"', '')
    
    tokens = word_tokenize(text.lower())
    
    stop_words = set(stopwords.words('english'))
    
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

def preprocess_keywords(keyword_string):
    keywords = [keyword.strip().lower().replace('"', '') for keyword in keyword_string.split(',')]
    return keywords

# keyword list
keywords = preprocess_keywords("""
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
    "emotional intelligence tools", "stress tracking", "self-improvement tracker",
                           
    "depression", "anxiety", "ADHD", "attention deficit hyperactivity disorder", 
    "bipolar disorder", "schizophrenia", "OCD", "obsessive-compulsive disorder", 
    "PTSD", "post-traumatic stress disorder", "panic disorder", "generalized anxiety disorder", 
    "GAD", "social anxiety", "phobias", "eating disorders", "anorexia", "bulimia", 
    "binge eating disorder", "autism", "autism spectrum disorder", "ASD", "borderline personality disorder", 
    "BPD", "dissociative identity disorder", "DID", "postpartum depression", 
    "seasonal affective disorder", "SAD", "insomnia", "sleep disorders", 
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

def is_mental_health_related(text, keywords):
    # Check if the given text contains any of the keywords
    if text:
        tokens = preprocess_text(text)
        return any(keyword in tokens for keyword in keywords)
    
    return False

def filter(data, keywords):
    # Filter reviews in the data for mental health relevance
    if 'Reviews' in data and isinstance(data['Reviews'], list):
        filtered_reviews = [review for review in data['Reviews'] if is_mental_health_related(review.get('Review', ''), keywords)]
        return {
            "App Name": data.get("App Name", "Unknown App"),
            "Developer": data.get("Developer", "Unknown Developer"),
            "Ratings": data.get("Ratings", "No Ratings"),
            "Description": data.get("Description", "No Description"),
            "Reviews": filtered_reviews
        }
    else:
        print(f"No 'Reviews' found in app data: {data.get('App Name', 'Unknown App')}")
        return data 

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
                filtered_data = filter(data, keywords)
                output_file_path = os.path.join(output_dir, filename)

                with open(output_file_path, 'w', encoding='utf-8') as file:
                    json.dump(filtered_data, file, indent=4, ensure_ascii=False)
            print(f'Processed {filename} and saved to {output_file_path}')
    
    if not files_processed:
        print(f"No JSON files found in {input_dir}")

if __name__ == "__main__":
    input_dir = 'Data/ScrapedData' 
    output_dir = 'Data/FilteredData' 
    
    process_directory(input_dir, output_dir)

    print(f"Keyword extraction completed. Check the '{output_dir}' folder.")