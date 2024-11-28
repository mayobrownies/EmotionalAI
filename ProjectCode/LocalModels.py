import json
import ollama
import re

model_name = "gemma2:latest"

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        print(f"Loaded data from {file_path}") 
        return data
    
def filter_data_by_label(query, data_path):
    data = load_data(data_path)
    filtered_apps = []
    query_lower = query.lower()
    num_apps = 0
    for app in data:
        matched_labels = [label for label in app['labels'] if label in query_lower]
        if matched_labels:
            # Return apps with labels that match the queried conditions
            filtered_apps.append({
                "name": app["name"],
                "labels": matched_labels,
                "reviews": app["reviews"],
                "sentiment": app["sentiment"]
            })
            num_apps += 1
        if num_apps > 5:
            return filtered_apps
    return filtered_apps

def clean_text(text):
    cleaned_text = text.strip()
    
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)

    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text
    
data_path = "C:/Emotional AI/Data/CompleteData/StringData.json"

# update mental health condition
condition = 'ptsd'

data = filter_data_by_label(condition, data_path)

data_string = ""

for app in data:
    # Clean the text before appending it to data_string to reduce token count
    name = clean_text(app['name'])
    labels = clean_text(' '.join(app['labels']))
    reviews = clean_text(app['reviews'])
    sentiment = clean_text(str(app['sentiment']))

    data_string += f"name:{name}\n"
    data_string += f"labels:{labels}\n"
    data_string += f"reviews:{reviews}\n"
    data_string += f"sentiment:{sentiment}\n"
    data_string += "\n"

text_path = "C:/Emotional AI/Data/CompleteData/output.txt"

with open(text_path, 'w', encoding='utf-8') as file:
    file.write(data_string)

question = f"""
Given the following data: {data_string}, tell me around 5 apps I should use to help with {condition}.
The apps you recommend must come from the provided dataset. Check the app labels for {condition}. If there aren't enough apps, don't try to pad the response.
Try to provide a variety of apps.
Provide me with an overview of what users think of the app and some review quotes to support why I should use the app.
"""

response = ollama.chat(model=model_name, messages = [
    {
        'role': 'user',
        'content': question
    },
])

print(response['message']['content'])