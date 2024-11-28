import json

def filter_data_by_label(query, data_path):
    data = load_data(data_path)
    filtered_apps = []
    query_lower = query.lower()
    num_apps = 0
    for app in data:
        matched_labels = [label for label in app['labels'] if label in query_lower]
        if matched_labels:
            # return apps with labels that match the queried conditions
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

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        print(f"Loaded data from {file_path}")
        return data

# use token reduced data and make a query (testing)
data_path = "C:/Emotional AI/Data/CompleteData/StringData.json"
query = "Can I get support for my insomnia?"
res = filter_data_by_label(query, data_path)

data_string = ""

for app in res:
        data_string += f"name:{app['name']}"
        data_string += f"labels:{' '.join(app['labels'])}"
        data_string += f"reviews:{app['reviews']}"
        data_string += "\n"

text_path = "C:/Emotional AI/Data/CompleteData/output.txt"

with open(text_path, 'w', encoding='utf-8') as file:
    file.write(data_string)
