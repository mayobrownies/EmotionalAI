import json

max_len = 200

def filter_data_by_label(query, data_path):
    data = load_data(data_path)
    filtered_apps = []
    num_apps = 0
    for app in data:
        matched_labels = [label for label in app['labels'] if label in query.lower()]
        if matched_labels:
            # filtered_reviews = ""
            # for label in app['labels']:
            #     for review in app["reviews"]:
            #         review_text = review["Review"]
            #         review_text = review_text.split(".")
            #         for sentence in review_text:
            #             if label.lower() in sentence.lower():
            #                 filtered_reviews += sentence.strip() + '.'
            #                 filtered_reviews += '\n'
                            
            #                 if len(filtered_reviews) > max_len:
            #                     break
            #         if len(filtered_reviews) > max_len:
            #             break

            # return apps with labels that match the queried conditions
            filtered_apps.append({
                "name": app["name"],
                "developer": app["developer"],
                "rating": app["rating"],
                "labels": matched_labels,
                "reviews": app["reviews"],
                "sentiment": app["sentiment"]
            })
            num_apps += 1
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
    data_string += f"name:{app['name']}\n"
    data_string += f"labels:{' '.join(app['labels'])}\n"
    data_string += f"reviews:{app['reviews']}\n"
    data_string += "\n"

text_path = "C:/Emotional AI/Data/CompleteData/output.txt"

with open(text_path, 'w', encoding='utf-8') as file:
    file.write(data_string)
