from google_play_scraper import app, reviews_all, Sort
import os
import json

def android_scrape(app_id):
    app_details = app(
        app_id,
        lang='en',
        country='us'
    )

    app_title = app_details['title']
    app_developer = app_details['developer']
    star_rating = f"{round(app_details['score'], 1)} out of 5"
    app_description = app_details['description']

    reviews_list = [] 

    reviews = reviews_all(
        app_id,
        sleep_milliseconds=0,
        lang='en',
        country='us',
        sort=Sort.MOST_RELEVANT
    )

    for review in reviews:
        reviews_list.append({
            "Rating": (f"{round(review['score'], 1)} out of 5"), 
            "Review": review['content'] 
        })

    # this is how the data is formatted
    app_data = {
        "App Name": app_title,
        "Developer": app_developer,
        "Ratings": star_rating,
        "Description": app_description,
        "Reviews": reviews_list
    }

    folder_path = "C:/Emotional AI/Data/ScrapedData"
    safe_app_name = "".join(c for c in app_title if c.isalnum() or c in (' ', '_')).rstrip()
    file_name = f"{safe_app_name}.json"
    file_path = os.path.join(folder_path, file_name)

    os.makedirs(folder_path, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(app_data, file, indent=4, ensure_ascii=False)

    print(f"Data for '{app_title}' saved to '{file_path}'")

if __name__ == "__main__":
    # enter google play app id to scrape data
    app_id = input("Enter the app id: ")
    android_scrape(app_id)
