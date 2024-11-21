from google_play_scraper import app, reviews_all, Sort
import os
import json

def android_scrape(app_id):
    # Fetch app details using the Google Play Scraper
    app_details = app(
        app_id,
        lang='en',
        country='us'
    )

    # Extract relevant information from the app details
    app_title = app_details['title']
    app_developer = app_details['developer']
    star_rating = f"{round(app_details['score'], 1)} out of 5"  # Round the rating to one decimal place
    app_description = app_details['description']

    reviews_list = []  # Initialize a list to store reviews

    # Fetch all reviews for the app
    reviews = reviews_all(
        app_id,
        sleep_milliseconds=0,
        lang='en',
        country='us',
        sort=Sort.MOST_RELEVANT
    )

    # Process each review and store relevant data
    for review in reviews:
        reviews_list.append({
            "Rating": (f"{round(review['score'], 1)} out of 5"),  # Round and format the review score
            "Review": review['content']  # Get the review content
        })

    # Create a dictionary to hold all app data
    app_data = {
        "App Name": app_title,
        "Developer": app_developer,
        "Ratings": star_rating,
        "Description": app_description,
        "Reviews": reviews_list
    }

    # Define the folder path to save the data
    folder_path = "C:/Emotional AI/Data/ScrapedData"
    # Replace any invalid characters in the app title for filenames
    safe_app_name = "".join(c for c in app_title if c.isalnum() or c in (' ', '_')).rstrip()
    file_name = f"{safe_app_name}.json"  # Ensure the file name has a .json extension
    file_path = os.path.join(folder_path, file_name)

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Save the app data to a JSON file
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(app_data, file, indent=4, ensure_ascii=False)

    # Print a confirmation message
    print(f"Data for '{app_title}' saved to '{file_path}'")

if __name__ == "__main__":
    # Prompt the user for the app ID and initiate scraping
    app_id = input("Enter the app id: ")
    android_scrape(app_id)
