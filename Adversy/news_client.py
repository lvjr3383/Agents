import os
import requests
from dotenv import load_dotenv

class NewsAPIClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("NEWS_API_KEY")
        if not self.api_key:
            raise ValueError("NEWS_API_KEY not found in .env file")
        
        self.base_url = "https://newsapi.org/v2/everything"

    def get_articles(self, name: str):
        query = f'"{name}"'
        print(f"Constructed Query: {query}") 

        params = {
            "q": query,
            "apiKey": self.api_key,
            "language": "en",
            "sortBy": "publishedAt"
        }

        try:
            print("🔍 Searching for articles...")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            articles = data.get("articles", [])
            print(f"✅ Found {len(articles)} articles.")
            return articles

        except requests.exceptions.RequestException as e:
            print(f"❌ Network error occurred: {e}")
            return []
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
            return []