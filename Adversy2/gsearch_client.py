import os
import requests
from dotenv import load_dotenv

class GoogleSearchClient:
    """A client to interact with the Google Custom Search JSON API."""

    def __init__(self):
        """
        Initializes the client by loading Google API credentials.
        """
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.pse_id = os.getenv("GOOGLE_PSE_ID")
        
        if not self.api_key or not self.pse_id:
            raise ValueError("GOOGLE_API_KEY or GOOGLE_PSE_ID not found in .env file")
        
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def search(self, name: str, keywords: list = None):
        """
        Searches the Google PSE for a name, optionally with keywords.

        Args:
            name (str): The name of the person to search for.
            keywords (list, optional): A list of keywords to add to the search.

        Returns:
            list: A list of search result items, formatted like our other articles.
        """
        if keywords:
            keyword_query = " OR ".join(keywords)
            query = f'"{name}" AND ({keyword_query})'
        else:
            query = f'"{name}"'
        
        print(f"Constructed Google Query: {query}")

        params = {
            "q": query,
            "key": self.api_key,
            "cx": self.pse_id,
            "num": 10 # Request the top 10 results
        }

        try:
            print("🔍 Searching Google...")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()

            data = response.json()
            search_items = data.get("items", [])
            print(f"✅ Found {len(search_items)} results from Google.")

            # --- Format the results to match our standard "article" structure ---
            formatted_results = []
            for item in search_items:
                formatted_results.append({
                    "title": item.get("title"),
                    "url": item.get("link"),
                    "description": item.get("snippet"),
                    "source": item.get("displayLink") # e.g., 'www.nytimes.com'
                })
            
            return formatted_results

        except requests.exceptions.HTTPError as e:
            # Try to get more specific error info from Google's response
            error_details = e.response.json().get("error", {})
            message = error_details.get("message", "No additional details")
            print(f"❌ HTTP error occurred: {e.response.status_code} - {message}")
            return []
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error occurred: {e}")
            return []