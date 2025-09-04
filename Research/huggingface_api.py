import requests

def fetch_huggingface_spaces(count, category=None):
    """
    Fetches top HuggingFace Spaces by likes, with an optional category filter.
    The category is used as a tag to filter the spaces.
    """
    if count not in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        raise ValueError("Count must be 5, 10, 15, 20, 25, 30, 35, 40, 45, or 50")
    
    # FIX: Correct the URL string so it's a valid URL, not a markdown link.
    url = "https://huggingface.co/api/spaces"
    params = {
        "sort": "likes",
        "direction": "-1",
        "limit": count
    }

    if category and category != "spaces":
        # The API uses the 'search' parameter to filter by tags.
        # We replace spaces in the category name with hyphens to match the tags.
        params["search"] = category.replace(" ", "-")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        spaces = response.json()
        
        if not spaces:
            return []
        
        return [
            {
                "name": space["id"],
                "url": f"https://huggingface.co/spaces/{space['id']}",
                "stars": space["likes"]
            }
            for space in spaces[:count]
        ]
    except requests.exceptions.RequestException as e:
        raise Exception(f"HuggingFace API error: {str(e)}")
