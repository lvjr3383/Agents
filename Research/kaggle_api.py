import os
import re
from kaggle.api.kaggle_api_extended import KaggleApi
from tabulate import tabulate
import json

def fetch_kaggle_code(count, search_term, sort_by="hotness"):
    """
    Fetches Kaggle code notebooks for a given search term and sorts them by a specified criterion.
    The 'kaggle' library automatically handles authentication via environment variables.
    """
    if count not in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        raise ValueError("Count must be 5, 10, 15, 20, 25, 30, 35, 40, 45, or 50")
    
    # Authenticate with the Kaggle API using environment variables
    api = KaggleApi()
    api.authenticate()

    try:
        results = api.kernels_list(
            search=search_term,
            sort_by="voteCount",
            page_size=count
        )

        formatted_results = []
        for item in results:
            formatted_results.append({
                "name": item.title,
                "url": f"[https://www.kaggle.com/](https://www.kaggle.com/){item.ref}",
                "votes": getattr(item, 'totalVotes', 0)
            })
        
        return formatted_results
    except Exception as e:
        raise Exception(f"Kaggle API error: {str(e)}")