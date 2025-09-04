import os
import requests
from dotenv import load_dotenv

load_dotenv()

def fetch_github_repos(topic, count):
    """Fetch top GitHub repos for a given topic using GitHub API."""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN not found in .env file")
    
    topic = topic.replace(" ", "+")
    url = f"https://api.github.com/search/repositories?q=topic:{topic}&sort=stars&order=desc&per_page={count}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        repos = data.get("items", [])
        if not repos:
            return []
        
        return [
            {
                "name": repo["full_name"],
                "stars": repo["stargazers_count"],
                "url": repo["html_url"],
                "description": repo["description"]
            }
            for repo in repos
        ]
    except requests.exceptions.RequestException as e:
        raise Exception(f"GitHub API error: {str(e)}")