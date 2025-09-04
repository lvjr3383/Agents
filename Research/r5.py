import re
import os
from dotenv import load_dotenv
from tabulate import tabulate
import requests
import arxiv
import json
from kaggle.api.kaggle_api_extended import KaggleApi

load_dotenv()

# --- API Fetching Functions (Centralized for Robustness) ---

def fetch_github_repos(topic, count):
    """Fetches top GitHub repos for a given topic."""
    api_url = "https://api.github.com/search/repositories"
    params = {"q": f"{topic} in:name", "sort": "stars", "order": "desc", "per_page": count}
    headers = {"Authorization": f"token {os.getenv('GITHUB_TOKEN')}"}
    
    try:
        response = requests.get(api_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        repos = response.json().get("items", [])
        
        return [
            {
                "name": repo["full_name"],
                "url": repo["html_url"],
                "stars": repo["stargazers_count"]
            }
            for repo in repos[:count]
        ]
    except requests.exceptions.RequestException as e:
        raise Exception(f"GitHub API error: {str(e)}")

def fetch_huggingface_spaces(count, category=None):
    """Fetches top HuggingFace Spaces by likes, with an optional category filter."""
    if count not in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        raise ValueError("Count must be 5, 10, 15, 20, 25, 30, 35, 40, 45, or 50")
    
    url = "https://huggingface.co/api/spaces"
    params = {"sort": "likes", "direction": "-1", "limit": count}

    if category and category != "spaces":
        params["search"] = category.replace(" ", "-")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        spaces = response.json()
        
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

def fetch_arxiv_papers(category, count):
    """Fetch recent ArXiv papers for the given category."""
    if count not in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        raise ValueError("Count must be 5, 10, 15, 20, 25, 30, 35, 40, 45, or 50")
    
    category_map = {
        "cs": "cs",
        "cs.ai": "cs.AI",
        "cs.lg": "cs.LG",
        "cs.cv": "cs.CV",
        "cs.hc": "cs.HC",
        "cs.ma": "cs.MA",
        "cs.ro": "cs.RO"
    }
    cat_code = category_map.get(category)
    if not cat_code:
        raise ValueError("Invalid category, must be 'cs', 'cs.AI', etc.")
    
    search = arxiv.Search(
        query=f"cat:{cat_code}",
        max_results=count,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    client = arxiv.Client()

    try:
        results = list(client.results(search))
        
        formatted_results = []
        for r in results:
            formatted_results.append({
                "name": r.title,
                "url": r.entry_id,
                "date": r.published.date()
            })
        
        return formatted_results
    except Exception as e:
        raise Exception(f"ArXiv API error: {str(e)}")

def fetch_kaggle_code(count, search_term):
    """
    Fetches Kaggle code notebooks for a given search term and sorts them by vote count.
    """
    if count not in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        raise ValueError("Count must be 5, 10, 15, 20, 25, 30, 35, 40, 45, or 50")
    
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
                "url": f"https://www.kaggle.com/{item.ref}",
                "votes": getattr(item, 'totalVotes', 0)
            })
        
        return formatted_results
    except Exception as e:
        raise Exception(f"Kaggle API error: {str(e)}")

# --- Agent Parsing Functions (Updated for New Categories) ---

def parse_github_query(query):
    match = re.match(r"top\s+(\d+)\s+([\w\s-]+)", query.lower().strip())
    if not match:
        return None, None
    count, topic = match.groups()
    count = int(count)
    if count not in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        return None, None
    valid_topics = [
        "python", "machine learning", "deep learning", "ai", "generative ai",
        "data science", "computer vision", "natural language processing",
        "reinforcement learning", "big data", "web development", "blockchain",
        "cybersecurity", "tensorflow"
    ]
    if topic not in valid_topics:
        return None, None
    return count, topic

def parse_huggingface_query(query):
    match = re.match(r"top\s+(\d+)\s+(spaces|[\w\s-]+)", query.lower().strip())
    if not match:
        return None, None
    count = int(match.group(1))
    category = match.group(2)
    if count not in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        return None, None
    valid_categories = ["spaces", "image generation", "video generation", "text generation", "language translation", "speech synthesis", "3d modeling", "object detection", "text analysis"]
    if category not in valid_categories:
        return None, None
    return count, category

def parse_arxiv_query(query):
    match = re.match(r"recent\s+(\d+)\s+([\w\s.-]+)", query.lower().strip())
    if not match:
        return None, None
    count, category = match.groups()
    count = int(count)
    if count not in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        return None, None
    valid_categories = ["cs", "cs.ai", "cs.lg", "cs.cv", "cs.hc", "cs.ma", "cs.ro"]
    if category not in valid_categories:
        return None, None
    return count, category

def parse_kaggle_query(query):
    match = re.match(r"top\s+(\d+)\s+([\w\s-]+)", query.lower().strip())
    if not match:
        return None, None
    count, search_term = match.groups()
    count = int(count)
    if count not in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        return None, None
    valid_terms = ["python", "r", "sql", "nlp", "random forest", "deep learning", "computer vision", "text classification"]
    if search_term not in valid_terms:
        return None, None
    return count, search_term

# --- Agent Functions (with new UX and Personas) ---

def gitty_agent():
    print("\nGitty here! I'm your guide to the best of the open-source world. What are you looking for? (e.g., 'top 5 python')")
    while True:
        query = input("> ")
        if query.lower().strip() == "done":
            return
        count, topic = parse_github_query(query)
        if not topic:
            print("Oops! I don't recognize that topic or count. Try something like 'top 5 python'.")
            continue
        try:
            print(f"\nFetching top {count} GitHub repos for '{topic}'...\n")
            results = fetch_github_repos(topic, count)
            if not results:
                print(f"No repos found for '{topic}'. Maybe try a different search?")
                continue
            print(f"Gitty's Top {count} GitHub Repos:\n")
            table_data = [
                [i, item['name'][:30] + "..." if len(item['name']) > 30 else item['name'],
                 item['url'][:50] + "..." if len(item['url']) > 50 else item['url'],
                 item['stars']]
                for i, item in enumerate(results, 1)
            ]
            print(tabulate(table_data, headers=["Rank", "Name", "Link", "Stars"], tablefmt="fancy_grid", maxcolwidths=[5, 30, 50, 10]))
        except Exception as e:
            print(f"Error fetching repos: {str(e)}")
        print("\nReady for another GitHub quest or are you 'done' with me?\n")

def huggy_agent():
    print("\nHuggy here! Welcome to the world of AI demos and apps. I can find the most popular Spaces for you.")
    valid_categories = ["spaces", "image generation", "video generation", "text generation", "language translation", "speech synthesis", "3d modeling", "object detection", "text analysis"]
    print("My available categories are:")
    for i, cat in enumerate(valid_categories, 1):
        print(f"  {i}. {cat.title()}")
    print("\nWhat are you looking for? (e.g., 'top 5 spaces' or 'top 10 image generation')")
    while True:
        query = input("> ")
        if query.lower().strip() == "done":
            return
        count, category = parse_huggingface_query(query)
        if not category:
            print("Hmm, that didn't quite match. Try something like 'top 5 spaces' or choose from my list!")
            continue
        try:
            print(f"\nFetching top {count} HuggingFace Spaces for '{category}'...\n")
            if category == "spaces":
                results = fetch_huggingface_spaces(count)
            else:
                results = fetch_huggingface_spaces(count, category=category)
            if not results:
                print(f"No spaces found for '{category}'. Let's try a different category!")
                continue
            print(f"Huggy's Top {count} HuggingFace Spaces:\n")
            table_data = [
                [i, item['name'][:30] + "..." if len(item['name']) > 30 else item['name'],
                 item['url'][:50] + "..." if len(item['url']) > 50 else item['url'],
                 item['stars']]
                for i, item in enumerate(results, 1)
            ]
            print(tabulate(table_data, headers=["Rank", "Name", "Link", "Likes"], tablefmt="fancy_grid", maxcolwidths=[5, 30, 50, 10]))
        except Exception as e:
            print(f"Error fetching Spaces: {str(e)}")
        print("\nReady for another Hugging Face adventure or are you 'done' for now?\n")

def arxy_agent():
    print("\nArxy here! I'm your window into the latest academic research. What papers can I find for you?")
    valid_categories = ["cs", "cs.ai", "cs.lg", "cs.cv", "cs.hc", "cs.ma", "cs.ro"]
    print("My available categories are:")
    for i, cat in enumerate(valid_categories, 1):
        print(f"  {i}. {cat}")
    print("\nTell me what you're looking for, for example: 'recent 5 cs.ai'")
    while True:
        query = input("> ")
        if query.lower().strip() == "done":
            return
        count, category = parse_arxiv_query(query)
        if not category:
            print("That's not a valid query. Please try again with a count and a valid category from my list!")
            continue
        try:
            print(f"\nFetching the {count} most recent ArXiv papers for '{category}'...\n")
            results = fetch_arxiv_papers(category, count)
            if not results:
                print(f"I couldn't find any recent papers for '{category}'.")
                continue
            print(f"Arxy's Top {count} Recent ArXiv Papers:\n")
            table_data = [
                [i, item['name'][:30] + "..." if len(item['name']) > 30 else item['name'],
                 item['url'][:50] + "..." if len(item['url']) > 50 else item['url'],
                 item['date']]
                for i, item in enumerate(results, 1)
            ]
            print(tabulate(table_data, headers=["Rank", "Name", "Link", "Date"], tablefmt="fancy_grid", maxcolwidths=[5, 30, 50, 10]))
        except Exception as e:
            print(f"Error fetching papers: {str(e)}")
        print("\nWant to search more papers or are you 'done' with me?\n")

def kaggy_agent():
    print("\nKaggy here! I'm your go-to for the most popular code notebooks in the data science world.")
    valid_terms = ["python", "r", "sql", "nlp", "random forest", "deep learning", "computer vision", "text classification"]
    print("My available search terms are:")
    for i, term in enumerate(valid_terms, 1):
        print(f"  {i}. {term.title()}")
    print("\nWhat are you looking for? (e.g., 'top 10 python' or 'top 5 nlp')")
    while True:
        query = input("> ")
        if query.lower().strip() == "done":
            return
        count, search_term = parse_kaggle_query(query)
        if not search_term:
            print("I'm not familiar with that search term. Please try again with a count and a term from my list!")
            continue
        try:
            print(f"\nFetching top {count} Kaggle notebooks for '{search_term}'...\n")
            results = fetch_kaggle_code(count, search_term)
            if not results:
                print(f"I couldn't find any notebooks for '{search_term}'.")
                continue
            print(f"Kaggy's Top {count} Kaggle Notebooks:\n")
            table_data = [
                [i, item['name'][:30] + "..." if len(item['name']) > 30 else item['name'],
                 item['url'][:50] + "..." if len(item['url']) > 50 else item['url'],
                 item['votes']]
                for i, item in enumerate(results, 1)
            ]
            print(tabulate(table_data, headers=["Rank", "Name", "Link", "Votes"], tablefmt="fancy_grid", maxcolwidths=[5, 30, 50, 10]))
        except Exception as e:
            print(f"Error fetching notebooks: {str(e)}")
        print("\nWant to see more notebooks or are you 'done' with me?\n")

def main():
    """Main CLI loop for Research Agent."""
    print("Hi there! I'm Researchy, your friendly knowledge conductor. I can help you find top repositories, apps, and papers from across the web. Which library of knowledge would you like to explore first?")
    while True:
        print("\nChoose an agent to get started:")
        print("  1. Gitty (GitHub)")
        print("  2. Huggy (HuggingFace)")
        print("  3. Arxy (Arxiv)")
        print("  4. Kaggy (Kaggle)")
        print("Or type 'exit' to say goodbye.")
        source = input("> ").lower().strip()
        if source == "exit":
            print("\nIt was great exploring with you! Feel free to call on me anytime you need a research boost. Until next time, adieu!")
            break
        if source == "github" or source == "1":
            gitty_agent()
        elif source == "huggingface" or source == "2":
            huggy_agent()
        elif source == "arxiv" or source == "3":
            arxy_agent()
        elif source == "kaggle" or source == "4":
            kaggy_agent()
        else:
            print("Invalid choice. Please pick a number or type an agent's name.")
            continue
        print("\nBack to Researchy! Ready to dive into another source of knowledge, or is it time to bid adieu?")

if __name__ == "__main__":
    main()