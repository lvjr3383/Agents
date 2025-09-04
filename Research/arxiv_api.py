import arxiv
from datetime import datetime
import pytz

def fetch_arxiv_papers(category, count):
    """Fetch recent ArXiv papers for the given category."""
    if count not in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        raise ValueError("Count must be 5, 10, 15, 20, 25, 30, 35, 40, 45, or 50")
    
    # Map category to ArXiv category code
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
        raise ValueError("Invalid category, must be 'cs', 'cs.AI', or 'cs.LG'")
    
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