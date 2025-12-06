# run_screening.py
import datetime
from dotenv import load_dotenv
load_dotenv()

from neggy_agent import NeggyAgent
from neggy_google import GoogleNeggyAgent
from genny_agent import GennyAgent
from repo_agent import RepoAgent
from news_client import NewsAPIClient
from gsearch_client import GoogleSearchClient

def _tag_provider(items, provider: str):
    """
    Attach provider metadata (NewsAPI or Google PSE) so the UI can display origin.
    """
    for item in items or []:
        item["source_provider"] = provider
    return items

def _initialize_status(items, default: str = "Pending"):
    """
    Ensure each result has an analyst status flag for UI tagging.
    """
    for item in items or []:
        item.setdefault("status", default)
    return items

def _initialize_comments(items):
    """
    Seed empty analyst comment fields for UI editing.
    """
    for item in items or []:
        item.setdefault("comment", "")
    return items

def run_screening(name: str, mode: str = "both"):
    console = None  # silence Rich in the UI

    adverse = []
    general = []
    google_error = None

    if mode in ["newsapi", "both"]:
        neggy = NeggyAgent(console)
        news_client = NewsAPIClient()
        adv, gen = neggy.run(name, news_client)
        adverse.extend(_tag_provider(adv, "NewsAPI"))
        general.extend(_tag_provider(gen, "NewsAPI"))

    if mode in ["google", "both"]:
        try:
            neggy_g = GoogleNeggyAgent(console)
            google_client = GoogleSearchClient()
            adv_g, gen_g = neggy_g.run(name, google_client)
            adverse.extend(_tag_provider(adv_g, "Google PSE"))
            general.extend(_tag_provider(gen_g, "Google PSE"))
        except Exception as e:
            google_error = f"Google PSE search unavailable: {e}"

    # Dedupe by URL
    seen = set()
    final_adverse = []
    final_general = []
    for item in adverse + general:
        url = item.get("url", "")
        if url and url not in seen:
            seen.add(url)
            (final_adverse if item.get("risk", 0) > 0 else final_general).append(item)

    genny = GennyAgent(console)
    final_general = genny.run(final_general)

    _initialize_status(final_adverse)
    _initialize_status(final_general)
    _initialize_comments(final_adverse)
    _initialize_comments(final_general)

    repo = RepoAgent(console)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_md = repo.generate_report(name, final_adverse, final_general, timestamp, write_file=False)

    results = {
        "name": name,
        "mode": mode,
        "adverse": final_adverse,
        "general": final_general,
        "report_md": report_md,
        "timestamp": datetime.datetime.now().isoformat()
    }
    if mode in ["google", "both"] and google_error:
        # Propagate any Google-side failures so UI can surface it.
        results["google_error"] = google_error
    return results
