# utils/summary.py
import requests
from bs4 import BeautifulSoup
import re

def get_quick_summary(url: str, timeout: int = 10, sentences: int = 3) -> str:
    """Return first few clean sentences from article. Graceful fallback."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Jack Bank Adverse Media Tool – Research Only)"
        }
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "iframe", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        clean = [s.strip() for s in sentences if len(s.strip()) > 25][:sentences]

        if clean:
            return " ".join(clean)
        return "Summary not available (paywall / dynamic content)"
    except Exception:
        return "Summary unavailable (blocked / timeout)"
