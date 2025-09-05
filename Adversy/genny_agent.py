import json
from news_client import NewsAPIClient
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class GennyAgent:
    def __init__(self, console):
        self.console = console
        self.news_client = NewsAPIClient()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        # --- CHANGE: Load keywords from config file ---
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
                self.keywords = config.get("keywords", {})
            if not self.keywords:
                raise ValueError("Keywords not found in config.json")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.console.print(f"[bold red]Error loading config.json: {e}[/bold red]")
            self.keywords = {} # Fallback to empty dict
        # --- END OF CHANGE ---

    def analyze_sentiment(self, text: str) -> str:
        score = self.sentiment_analyzer.polarity_scores(text)
        if score['compound'] >= 0.05:
            return "Positive"
        elif score['compound'] <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    def present_intel(self, general_articles: list):
        self.console.print("\n[bold blue]>>> Genny Activated <<<[/bold blue]")
        self.console.print(f"I have received {len(general_articles)} general news articles for your review.")

        if not general_articles:
            self.console.print("[blue]There are no general news articles to display.[/blue]")
            return []

        # Genny doesn't need to enrich the data, as Neggy has already done so.
        # We just need to make sure the data passed in is complete.
        # However, for consistency and future-proofing, let's keep the enrichment here.
        enriched_articles = []
        for article in general_articles:
            title = article.get("title", "No Title")
            sentiment = self.analyze_sentiment(title)
            article["risk_score"] = 0 # General news has no keyword risk
            article["sentiment"] = sentiment
            enriched_articles.append(article)

        return enriched_articles