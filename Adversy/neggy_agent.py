import json
from news_client import NewsAPIClient
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class NeggyAgent:
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

    # ... (rest of the file is unchanged) ...
    def analyze_sentiment(self, text: str) -> str:
        score = self.sentiment_analyzer.polarity_scores(text)
        if score['compound'] >= 0.05:
            return "Positive"
        elif score['compound'] <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    def determine_confidence(self, risk_score: int, sentiment: str) -> str:
        if risk_score > 8 and sentiment == "Negative":
            return "High"
        elif risk_score > 5 or sentiment == "Negative":
            return "Medium"
        else:
            return "Low"

    def investigate(self, name: str):
        self.console.print("\n[bold green]>>> Neggy Activated <<<[/bold green]")
        self.console.print(f"I am conducting a comprehensive search for '{name}' and will report any adverse findings.")

        all_articles = self.news_client.get_articles(name)

        if not all_articles:
            self.console.print("[green]My search is complete. No articles of any kind were found.[/green]")
            return [], []

        adverse_hits = []
        general_news = []

        for article in all_articles:
            title = article.get("title", "")
            description = article.get("description", "") or ""
            content_to_scan = (title + " " + description).lower()

            found_keywords = {kw for kw in self.keywords if kw in content_to_scan}

            source_name = article.get("source", {}).get("name", "N/A")
            try:
                published_at = datetime.fromisoformat(article.get("publishedAt", "").replace("Z", "+00:00"))
                formatted_date = published_at.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                formatted_date = "N/A"

            processed_article = {
                "title": title,
                "source": source_name,
                "published_at": formatted_date,
                "url": article.get("url", "#"),
            }

            if found_keywords:
                risk_score = sum(self.keywords[kw] for kw in found_keywords)
                sentiment = self.analyze_sentiment(title)
                confidence = self.determine_confidence(risk_score, sentiment)

                processed_article["category"] = ", ".join(found_keywords).title()
                processed_article["risk_score"] = risk_score
                processed_article["sentiment"] = sentiment
                processed_article["confidence"] = confidence
                adverse_hits.append(processed_article)
            else:
                processed_article["category"] = "General News"
                general_news.append(processed_article)

        if not adverse_hits:
            self.console.print(f"[green]My search is complete. No direct adverse media hits found among the {len(all_articles)} articles.[/green]")

        adverse_hits.sort(key=lambda x: x["risk_score"], reverse=True)

        return adverse_hits, general_news