import json
from gsearch_client import GoogleSearchClient # Changed import
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class GoogleNeggyAgent: # Renamed class
    def __init__(self, console):
        self.console = console
        self.search_client = GoogleSearchClient() # Use the Google client
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
                self.keywords = config.get("keywords", {})
            if not self.keywords:
                raise ValueError("Keywords not found in config.json")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.console.print(f"[bold red]Error loading config.json: {e}[/bold red]")
            self.keywords = {}

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
        self.console.print("\n[bold blue]>>> Google Neggy Activated <<<[/bold blue]")
        self.console.print(f"I am conducting a comprehensive Google search for '{name}' and will report any adverse findings.")
        
        # This agent performs the broad search and analysis
        all_results = self.search_client.search(name)
        
        if not all_results:
            self.console.print("[blue]My Google search is complete. No articles of any kind were found.[/blue]")
            return [], []

        adverse_hits = []
        general_news = []

        for item in all_results:
            title = item.get("title", "")
            description = item.get("description", "") or ""
            content_to_scan = (title + " " + description).lower()
            
            found_keywords = {kw for kw in self.keywords if kw in content_to_scan}
            
            processed_item = {
                "title": title,
                "source": item.get("source", "N/A"),
                "published_at": "N/A",  # Google Search doesn't reliably provide dates
                "url": item.get("url", "#"),
            }

            if found_keywords:
                risk_score = sum(self.keywords[kw] for kw in found_keywords)
                sentiment = self.analyze_sentiment(title)
                confidence = self.determine_confidence(risk_score, sentiment)
                
                processed_item["category"] = ", ".join(found_keywords).title()
                processed_item["risk_score"] = risk_score
                processed_item["sentiment"] = sentiment
                processed_item["confidence"] = confidence
                adverse_hits.append(processed_item)
            else:
                processed_item["category"] = "General News"
                general_news.append(processed_item)
        
        if not adverse_hits:
            self.console.print(f"[blue]My Google search is complete. No direct adverse media hits found among the {len(all_results)} results.[/blue]")
        
        adverse_hits.sort(key=lambda x: x["risk_score"], reverse=True)
        
        return adverse_hits, general_news