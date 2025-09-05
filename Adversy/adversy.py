from neggy_agent import NeggyAgent
from neggy_google import GoogleNeggyAgent # Import our new agent
from genny_agent import GennyAgent
from repo_agent import RepoAgent
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel

def display_results(console: Console, results: list, title: str, is_adverse: bool = False):
    """Helper function to display articles in a table."""
    if not results:
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("No.", style="dim", width=4, justify="right")
    table.add_column("Date", style="dim", width=12)
    table.add_column("Category", style="yellow")
    if is_adverse:
        table.add_column("Risk", style="bold", justify="right")
        table.add_column("Sentiment", style="bold")
        table.add_column("Confidence", style="bold")
    table.add_column("Source", style="cyan")
    table.add_column("Title", style="green")
    table.add_column("URL", style="blue")

    for idx, article in enumerate(results, 1):
        row = [
            str(idx),
            article.get("published_at", "N/A"),
            article["category"],
        ]

        if is_adverse:
            risk_score = article.get("risk_score", 0)
            sentiment = article.get("sentiment", "Neutral")
            confidence = article.get("confidence", "Low")
            
            risk_text = Text(str(risk_score))
            if risk_score > 8: risk_text.stylize("bold red")
            elif risk_score > 5: risk_text.stylize("bold yellow")

            sentiment_text = Text(sentiment)
            if sentiment == "Negative": sentiment_text.stylize("bold red")
            elif sentiment == "Positive": sentiment_text.stylize("bold green")

            confidence_text = Text(confidence)
            if confidence == "High": confidence_text.stylize("bold red")
            elif confidence == "Medium": confidence_text.stylize("bold yellow")
            
            row.extend([risk_text, sentiment_text, confidence_text])

        row.extend([
            article["source"],
            article["title"],
            article["url"]
        ])
        table.add_row(*row)
    
    console.print(Panel(table, title=title, border_style="green" if is_adverse else "blue", expand=False))

def main():
    console = Console()
    neggy_newsapi = NeggyAgent(console)
    neggy_google = GoogleNeggyAgent(console)
    genny = GennyAgent(console)
    repo = RepoAgent(console)

    console.print(Panel("[bold cyan]Welcome to the Adversy Suite[/bold cyan]\nI am Adversy, your primary interface for media screening.\nYou can type 'exit' or 'quit' at any time to stop.",
                          title="[bold]Adversy Conductor[/bold]", border_style="cyan"))

    while True:
        try:
            # --- NEW: Agent Selection ---
            console.print("\nPlease select a specialist to deploy for the investigation:")
            console.print("  [bold]1[/bold]: Neggy (Uses NewsAPI for recent news)")
            console.print("  [bold]2[/bold]: Neggy-G (Uses Google Search for broader web results)")
            
            choice = console.input("Enter your choice (1 or 2): ")
            
            if choice == '1':
                active_neggy = neggy_newsapi
            elif choice == '2':
                active_neggy = neggy_google
            else:
                console.print("[bold red]Invalid choice. Please try again.[/bold red]")
                continue
            # --- END OF NEW ---
            
            name = console.input("\n[bold yellow]Please enter the full name you wish to screen: [/bold yellow]")

            if name.lower() in ["exit", "quit"]:
                console.print("[bold cyan]Session terminated. Goodbye![/bold cyan]")
                break
            
            if not name:
                console.print("[bold red]Name cannot be empty.[/bold red]")
                continue

            adverse_hits, general_news = active_neggy.investigate(name)
            
            if adverse_hits:
                display_results(console, adverse_hits, f"{active_neggy.__class__.__name__}'s Report: Adverse Media Hits for '{name}'", is_adverse=True)
            
            genny_report = []
            if general_news:
                prompt = console.input(f"\nThis agent has filtered out {len(general_news)} general articles. Deploy Genny to see them? (y/n): ")
                if prompt.lower() == 'y':
                    genny_report = genny.present_intel(general_news)
                    display_results(console, genny_report, f"Genny's Report: General News for '{name}'")
            
            if adverse_hits or genny_report:
                report_prompt = console.input(f"\nWould you like to generate a full compliance report for '{name}'? (y/n): ")
                if report_prompt.lower() == 'y':
                    repo.generate_report(name, adverse_hits, genny_report)

        except KeyboardInterrupt:
            console.print("\n[bold cyan]Session terminated. Goodbye![/bold cyan]")
            break
        except Exception as e:
            console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")

if __name__ == "__main__":
    main()