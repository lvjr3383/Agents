from datetime import datetime

class RepoAgent:
    def __init__(self, console):
        self.console = console

    def generate_report(self, person_name: str, adverse_hits: list, general_news: list):
        if not adverse_hits and not general_news:
            self.console.print("[yellow]No data available to generate a report.[/yellow]")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Compliance_Report_{person_name.replace(' ', '_')}_{timestamp}.md"

        self.console.print(f"\n[bold blue]>>> Repo Activated <<<[/bold blue]")
        self.console.print(f"Generating compliance report: [bold cyan]{filename}[/bold cyan]...")

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # Report Header
                f.write(f"# Adverse Media Screening Report\n\n")
                f.write(f"**Subject:** {person_name}\n")
                f.write(f"**Date of Report:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("---\n\n")

                # Executive Summary
                f.write(f"## Executive Summary\n\n")
                if adverse_hits:
                    highest_risk = max(hit['risk_score'] for hit in adverse_hits) if adverse_hits else 0
                    high_confidence_count = sum(1 for hit in adverse_hits if hit.get('confidence') == 'High')
                    f.write(f"- **Adverse Hits Found:** {len(adverse_hits)}\n")
                    f.write(f"- **High-Confidence Hits:** {high_confidence_count}\n")
                    f.write(f"- **Highest Risk Score Detected:** {highest_risk}\n\n")
                else:
                    f.write(f"- **No direct adverse media hits were found.**\n\n")
                
                f.write(f"A total of {len(general_news)} general news articles were also reviewed for context.\n\n")

                # Detailed Adverse Findings
                if adverse_hits:
                    f.write("---\n\n")
                    f.write("## Detailed Adverse Findings\n\n")
                    f.write("| No. | Date       | Confidence | Risk | Sentiment | Category         | Title                                      | URL |\n")
                    f.write("|-----|------------|------------|------|-----------|------------------|--------------------------------------------|-----|\n")
                    for idx, article in enumerate(adverse_hits, 1):
                        f.write(f"| {idx} | {article['published_at']} | {article.get('confidence', 'N/A')} | {article.get('risk_score', 'N/A')} | {article.get('sentiment', 'N/A')} | {article.get('category', 'N/A')} | {article.get('title', 'N/A')} | [Link]({article.get('url', '#')}) |\n")
                    f.write("\n")

                # --- THIS BLOCK WAS MISSING ---
                if general_news:
                    f.write("---\n\n")
                    f.write("## General News for Context\n\n")
                    f.write("| No. | Date       | Source          | Title                                      |\n")
                    f.write("|-----|------------|-----------------|--------------------------------------------|\n")
                    for idx, article in enumerate(general_news, 1):
                        f.write(f"| {idx} | {article['published_at']} | {article['source']} | {article['title']} |\n")
                # --- END OF MISSING BLOCK ---

                # Definitions Legend
                f.write("\n---\n\n")
                f.write("## Definitions\n\n")
                f.write("**Risk Score:** A score calculated based on the severity of keywords found in the article (e.g., 'Fraud' > 'Lawsuit'). Higher scores indicate more severe topics.\n\n")
                f.write("**Sentiment:** An AI-generated score of the headline's emotional tone (Positive, Negative, Neutral). Negative sentiment indicates the use of alarming or emotionally charged language.\n\n")
                f.write("**Confidence Level:** A synthesized rating based on both Risk and Sentiment. A 'High' confidence hit has a high risk score and negative sentiment, strongly suggesting it requires manual review.\n")

            self.console.print("[bold green]✅ Report generated successfully.[/bold green]")

        except Exception as e:
            self.console.print(f"[bold red]❌ Error generating report: {e}[/bold red]")