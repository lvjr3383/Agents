from datetime import datetime

class RepoAgent:
    def __init__(self, console):
        self.console = console

    def generate_report(self, person_name: str, adverse_hits: list, general_news: list, timestamp: str = None, write_file: bool = True):
        """
        Build a markdown report. Optionally write it to disk (for CLI), but always return the markdown string for UI use.
        """
        if not adverse_hits and not general_news:
            if self.console:
                self.console.print("[yellow]No data available to generate a report.[/yellow]")
            return ""

        ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Compliance_Report_{person_name.replace(' ', '_')}_{ts}.md"

        lines = []
        lines.append("# Adverse Media Screening Report\n")
        lines.append(f"**Subject:** {person_name}")
        lines.append(f"**Date of Report:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("---\n")

        lines.append("## Executive Summary\n")
        if adverse_hits:
            highest_risk = max(hit.get('risk_score', hit.get('risk', 0)) for hit in adverse_hits)
            high_confidence_count = sum(1 for hit in adverse_hits if hit.get('confidence') == 'High')
            lines.append(f"- **Adverse Hits Found:** {len(adverse_hits)}")
            lines.append(f"- **High-Confidence Hits:** {high_confidence_count}")
            lines.append(f"- **Highest Risk Score Detected:** {highest_risk}\n")
        else:
            lines.append("- **No direct adverse media hits were found.**\n")
        lines.append(f"A total of {len(general_news)} general news articles were also reviewed for context.\n")

        if adverse_hits:
            lines.append("---")
            lines.append("## Detailed Adverse Findings\n")
            lines.append("| No. | Date       | Confidence | Risk | Sentiment | Category         | Title                                      | URL |")
            lines.append("|-----|------------|------------|------|-----------|------------------|--------------------------------------------|-----|")
            for idx, article in enumerate(adverse_hits, 1):
                lines.append(f"| {idx} | {article.get('published_at','N/A')} | {article.get('confidence', 'N/A')} | {article.get('risk_score', article.get('risk','N/A'))} | {article.get('sentiment', 'N/A')} | {article.get('category', 'N/A')} | {article.get('title', 'N/A')} | [Link]({article.get('url', '#')}) |")
            lines.append("")

        if general_news:
            lines.append("---")
            lines.append("## General News for Context\n")
            lines.append("| No. | Date       | Category      | Source          | Title                                      |")
            lines.append("|-----|------------|---------------|-----------------|--------------------------------------------|")
            for idx, article in enumerate(general_news, 1):
                lines.append(f"| {idx} | {article.get('published_at','N/A')} | {article.get('category','General News')} | {article.get('source','N/A')} | {article.get('title','N/A')} |")

        lines.append("\n---\n")
        lines.append("## Definitions\n")
        lines.append("**Risk Score:** A score calculated based on the severity of keywords found in the article (e.g., 'Fraud' > 'Lawsuit'). Higher scores indicate more severe topics.")
        lines.append("**Sentiment:** An AI-generated score of the headline's emotional tone (Positive, Negative, Neutral). Negative sentiment indicates the use of alarming or emotionally charged language.")
        lines.append("**Confidence Level:** A synthesized rating based on both Risk and Sentiment. A 'High' confidence hit has a high risk score and negative sentiment, strongly suggesting it requires manual review.")

        markdown_report = "\n".join(lines)

        if write_file:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(markdown_report)
                if self.console:
                    self.console.print(f"\n[bold blue]>>> Repo Activated <<<[/bold blue]")
                    self.console.print(f"Generating compliance report: [bold cyan]{filename}[/bold cyan]...")
                    self.console.print("[bold green]✅ Report generated successfully.[/bold green]")
            except Exception as e:
                if self.console:
                    self.console.print(f"[bold red]❌ Error generating report: {e}[/bold red]")

        return markdown_report
