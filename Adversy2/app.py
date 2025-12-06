# app.py – Gradio UI for Adversy2
import datetime
import gradio as gr
import os
from run_screening import run_screening
from database import save_case, load_case, list_recent_cases
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------
def normalize_mode(label: str) -> str:
    label = (label or "").lower()
    if "newsapi" in label or "recent" in label:
        return "newsapi"
    if "google" in label or "deep" in label:
        return "google"
    return "both"

def display_source(item: dict) -> str:
    """
    Show domain/source along with which search provider returned the hit.
    """
    base = item.get("source", "") or ""
    provider = item.get("source_provider") or ""
    provider = provider.strip()
    provider = provider if provider else ""
    if provider and base:
        return f"{base} ({provider})"
    if provider:
        return provider
    return base or "N/A"

def results_to_tables(results, limit: int = None):
    adverse = results.get("adverse", [])
    general = results.get("general", [])
    if limit:
        adverse = adverse[:limit]
        general = general[:limit]
    adverse_rows = [
        [
            i + 1,
            a.get("risk", 0),
            a.get("category", ""),
            a.get("confidence", ""),
            display_source(a),
            a.get("title", ""),
            f"[Link]({a.get('url','')})",
        ]
        for i, a in enumerate(adverse)
    ]
    general_rows = [
        [
            i + 1,
            g.get("category", ""),
            display_source(g),
            g.get("title", ""),
            f"[Link]({g.get('url','')})",
        ]
        for i, g in enumerate(general)
    ]
    return adverse_rows, general_rows

def _safe_filename(name: str) -> str:
    keep = [c if c.isalnum() or c in ("-", "_") else "_" for c in name.strip() or "case"]
    return "".join(keep)

def export_pdf(state_payload):
    """
    Export current screening results to a PDF on the Desktop.
    """
    if not state_payload or "results" not in state_payload:
        return "Run a screening first."

    results = state_payload.get("results", {})
    name = results.get("name") or state_payload.get("name") or "Case"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_dir = os.path.expanduser("~/Desktop")
    os.makedirs(dest_dir, exist_ok=True)
    filename = f"Adverse_Report_{_safe_filename(name)}_{ts}.pdf"
    filepath = os.path.join(dest_dir, filename)

    doc = SimpleDocTemplate(filepath, pagesize=letter)
    styles = getSampleStyleSheet()
    elems = []

    elems.append(Paragraph("Adverse Media Screening Report", styles["Title"]))
    elems.append(Paragraph(f"Subject: {name}", styles["Normal"]))
    elems.append(Paragraph(f"Mode: {results.get('mode','').upper()}", styles["Normal"]))
    elems.append(Paragraph(f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}", styles["Normal"]))
    elems.append(Spacer(1, 12))

    elems.append(Paragraph(f"Adverse hits: {len(results.get('adverse', []))} | General: {len(results.get('general', []))}", styles["Heading3"]))
    elems.append(Spacer(1, 6))

    adverse = results.get("adverse", [])
    if adverse:
        elems.append(Paragraph("Adverse Hits", styles["Heading2"]))
        data = [["#", "Risk", "Category", "Confidence", "Source", "Title", "URL"]]
        for i, a in enumerate(adverse, 1):
            data.append([
                i,
                a.get("risk", a.get("risk_score", "")),
                a.get("category", ""),
                a.get("confidence", ""),
                display_source(a),
                a.get("title", ""),
                a.get("url", ""),
            ])
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        elems.append(table)
        elems.append(Spacer(1, 12))

    general = results.get("general", [])
    if general:
        elems.append(Paragraph("General News", styles["Heading2"]))
        data = [["#", "Category", "Source", "Title", "URL"]]
        for i, g in enumerate(general, 1):
            data.append([
                i,
                g.get("category", ""),
                display_source(g),
                g.get("title", ""),
                g.get("url", ""),
            ])
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        elems.append(table)

    try:
        doc.build(elems)
        return f"PDF exported to Desktop: {filepath}"
    except Exception as e:
        return f"Export failed: {e}"
# --------------------------------------------------------------
# Analyst workflow
# --------------------------------------------------------------
def run_analyst(name: str, mode_label: str):
    name = (name or "").strip()
    if not name:
        return (
            "Please enter a name to screen.",
            "",
            [],
            [],
            {},
            gr.update(),
            gr.update(value=""),
            gr.update(value=""),
        )
    mode = normalize_mode(mode_label)
    results = run_screening(name, mode)
    full_adverse_rows, full_general_rows = results_to_tables(results, limit=None)
    adverse_rows, general_rows = results_to_tables(results, limit=25)
    state_payload = {
        "adverse": full_adverse_rows,
        "general": full_general_rows,
        "results": results,
        "name": name,
        "mode": mode,
    }
    google_note = ""
    if results.get("google_error"):
        google_note = f"\n*{results['google_error']}*"
    stats = f"""
**Screening Complete**  
**{name}** | Mode: **{mode.upper()}** | {datetime.datetime.now():%Y-%m-%d %H:%M}  
Adverse hits: **{len(results['adverse'])}** | General: **{len(results['general'])}**  
Click **Save Case** to store this run.{google_note}
    """
    return (
        stats,
        manual_review(state_payload),
        adverse_rows,
        general_rows,
        state_payload,
        gr.update(value=""),
        gr.update(value=""),
    )
def save_case_handler(state_payload):
    if not state_payload or "results" not in state_payload:
        return "Run a screening first, then click Submit to Compliance."
    name = state_payload.get("name") or "Unknown"
    mode = state_payload.get("mode") or "both"
    results = state_payload.get("results", {})
    case_id = save_case(name, mode, results)
    return f"Submitted to Compliance – Token: `{case_id}`. Great work!"
# --------------------------------------------------------------
# Compliance workflow
# --------------------------------------------------------------
def list_cases():
    cases = list_recent_cases(5)
    if not cases:
        return "No cases yet. Ask the Analyst to run a screening first."
    lines = ["Recent cases:\n"]
    for c in cases:
        lines.append(f"• {c['date']} – **{c['name']}** (Token: `{c['id']}`)")
    lines.append("\nPaste a token or name below to load.")
    return "\n".join(lines)
def load_case_report(token_or_name: str):
    token_or_name = (token_or_name or "").strip()
    if not token_or_name:
        return "Enter a case token or name.", {}
    case = load_case(case_id=token_or_name) or load_case(name=token_or_name)
    if not case:
        return "Case not found. Please check the token/name.", {}
    results = case["results"]
    return f"""# Compliance Report – {case['name']}
**Case ID:** `{case['id']}` | **Date:** {case['timestamp'][:19].replace('T',' ')} | **Mode:** {case['mode'].upper()}`
{results.get('report_md','(No report available)')}
""", {
        "results": results,
        "name": case.get("name"),
        "mode": case.get("mode"),
        "id": case.get("id"),
    }
def update_adverse_page(page, state_payload):
    rows = state_payload.get("adverse", []) if state_payload else []
    return rows
def update_general_page(page, state_payload):
    rows = state_payload.get("general", []) if state_payload else []
    return rows
def show_all_general(state_payload):
    return state_payload.get("general", []) if state_payload else []
def show_all_adverse(state_payload):
    return state_payload.get("adverse", []) if state_payload else []
def manual_review(state_payload):
    if not state_payload:
        return "Run a screening first."
    items = state_payload.get("results", {}).get("adverse", [])
    flagged = [a for a in items if a.get("risk", 0) >= 8 or str(a.get("confidence", "")).lower() == "high"]
    flagged = flagged[:5]
    if not flagged:
        return "No high-priority hits (risk ≥ 8 or confidence High)."
    lines = ["**Manual Review Priority (Top 5)**"]
    for a in flagged:
        lines.append(f"- Risk {a.get('risk',0)} | {a.get('confidence','')} | {a.get('title','')} ({a.get('source','')}) [Link]({a.get('url','')})")
    return "\n".join(lines)

def handle_table_edit(kind: str, table_rows, state_payload):
    # Inline editing removed; keep tables view-only.
    return table_rows, state_payload, "Editing disabled."
# --------------------------------------------------------------
# Gradio UI – bank-ready
# --------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft(), title="Jack Bank – Adverse Media Cockpit") as demo:
    gr.Markdown("# Jack Bank\n### Adverse Media Screening Cockpit – Agentic AML Suite")
    with gr.Tabs():
        with gr.Tab("Analyst Workspace"):
            gr.Markdown("Enter the subject and choose scope. We'll run the screen, score adverse hits, and save a case token.")
            name_in = gr.Textbox(label="Entity / Person Name", placeholder="e.g., Sam Bankman-Fried")
            mode_radio = gr.Radio(
                ["Both (Recommended)", "NewsAPI (Recent)", "Google PSE (Deep)"],
                value="Both (Recommended)",
                label="Search Mode"
            )
            run_btn = gr.Button("Run Screening", variant="primary")
            save_btn = gr.Button("Submit to Compliance", variant="secondary")
            export_btn = gr.Button("Export to PDF", variant="secondary")
            stats_md = gr.Markdown()
            manual_review_md = gr.Markdown()
            adverse_df = gr.Dataframe(
                headers=["#", "Risk", "Category", "Confidence", "Source", "Title", "URL"],
                datatype=["number", "number", "str", "str", "str", "markdown", "markdown"],
                interactive=False,
                wrap=True,
                height=500,
            )
            show_all_adverse_btn = gr.Button("Show All Adverse", variant="secondary")
            general_df = gr.Dataframe(
                headers=["#", "Category", "Source", "Title", "URL"],
                datatype=["number", "str", "str", "markdown", "markdown"],
                interactive=False,
                wrap=True,
                height=360,
            )
            show_all_general_btn = gr.Button("Show All General", variant="secondary")
            state_payload = gr.State()
            save_status = gr.Markdown(value="Click **Submit to Compliance** to store this run.")
            export_status = gr.Markdown()
            run_btn.click(
                run_analyst,
                inputs=[name_in, mode_radio],
                outputs=[
                    stats_md,
                    manual_review_md,
                    adverse_df,
                    general_df,
                    state_payload,
                    save_status,
                    export_status,
                ],
            )
            show_all_adverse_btn.click(show_all_adverse, inputs=state_payload, outputs=adverse_df)
            show_all_general_btn.click(show_all_general, inputs=state_payload, outputs=general_df)
            save_btn.click(save_case_handler, inputs=state_payload, outputs=save_status)
            export_btn.click(export_pdf, inputs=state_payload, outputs=export_status)
        with gr.Tab("Compliance Review"):
            gr.Markdown("Retrieve a saved case for review or approval.")
            recent_md = gr.Markdown()
            refresh_btn = gr.Button("List Recent Cases")
            token_in = gr.Textbox(label="Case Token or Name", placeholder="Paste token or name here")
            load_btn = gr.Button("Load Case")
            report_md = gr.Markdown()
            compliance_state = gr.State()
            refresh_btn.click(list_cases, outputs=recent_md)
            load_btn.click(load_case_report, inputs=token_in, outputs=[report_md, compliance_state])
if __name__ == "__main__":
    env_port = os.getenv("GRADIO_SERVER_PORT")
    port = int(env_port) if env_port else None
    demo.launch(server_name="127.0.0.1", server_port=port, share=False, inbrowser=True)
