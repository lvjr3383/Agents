import collections
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

import spacy

# -------------------------
# spaCy model setup
# -------------------------

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If the model is not installed, we keep nlp=None and handle it gracefully in the UI.
    nlp = None


# -------------------------
# Shared helpers
# -------------------------

def placeholder_plot(title: str, msg: str, height: float = 3.0):
    """Return a simple matplotlib figure with centered message."""
    fig, ax = plt.subplots(figsize=(6, height))
    ax.text(
        0.5,
        0.5,
        msg,
        ha="center",
        va="center",
        fontsize=12,
        color="#6b7280",
        transform=ax.transAxes,
        wrap=True,
    )
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def require_nlp():
    """Return None if nlp is available, else an error message string."""
    if nlp is None:
        return (
            "spaCy model 'en_core_web_sm' is not installed.\n\n"
            "Please run the following command in your environment:\n\n"
            "    python -m spacy download en_core_web_sm\n\n"
            "Then restart this app."
        )
    return None


# -------------------------
# POS Tagging helpers
# -------------------------

# Dark-theme friendly colors: strong backgrounds, white text
POS_COLOR_MAP = {
    "NOUN": "#1d4ed8",  # blue
    "PROPN": "#1d4ed8",
    "VERB": "#16a34a",  # green
    "AUX": "#16a34a",
    "ADJ": "#f97316",   # orange
    "ADV": "#e11d48",   # red/pink
}


def color_text_pos(doc):
    """Return HTML highlighting tokens by coarse POS type."""
    if doc is None or len(doc) == 0:
        return "<p>No text.</p>"

    spans = []
    for token in doc:
        bg = POS_COLOR_MAP.get(token.pos_, "#4b5563")  # default gray
        spans.append(
            f"<span style='background-color:{bg}; color:#ffffff; "
            f"padding:2px 4px; margin:1px; border-radius:3px;'>"
            f"{token.text}</span>"
        )
    return "<div style='line-height:1.8'>" + " ".join(spans) + "</div>"


def pos_distribution_plot(doc):
    """Bar plot of POS tag counts."""
    if doc is None or len(doc) == 0:
        return placeholder_plot(
            "POS Tag Distribution",
            "Enter text and click Analyze to see POS tag counts.",
            height=3.0,
        )

    counter = collections.Counter(tok.pos_ for tok in doc)
    labels = list(counter.keys())
    counts = [counter[l] for l in labels]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    bars = ax.bar(x, counts)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title("POS Tag Distribution")

    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            str(c),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    return fig


def analyze_pos(text: str):
    """Main POS analysis function for the POS tab."""
    err = require_nlp()
    if err is not None:
        # nlp not available
        empty_table = []
        empty_plot = placeholder_plot(
            "POS Tag Distribution",
            "spaCy model missing. See the message on the left.",
            height=3.0,
        )
        return (
            f"### POS Tagging\n\n{err}",
            empty_table,
            empty_plot,
            "<p>No text to analyze.</p>",
        )

    text = text.strip()
    if not text:
        empty_table = []
        empty_plot = placeholder_plot(
            "POS Tag Distribution",
            "Enter text and click Analyze to see POS tag counts.",
            height=3.0,
        )
        return (
            "### POS Tagging\n\nPlease enter some text.",
            empty_table,
            empty_plot,
            "<p>No text to analyze.</p>",
        )

    doc = nlp(text)

    # POS counts for descriptive summary
    counter = collections.Counter(tok.pos_ for tok in doc)
    top_parts = ", ".join(
        f"{pos} ({count})" for pos, count in counter.most_common(4)
    )

    # Summary
    summary_md = (
        "### POS Tagging\n"
        f"Tokens: **{len(doc)}**  ·  "
        f"Unique POS tags: **{len(counter)}**\n\n"
        f"Most frequent POS tags: {top_parts}"
    )

    # Token table
    token_rows = []
    for i, tok in enumerate(doc):
        token_rows.append(
            [
                i,
                tok.text,
                tok.lemma_,
                tok.pos_,
                tok.tag_,
                tok.dep_,
                tok.head.text,
            ]
        )

    # Distribution plot
    pos_plot = pos_distribution_plot(doc)

    # Highlighted text
    highlight_html = color_text_pos(doc)

    return summary_md, token_rows, pos_plot, highlight_html


# -------------------------
# NER helpers
# -------------------------

# Dark-theme friendly entity colors
NER_COLOR_MAP = {
    "PERSON": "#b91c1c",   # strong red
    "ORG": "#1d4ed8",      # blue
    "GPE": "#15803d",      # green
    "LOC": "#7c3aed",      # purple
    "DATE": "#9d174d",     # pink
}


def color_text_ner(doc):
    """HTML highlighting entity spans."""
    if doc is None or len(doc) == 0:
        return "<p>No text.</p>"

    ent_spans = []
    for ent in doc.ents:
        ent_spans.append((ent.start_char, ent.end_char, ent.label_))

    text = doc.text
    if not ent_spans:
        safe = text.replace("<", "&lt;").replace(">", "&gt;")
        return f"<p>{safe}</p>"

    ent_spans.sort()  # by start_char
    html_parts = []
    cursor = 0

    for start, end, label in ent_spans:
        # plain text before entity
        if cursor < start:
            plain = text[cursor:start]
            html_parts.append(
                plain.replace("<", "&lt;").replace(">", "&gt;")
            )

        chunk = text[start:end]
        bg = NER_COLOR_MAP.get(label, "#4b5563")
        html_parts.append(
            f"<span style='background-color:{bg}; color:#ffffff; "
            f"padding:2px 4px; margin:1px; border-radius:3px;'>"
            f"{chunk} <span style='font-size:0.75rem; opacity:0.9;'>[{label}]</span>"
            f"</span>"
        )
        cursor = end

    # trailing text
    if cursor < len(text):
        tail = text[cursor:]
        html_parts.append(
            tail.replace("<", "&lt;").replace(">", "&gt;")
        )

    return "<div style='line-height:1.8'>" + "".join(html_parts) + "</div>"


def entity_distribution_plot(doc):
    """Bar plot of entity label counts."""
    if doc is None or len(doc.ents) == 0:
        return placeholder_plot(
            "Entity Type Distribution",
            "Enter text with entities (people, places, organizations, etc.) "
            "and click Analyze to see entity counts.",
            height=3.0,
        )

    counter = collections.Counter(ent.label_ for ent in doc.ents)
    labels = list(counter.keys())
    counts = [counter[l] for l in labels]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    bars = ax.bar(x, counts)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title("Entity Type Distribution")

    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            str(c),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    return fig


def analyze_ner(text: str):
    """Main NER analysis function for the NER tab."""
    err = require_nlp()
    if err is not None:
        empty_token_table = []
        empty_ent_table = []
        empty_plot = placeholder_plot(
            "Entity Type Distribution",
            "spaCy model missing. See the message on the left.",
            height=3.0,
        )
        return (
            f"### Named Entity Recognition\n\n{err}",
            empty_token_table,
            "<p>No text to analyze.</p>",
            empty_ent_table,
            empty_plot,
        )

    text = text.strip()
    if not text:
        empty_token_table = []
        empty_ent_table = []
        empty_plot = placeholder_plot(
            "Entity Type Distribution",
            "Enter text and click Analyze to see entity counts.",
            height=3.0,
        )
        return (
            "### Named Entity Recognition\n\nPlease enter some text.",
            empty_token_table,
            "<p>No text to analyze.</p>",
            empty_ent_table,
            empty_plot,
        )

    doc = nlp(text)

    # Entity counts for descriptive summary
    counter = collections.Counter(ent.label_ for ent in doc.ents)
    if counter:
        type_summary = ", ".join(
            f"{label} ({count})" for label, count in counter.most_common()
        )
    else:
        type_summary = "none detected"

    summary_md = (
        "### Named Entity Recognition\n"
        f"Tokens: **{len(doc)}**  ·  "
        f"Entities detected: **{len(doc.ents)}**\n\n"
        f"Entity types: {type_summary}"
    )

    # Token table with BIO-ish tags
    token_rows = []
    for i, tok in enumerate(doc):
        ent_iob = tok.ent_iob_
        ent_label = tok.ent_type_ if tok.ent_type_ else "O"
        token_rows.append(
            [
                i,
                tok.text,
                ent_iob,
                ent_label,
            ]
        )

    # Entity table
    ent_rows = []
    for ent in doc.ents:
        ent_rows.append(
            [
                ent.text,
                ent.label_,
                ent.start_char,
                ent.end_char,
            ]
        )

    # Highlighted text
    highlight_html = color_text_ner(doc)

    # Distribution plot
    ent_plot = entity_distribution_plot(doc)

    return summary_md, token_rows, highlight_html, ent_rows, ent_plot


# -------------------------
# Gradio UI
# -------------------------

with gr.Blocks() as demo:
    gr.Markdown("## NLP-101: Multi-Class Token Lab (POS + NER)")

    with gr.Tabs():
        # ======================
        # POS TAB
        # ======================
        with gr.Tab("POS Tagging"):
            with gr.Row():
                # LEFT COLUMN
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Input Text (POS)")
                    pos_example_dropdown = gr.Dropdown(
                        choices=[
                            "Simple sentence",
                            "Question",
                            "Dialogue",
                        ],
                        label="Examples (optional)",
                        value=None,
                    )
                    pos_text = gr.Textbox(
                        label="Enter a sentence or short paragraph",
                        lines=6,
                        placeholder="Example:\nThe quick brown fox jumps over the lazy dog.",
                    )
                    pos_info = gr.Markdown("")
                    pos_btn = gr.Button("Analyze POS", variant="primary")

                # MIDDLE COLUMN
                with gr.Column(scale=1):
                    gr.Markdown("### 2. Processing (POS Black Box)")
                    with gr.Group():
                        gr.Markdown("#### Stage 1 · Tokenization")
                        gr.Markdown(
                            "The text is split into tokens. Each token has its own lemma and tag."
                        )
                        pos_token_table = gr.Dataframe(
                            headers=[
                                "Index",
                                "Token",
                                "Lemma",
                                "POS",
                                "Tag",
                                "Dep",
                                "Head",
                            ],
                            datatype=[
                                "number",
                                "str",
                                "str",
                                "str",
                                "str",
                                "str",
                                "str",
                            ],
                            interactive=False,
                            label="Tokens",
                        )

                    with gr.Group():
                        gr.Markdown("#### Stage 2 · POS Tags")
                        gr.Markdown(
                            "We count how many nouns, verbs, adjectives, etc. appear in your text."
                        )
                        pos_plot = gr.Plot(label="POS Tag Distribution")

                # RIGHT COLUMN
                with gr.Column(scale=1):
                    gr.Markdown("### 3. POS Output")
                    pos_summary = gr.Markdown(label="POS Summary")
                    gr.Markdown("#### POS-Colored Text")
                    gr.Markdown(
                        "Tokens are colored based on their coarse POS tag "
                        "(nouns, verbs, adjectives, etc.)."
                    )
                    pos_highlight = gr.HTML(label="POS Highlighted Text")

        # ======================
        # NER TAB
        # ======================
        with gr.Tab("Named Entity Recognition"):
            with gr.Row():
                # LEFT COLUMN
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Input Text (NER)")
                    ner_example_dropdown = gr.Dropdown(
                        choices=[
                            "News-style sentence",
                            "People and organizations",
                            "Locations and dates",
                        ],
                        label="Examples (optional)",
                        value=None,
                    )
                    ner_text = gr.Textbox(
                        label="Enter a sentence or short paragraph",
                        lines=6,
                        placeholder=(
                            "Example:\n"
                            "Barack Obama was born in Hawaii and served as the 44th President of the United States."
                        ),
                    )
                    ner_info = gr.Markdown("")
                    ner_btn = gr.Button("Analyze NER", variant="primary")

                # MIDDLE COLUMN
                with gr.Column(scale=1):
                    gr.Markdown("### 2. Processing (NER Black Box)")
                    with gr.Group():
                        gr.Markdown("#### Stage 1 · Token-Level Tags")
                        gr.Markdown(
                            "Each token receives a BIO tag (B/I/O) and an entity label if applicable."
                        )
                        ner_token_table = gr.Dataframe(
                            headers=["Index", "Token", "IOB", "Entity Label"],
                            datatype=["number", "str", "str", "str"],
                            interactive=False,
                            label="Tokens with NER tags",
                        )

                    with gr.Group():
                        gr.Markdown("#### Stage 2 · Entity Spans")
                        gr.Markdown(
                            "Detected entities are grouped into spans with their label and character offsets."
                        )
                        ner_ent_table = gr.Dataframe(
                            headers=["Text", "Label", "Start Char", "End Char"],
                            datatype=["str", "str", "number", "number"],
                            interactive=False,
                            label="Entities",
                        )

                    with gr.Group():
                        gr.Markdown("#### Stage 3 · Entity Types")
                        gr.Markdown(
                            "We count how many entities of each type appear in your text."
                        )
                        ner_plot = gr.Plot(label="Entity Type Distribution")

                # RIGHT COLUMN
                with gr.Column(scale=1):
                    gr.Markdown("### 3. NER Output")
                    ner_summary = gr.Markdown(label="NER Summary")
                    gr.Markdown("#### Entity-Highlighted Text")
                    gr.Markdown(
                        "Entity spans (PERSON, ORG, GPE, etc.) are highlighted in the text with labels."
                    )
                    ner_highlight = gr.HTML(label="Entity-Highlighted Text")

    # -------------------------
    # Example callbacks
    # -------------------------

    def load_pos_example(kind):
        if kind == "Simple sentence":
            return "The quick brown fox jumps over the lazy dog.", ""
        elif kind == "Question":
            return "When does the next train to New York leave from this station?", ""
        elif kind == "Dialogue":
            return (
                '"I will call you tomorrow," she said, as she walked out of the office.',
                "",
            )
        return gr.update(), ""

    pos_example_dropdown.change(
        fn=load_pos_example,
        inputs=[pos_example_dropdown],
        outputs=[pos_text, pos_info],
    )

    def load_ner_example(kind):
        if kind == "News-style sentence":
            return (
                "Apple announced its latest iPhone model in California on Tuesday.",
                "",
            )
        elif kind == "People and organizations":
            return (
                "Satya Nadella is the CEO of Microsoft, which is headquartered in Redmond.",
                "",
            )
        elif kind == "Locations and dates":
            return (
                "The conference will be held in London from March 3rd to March 5th, 2025.",
                "",
            )
        return gr.update(), ""

    ner_example_dropdown.change(
        fn=load_ner_example,
        inputs=[ner_example_dropdown],
        outputs=[ner_text, ner_info],
    )

    # -------------------------
    # Main button callbacks
    # -------------------------

    pos_btn.click(
        fn=analyze_pos,
        inputs=[pos_text],
        outputs=[pos_summary, pos_token_table, pos_plot, pos_highlight],
    )

    ner_btn.click(
        fn=analyze_ner,
        inputs=[ner_text],
        outputs=[ner_summary, ner_token_table, ner_highlight, ner_ent_table, ner_plot],
    )

if __name__ == "__main__":
    demo.launch()
