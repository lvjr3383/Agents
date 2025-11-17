import string

import numpy as np
import torch
import gradio as gr
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.decomposition import PCA


# -------------------------
# Model Setup
# -------------------------

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

id2label = model.config.id2label  # e.g. {0: 'negative', 1: 'neutral', 2: 'positive'}

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "by", "for", "and", "or", "to", "from", "with", "that", "this", "it",
    "is", "was", "are", "were", "be", "been", "am", "i", "you", "he", "she", "they", "we"
}


# -------------------------
# Core helpers
# -------------------------

def softmax(logits):
    exps = np.exp(logits - np.max(logits))
    return exps / exps.sum()


def encode_text(text: str):
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        return_attention_mask=True
    ).to(device)


def get_sentence_embedding(text: str):
    """Mean-pool the last hidden layer over non-pad tokens to get a sentence vector."""
    inputs = encode_text(text)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0]  # [seq_len, dim]
        mask = inputs["attention_mask"][0].unsqueeze(-1)  # [seq_len, 1]
        masked_hidden = hidden * mask
        summed = masked_hidden.sum(dim=0)
        counts = mask.sum(dim=0).clamp(min=1.0)
        sent_vec = (summed / counts).cpu().numpy()
    return sent_vec


# Tiny “teaching set” to build sentiment centroids
SENTENCE_TEMPLATES = {
    "negative": [
        "I hated this movie. It was boring and a complete waste of time.",
        "The food was terrible and the service was very slow.",
    ],
    "neutral": [
        "The movie was okay, nothing special but not bad either.",
        "The restaurant was fine. Some things were good and some were average.",
    ],
    "positive": [
        "I absolutely loved this movie. The acting and story were fantastic.",
        "The food was amazing and the staff were incredibly friendly.",
    ],
}

# Precompute centroids in embedding space
SENTIMENT_CENTROIDS = {}
for label, texts in SENTENCE_TEMPLATES.items():
    vecs = [get_sentence_embedding(t) for t in texts]
    SENTIMENT_CENTROIDS[label] = np.mean(np.stack(vecs, axis=0), axis=0)


def get_embeddings_and_probs(text):
    """
    Run the model once to get:
    - tokens
    - last-layer hidden states
    - attention mask
    - class probabilities
    - predicted label
    """
    inputs = encode_text(text)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1][0].cpu().numpy()  # [seq_len, hidden_dim]
        logits = outputs.logits[0].cpu().numpy()

    probs = softmax(logits)
    pred_id = int(np.argmax(probs))
    pred_label = id2label[pred_id]

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu().tolist())
    attention_mask = inputs["attention_mask"][0].cpu().numpy()

    return tokens, hidden_states, attention_mask, probs, pred_id, pred_label


def build_token_table(tokens):
    return [[i, tok] for i, tok in enumerate(tokens)]


def classify_token(tok: str) -> str:
    # For coloring: content vs stopword vs punctuation vs special
    if tok in tokenizer.all_special_tokens:
        return "special"
    stripped = tok.lstrip("Ġ")
    if stripped in string.punctuation:
        return "punctuation"
    if stripped.lower() in STOPWORDS:
        return "stopword"
    return "content"


def display_token(tok: str) -> str:
    # Clean up leading Ġ markers for readability
    return tok.lstrip("Ġ")


# -------------------------
# Plot helpers (matplotlib)
# -------------------------

def placeholder_plot(title: str, msg: str, height: float = 3.0):
    """Return a simple matplotlib figure with centered message."""
    fig, ax = plt.subplots(figsize=(6, height))
    ax.text(
        0.5, 0.5, msg,
        ha="center", va="center",
        fontsize=12, color="#6b7280",
        transform=ax.transAxes,
        wrap=True,
    )
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_embeddings_pca(hidden_states, tokens):
    """
    PCA projection of token embeddings to 2D.
    Filters out obvious special tokens so the plot is more meaningful.
    """
    clean_indices = []
    for i, tok in enumerate(tokens):
        if tok in tokenizer.all_special_tokens:
            continue
        if tok.startswith("<") and tok.endswith(">"):
            continue
        clean_indices.append(i)

    if len(clean_indices) < 2:
        return placeholder_plot(
            "Token Embeddings (PCA 2D)",
            "Add a longer review or more varied text to see token positions."
        )

    hs = hidden_states[clean_indices, :]
    toks = [tokens[i] for i in clean_indices]

    pca = PCA(n_components=2)
    coords = pca.fit_transform(hs)
    explained = pca.explained_variance_ratio_ * 100

    kinds = [classify_token(t) for t in toks]
    colors = {
        "special": "#6b7280",
        "punctuation": "#22c55e",
        "stopword": "#f59e0b",
        "content": "#3b82f6",
    }
    cvals = [colors.get(k, "#3b82f6") for k in kinds]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(coords[:, 0], coords[:, 1], c=cvals, s=50, alpha=0.9, edgecolors="#ffffff", linewidths=0.8)

    for (x, y, tok) in zip(coords[:, 0], coords[:, 1], toks):
        ax.text(x, y, display_token(tok), fontsize=9, ha="center", va="bottom", color="#0f172a")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Token Embeddings (PCA 2D) · PC1 {explained[0]:.1f}% | PC2 {explained[1]:.1f}%")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Legend explaining roles
    legend_elements = []
    for k, col in colors.items():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          label=k, markerfacecolor=col, markersize=8))
    ax.legend(handles=legend_elements, title="Token Type", loc="best")

    fig.tight_layout()
    return fig


def plot_sentence_position(sentence_vec):
    """Position the user review vs sentiment centroids in 2D."""
    if sentence_vec is None or sentence_vec.ndim != 1:
        return placeholder_plot(
            "Sentence Position in Sentiment Space",
            "Run analysis to position your review among sentiment centroids.",
            height=3.0,
        )

    labels = []
    vecs = []

    labels.append("your_review")
    vecs.append(sentence_vec)

    for lab, v in SENTIMENT_CENTROIDS.items():
        labels.append(lab)
        vecs.append(v)

    X = np.stack(vecs, axis=0)

    if X.shape[0] < 2:
        return placeholder_plot(
            "Sentence Position in Sentiment Space",
            "Not enough reference points to show a position map.",
            height=3.0,
        )

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    colors = {
        "your_review": "#0ea5e9",
        "negative": "#ef4444",
        "neutral": "#f59e0b",
        "positive": "#22c55e",
    }
    for (x, y, lab) in zip(coords[:, 0], coords[:, 1], labels):
        ax.scatter(x, y, c=colors.get(lab, "#6b7280"), s=70, edgecolors="#ffffff", linewidths=0.9)
        ax.text(x, y, lab, fontsize=9, ha="center", va="bottom")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Sentence Position in Sentiment Space")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def build_prob_bar(prob_rows):
    """Mini bar chart for class probabilities."""
    if not prob_rows:
        return placeholder_plot(
            "Mini View: Class Probabilities",
            "Run analysis to populate class probabilities.",
            height=2.2,
        )

    labels = [row[0] for row in prob_rows]
    probs = [row[1] for row in prob_rows]

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    x = np.arange(len(labels))
    colors = ["#ef4444", "#f59e0b", "#22c55e"][: len(labels)]

    bars = ax.bar(x, probs, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Mini View: Class Probabilities")

    for bar, p in zip(bars, probs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{p:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    return fig


def build_importance_bar(words, importances, top_n=8):
    """Horizontal bar chart of top influential words."""
    if not words:
        return placeholder_plot(
            "Top Words by Influence",
            "Run analysis to see which words influenced the prediction.",
            height=2.6,
        )

    pairs = list(zip(words, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:top_n]

    labels = [w for w, _ in top]
    scores = [s for _, s in top]

    y = np.arange(len(labels))[::-1]

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.barh(y, scores, color="#0ea5e9")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Relative Influence")
    ax.set_title("Top Words by Influence (Mask-Based)")

    for val, yy in zip(scores, y):
        ax.text(val + 0.01, yy, f"{val:.2f}", va="center", fontsize=9)

    fig.tight_layout()
    return fig


# -------------------------
# Word importance
# -------------------------

def compute_word_importance(text, pred_id, base_prob):
    """
    Leave-one-out importance using [MASK].
    Returns: words, normalized_importances, raw_deltas.
    """
    words = text.split()
    if not words:
        return [], [], []

    importances = []
    deltas = []

    for i in range(len(words)):
        masked_words = words.copy()
        masked_words[i] = "[MASK]"
        masked_text = " ".join(masked_words)

        inputs = encode_text(masked_text)
        with torch.no_grad():
            logits = model(**inputs).logits[0].cpu().numpy()
        probs = softmax(logits)
        prob_pred = probs[pred_id]
        delta = base_prob - prob_pred  # how much confidence drops when this word is removed
        importances.append(max(delta, 0.0))
        deltas.append(delta)

    max_imp = max(importances) if importances else 0.0
    if max_imp > 0:
        importances = [imp / max_imp for imp in importances]
    else:
        importances = [0.0 for _ in importances]

    return words, importances, deltas


def color_text(words, importances):
    """HTML with red highlight proportional to importance."""
    if not words:
        return "<p>No text.</p>"

    spans = []
    for w, imp in zip(words, importances):
        intensity = int(255 - imp * 120)  # 255 -> ~135 for stronger red
        color = f"rgb(255,{intensity},{intensity})"
        spans.append(
            f"<span style='background-color:{color}; padding:2px; margin:1px; border-radius:3px;'>{w}</span>"
        )
    return "<div style='line-height:1.8'>" + " ".join(spans) + "</div>"


# -------------------------
# Main explanation function
# -------------------------

def explain_sentiment(review_text):
    """
    Run the model and return everything for the three-column UI.
    """
    if not review_text.strip():
        empty_probs = [["negative", 0.0], ["neutral", 0.0], ["positive", 0.0]]
        empty_pca = placeholder_plot(
            "Token Embeddings (PCA 2D)",
            "Add text and click Analyze to see token positions."
        )
        empty_bar = build_prob_bar([])
        empty_sent = placeholder_plot(
            "Sentence Position in Sentiment Space",
            "Run analysis to position your review among sentiment centroids.",
        )
        empty_imp = build_importance_bar([], [])
        return (
            "Please enter a review.",
            empty_probs,
            "<p>No text to analyze.</p>",
            [],          # token table
            empty_pca,   # token embeddings
            empty_bar,   # mini prob bar
            empty_sent,  # sentence position
            empty_imp,   # top words bar
        )

    tokens, hidden_states, attention_mask, probs, pred_id, pred_label = get_embeddings_and_probs(review_text)

    base_prob = float(probs[pred_id])
    summary_text = f"### Predicted Sentiment\n**{pred_label}** (confidence: `{base_prob:.3f}`)"
    prob_rows = [[id2label[i], float(p)] for i, p in enumerate(probs)]

    words, importances, deltas = compute_word_importance(review_text, pred_id, base_prob)
    highlighted_html = color_text(words, importances)
    prob_bar_fig = build_prob_bar(prob_rows)

    token_rows = build_token_table(tokens)
    emb_fig = plot_embeddings_pca(hidden_states, tokens)

    sent_vec = get_sentence_embedding(review_text)
    sent_pos_fig = plot_sentence_position(sent_vec)
    importance_bar_fig = build_importance_bar(words, importances)

    return (
        summary_text,
        prob_rows,
        highlighted_html,
        token_rows,
        emb_fig,
        prob_bar_fig,
        sent_pos_fig,
        importance_bar_fig,
    )


# -------------------------
# A/B comparison helpers
# -------------------------

def basic_sentiment_core(text: str):
    if not text.strip():
        return "No text.", [0.0, 0.0, 0.0]
    _, _, _, probs, pred_id, pred_label = get_embeddings_and_probs(text)
    label = pred_label
    probs_list = [float(p) for p in probs]
    return label, probs_list


def compare_reviews(text_a: str, text_b: str):
    if not text_a.strip() and not text_b.strip():
        fig = placeholder_plot(
            "Comparison",
            "Enter at least one review to compare.",
            height=2.5,
        )
        return "No input.", "No input.", fig

    label_a, probs_a = basic_sentiment_core(text_a or "")
    label_b, probs_b = basic_sentiment_core(text_b or "")

    msg_a = f"**Review A** → {label_a}"
    msg_b = f"**Review B** → {label_b}"

    labels = [id2label[i] for i in range(len(probs_a))]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6, 3))
    width = 0.35
    bars1 = ax.bar(x - width / 2, probs_a, width, label="Review A", color="#3b82f6")
    bars2 = ax.bar(x + width / 2, probs_b, width, label="Review B", color="#f97316")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Sentiment Distribution – A vs B")
    ax.legend()

    for bar in list(bars1) + list(bars2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    return msg_a, msg_b, fig


# -------------------------
# Example gallery
# -------------------------

EXAMPLES = {
    "Very positive movie": (
        "I absolutely loved this movie. The acting, direction, and music were all fantastic!",
        "positive",
    ),
    "Very negative restaurant": (
        "This was the worst dining experience I've had in years. The food was cold and the staff was rude.",
        "negative",
    ),
    "Neutral / mixed review": (
        "The movie had a strong start but lost steam in the second half. Overall it was just okay.",
        "neutral",
    ),
    "Sarcastic line": (
        "Great, another three-hour movie with zero character development. Exactly what I needed.",
        "negative (model may get confused)",
    ),
}


# -------------------------
# Gradio UI: Left - Middle - Right
# -------------------------

with gr.Blocks() as demo:
    gr.Markdown("## NLP-101: Sentiment Classification Visualizer")

    with gr.Row():
        # LEFT COLUMN
        with gr.Column(scale=1):
            gr.Markdown("### 1. Input Review")

            sample_dropdown = gr.Dropdown(
                choices=list(EXAMPLES.keys()),
                label="Example Gallery (optional)",
                value=None,
                interactive=True,
            )

            review_in = gr.Textbox(
                label="Enter a short movie/restaurant review",
                lines=7,
                placeholder=(
                    "Example:\n"
                    "The movie was amazing with mind blowing action episodes and visual effects."
                )
            )
            expected_label_md = gr.Markdown("")

            example_btn = gr.Button("Load Simple Example")
            run_btn = gr.Button("Analyze Sentiment", variant="primary")

        # MIDDLE COLUMN
        with gr.Column(scale=1):
            gr.Markdown("### 2. Processing (Black Box)")
            gr.Markdown(
                "Follow the flow: tokenize → embed → classify. Each card shows what changes and why it matters."
            )

            # Stage 1 – Tokenization
            with gr.Group():
                gr.Markdown("#### Stage 1 · Tokenization  <small style='color:#6b7280'>1/3</small>")
                gr.Markdown(
                    "Text is split into the subword tokens the model actually sees. "
                    "If it isn't tokenized, it isn't learned."
                )
                gr.Markdown(
                    "<small>Tip: watch for special tokens like `<s>` and `</s>`—they frame the sequence.</small>"
                )
                token_table = gr.Dataframe(
                    headers=["Index", "Token"],
                    label="Tokens",
                    datatype=["number", "str"],
                    interactive=False
                )

            # Stage 2 – Embeddings
            with gr.Group():
                gr.Markdown("#### Stage 2 · Embeddings  <small style='color:#6b7280'>2/3</small>")
                gr.Markdown(
                    "Tokens become vectors. We project them to 2D with PCA to show their relative positions."
                )
                gr.Markdown(
                    "<small>What to look for: clusters indicate similar meaning; "
                    "outliers are special tokens or typos.</small>"
                )
                emb_plot = gr.Plot(label="Token Embeddings (PCA Projection)")
                sentence_plot = gr.Plot(label="Sentence Position in Sentiment Space")

            # Stage 3 – Classifier
            with gr.Group():
                gr.Markdown("#### Stage 3 · Classifier  <small style='color:#6b7280'>3/3</small>")
                gr.Markdown(
                    "A small head reads the sentence representation and produces sentiment scores."
                )
                gr.HTML(
                    "<div style='display:flex; gap:6px; flex-wrap:wrap;'>"
                    "<span style='padding:6px 10px; background:#e5e7eb; border-radius:999px;'>Embedding</span>"
                    "<span style='padding:6px 10px; background:#e5e7eb; border-radius:999px;'>Linear</span>"
                    "<span style='padding:6px 10px; background:#e5e7eb; border-radius:999px;'>Softmax</span>"
                    "<span style='padding:6px 10px; background:#dbeafe; border-radius:999px;'>Probabilities</span>"
                    "</div>"
                )
                prob_mini_plot = gr.Plot(label="Mini Probabilities")
                gr.Markdown(
                    "<small>Bias lives here: the head turns embeddings into the probabilities shown on the right.</small>"
                )

        # RIGHT COLUMN
        with gr.Column(scale=1):
            gr.Markdown("### 3. Sentiment Output")
            summary_out = gr.Markdown(label="Prediction")
            prob_table = gr.Dataframe(
                headers=["Label", "Probability"],
                label="Class Probabilities",
                datatype=["str", "number"],
                interactive=False
            )
            gr.Markdown("#### Word Importance")
            gr.Markdown(
                "Words in the review are highlighted by how much they influence the predicted sentiment "
                "(darker = more influence)."
            )
            highlighted_out = gr.HTML(label="Word Importance (Why this prediction?)")
            importance_plot = gr.Plot(label="Top Words by Influence")

    # A/B comparison accordion
    with gr.Accordion("Extra: Compare Two Reviews (A/B Mode)", open=False):
        with gr.Row():
            with gr.Column():
                review_a = gr.Textbox(label="Review A", lines=4)
            with gr.Column():
                review_b = gr.Textbox(label="Review B", lines=4)
        compare_btn = gr.Button("Compare A vs B")
        compare_a_md = gr.Markdown()
        compare_b_md = gr.Markdown()
        compare_fig = gr.Plot()

    # Example handlers
    def load_example():
        return (
            "The movie was amazing with mind blowing action episodes and visual effects, "
            "I absolutely loved it from start to finish."
        )

    def load_from_gallery(example_name):
        if example_name is None:
            return gr.update(), ""
        text, expected = EXAMPLES[example_name]
        return text, f"**Expected label (for teaching):** {expected}"

    sample_dropdown.change(
        fn=load_from_gallery,
        inputs=[sample_dropdown],
        outputs=[review_in, expected_label_md],
    )

    example_btn.click(fn=load_example, inputs=[], outputs=[review_in])

    # Main analysis
    run_btn.click(
        fn=explain_sentiment,
        inputs=[review_in],
        outputs=[
            summary_out,
            prob_table,
            highlighted_out,
            token_table,
            emb_plot,
            prob_mini_plot,
            sentence_plot,
            importance_plot,
        ],
    )

    # A/B comparison callback
    compare_btn.click(
        fn=compare_reviews,
        inputs=[review_a, review_b],
        outputs=[compare_a_md, compare_b_md, compare_fig],
    )


if __name__ == "__main__":
    demo.launch()
