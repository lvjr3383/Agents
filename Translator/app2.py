# app_multi.py — NLP-101 English → (Telugu | Spanish | Chinese) on CPU with visualization
import time
import pandas as pd
import plotly.graph_objects as go
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Optional romanization libs
from indic_transliteration import sanscript
try:
    from pypinyin import pinyin, Style
    HAS_PYPINYIN = True
except Exception:
    HAS_PYPINYIN = False

# --- Model setup (CPU) ---
MODEL_NAME = "facebook/nllb-200-distilled-600M"
# Source: English (Latin)
SRC_LANG_CODE = "eng_Latn"

# Target language menu and NLLB codes
LANG_MENU = {
    "Telugu (తెలుగు)": "tel_Telu",
    "Spanish (Español)": "spa_Latn",
    "Chinese (简体中文)": "zho_Hans",   # Simplified Mandarin
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG_CODE)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).eval()
DEVICE = torch.device("cpu")
model.to(DEVICE)

def utf8_table(text: str) -> pd.DataFrame:
    rows = []
    for ch in text:
        rows.append({
            "char": ch,
            "codepoint": f"U+{ord(ch):04X}",
            "utf8_bytes": " ".join([f"{b:02X}" for b in ch.encode('utf-8')]),
        })
    return pd.DataFrame(rows)

def romanize(text: str, lang_code: str) -> str:
    """Return a romanized form when appropriate; else a helpful note."""
    if lang_code == "tel_Telu":
        try:
            return sanscript.transliterate(text, sanscript.TELUGU, sanscript.ITRANS)
        except Exception:
            return "(Telugu romanization unavailable)"
    if lang_code == "zho_Hans":
        if not HAS_PYPINYIN:
            return "(Install 'pypinyin' to show Pinyin)"
        try:
            pieces = pinyin(text, style=Style.TONE3, strict=False)
            # Flatten list of lists and join with spaces
            return " ".join(token for lst in pieces for token in lst if token)
        except Exception:
            return "(Pinyin generation failed)"
    # spa_Latn is already Latin script
    return text

def plot_line(series, title, ylab, mode="greedy"):
    x = list(range(1, len(series) + 1))
    color = "#0074D9" if mode == "greedy" else "#FF851B"  # blue vs orange
    title_suffix = " (Greedy decoding)" if mode == "greedy" else " (Sampling decoding)"
    fig = go.Figure(
        data=go.Scatter(
            x=x, y=series, mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=6, color=color, line=dict(width=1, color="black"))
        )
    )
    fig.update_layout(
        title=title + title_suffix,
        xaxis_title="Decode step",
        yaxis_title=ylab,
        height=280,
        margin=dict(t=40, b=40, l=40, r=10),
        plot_bgcolor="#FFF7E6" if mode == "sampling" else "#E6F2FF"
    )
    return fig

def run_pipeline(english: str,
                 target_label: str,
                 do_sample: bool,
                 temperature: float,
                 top_p: float):

    if not english.strip():
        return ("", pd.DataFrame(), pd.DataFrame(), "", "", go.Figure(), go.Figure())

    # Resolve target language id
    tgt_code = LANG_MENU[target_label]
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_code)

    # --- Tokenize source (encoder input)
    enc_inputs = tokenizer(english, return_tensors="pt").to(DEVICE)
    src_tokens = tokenizer.convert_ids_to_tokens(enc_inputs.input_ids[0])
    src_ids = enc_inputs.input_ids[0].tolist()

    # --- Generate with per-step scores
    gen_kwargs = dict(
        forced_bos_token_id=forced_bos,
        max_new_tokens=80,
        output_scores=True,
        return_dict_in_generate=True
    )
    if do_sample:
        gen_kwargs.update(dict(do_sample=True, temperature=float(temperature), top_p=float(top_p)))
    else:
        gen_kwargs.update(dict(do_sample=False))

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc_inputs, **gen_kwargs)
    _latency = time.perf_counter() - t0

    # Sequence (decoder ids) and final text
    seq = out.sequences[0]
    translated = tokenizer.decode(seq, skip_special_tokens=True)
    roman = romanize(translated, tgt_code)

    # --- Per-step probabilities from generation scores
    scores = out.scores or []
    steps = min(len(scores), len(seq) - 1)

    step_rows = []
    chosen_probs = []
    entropies = []

    for t in range(steps):
        logits = scores[t][0]                       # (vocab,)
        probs = torch.softmax(logits, dim=-1)
        chosen_id = int(seq[t+1].item())
        chosen_prob = float(probs[chosen_id].item())

        entropy = float(-(probs * (probs.clamp_min(1e-12)).log()).sum().item())

        # top-k display
        topk_prob, topk_id = torch.topk(probs, k=5)
        topk_prob = topk_prob.tolist()
        topk_id = topk_id.tolist()
        topk_tok = tokenizer.convert_ids_to_tokens(topk_id)

        step_rows.append({
            "step": t+1,
            "chosen_token": tokenizer.convert_ids_to_tokens([chosen_id])[0],
            "chosen_prob": round(chosen_prob, 4),
            "top1": f"{topk_tok[0]} ({topk_prob[0]:.3f})",
            "top2": f"{topk_tok[1]} ({topk_prob[1]:.3f})",
            "top3": f"{topk_tok[2]} ({topk_prob[2]:.3f})",
            "top4": f"{topk_tok[3]} ({topk_prob[3]:.3f})",
            "top5": f"{topk_tok[4]} ({topk_prob[4]:.3f})",
        })

        chosen_probs.append(chosen_prob)
        entropies.append(entropy)

    # --- Middle panel markdown & bytes
    tokens_md = (
        f"**Input (English):** `{english}`\n\n"
        f"**English tokens:** `{', '.join(src_tokens)}`\n\n"
        f"**Token IDs:** `{src_ids}`\n\n"
        f"_`▁` denotes a word boundary in subword tokenization._"
    )
    bytes_df = utf8_table(english)
    topk_df = pd.DataFrame(step_rows)

    # --- Plots
    mode = "sampling" if do_sample else "greedy"
    prob_fig = plot_line(chosen_probs, "Chosen-Token Probability per Step", "Probability", mode=mode)
    ent_fig  = plot_line(entropies, "Entropy per Step (bits)", "Entropy", mode=mode)

    return (tokens_md, bytes_df, topk_df, translated, roman, prob_fig, ent_fig)

with gr.Blocks(title="NLP-101: English → Telugu/Spanish/Chinese") as demo:
    gr.Markdown("### NLP-101 — English → Telugu / Spanish / Chinese (CPU, visualization of inference steps)")

    with gr.Row():
        # Left: controls
        with gr.Column(scale=1):
            english = gr.Textbox(label="Input (English)", value="I want to learn Telugu", lines=3)
            target  = gr.Dropdown(choices=list(LANG_MENU.keys()), value="Telugu (తెలుగు)", label="Target language")
            do_sample = gr.Checkbox(label="Use sampling (instead of greedy)", value=False)
            temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p (nucleus)")
            run_btn = gr.Button("Translate & Visualize", variant="primary")

        # Middle: internals
        with gr.Column(scale=2):
            mid_md = gr.Markdown(label="Tokenization")
            utf8_df = gr.Dataframe(label="UTF-8 view (input bytes)")
            topk_df = gr.Dataframe(label="Decode Timeline — Top-k per step")

        # Right: outputs & plots
        with gr.Column(scale=2):
            out_text  = gr.Textbox(label="Translation", lines=2)
            out_roman = gr.Textbox(label="Romanization / Latinized", lines=1)
            prob_plot = gr.Plot(label="Confidence per step")
            ent_plot  = gr.Plot(label="Entropy per step")

    run_btn.click(
        fn=run_pipeline,
        inputs=[english, target, do_sample, temperature, top_p],
        outputs=[mid_md, utf8_df, topk_df, out_text, out_roman, prob_plot, ent_plot]
    )

if __name__ == "__main__":
    demo.launch(share=False)
