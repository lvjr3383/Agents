# app3.py — NLP-101 Eng→Telugu (CS229-aligned, Mac-stable v4)
# Reliable visuals only: Unicode table, Token length bars, UTF-8 bytes per char,
# Decode top-k timeline, Chosen-prob over time, Entropy per step, Telugu + Roman, Latency.

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr
import torch, numpy as np, time, re, traceback, unicodedata
import plotly.graph_objects as go
import pandas as pd
from indic_transliteration import sanscript

MODEL = "facebook/nllb-200-distilled-600M"

# Load once (force eager just in case, though we don't plot attentions now)
tokenizer = AutoTokenizer.from_pretrained(MODEL, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, attn_implementation="eager").eval()

def _get_telugu_bos_id(tok):
    try:
        return tok.lang_code_to_id["tel_Telu"]
    except Exception:
        bos = tok.convert_tokens_to_ids("tel_Telu")
        if bos is None:
            raise ValueError("Could not resolve BOS id for 'tel_Telu'.")
        return bos

TEL_BOS_ID = _get_telugu_bos_id(tokenizer)

# ---------- helpers ----------

def unicode_table_md(s: str) -> str:
    rows = ["| char | codepoint | name |", "|---:|:---|:---|"]
    for ch in s:
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = "UNNAMED"
        rows.append(f"| `{ch}` | `U+{ord(ch):04X}` | {name} |")
    return "\n".join(rows)

def plot_token_lengths(tokens):
    # Visualize SentencePiece split: length of token (without the word-boundary marker ▁)
    labels = [t for t in tokens]
    lengths = [len(t.replace("▁", "")) for t in tokens]
    fig = go.Figure(data=go.Bar(x=list(range(len(labels))), y=lengths, text=labels))
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        title="Subword Token Lengths (SentencePiece)",
        xaxis_title="Token index", yaxis_title="Length (chars, '▁' removed)",
        height=360, margin=dict(t=40, b=40, l=40, r=10)
    )
    return fig

def plot_utf8_bytes_per_char(s: str):
    idx = list(range(len(s)))
    bcounts = [len(ch.encode("utf-8")) for ch in s] if s else []
    labels = [ch if ch != " " else "␣" for ch in s]
    fig = go.Figure(data=go.Bar(x=idx, y=bcounts, text=labels))
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        title="UTF-8 Bytes per Character (Input)",
        xaxis_title="Character index", yaxis_title="Bytes",
        height=360, margin=dict(t=40, b=40, l=40, r=10)
    )
    return fig

def softmax_np(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-9)

def decode_with_scores(enc_inputs, do_sample, temperature, top_p, max_new_tokens=60):
    """
    Generation (greedy or sampling);
    returns: telugu_text, topk_df, avg_conf, chosen_prob_series, entropy_series
    """
    with torch.no_grad():
        gen = model.generate(
            **enc_inputs,
            forced_bos_token_id=TEL_BOS_ID,
            max_new_tokens=max_new_tokens,
            do_sample=bool(do_sample),
            temperature=float(temperature) if do_sample else None,
            top_p=float(top_p) if do_sample else None,
            num_beams=1,  # align scores with sampling
            output_scores=True,
            return_dict_in_generate=True
        )

    seq = gen.sequences[0].tolist()
    telugu = tokenizer.decode(gen.sequences[0], skip_special_tokens=True)
    telugu = re.sub(r"\s+([.,!?;:])", r"\1", telugu)

    k = 5
    rows, chosen_probs, entropies = [], [], []
    for t, logits in enumerate(gen.scores):
        # probs for next token
        p = softmax_np(logits[0].detach().cpu().numpy())
        # entropy (nats) -> convert to bits for intuition
        H = float(-(p * (np.log(p + 1e-12))).sum() / np.log(2))
        entropies.append(H)

        top_ids = np.argpartition(p, -k)[-k:]
        top_ids = top_ids[np.argsort(-p[top_ids])]
        top_tokens = tokenizer.convert_ids_to_tokens(top_ids.tolist())
        top_vals = [float(p[i]) for i in top_ids]

        chosen_id = seq[min(t + 1, len(seq) - 1)]
        chosen_prob = float(p[chosen_id]) if chosen_id < p.shape[0] else 0.0
        chosen_probs.append(chosen_prob)

        rows.append([
            t + 1,
            top_tokens[0], round(top_vals[0], 4),
            top_tokens[1] if len(top_tokens) > 1 else "", round(top_vals[1], 4) if len(top_vals) > 1 else 0.0,
            top_tokens[2] if len(top_tokens) > 2 else "", round(top_vals[2], 4) if len(top_vals) > 2 else 0.0,
            top_tokens[3] if len(top_tokens) > 3 else "", round(top_vals[3], 4) if len(top_vals) > 3 else 0.0,
            top_tokens[4] if len(top_tokens) > 4 else "", round(top_vals[4], 4) if len(top_vals) > 4 else 0.0,
            round(chosen_prob, 4)
        ])

    columns = [
        "step",
        "top1_token", "top1_prob",
        "top2_token", "top2_prob",
        "top3_token", "top3_prob",
        "top4_token", "top4_prob",
        "top5_token", "top5_prob",
        "chosen_prob"
    ]
    topk_df = pd.DataFrame(rows, columns=columns)
    avg_conf = float(np.mean(chosen_probs)) if chosen_probs else 0.0

    return telugu, topk_df, avg_conf, chosen_probs, entropies

# --- Replace existing plot_line function with this ---
def plot_line(series, title, ylab, mode="greedy"):
    """Plot chosen-token probability or entropy per step, with visual cue for decoding mode."""
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

def plot_latency(lat):
    stages = list(lat.keys())
    ms = [lat[s] for s in stages]
    fig = go.Figure(data=go.Bar(x=stages, y=ms))
    fig.update_layout(
        title="Latency Breakdown (ms)",
        xaxis_title="Stage", yaxis_title="Milliseconds",
        height=280, margin=dict(t=40, b=40, l=40, r=10)
    )
    return fig

# ---------- Gradio callback ----------

def run_pipeline(english, do_sample, temperature, top_p):
    try:
        t0 = time.perf_counter()

        # CS229-1: Tokenize + Unicode
        uni_md = unicode_table_md(english)
        t1 = time.perf_counter()
        enc = tokenizer(english, return_tensors="pt")
        src_ids = enc["input_ids"][0].tolist()
        src_tokens = tokenizer.convert_ids_to_tokens(src_ids)
        t2 = time.perf_counter()

        # Middle visuals (always reliable)
        token_len_fig = plot_token_lengths(src_tokens)
        utf8_fig = plot_utf8_bytes_per_char(english)
        t3 = time.perf_counter()

        # CS229-3/4/6: Predict probs + (Greedy|Sampling) + Repeat
        telugu, topk_df, avg_conf, chosen_probs, entropies = decode_with_scores(
            enc, do_sample, temperature, top_p
        )
        t4 = time.perf_counter()

        # CS229-5: Detokenize + Transliteration
        roman = sanscript.transliterate(telugu, sanscript.TELUGU, sanscript.ITRANS)
        t5 = time.perf_counter()

        # Dynamics plots
        mode = "sampling" if do_sample else "greedy"
        conf_line = plot_line(chosen_probs, "Chosen-Token Probability per Step", "Probability", mode=mode)
        H_line = plot_line(entropies, "Entropy per Step (bits)", "Entropy (bits)", mode=mode)

        # Latency
        latency = {
            "tokenize": round((t2 - t1) * 1000, 2),
            "middle_plots": round((t3 - t2) * 1000, 2),
            "decode": round((t4 - t3) * 1000, 2),
            "postprocess": round((t5 - t4) * 1000, 2),
            "total": round((t5 - t0) * 1000, 2),
        }
        lat_fig = plot_latency(latency)

        # Middle summary
        mid_md = (
            f"### Input → Machine View\n\n"
            f"{uni_md}\n\n"
            f"**Tokens/IDs:** `{', '.join(src_tokens)}`\n\n"
            f"**Source length:** `{len(src_tokens)}` tokens\n"
            "_`▁` denotes word boundary in subword tokenization._"
        )

        # Right summary
        right_md = (
            f"**Confidence (avg next-token prob):** `{avg_conf:.3f}`\n\n"
            f"**Decoding:** `{'Sampling' if do_sample else 'Greedy (argmax)'}`"
            f"{' · temp=' + str(temperature) + ', top_p=' + str(top_p) if do_sample else ''}"
        )

        if topk_df.empty:
            topk_df = pd.DataFrame([["", "", 0.0, "", 0.0, "", 0.0, "", 0.0, "", 0.0, 0.0]],
                                   columns=["step","top1_token","top1_prob","top2_token","top2_prob",
                                            "top3_token","top3_prob","top4_token","top4_prob",
                                            "top5_token","top5_prob","chosen_prob"])

        return (
            mid_md, token_len_fig, utf8_fig,        # middle column
            telugu, roman, right_md,                # right text
            topk_df, conf_line, H_line, lat_fig     # right plots/tables
        )
    except Exception:
        tb = traceback.format_exc()
        err_md = f"**⛔ Pipeline error**\n\n```\n{tb}\n```"
        empty_df = pd.DataFrame()
        return (err_md, go.Figure(), go.Figure(), "", "", "", empty_df, go.Figure(), go.Figure(), go.Figure())

# ---------- Layout ----------

with gr.Blocks(title="NLP-101: English → Telugu (CS229-aligned)") as demo:
    gr.Markdown("## NLP-101: English → Telugu — See the CS229 Inference Steps in Action")

    with gr.Row():
        # Left: Input + Controls
        with gr.Column(scale=1):
            english = gr.Textbox(label="Input (English)", placeholder="Type a short sentence…", lines=3, value="I want to learn Telugu")
            do_sample = gr.Checkbox(label="Use sampling (instead of greedy)", value=False)
            with gr.Row():
                temperature = gr.Slider(0.2, 1.5, value=0.8, step=0.05, label="Temperature")
                top_p = gr.Slider(0.5, 1.0, value=0.9, step=0.01, label="Top-p (nucleus)")
            run_btn = gr.Button("Translate & Visualize", variant="primary")

        # Middle: Processing visuals (always visible)
        with gr.Column(scale=2):
            mid_md = gr.Markdown(label="Input → Machine View")
            token_len_plot = gr.Plot(label="Subword Token Lengths")
            utf8_plot = gr.Plot(label="UTF-8 Bytes per Character")

        # Right: Outputs & dynamics
        with gr.Column(scale=2):
            telugu_out = gr.Textbox(label="Telugu", lines=2)
            roman_out = gr.Textbox(label="Roman (ITRANS)", lines=1)
            right_md = gr.Markdown(label="Confidence & Decoding")
            topk_df = gr.Dataframe(label="Decode Timeline — Top-k per step")
            conf_plot = gr.Plot(label="Chosen-Prob per Step")
            ent_plot = gr.Plot(label="Entropy per Step")
            lat_plot = gr.Plot(label="Latency")

    run_btn.click(
        fn=run_pipeline,
        inputs=[english, do_sample, temperature, top_p],
        outputs=[mid_md, token_len_plot, utf8_plot, telugu_out, roman_out, right_md, topk_df, conf_plot, ent_plot, lat_plot],
    )

if __name__ == "__main__":
    demo.launch(share=False)
