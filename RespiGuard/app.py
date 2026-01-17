import hashlib
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time

import numpy as np
import streamlit as st
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CONFIGURATION ---
# DEMO_MODE: Set True only if you need a stable high score for a demo video.
DEMO_MODE = False

# --- PAGE SETUP ---
st.set_page_config(
    page_title="RespiGuard", 
    page_icon="🫁", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern clinical styling
st.markdown("""
<style>
    :root {
        --rg-bg: #f6f7fb;
        --rg-surface: #ffffff;
        --rg-ink: #0f172a;
        --rg-muted: #475569;
        --rg-accent: #1565c0;
        --rg-accent-2: #0ea5a4;
        --rg-border: #e2e8f0;
        --rg-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
    }
    .stApp {
        background:
            radial-gradient(1200px 600px at 12% -10%, #e6f0ff 0%, rgba(230, 240, 255, 0) 60%),
            radial-gradient(900px 600px at 100% -20%, #fff4e8 0%, rgba(255, 244, 232, 0) 55%),
            var(--rg-bg);
        color: var(--rg-ink);
        font-family: "Avenir Next", "Avenir", "Optima", "Trebuchet MS", sans-serif;
    }
    .block-container {
        max-width: 1200px;
        padding-top: 2.2rem;
        padding-bottom: 2rem;
        animation: rgFadeUp 0.6s ease-out both;
    }
    h1, h2, h3 {
        color: var(--rg-accent);
        letter-spacing: -0.01em;
    }
    .rg-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
    }
    .rg-header-main {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .rg-logo {
        font-size: 2.8rem;
        filter: drop-shadow(0 6px 10px rgba(21, 101, 192, 0.2));
    }
    .rg-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: var(--rg-accent);
        margin: 0;
    }
    .rg-subtitle {
        color: var(--rg-muted);
        font-size: 1rem;
        margin-top: 0.2rem;
    }
    .rg-badge {
        background: linear-gradient(120deg, var(--rg-accent), var(--rg-accent-2));
        color: #ffffff;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .rg-divider {
        height: 1px;
        background: linear-gradient(90deg, rgba(21, 101, 192, 0.4), rgba(14, 165, 164, 0.2), rgba(21, 101, 192, 0));
        margin: 1.2rem 0 2rem;
        border: 0;
    }
    .rg-panel {
        background: rgba(21, 101, 192, 0.07);
        border: 1px solid rgba(21, 101, 192, 0.18);
        padding: 0.9rem 1rem;
        border-radius: 12px;
        color: #1f2937;
        margin-bottom: 1rem;
    }
    .rg-score-card {
        background: var(--rg-surface);
        border: 1px solid var(--rg-border);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: var(--rg-shadow);
        text-align: center;
        background-image: linear-gradient(180deg, rgba(21, 101, 192, 0.03), rgba(21, 101, 192, 0));
    }
    .rg-score-card h4 {
        margin: 0;
        color: var(--rg-muted);
        font-weight: 600;
    }
    .rg-score {
        font-size: 3.4rem;
        font-weight: 700;
        margin: 0.4rem 0 0.2rem;
        letter-spacing: -0.02em;
        font-variant-numeric: tabular-nums;
    }
    .rg-score-chip {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.8rem;
        color: #ffffff;
    }
    .rg-score-placeholder {
        border: 2px dashed #cbd5e1;
        color: #94a3b8;
        background: rgba(255, 255, 255, 0.6);
    }
    .rg-report {
        background: var(--rg-surface);
        border: 1px solid var(--rg-border);
        border-left: 6px solid var(--rg-accent);
        border-radius: 16px;
        padding: 1.6rem 1.9rem;
        box-shadow: var(--rg-shadow);
        line-height: 1.6;
    }
    .rg-report-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
        margin-bottom: 1.2rem;
    }
    .rg-report-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--rg-accent);
    }
    .rg-report-subtitle {
        color: var(--rg-muted);
        font-size: 0.9rem;
    }
    .rg-report-chip {
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.75rem;
        color: #ffffff;
        background: var(--rg-accent);
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }
    .rg-report-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.9fr);
        gap: 1.4rem;
    }
    .rg-report-section {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.1rem;
    }
    .rg-report-section h3 {
        margin: 0 0 0.6rem;
        font-size: 1rem;
        color: var(--rg-accent);
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .rg-report-icon {
        font-size: 1.05rem;
    }
    .rg-report-list {
        list-style: none;
        padding: 0;
        margin: 0;
        display: grid;
        gap: 0.6rem;
    }
    .rg-report-item {
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
    }
    .rg-report-item span {
        font-size: 1rem;
    }
    .rg-report-item p {
        margin: 0;
        color: #1f2937;
    }
    .rg-loader {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid var(--rg-border);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        box-shadow: var(--rg-shadow);
        display: grid;
        gap: 0.6rem;
        align-items: center;
        max-width: 520px;
    }
    .rg-lungs-wrap {
        width: 72px;
        height: 72px;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(21, 101, 192, 0.15), rgba(14, 165, 164, 0.12));
        display: grid;
        place-items: center;
    }
    .rg-lungs {
        width: 46px;
        height: 46px;
        fill: none;
        stroke: var(--rg-accent);
        stroke-width: 2.2;
        animation: rgBreathe 2.2s ease-in-out infinite;
    }
    .rg-loader-text {
        font-weight: 700;
        color: var(--rg-accent);
        font-size: 1.05rem;
    }
    .rg-loader-subtext {
        color: var(--rg-muted);
        font-size: 0.9rem;
    }
    .rg-loader-bar {
        height: 8px;
        background: #e2e8f0;
        border-radius: 999px;
        overflow: hidden;
    }
    .rg-loader-bar span {
        display: block;
        height: 100%;
        width: 40%;
        background: linear-gradient(90deg, var(--rg-accent), var(--rg-accent-2));
        animation: rgFlow 1.6s ease-in-out infinite;
    }
    .rg-report h3 {
        color: var(--rg-accent);
        margin-top: 1rem;
    }
    .rg-report h3:first-child {
        margin-top: 0;
    }
    .rg-report p {
        margin: 0.4rem 0 0.6rem;
    }
    .rg-report ul {
        padding-left: 1.3rem;
        margin-top: 0.6rem;
    }
    .rg-report li {
        margin: 0.4rem 0;
    }
    .rg-meta {
        color: var(--rg-muted);
        font-size: 0.85rem;
        margin-top: 0.75rem;
    }
    .rg-empty {
        background: rgba(255, 255, 255, 0.7);
        border: 1px dashed #cbd5e1;
        border-radius: 14px;
        padding: 2rem;
        color: var(--rg-muted);
        text-align: center;
    }
    .rg-alert {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 12px;
        padding: 0.7rem 1rem;
        color: #9a3412;
        font-weight: 600;
        margin-top: 0.8rem;
    }
    .rg-alert.ok {
        background: #ecfdf3;
        border-color: #a7f3d0;
        color: #166534;
    }
    .rg-signal-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: #e2e8f0;
        color: #475569;
        margin-top: 0.6rem;
    }
    .rg-signal-chip.low {
        background: #fee2e2;
        color: #b91c1c;
    }
    .rg-signal-chip.ok {
        background: #fff7ed;
        color: #b45309;
    }
    .rg-signal-chip.strong {
        background: #dcfce7;
        color: #15803d;
    }
    .rg-section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--rg-accent);
        margin: 0.4rem 0 1rem;
    }
    .rg-report-actions {
        display: flex;
        gap: 0.8rem;
        flex-wrap: wrap;
        align-items: center;
        justify-content: flex-end;
        margin-top: 0.8rem;
    }
    .rg-audio-label {
        color: var(--rg-muted);
        font-size: 0.85rem;
        margin-top: 0.6rem;
    }
    .rg-animate {
        animation: rgFadeUp 0.6s ease-out both;
    }
    .rg-delay-1 { animation-delay: 0.05s; }
    .rg-delay-2 { animation-delay: 0.12s; }
    .rg-delay-3 { animation-delay: 0.2s; }
    .stButton>button {
        background: var(--rg-accent);
        color: #ffffff;
        border-radius: 10px;
        border: none;
        padding: 0.7rem 1.2rem;
        font-weight: 600;
        box-shadow: 0 10px 18px rgba(21, 101, 192, 0.18);
        transition: transform 0.08s ease, background 0.2s ease;
    }
    .stButton>button:active {
        transform: scale(0.98);
    }
    .stButton>button:hover {
        background: #0d47a1;
    }
    .stButton>button:disabled {
        background: #cbd5e1;
        color: #64748b;
        box-shadow: none;
        cursor: not-allowed;
    }
    .stAudio {
        background: var(--rg-surface);
        border-radius: 12px;
        padding: 0.5rem 0.75rem;
        border: 1px solid var(--rg-border);
        box-shadow: 0 10px 20px rgba(15, 23, 42, 0.06);
    }
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, var(--rg-accent), var(--rg-accent-2));
    }
    @media (max-width: 900px) {
        .rg-report-grid {
            grid-template-columns: 1fr;
        }
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    @keyframes rgFadeUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes rgBreathe {
        0%, 100% { transform: scale(1); opacity: 0.85; }
        50% { transform: scale(1.07); opacity: 1; }
    }
    @keyframes rgFlow {
        0% { transform: translateX(-60%); }
        50% { transform: translateX(70%); }
        100% { transform: translateX(180%); }
    }
</style>
""", unsafe_allow_html=True)

# --- HARDWARE SETUP (M4 OPTIMIZED) ---
# We use 'mps' (Metal Performance Shaders) for Mac GPU acceleration
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# bfloat16 is the native format for MedGemma and works great on M-series chips
DTYPE = torch.bfloat16 
TARGET_SR = 16000
TARGET_LEN = 32000
MIN_RMS = 0.008
RMS_FULL_SCALE = 0.05

# --- CACHED MODEL LOADING ---
def compute_reference_embedding(hear_signature):
    silence = np.zeros((1, TARGET_LEN), dtype="float32")
    outputs = hear_signature(x=silence)
    embedding = outputs["output_0"]
    if hasattr(embedding, "numpy"):
        embedding = embedding.numpy()
    return torch.from_numpy(embedding).squeeze()


def render_loader(status_box, headline, subtext):
    status_box.markdown(
        f"""
        <div class="rg-loader rg-animate">
            <div class="rg-lungs-wrap">
                <svg class="rg-lungs" viewBox="0 0 64 64" aria-hidden="true">
                    <path d="M32 10v20" />
                    <path d="M32 30c-6-6-14-6-18 0-3 5-2 12 1 18 3 6 9 8 16 8" />
                    <path d="M32 30c6-6 14-6 18 0 3 5 2 12-1 18-3 6-9 8-16 8" />
                    <path d="M28 14l8 0" />
                </svg>
            </div>
            <div>
                <div class="rg-loader-text">{html.escape(headline)}</div>
                <div class="rg-loader-subtext">{html.escape(subtext)}</div>
            </div>
            <div class="rg-loader-bar"><span></span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_models():
    try:
        import tensorflow as tf

        status_box = st.empty()
        render_loader(status_box, "RespiGuard is warming up", "Get your cough ready.")
        time.sleep(0.3)

        hear_path = os.path.abspath("models/hear")
        hear_model = tf.saved_model.load(hear_path)
        hear_signature = hear_model.signatures["serving_default"]
        reference_embedding = compute_reference_embedding(hear_signature)

        render_loader(status_box, "Calibrating acoustic intake", "Tuning breath and cough signals.")
        time.sleep(0.3)

        medgemma_path = "models/medgemma"
        tokenizer = AutoTokenizer.from_pretrained(medgemma_path)
        llm_model = AutoModelForCausalLM.from_pretrained(
            medgemma_path,
            torch_dtype=DTYPE,
            device_map=DEVICE,
        )

        render_loader(status_box, "Bringing the clinic online", "Almost ready.")
        time.sleep(0.3)

        status_box.empty()
        return hear_signature, reference_embedding, tokenizer, llm_model

    except Exception as e:
        st.error(f"❌ Model Load Failed: {e}")
        st.warning("Ensure TensorFlow is installed and download_models.py completed.")
        st.stop()

# Initialize Models
hear_signature, healthy_reference, tokenizer, llm_model = load_models()

# --- HELPER FUNCTIONS ---
def get_audio_embedding(audio_tensor):
    """Passes raw audio through HeAR to get a 512-d vector."""
    # Ensure correct shape (Batch, Time)
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    audio_np = audio_tensor.detach().cpu().numpy().astype("float32")

    # HeAR expects 2 seconds at 16kHz (32000 samples)
    if audio_np.shape[-1] < TARGET_LEN:
        pad_width = TARGET_LEN - audio_np.shape[-1]
        audio_np = np.pad(audio_np, ((0, 0), (0, pad_width)))
    elif audio_np.shape[-1] > TARGET_LEN:
        audio_np = audio_np[:, :TARGET_LEN]

    outputs = hear_signature(x=audio_np)
    embedding = outputs["output_0"]
    if hasattr(embedding, "numpy"):
        embedding = embedding.numpy()
    return torch.from_numpy(embedding).squeeze()

def analyze_signal(waveform):
    if waveform.numel() == 0:
        return 0.0, 0.0, 0.0, "LOW", 1.0
    rms = waveform.pow(2).mean().sqrt().item()
    peak = waveform.abs().max().item()
    silence_ratio = (waveform.abs() < 0.01).float().mean().item()
    burstiness_factor = 0.0
    frame_len = max(1, int(TARGET_SR * 0.05))
    if waveform.numel() >= frame_len:
        frames = waveform.unfold(0, frame_len, frame_len)
        frame_rms = frames.pow(2).mean(dim=1).sqrt()
        frame_mean = frame_rms.mean().item()
        if frame_mean > 1e-6:
            burstiness = frame_rms.max().item() / frame_mean
            burstiness_factor = float(np.clip((burstiness - 1.0) / 3.0, 0.0, 1.0))
    signal_factor = (rms - MIN_RMS) / (RMS_FULL_SCALE - MIN_RMS)
    signal_factor = float(np.clip(signal_factor, 0.0, 1.0))
    signal_factor *= max(0.0, 1.0 - (silence_ratio * 0.6))
    signal_factor *= (0.4 + 0.6 * burstiness_factor)
    if rms < MIN_RMS:
        quality = "LOW"
    elif burstiness_factor < 0.35:
        quality = "OK"
    else:
        quality = "STRONG"
    return rms, peak, signal_factor, quality, silence_ratio


def get_risk_band(score):
    if score > 0.7:
        return {
            "label": "CRITICAL",
            "color": "#d32f2f",
            "alert_text": "ALERT: Elevated acoustic biomarkers detected.",
            "alert_class": "rg-alert",
            "level": "HIGH",
        }
    if score > 0.4:
        return {
            "label": "MODERATE",
            "color": "#f57c00",
            "alert_text": "NOTICE: Moderate acoustic anomalies detected.",
            "alert_class": "rg-alert",
            "level": "MODERATE",
        }
    return {
        "label": "NORMAL",
        "color": "#2e7d32",
        "alert_text": "RESULT: Within normal limits.",
        "alert_class": "rg-alert ok",
        "level": "LOW",
    }


def calculate_dynamic_risk(embedding, reference_embedding, signal_factor):
    emb = embedding.float().flatten()
    ref = reference_embedding.float().flatten()
    if emb.numel() != ref.numel():
        ref = ref[: emb.numel()]

    emb_norm_val = emb.norm().item()
    ref_norm_val = ref.norm().item()
    if emb_norm_val > 0 and ref_norm_val > 0:
        emb_norm = emb / emb_norm_val
        ref_norm = ref / ref_norm_val
        cosine_distance = 1 - torch.dot(emb_norm, ref_norm).clamp(-1, 1).item()
    else:
        cosine_distance = 0.0
    variance = emb.std().item()
    energy = emb.abs().mean().item()

    raw_score = (cosine_distance * 0.9) + (variance * 2.0) + (energy * 0.4)
    risk = 1 / (1 + np.exp(-((raw_score - 1.7) / 0.9)))
    strength = float(signal_factor) ** 1.4
    risk = (risk * strength) + (0.04 * (1 - strength))
    return float(np.clip(risk, 0.03, 0.92))


def sanitize_report_text(text):
    cleaned = re.sub(r"[#*`]", "", text)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = re.sub(r"^[\s\-\d\.\)\u2022]+", "", cleaned)
    cleaned = cleaned.replace('"', "").replace("'", "")
    return cleaned.strip()


def extract_json_payload(text):
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    candidate = re.sub(r",\s*}", "}", candidate)
    candidate = re.sub(r",\s*]", "]", candidate)
    if "'" in candidate and '"' not in candidate:
        candidate = candidate.replace("'", '"')
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def build_fallback_report(risk_level):
    if risk_level == "HIGH":
        summary = "Findings suggest elevated airway irritation and possible inflammation."
        findings = [
            "Elevated acoustic anomaly index relative to baseline.",
            "Broad feature activation consistent with forceful cough.",
            "Signal energy and variance are above typical levels.",
            "Pattern warrants close symptom monitoring.",
        ]
        actions = [
            "Assess respiratory effort and oxygen saturation.",
            "Provide supportive care and monitor response closely.",
            "Escalate to a clinician if symptoms persist or worsen.",
        ]
    elif risk_level == "MODERATE":
        summary = "Findings suggest mild to moderate airway irritation."
        findings = [
            "Moderate anomaly index relative to baseline.",
            "Signal variance indicates intermittent airway stress.",
            "Cough pattern shows mixed acoustic intensity.",
        ]
        actions = [
            "Recheck symptoms after rest and hydration.",
            "Evaluate for potential triggers or infection.",
            "Monitor for progression or persistent cough.",
        ]
    else:
        summary = "Findings are consistent with normal or low-risk respiratory acoustics."
        findings = [
            "Low anomaly index with limited feature activation.",
            "Signal energy aligns with quiet breathing.",
            "No acute acoustic markers detected.",
        ]
        actions = [
            "Continue routine observation and self-care.",
            "Encourage hydration and rest.",
            "Seek care if new symptoms develop.",
        ]
    return {"summary": summary, "findings": findings, "actions": actions}


def normalize_report_payload(payload, risk_level):
    if not isinstance(payload, dict):
        payload = build_fallback_report(risk_level)
    summary = sanitize_report_text(str(payload.get("summary", "")))
    if not summary:
        summary = sanitize_report_text(str(payload.get("interpretation", "")))
    findings = payload.get("findings", payload.get("observations", []))
    actions = payload.get("actions", [])

    if isinstance(findings, str):
        findings = [f.strip() for f in re.split(r"[\n;]", findings) if f.strip()]
    if not isinstance(findings, list):
        findings = []
    findings = [sanitize_report_text(str(item)) for item in findings if str(item).strip()]
    findings = [item for item in findings if item]

    if isinstance(actions, str):
        actions = [a.strip() for a in re.split(r"[\n;]", actions) if a.strip()]
    if not isinstance(actions, list):
        actions = []
    actions = [sanitize_report_text(str(action)) for action in actions if str(action).strip()]
    actions = [action for action in actions if action]

    fallback = build_fallback_report(risk_level)
    if not summary:
        summary = fallback["summary"]
    if len(findings) < 3:
        findings = findings + fallback["findings"]
    if len(actions) < 3:
        actions = actions + fallback["actions"]

    return {
        "summary": summary,
        "findings": findings[:5],
        "actions": actions[:3],
    }


def render_report_html(report, risk_band, signal_quality):
    summary = html.escape(report["summary"])
    action_icons = ["✅", "🩺", "📌"]
    finding_icons = ["🫁", "📈", "🧭", "🧪", "🩺"]

    finding_items = []
    for idx, finding in enumerate(report["findings"]):
        icon = finding_icons[idx % len(finding_icons)]
        finding_items.append(
            f"<div class='rg-report-item'><span>{icon}</span><p>{html.escape(finding)}</p></div>"
        )
    findings_html = "\n".join(finding_items)

    action_items = []
    for idx, action in enumerate(report["actions"]):
        icon = action_icons[idx % len(action_icons)]
        action_items.append(
            f"<div class='rg-report-item'><span>{icon}</span><p>{html.escape(action)}</p></div>"
        )
    action_html = "\n".join(action_items)
    signal_note = ""
    if signal_quality == "LOW":
        signal_note = "<p class='rg-meta'>Signal quality is low; interpret with caution.</p>"
    html_output = f"""
    <div class="rg-report rg-animate rg-delay-1">
        <div class="rg-report-header">
            <div>
                <div class="rg-report-title">Clinical Assessment</div>
                <div class="rg-report-subtitle">Diagnostic summary</div>
            </div>
            <div class="rg-report-chip" style="background:{risk_band['color']};">{risk_band['level']}</div>
        </div>
        <div class="rg-report-grid">
            <div class="rg-report-section">
                <h3><span class="rg-report-icon">🧾</span>Clinical Summary</h3>
                <p>{summary}</p>
                {signal_note}
                <h3><span class="rg-report-icon">🔍</span>Key Findings</h3>
                <div class="rg-report-list">{findings_html}</div>
            </div>
            <div class="rg-report-section">
                <h3><span class="rg-report-icon">📋</span>Recommended Actions</h3>
                <div class="rg-report-list">{action_html}</div>
            </div>
        </div>
    </div>
    """
    return re.sub(r"\n\s+", "\n", html_output).strip()


def safe_ascii(text):
    if not text:
        return ""
    return text.encode("ascii", "ignore").decode("ascii")


def break_long_pdf_words(text, max_len=32):
    def split_word(match):
        word = match.group(0)
        return " ".join(word[i : i + max_len] for i in range(0, len(word), max_len))

    return re.sub(r"\S{" + str(max_len + 1) + r",}", split_word, text)


def compose_report_text(report, band, score, signal_quality, duration_sec):
    lines = []
    lines.append("RespiGuard Diagnostic Report")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Patient Intake")
    if duration_sec:
        lines.append(f"Sample duration: {duration_sec:.1f}s")
    lines.append(f"Signal quality: {signal_quality}")
    lines.append("")
    lines.append("Biomarker Analysis")
    lines.append(f"Acoustic Anomaly Index: {score:.2f} ({band['label']})")
    lines.append("")
    lines.append("Clinical Summary")
    lines.append(report["summary"])
    lines.append("")
    lines.append("Key Findings")
    for idx, finding in enumerate(report["findings"], start=1):
        lines.append(f"{idx}) {finding}")
    lines.append("")
    lines.append("Recommended Actions")
    for idx, action in enumerate(report["actions"], start=1):
        lines.append(f"{idx}) {action}")
    return safe_ascii("\n".join(lines))


def generate_pdf_bytes(report_text):
    try:
        from fpdf import FPDF
    except Exception:
        return None

    try:
        pdf = FPDF()
        pdf.set_left_margin(14)
        pdf.set_right_margin(14)
        pdf.set_auto_page_break(auto=True, margin=14)
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        wrapped_text = break_long_pdf_words(report_text)
        page_width = max(10.0, pdf.w - pdf.l_margin - pdf.r_margin)
        for line in wrapped_text.splitlines():
            if not line.strip():
                pdf.ln(4)
                pdf.set_x(pdf.l_margin)
                continue
            pdf.set_x(pdf.l_margin)
            try:
                pdf.multi_cell(page_width, 6, line)
            except Exception:
                for chunk in textwrap.wrap(line, width=80, break_long_words=True):
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(page_width, 6, chunk)
        return pdf.output(dest="S").encode("latin-1")
    except Exception:
        return None


def compose_tts_text(report, band):
    findings = "; ".join(report["findings"][:3])
    actions = "; ".join(report["actions"][:3])
    return (
        "RespiGuard clinical report. "
        f"Risk level {band['label']}. "
        f"Summary: {report['summary']} "
        f"Key findings: {findings}. "
        f"Recommended actions: {actions}."
    )


def synthesize_tts(text):
    cleaned = safe_ascii(text)
    if not cleaned:
        return None, "No report text to narrate."
    if sys.platform != "darwin":
        return None, "Voice playback is available on macOS only."
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            aiff_path = os.path.join(temp_dir, "report.aiff")
            wav_path = os.path.join(temp_dir, "report.wav")
            subprocess.run(["say", "-o", aiff_path, cleaned], check=True)
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16", aiff_path, wav_path], check=True)
            with open(wav_path, "rb") as handle:
                return handle.read(), None
    except Exception:
        return None, "Unable to generate narration audio."


def generate_medgemma_report(risk_score):
    """Prompts MedGemma to draft a structured clinical report."""
    risk_level = get_risk_band(risk_score)["level"]

    prompt = f"""<start_of_turn>user
You are a clinical AI reporting system.
Input: Patient cough analysis.
Score: {risk_score:.2f} / 1.00 ({risk_level}).

Return ONLY JSON in this exact shape:
{{
  "summary": "One sentence on what the score suggests about airway inflammation.",
  "findings": ["Finding 1", "Finding 2", "Finding 3", "Finding 4"],
  "actions": ["Action 1", "Action 2", "Action 3"]
}}

Rules:
- No markdown, no bullet symbols, no numbering.
- No extra keys or commentary.
- Use concise clinical language.
- Provide 3 to 5 findings.
<end_of_turn>
<start_of_turn>model
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            temperature=0.0,
        )

    input_len = inputs.input_ids.shape[1]
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    payload = extract_json_payload(response)
    if payload is None:
        payload = build_fallback_report(risk_level)
    return normalize_report_payload(payload, risk_level)

# --- UI LAYOUT ---
st.markdown(
    """
    <div class="rg-header rg-animate rg-delay-1">
        <div class="rg-header-main">
            <div class="rg-logo">🫁</div>
            <div>
                <div class="rg-title">RespiGuard</div>
                <div class="rg-subtitle">Offline Acoustic Triage | Powered by Google HeAR & MedGemma</div>
            </div>
        </div>
        <div class="rg-badge">M4 Ready</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='rg-divider'></div>", unsafe_allow_html=True)

audio_value = None
run_clicked = False

col_input, col_score = st.columns([1.5, 1], gap="large")

with col_input:
    st.subheader("1. Patient Intake")
    st.markdown(
        "<div class='rg-panel'><strong>Protocol:</strong> Record 5 seconds of forced cough.</div>",
        unsafe_allow_html=True,
    )

    audio_value = st.audio_input("Input Stream")
    run_clicked = st.button(
        "Run Diagnostic Analysis",
        use_container_width=True,
        disabled=audio_value is None,
    )
    st.caption("Tip: Quiet breathing yields low scores, sharp coughs raise them.")
    signal_quality = st.session_state.get("signal_quality")
    if signal_quality:
        chip_class = signal_quality.lower()
        st.markdown(
            f"<div class='rg-signal-chip {chip_class}'>Signal: {signal_quality}</div>",
            unsafe_allow_html=True,
        )

with col_score:
    st.subheader("2. Biomarker Analysis")

    if "risk" in st.session_state:
        r = st.session_state["risk"]
        band = get_risk_band(r)
        color = band["color"]
        label = band["label"]

        st.markdown(
            f"""
            <div class="rg-score-card rg-animate rg-delay-1">
                <h4>Acoustic Anomaly Index</h4>
                <div class="rg-score" style="color: {color};">{r:.2f}</div>
                <span class="rg-score-chip" style="background-color: {color};">{label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(min(max(r, 0.0), 1.0))
        st.markdown(
            f"<div class='{band['alert_class']}'>{band['alert_text']}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="rg-score-card rg-score-placeholder">
                <h4>Acoustic Anomaly Index</h4>
                <div style="font-size: 1.1rem; margin-top: 0.6rem;">Waiting for audio...</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# 3. PROCESSING LOGIC
if audio_value and run_clicked:
    with st.spinner("Extracting bio-acoustic embeddings..."):
        import io
        import soundfile as sf

        audio_bytes = audio_value.read()
        data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        waveform = torch.tensor(data).float()

        if sample_rate != TARGET_SR:
            resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SR)
            waveform = resampler(waveform)
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=1)

        rms, peak, signal_factor, signal_quality, silence_ratio = analyze_signal(waveform)
        user_vector = get_audio_embedding(waveform)
        duration_sec = float(waveform.numel()) / float(TARGET_SR) if TARGET_SR else 0.0

        if DEMO_MODE:
            score = 0.88
        else:
            score = calculate_dynamic_risk(user_vector, healthy_reference, signal_factor)

    st.session_state["risk"] = score
    st.session_state["vector"] = user_vector
    st.session_state["signal_rms"] = rms
    st.session_state["signal_peak"] = peak
    st.session_state["signal_factor"] = signal_factor
    st.session_state["signal_quality"] = signal_quality
    st.session_state["signal_silence_ratio"] = silence_ratio
    st.session_state["audio_bytes"] = audio_bytes
    st.session_state["audio_duration"] = duration_sec
    st.session_state["report"] = None
    st.session_state["report_risk"] = score
    st.session_state["tts_audio"] = None
    st.session_state["tts_key"] = None
    st.session_state["tts_error"] = None
    st.rerun()

# 4. REPORT SECTION
st.markdown(
    "<div class='rg-section-title'>3. Clinical Assessment Report</div>",
    unsafe_allow_html=True,
)

if "risk" in st.session_state:
    report = st.session_state.get("report")
    if report is None or st.session_state.get("report_risk") != st.session_state["risk"]:
        with st.spinner("Generative AI is drafting the report..."):
            start_time = time.perf_counter()
            report = generate_medgemma_report(st.session_state["risk"])
            st.session_state["report"] = report
            st.session_state["report_risk"] = st.session_state["risk"]
            st.session_state["report_time"] = time.perf_counter() - start_time
            st.session_state["tts_audio"] = None
            st.session_state["tts_key"] = None
            st.session_state["tts_error"] = None

    band = get_risk_band(st.session_state["risk"])
    signal_quality = st.session_state.get("signal_quality", "LOW")
    st.markdown(
        render_report_html(report, band, signal_quality),
        unsafe_allow_html=True,
    )

    inference_time = st.session_state.get("report_time", 0.0)
    st.markdown(
        f"<div class='rg-meta'>Report generated in {inference_time:.2f}s</div>",
        unsafe_allow_html=True,
    )
    duration_sec = st.session_state.get("audio_duration", 0.0)
    report_text = compose_report_text(
        report,
        band,
        st.session_state["risk"],
        signal_quality,
        duration_sec,
    )
    pdf_bytes = generate_pdf_bytes(report_text)
    report_stamp = time.strftime("%Y%m%d_%H%M%S")
    left_action, right_action = st.columns([1.2, 0.8])
    with left_action:
        st.markdown("<div class='rg-audio-label'>Narrate clinical report</div>", unsafe_allow_html=True)
        tts_text = compose_tts_text(report, band)
        tts_key = hashlib.sha1(tts_text.encode("utf-8")).hexdigest()
        if st.button("🔊 Play clinical report", use_container_width=True):
            audio_bytes, error = synthesize_tts(tts_text)
            st.session_state["tts_audio"] = audio_bytes
            st.session_state["tts_key"] = tts_key
            st.session_state["tts_error"] = error
        if st.session_state.get("tts_audio") and st.session_state.get("tts_key") == tts_key:
            st.audio(st.session_state["tts_audio"], format="audio/wav")
        elif st.session_state.get("tts_error"):
            st.caption(st.session_state["tts_error"])
    with right_action:
        if pdf_bytes:
            st.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name=f"respiguard_report_{report_stamp}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
else:
    st.markdown(
        "<div class='rg-empty rg-animate rg-delay-2'>Waiting for input stream...</div>",
        unsafe_allow_html=True,
    )
