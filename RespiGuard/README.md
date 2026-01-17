# RespiGuard: Offline-First Acoustic Triage Assistant

RespiGuard is a privacy-first, offline AI application designed to bring specialist-level respiratory screening to low-resource and remote clinical environments. The goal is simple: turn a standard laptop into a bio-acoustic triage tool that helps frontline staff make faster, more informed decisions without needing internet access.

## Your team
Solo builder. I handled product, ML integration, and the Streamlit UI.

## The problem
Respiratory conditions (pneumonia, COPD, tuberculosis, asthma exacerbations) are among the leading causes of morbidity and mortality worldwide. In rural and low-resource clinics, early screening is limited by two bottlenecks:

- Infrastructure: Imaging tools and specialized clinicians are scarce or unavailable.
- Connectivity: Many AI diagnostics require cloud access, which fails in intermittent or no-connectivity settings.

This creates a gap where preventable deterioration can go unnoticed until it becomes an emergency.

## Why we built this
I wanted a tool that works where the infrastructure does not. The Edge AI track is a natural fit because the hardest clinics to reach are also the least connected. The design goal was clear: make the entire experience run locally, keep patient audio on-device, and still provide a clinically useful summary rather than a raw score.

## What we built
RespiGuard uses two HAI-DEF models in a local, offline pipeline:

- Listen (HeAR): Google’s Health Acoustic Representations model encodes a short cough sample into a compact acoustic embedding that captures subtle airway signals. The model consumes a 2-second window, even if the user records a longer 5-second sample.
- Reason (MedGemma): Google’s MedGemma 1.5 4B IT model turns those signals into a concise clinical summary and recommended actions, so the output is not just a score but a useful clinical note.

Both models run locally. No audio leaves the device.

## How it works (technical feasibility)
Core pipeline:
1) Capture a cough sample in the Streamlit UI.
2) Resample to 16k and normalize, then run HeAR locally to obtain an embedding.
3) Compare the embedding to a healthy reference signature and combine distance, variance, and energy into a risk score.
4) Use MedGemma locally to generate a structured clinical report (summary, 3–5 findings, 3 actions).

Implementation notes:
- HeAR runs via a TensorFlow SavedModel signature (local folder under models/hear).
- MedGemma runs via PyTorch and transformers (local folder under models/medgemma).
- The UI is fully offline; model downloads happen once via download_models.py after accepting Hugging Face terms.
- The app is optimized for Apple Silicon (M-series) using MPS where available.

Privacy and reliability:
- Audio stays on-device; no cloud calls at inference time.
- Works in low-connectivity settings, supporting the Edge AI prize requirements.

## Impact potential
RespiGuard targets early respiratory triage in settings where imaging and specialist care are unavailable. Even a lightweight acoustic screen that flags elevated risk can:
- speed up referral for severe cases,
- prioritize limited resources, and
- reduce delays in treatment for high-risk patients.

## Lessons learned
- Edge-first UX matters as much as the model: a clean, readable clinical note builds trust.
- Signal quality gating reduces false alarms from very quiet samples.
- Offline constraints force better engineering discipline (local models, deterministic outputs, no hidden dependencies).

## Setup and run (reproducible)
1) Create a local environment:
   - python3 -m venv venv
   - source venv/bin/activate

2) Install dependencies:
   - pip install --upgrade pip
   - pip install -r requirements.txt

3) Accept Hugging Face terms:
   - google/hear
   - google/medgemma-1.5-4b-it

4) Download models locally:
   - python3 download_models.py

5) Run the app:
   - streamlit run app.py

Notes for Apple Silicon:
- TensorFlow on macOS is most stable with Homebrew or python.org builds.
- If TensorFlow install fails, recreate the venv using the Homebrew python3.

## Submission checklist
- Video demo (<= 3 minutes): show the story, the live app, and proof of offline execution.
- Kaggle write-up (<= 3 pages): use the sections above as your base.
- Public code repository: this project plus the README, requirements, and reproducible steps.
