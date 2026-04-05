# AI Autopsy

Post-mortem diagnostic tool for failed AI builds. Paste a failed prompt, vague spec, or broken code snippet — and get an evidence-backed diagnosis of what went wrong upstream and why.

Built for the [Devpost Spec-Driven Development Learning Hackathon](https://devpost.com).

---

## What It Does

AI Autopsy runs your input through a Judge LLM against a proprietary failure taxonomy (17 modes, 5 categories). It returns a structured Post-Mortem Report with:

- **Primary + Secondary Failure Modes** — identified from the taxonomy, never without evidence
- **Evidence** — direct quotes or logic gaps from your input that prove the failure
- **Causal Chain** — how one failure created conditions for the next
- **Reconstructed Spec** — what the spec should have said
- **Remediation Plan** — what to fix first and how
- **Verdict** — Inconclusive / Recoverable / Critical

---

## Setup

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd ai-autopsy
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Add your Anthropic API key

```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder with your real key:

```
ANTHROPIC_API_KEY=sk-ant-api03-...
```

Get a key at [console.anthropic.com](https://console.anthropic.com) → API Keys.

### 4. Run the server

```bash
uvicorn main:app --reload
```

### 5. Open the app

Go to [http://localhost:8000](http://localhost:8000)

---

## Usage

1. Select an input type: **Prompt + Output**, **Spec / Requirements**, or **Code Snippet**
2. Paste your input (max 5,000 characters)
3. Click **Run Autopsy** or press `⌘ + Enter`
4. Read the report — copy it to Clipboard for use in GitHub Issues, Notion, or Slack

Try the built-in example (click **Try an example**) to see the anchor case study in action.

---

## Failure Taxonomy

17 failure modes across 5 categories:

| Category | Modes |
|---|---|
| Prompt & Framing | FM-01 through FM-04 |
| Spec & Planning | FM-05 through FM-07 |
| Architecture & Design | FM-08 through FM-10 |
| Context & Execution | FM-11 through FM-14 |
| Decision & Governance | FM-15 through FM-17 |

Plus FM-00 (insufficient data) for inputs that lack enough diagnostic signal.

Full taxonomy in [`data/taxonomy.json`](data/taxonomy.json).

---

## Project Structure

```
ai-autopsy/
  main.py              ← FastAPI backend + Judge LLM logic
  index.html           ← Single-page frontend
  requirements.txt
  .env.example
  .env                 ← Your API key (gitignored)
  data/
    taxonomy.json      ← 17 failure modes + null state
  docs/
    scope.md
    prd.md
    tech-spec.md
    build-checklist.md
```

---

## Tech Stack

- **Backend:** Python / FastAPI + Uvicorn
- **LLM:** Claude (via Anthropic API)
- **Taxonomy:** Local JSON — human-editable, no code changes needed to add modes
- **Frontend:** Vanilla HTML/CSS/JS + marked.js for Markdown rendering

---

## License

MIT
