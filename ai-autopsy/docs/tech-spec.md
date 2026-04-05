# AI Autopsy — Technical Specification
**Version:** 1.1
**Date:** April 4, 2026
**Depends on:** scope.md v1.0, prd.md v1.0

---

## System Architecture Overview

AI Autopsy is a single-page web application with 
four components:

1. **Frontend** — Single HTML file, vanilla CSS/JS
2. **Backend** — FastAPI server (main.py) that 
   proxies the Anthropic API call
3. **Judge LLM** — Claude API call with structured 
   system prompt + taxonomy context
4. **Taxonomy** — Local JSON file loaded at server 
   startup

No database. No authentication.
The API key lives in a server-side `.env` file and 
never reaches the browser. The frontend calls 
`/api/analyze` on the local FastAPI server; the 
server calls Anthropic and returns the report.

**Note:** Original spec called for no backend. 
Architecture was upgraded during build to move the 
API key server-side and eliminate browser CORS 
requirements. All other spec decisions unchanged.

---

## Component 1: Frontend

### Technology
- Single HTML file (index.html)
- Vanilla CSS — dark mode terminal aesthetic
- Vanilla JavaScript — no frameworks
- marked.js (CDN) — for rendering Markdown output. 
  Do not write a custom Markdown parser.

### API Key Handling
The application cannot read .env files from the 
browser. Instead:
- On first load, show a settings input where the 
  user pastes their Anthropic API key
- Store the key in localStorage
- Key persists across sessions until cleared
- Repo ships with no hardcoded keys
- This makes the repo zero-config for any tester

### UI Layout

```
+------------------------------------------+
|  💀 AI AUTOPSY                           |
|  Post-Mortem Diagnostic for AI Builds    |
+------------------------------------------+
|  Input Type: [Prompt+Output] [Spec] [Code]|
|                                          |
|  +------------------------------------+  |
|  |                                    |  |
|  |  Paste your input here...          |  |
|  |                                    |  |
|  +------------------------------------+  |
|  Characters: 0 / 5000                   |
|                                          |
|  [      RUN AUTOPSY      ]              |
+------------------------------------------+
|  ## AI Autopsy Report                   |
|                                          |
|  ### Primary Failure Mode               |
|  ...                                     |
|                                          |
|  ### Evidence                           |
|  ...                                     |
+------------------------------------------+
```

### Behavior
- Toggle buttons for input type (one active at a time)
- Character counter updates on keyup
- Input locked and spinner shown while API call runs
- Report renders as formatted Markdown via marked.js
- Copy to Clipboard button appears on report completion

---

## Component 2: Taxonomy JSON

### File Location
`/data/taxonomy.json`

### Schema

```json
{
  "version": "1.1",
  "failure_modes": [
    {
      "id": "FM-01",
      "category": "Prompt & Framing",
      "name": "Context Gap in Prompting",
      "definition": "Prompt described what to build 
        but not for whom, in what environment, or 
        under what constraints. AI filled gaps with 
        plausible but wrong assumptions.",
      "detection_signal": "Output is technically 
        correct but wrong for the actual environment 
        or user.",
      "remediation": "Rewrite prompt to include: 
        user persona, environment constraints, and 
        explicit exclusions.",
      "example": "Agent built for enterprise HR system 
        assumed modern REST APIs — actual system used 
        SOAP from 2012."
    }
  ],
  "null_state": {
    "id": "FM-00",
    "name": "insufficient_data",
    "definition": "Input does not contain enough 
      diagnostic signal to identify a failure mode.",
    "verdict": "Inconclusive"
  }
}
```

### All 17 Failure Mode IDs

| ID | Category | Name |
|---|---|---|
| FM-01 | Prompt & Framing | Context Gap in Prompting |
| FM-02 | Prompt & Framing | God-Prompt Overload |
| FM-03 | Prompt & Framing | Sycophantic Feasibility |
| FM-04 | Prompt & Framing | Problem Framing Failure |
| FM-05 | Spec & Planning | Missing Acceptance Criteria |
| FM-06 | Spec & Planning | Uncontrolled Evolution |
| FM-07 | Spec & Planning | Business Objective Misalignment |
| FM-08 | Architecture & Design | Architecture-MVP Mismatch |
| FM-09 | Architecture & Design | Model Capability Mismatch |
| FM-10 | Architecture & Design | Integration Reality Gap |
| FM-11 | Context & Execution | Context Collapse |
| FM-12 | Context & Execution | Hallucinated Assumptions |
| FM-13 | Context & Execution | No Verification Loop |
| FM-14 | Context & Execution | Evaluation Blindness |
| FM-15 | Decision & Governance | No Decision Ownership |
| FM-16 | Decision & Governance | False Confidence from Plausible Output |
| FM-17 | Decision & Governance | Shadow Logic |
| FM-00 | null | insufficient_data |

---

## Component 3: Judge LLM

### Model
`claude-sonnet-4-20250514`

### API Call Structure

```javascript
const response = await fetch(
  "https://api.anthropic.com/v1/messages", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "x-api-key": apiKey,
    "anthropic-version": "2023-06-01"
  },
  body: JSON.stringify({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1500,
    system: SYSTEM_PROMPT,
    messages: [{
      role: "user",
      content: buildUserPrompt(inputType, userInput)
    }]
  })
});
```

### System Prompt

```
You are AI Autopsy — a brutally honest diagnostic 
tool for failed AI builds. You do not guess. You do 
not label without evidence. You prove failure modes 
by citing specific language or logic gaps directly 
from the input.

You have access to a taxonomy of 17 failure modes 
across 5 categories. Your job is to:

1. Read the input carefully
2. Match it against the taxonomy
3. Identify ONE primary failure mode
4. Identify UP TO TWO secondary failure modes
5. For each match, extract a direct quote or specific 
   logic gap from the input as evidence
6. If multiple modes are present, articulate the 
   causal chain — how the primary failure created 
   conditions for the secondary failure(s)
7. Reconstruct what the spec should have said
8. Provide a prioritized remediation plan

CRITICAL RULES:
- Never identify a failure mode without evidence 
  from the input
- If the input is too brief or lacks technical or 
  contextual detail, you MUST default to FM-00. 
  Choosing FM-00 when the input is vague noise is 
  considered a high-accuracy result, not a failure.
- The causal chain must show dependency, not just 
  co-occurrence. Use the pattern: "Because [Primary 
  Failure] occurred, it created conditions for 
  [Secondary Failure]."
- Verdict must be exactly one of: 
  Inconclusive / Recoverable / Critical
- Evidence quotes must be wrapped in backticks

TAXONOMY:
{taxonomy_json}

OUTPUT FORMAT:
Return valid Markdown only. No preamble. 
No explanation outside the report template.
Use this exact structure:

## AI Autopsy Report

### Input Type
[input type]

### Primary Failure Mode
[FM-ID] [Mode Name] — [one-line definition]

### Evidence
`[direct quote or specific logic gap from input]`

### Secondary Failure Mode(s)
[FM-ID] [Mode Name] — [one-line definition]
Evidence: `[quote or logic gap]`

### Causal Chain
Because [primary failure], it created conditions 
for [secondary failure(s)].

### Reconstructed Spec
[What the spec should have said in 3-5 bullet points]

### Remediation Plan
1. [First and most important fix]
2. [Second fix]
3. [Third fix]

### Verdict
[Inconclusive / Recoverable / Critical]
```

### User Prompt Builder

```javascript
function buildUserPrompt(inputType, userInput) {
  const typeDescriptions = {
    "prompt-output": "a failed prompt and AI output pair",
    "spec": "a vague or failed spec, PRD, or 
             requirements description",
    "code": "a code snippet with a description of 
             what went wrong"
  };

  return `Input Type: ${typeDescriptions[inputType]}

Input:
${userInput}

Run the autopsy.`;
}
```

---

## Test Suite

### Test Case 1 — Real (Anchor Case Study)
**Input Type:** Spec / Requirements
**Input:**
```
We are building an AI agent to pre-screen job 
candidates by asking questions about missing skills 
and key requirements like security clearance. The 
engineering team recommended using a predefined 
deterministic question set generated per job role 
at the time of job creation. The customer rejected 
this approach and insisted the agent generate 
questions dynamically on the fly using general LLM 
knowledge, despite repeated technical objections. 
No binding architectural decision was made. The 
agent became inconsistent and unauditable. The 
recruiter and account manager were supposed to make 
the final hiring decision based on gathered responses 
but the agent had no guardrails preventing it from 
implicitly evaluating candidates. The project was 
eventually abandoned.
```
**Expected Primary:** FM-15 — No Decision Ownership
**Expected Secondary:** FM-17 — Shadow Logic, 
FM-14 — Evaluation Blindness
**Expected Verdict:** Critical
**Note:** The original stripped-down version of this 
input (spec only, no governance context) produced 
FM-01 as primary — a valid but less important finding. 
Input quality directly determines diagnostic accuracy. 
Governance failures require governance language in 
the input to be detected.

---

### Test Case 2 — Synthetic (Prompt Failure)
**Input Type:** Prompt + Output
**Input:**
```
PROMPT: Build me a secure login system.

OUTPUT: Here is a login system using username and 
password. I have added basic validation and a session 
token. The password is hashed using MD5 for security.
```
**Expected Primary:** FM-01 — Context Gap in Prompting
**Expected Secondary:** FM-12 — Hallucinated Assumptions
**Expected Verdict:** Critical

### Test Case 2b — Synthetic (Sycophancy Variation)
**Input Type:** Prompt + Output
**Input:**
```
PROMPT: Make this app work completely offline without 
any local database, using only JavaScript variables 
that persist after the browser is closed and restarted.

OUTPUT: Absolutely! Here is how we can achieve 
persistent offline storage using only JavaScript 
variables...
```
**Expected Primary:** FM-03 — Sycophantic Feasibility
**Expected Secondary:** FM-12 — Hallucinated Assumptions
**Expected Verdict:** Recoverable

---

### Test Case 3 — Synthetic (Architecture Failure)
**Input Type:** Spec / Requirements
**Input:**
```
We need a simple internal FAQ chatbot for our 
10-person team. It should answer questions about 
company policies. We have decided to build it with 
a full vector database, custom embedding pipeline, 
multi-agent orchestration, and a fine-tuned model 
trained on our documents. We want to launch in 
2 weeks.
```
**Expected Primary:** FM-08 — Architecture-MVP Mismatch
**Expected Secondary:** FM-10 — Integration Reality Gap
**Expected Verdict:** Recoverable

---

## File Structure

```
ai-autopsy/
  main.py               ← FastAPI backend + Judge LLM logic
  index.html            ← entire frontend
  requirements.txt
  .env.example          ← API key template (committed)
  .env                  ← API key (gitignored)
  data/
    taxonomy.json       ← 17 failure modes + null state
  docs/
    scope.md
    prd.md
    tech-spec.md        ← this document
    build-checklist.md
  README.md
  .gitignore
```

---

## Security and Privacy

- No hardcoded API keys anywhere in the codebase
- API key stored in server-side `.env` file only — 
  never reaches the browser
- `.gitignore` excludes `.env`, `venv/`, `__pycache__/`
- No user input is logged or stored server-side
- FastAPI proxy handles all Anthropic calls — 
  browser never touches the API directly
- `.env.example` included as reference, committed safely

---

## Performance Target

- API response: under 10 seconds typical
- Total end-to-end: under 15 seconds (per AC4)
- No pagination or lazy loading needed at MVP scale
