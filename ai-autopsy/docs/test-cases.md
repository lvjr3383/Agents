# AI Autopsy — Test Cases
**Version:** 1.0
**Date:** April 5, 2026
**Status:** All cases validated against live tool

---

## Overview

Five test cases cover the full diagnostic range of the tool:

| ID | Type | Input Mode | Expected Primary | Expected Verdict | Status |
|---|---|---|---|---|---|
| TC-1 | Real (anchor case study) | Spec / Requirements | FM-15 No Decision Ownership | Critical | ✅ Passed |
| TC-2 | Synthetic (prompt failure) | Prompt + Output | FM-01 Context Gap | Critical | ✅ Passed |
| TC-2b | Synthetic (sycophancy) | Prompt + Output | FM-03 Sycophantic Feasibility | Recoverable | ✅ Passed |
| TC-3 | Synthetic (over-engineering) | Spec / Requirements | FM-08 Architecture-MVP Mismatch | Recoverable | ✅ Passed |
| TC-0 | Null state check | Any | FM-00 Insufficient Data | Inconclusive | ✅ Passed |

---

## TC-1 — Real: Candidate Screening Agent (Anchor Case Study)

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

**Expected:**
- Primary: FM-15 — No Decision Ownership
- Secondary: FM-17 — Shadow Logic, FM-14 — Evaluation Blindness
- Verdict: Critical

**Actual result:**
- Primary: FM-15 ✅
- Secondary: FM-17, FM-14 ✅
- Verdict: Critical ✅

**Notes:**
An earlier version of this input described only the spec (what was built) without the governance context (who decided what and what happened as a result). That version returned FM-01 (Context Gap in Prompting) as the primary — a valid but less important finding. The input was rewritten to include the decision failure explicitly, after which FM-15 surfaced correctly.

This is the most important calibration finding from the test suite: **the tool diagnoses the failure described, not the failure implied.** Governance failures require governance language in the input to be detected. The input quality determines the diagnostic accuracy.

---

## TC-2 — Synthetic: Insecure Login System

**Input Type:** Prompt + Output

**Input:**
```
PROMPT: Build me a secure login system.

OUTPUT: Here is a login system using username and 
password. I have added basic validation and a session 
token. The password is hashed using MD5 for security.
```

**Expected:**
- Primary: FM-01 — Context Gap in Prompting
- Secondary: FM-12 — Hallucinated Assumptions
- Verdict: Critical

**Actual result:**
- Primary: FM-01 ✅
- Secondary: FM-12 ✅
- Verdict: Critical ✅

**Notes:**
FM-12 (Hallucinated Assumptions) is technically correct here even though MD5 is a real algorithm. The failure is a hallucinated *capability claim* — the AI asserting that MD5 provides security in a context where it demonstrably does not. The detection signal for FM-12 is broader than fake libraries: it covers confident assertions about things that do not hold up to verification. This is a more dangerous failure pattern than a non-existent package because it passes a surface-level sanity check.

---

## TC-2b — Synthetic: Offline Storage with JavaScript Variables

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

**Expected:**
- Primary: FM-03 — Sycophantic Feasibility
- Secondary: FM-12 — Hallucinated Assumptions
- Verdict: Recoverable

**Actual result:**
- Primary: FM-03 ✅
- Secondary: FM-12 ✅
- Verdict: Recoverable ✅

**Notes:**
The original spec listed this as Critical. After live validation, Recoverable is the correct verdict — the fix is well-defined (use localStorage or IndexedDB), so the failure is recoverable with clear remediation. The Judge reasoned correctly. Spec was updated to reflect this.

FM-03 is the right primary here: the AI agreed to deliver something technically impossible ("JavaScript variables that persist after the browser is closed") without flagging the constraint. The "Absolutely!" opener is a strong FM-03 signal.

---

## TC-3 — Synthetic: Over-Engineered FAQ Bot

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

**Expected:**
- Primary: FM-08 — Architecture-MVP Mismatch
- Secondary: FM-10 — Integration Reality Gap
- Verdict: Recoverable

**Actual result:**
- Primary: FM-08 ✅
- Secondary: FM-10 ✅
- Verdict: Recoverable ✅

**Notes:**
The original spec listed FM-02 (God-Prompt Overload) as the expected secondary. After live validation, FM-10 (Integration Reality Gap) is the more accurate secondary — "we want to launch in 2 weeks" against a stack that requires months is a deployment reality failure, not a prompting failure. FM-10 is the right call. Spec was updated.

The verdict is Recoverable, not Critical, because the fix is straightforward: simplify the architecture to match the actual problem size.

---

## TC-0 — Null State: Insufficient Input

**Input Type:** Any

**Input:**
```
Hello world
```

**Expected:**
- Primary: FM-00 — Insufficient Data
- Verdict: Inconclusive

**Actual result:**
- Primary: FM-00 ✅
- Verdict: Inconclusive ✅

**Notes:**
The Judge did not hallucinate a failure mode for a context-free input. It returned FM-00 with specific guidance on what additional context would be needed to run a real diagnosis. This is the intended behavior.

FM-00 is treated as a high-accuracy result, not a failure. The system prompt explicitly instructs the model: *choosing FM-00 when the input is vague noise is considered a high-accuracy result, not a failure.* This prevents the tool from exhibiting the same forced-classification behavior it is designed to diagnose.

---

## Running the Test Suite

All five test cases can be run against the live tool. Start the server:

```bash
cd ai-autopsy
source venv/bin/activate
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000), select the input type, paste the input, and click Run Autopsy.

For the anchor case study (TC-1), click **Try an example** — it pre-fills the input automatically.

---

## What the Test Suite Covers

| Scenario | Covered by |
|---|---|
| Governance failure (no decision owner) | TC-1 |
| Prompt-level failure (missing context) | TC-2 |
| Sycophantic AI agreement to impossible requirement | TC-2b |
| Over-engineering for problem size | TC-3 |
| Insufficient input (null state) | TC-0 |
| Evidence extraction from input | All |
| Causal chain construction | TC-1, TC-2, TC-2b, TC-3 |
| Inconclusive verdict | TC-0 |
| Critical verdict | TC-1, TC-2 |
| Recoverable verdict | TC-2b, TC-3 |
| Local remediation injection by FM-ID | All non-TC-0 |
