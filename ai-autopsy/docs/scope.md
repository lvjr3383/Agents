# AI Autopsy — Scope Document
**Version:** 1.0  
**Date:** April 4, 2026  
**Hackathon:** Devpost Spec-Driven Development Learning Hackathon

## What We're Building

AI Autopsy is a web-based diagnostic tool that analyzes 
failed or underperforming AI builds and produces a 
structured post-mortem report. You paste in a failed 
prompt + output, a vague spec, or a short code snippet — 
and the tool tells you exactly what went wrong upstream 
and why.

## The Problem It Solves

Most AI projects don't fail because the code broke. 
They fail because the thinking broke 3 steps earlier — 
a vague requirement, a wrong architectural assumption, 
a customer decision that overruled engineering judgment. 
By the time output looks wrong, the root cause is 
invisible. Teams either shrug and rebuild, or ask the 
same AI that failed to explain why it failed. Neither 
works.

There is no structured tool for diagnosing AI build 
failures at the decision layer. AI Autopsy fills that gap.

## Who It's For

- Solutions Architects and consultants deploying AI 
  in enterprise environments
- Developers who vibe-coded something and can't figure 
  out why it degraded
- Technical PMs who need to explain to stakeholders 
  why an AI project failed

## What It Does

User pastes one of three input types:
1. Failed prompt + AI output pair
2. Vague spec, PRD, or project description
3. Short code snippet + description of what went wrong

The tool runs it through a Judge LLM against a 
proprietary failure taxonomy (17 modes, 5 categories) 
and outputs a structured Post-Mortem Report containing:

- **Failure Classification** — which mode(s) triggered
- **Evidence** — direct quotes or logic gaps from the 
  input that prove the failure mode
- **Causal Chain** — how one failure created conditions 
  for the next
- **Reconstructed Spec** — what the spec should have been
- **Remediation Plan** — what to fix first and how

## What It Does NOT Do

- Does not fix or rewrite your code
- Does not ingest full repositories or large codebases
- Does not connect to GitHub, Jira, or any external system
- Does not make hiring, legal, or compliance decisions
- Does not store, log, or retain any user input
- Does not always find a problem — returns "Inconclusive" 
  when evidence is insufficient

## The Failure Taxonomy (Core IP)

17 failure modes across 5 categories:

| Category | Modes |
|---|---|
| Prompt & Framing | 4 modes |
| Spec & Planning | 3 modes |
| Architecture & Design | 3 modes |
| Context & Execution | 4 modes |
| Decision & Governance | 3 modes |

Each mode includes: definition, detection signal, 
remediation action, and one real-world example.

**Anchor Case Study:** An enterprise AI agent built to 
pre-screen job candidates by gathering answers to targeted 
questions about skill gaps. The customer refused 
deterministic question sets in favor of free-flowing LLM 
generation. The agent became inconsistent, unauditable, 
and was eventually abandoned. Root cause: 4 simultaneous 
failure modes — No Decision Ownership, Architecture-MVP 
Mismatch, Business Objective Misalignment, Shadow Logic.

## MVP Scope (4–6 Hour Build)

### In Scope
- Single-page web UI (paste input, get report)
- 3 input types: prompt/output, spec/PRD, code snippet
- Judge LLM running against taxonomy JSON
- Evidence-matched Post-Mortem Report in Markdown
- 3 test cases: 1 real (candidate screening agent), 
  2 synthetic (deliberately broken scenarios)

### Explicitly Out of Scope for v1
- Repository ingestion
- GitHub integration
- Multi-file uploads
- User accounts or session history
- Batch processing
- Mobile optimization
- Consultant vs Auditor tone modes

## Input Constraints

- Maximum input: 5,000 characters
- Plain text only — no file uploads in v1
- For long specs: paste executive summary or 
  key requirements section only

## Output Format

- All Post-Mortem Reports rendered as Markdown
- Designed to be copy-pasted directly into 
  GitHub Issues, Notion, or Slack

## Taxonomy JSON Design Principle

- Each failure mode match must cite specific evidence 
  from the user's input
- The Judge does not label — it proves
- Taxonomy is human-editable without touching 
  application code (pluggable, version-controlled)

## Tech Stack (Proposed)

- Frontend: Single HTML page or Streamlit
- Backend: Claude API (Judge LLM via Anthropic)
- Taxonomy: Local JSON file
- Deployment: Replit or local run

## Success Criteria

The tool is working when it correctly identifies the 
primary failure mode in all 3 test cases, explains the 
causal chain in plain English, and produces a remediation 
plan a non-technical stakeholder can act on.

## Hackathon Submission Artifacts

- docs/scope.md (this document)
- docs/prd.md
- docs/tech-spec.md
- docs/build-checklist.md
- Public GitHub repo with source + README
- Devpost write-up
