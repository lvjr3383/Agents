# AI Autopsy — Product Requirements Document (PRD)
**Version:** 1.0  
**Date:** April 4, 2026  
**Depends on:** scope.md v1.0

## User Stories

### US1 — The Vibe-Coder Recovery
As a developer whose AI output stopped making sense, 
I want to paste the prompt and response that failed 
so I can identify the specific upstream decision that 
caused the breakdown — not just "the AI got it wrong."

### US2 — The Architect's Audit
As a Solutions Architect reviewing a client's failed 
AI initiative, I want to paste their requirements or 
project description so I can produce a defensible, 
evidence-backed diagnosis of why the project is 
heading toward failure.

### US3 — The Post-Mortem
As a Tech Lead, I want a Markdown-formatted Root Cause 
Analysis report so I can attach it directly to a Jira 
ticket, GitHub Issue, or Slack thread without reformatting.

## Functional Requirements

### FR1: Multi-Type Input Processor
- System accepts three input modes selectable by the user:
  - Prompt + Output — a prompt and the AI response it produced
  - Spec / Requirements — a PRD, scope doc, or requirements description
  - Code Snippet — a code block plus a short description of what went wrong
- Input is plain text only
- Maximum 5,000 characters per submission
- Input is stateless — nothing is stored or logged

### FR2: Taxonomy-Driven Judge
- System passes input + selected type to a Judge LLM
- Judge evaluates input against all 17 modes in taxonomy.json
- Judge identifies one Primary failure mode and up to 
  two Secondary failure modes
- Judge must not force a classification — if evidence 
  is insufficient, returns Inconclusive state

### FR3: Evidence Extraction
- For every identified failure mode, the Judge must 
  extract a direct quote or specific logic gap from 
  the input as supporting evidence
- The Judge does not label — it proves
- No failure mode is reported without evidence

### FR4: Causal Chain Construction
- Where multiple failure modes are identified, the 
  Judge must articulate how the primary failure created 
  conditions for the secondary failure(s)
- Output is a plain-English causal chain, not a list
- Prompt the Judge to look for dependency: "Because X 
  failed, it created conditions for Y"

### FR5: Markdown Report Generation
- Output follows a standardized Autopsy Report template:

  ## AI Autopsy Report
  
  ### Input Type
  [Prompt+Output / Spec / Code Snippet]
  
  ### Primary Failure Mode
  [Mode Name] — [One-line definition]
  
  ### Evidence
  "[Direct quote or logic gap from input]"
  
  ### Secondary Failure Mode(s)
  [If applicable]
  
  ### Causal Chain
  [Plain English: how one failure led to the next]
  
  ### Reconstructed Spec
  [What the spec should have said]
  
  ### Remediation Plan
  1. [First fix]
  2. [Second fix]
  3. [Third fix]
  
  ### Verdict
  [Inconclusive / Recoverable / Critical]

## Acceptance Criteria

### AC1 — Anchor Case Study
Given the candidate screening agent case study as input 
(Spec/Requirements type), the tool must identify 
No Decision Ownership as the primary failure mode 
with evidence extracted from the input.

### AC2 — Valid Markdown Output
The Post-Mortem Report must render correctly in any 
standard Markdown viewer including GitHub, Notion, 
and Slack without modification.

### AC3 — Inconclusive State
If the user pastes input with insufficient diagnostic 
signal (e.g., "Hello world" or a single sentence), 
the tool must return an Inconclusive verdict rather 
than hallucinating a failure mode.

### AC4 — Response Time
End-to-end analysis (paste → report) must complete 
in under 15 seconds under normal conditions.

### AC5 — Synthetic Test Cases
Tool must correctly identify the primary failure mode 
in both synthetic test cases (one prompt/output failure, 
one architecture failure) with evidence cited.

## Non-Functional Requirements

### NFR1 — Determinism
The Judge prompt must be structured to produce 
consistent failure mode classification for the same 
input across multiple runs. No random drift in 
primary failure identification.

### NFR2 — Pluggable Taxonomy
taxonomy.json must be human-editable without touching 
application code. Failure modes can be added, modified, 
or versioned independently of the application.

### NFR3 — Stateless / No Data Retention
No user input is stored, logged, or transmitted beyond 
the single API call required for analysis. Each 
analysis session is fully independent.

## UI Requirements

- Dark mode terminal aesthetic (matches Autopsy theme)
- Single page — no navigation, no login
- Input area: large text box with 3 toggle buttons 
  for input type selection
- "Run Autopsy" as the single primary action
- Results rendered as formatted Markdown below or 
  beside the input area
- Character counter visible on input area (max 5,000)

## Out of Scope (Confirmed)

- Consultant vs Auditor tone modes (v2)
- Failure mode confidence scoring (v2)
- Repo or file ingestion (v2)
- GitHub / Jira integration (v2)
- User accounts or history (v2)
- Batch processing (v2)
- Mobile optimization (v2)
