# AI Autopsy — Build Checklist
**Version:** 1.0
**Date:** April 4, 2026
**Depends on:** tech-spec.md v1.1

---

## Phase 1: Project Setup
- [ ] Confirm ai-autopsy/ folder structure matches 
      tech spec
- [ ] Create index.html (empty shell)
- [ ] Create data/taxonomy.json (empty shell)
- [ ] Create README.md (empty shell)
- [ ] Create .gitignore (node_modules, .env)
- [ ] Commit: "setup: project scaffold"

---

## Phase 2: Taxonomy JSON
- [ ] Define FM-00 null/insufficient_data state
- [ ] Define all 17 failure modes with full schema:
  - id, category, name, definition
  - detection_signal, remediation, example
- [ ] Validate JSON is well-formed (no syntax errors)
- [ ] Commit: "data: complete taxonomy.json"

---

## Phase 3: Frontend Shell
- [ ] Add marked.js via CDN in index.html
- [ ] Build dark mode terminal UI layout
- [ ] Add 3 input type toggle buttons
- [ ] Add textarea with character counter (max 5000)
- [ ] Add Run Autopsy button
- [ ] Add results panel (empty, hidden by default)
- [ ] Add API key settings input (localStorage)
- [ ] Add Clear Key button next to API key input
- [ ] Commit: "frontend: UI shell complete"

---

## Phase 4: Judge LLM Integration
- [ ] Load taxonomy.json at runtime
- [ ] Inject ONLY id, name, definition into system 
      prompt (not remediation or example — token 
      efficiency)
- [ ] Store remediation text locally — pull by FM-ID 
      after Judge returns classification
- [ ] Implement buildUserPrompt() function
- [ ] Implement Anthropic API call with correct headers
- [ ] Handle API key from localStorage
- [ ] Show spinner while API call runs
- [ ] Lock input during API call
- [ ] Commit: "feature: judge llm integration"

---

## Phase 5: Report Rendering
- [ ] Parse API response
- [ ] Render Markdown output via marked.js
- [ ] Inject remediation text from local taxonomy 
      by FM-ID returned from Judge
- [ ] Display report in results panel
- [ ] Add Copy to Clipboard button on report render
- [ ] Handle Inconclusive / FM-00 state gracefully
- [ ] Handle API errors gracefully (show error message)
- [ ] Commit: "feature: report rendering"

---

## Phase 6: Test Suite Validation
- [ ] Run Test Case 1 (Anchor — candidate screening)
  - [ ] Primary: FM-15 No Decision Ownership
  - [ ] Verdict: Critical
- [ ] Run Test Case 2 (Prompt failure — MD5 login)
  - [ ] Primary: FM-01 Context Gap in Prompting
  - [ ] Verdict: Critical
- [ ] Run Test Case 2b (Sycophancy — offline app)
  - [ ] Primary: FM-03 Sycophantic Feasibility
  - [ ] Verdict: Critical
- [ ] Run Test Case 3 (Architecture overkill — FAQ bot)
  - [ ] Primary: FM-08 Architecture-MVP Mismatch
  - [ ] Verdict: Critical
- [ ] Run FM-00 check (paste "Hello world")
  - [ ] Verdict: Inconclusive
- [ ] Commit: "test: all test cases validated"

---

## Phase 7: Hardening
- [ ] Confirm no API key appears in source or 
      git history
- [ ] Confirm .gitignore is correct
- [ ] Confirm input is capped at 5000 characters
- [ ] Confirm marked.js renders report correctly
- [ ] Confirm Copy to Clipboard works
- [ ] Confirm Clear Key button wipes localStorage
- [ ] Final commit: "hardening: pre-submission checks"

---

## Phase 8: Submission Prep
- [ ] Write README.md with setup instructions
- [ ] Push repo to GitHub (public)
- [ ] Zip docs/ folder for Devpost upload
- [ ] Record Loom demo video (3-5 minutes)
- [ ] Write Devpost story sections
- [ ] Submit on Devpost before April 29 @ 4pm CDT
