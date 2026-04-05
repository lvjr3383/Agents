# When AI Builds Fail, You're Asking the Wrong Question

*— and the tool we built to ask the right one*

---

Most AI post-mortems are useless.

Not because the teams doing them aren't smart. Because they do the post-mortem by asking the AI that failed to explain why it failed. The model gives a thoughtful, well-structured answer. Everyone nods. Nobody learns anything. The next project makes the same mistakes.

I've watched this pattern repeat for two decades — at IBM, at Deloitte, and now deploying AI agents at Salesforce. The failure is almost never in the code. By the time the output looks wrong, the actual root cause is three steps back: a vague requirement, a wrong architectural assumption, a customer decision that overruled engineering judgment with no accountability attached. The code was just following orders.

There is no structured tool for diagnosing AI build failures at the decision layer. So we built one.

---

## The Tool: AI Autopsy

AI Autopsy is a diagnostic tool that analyzes failed or underperforming AI builds and produces a structured post-mortem report.

You paste one of three input types:
- A failed prompt and AI output pair
- A vague spec, PRD, or requirements description
- A code snippet with a description of what went wrong

The tool runs it through a Judge LLM against a proprietary 17-mode failure taxonomy. It returns a structured Autopsy Report with a primary failure mode, supporting evidence quoted directly from your input, a causal chain, a reconstructed spec, and a verdict — Inconclusive, Recoverable, or Critical.

The critical design principle: **the Judge doesn't label — it proves.** No failure mode gets reported without a direct quote or specific logic gap from your input as evidence. If the input doesn't contain enough diagnostic signal, the tool returns Inconclusive rather than hallucinating a finding.

---

## The Taxonomy Is the Product

The application is 400 lines of Python and HTML. The taxonomy took weeks.

17 failure modes across 5 categories, built from real enterprise AI engagements — not textbook definitions. Each mode has a definition, a detection signal, a remediation action, and a real-world example.

| ID | Category | Failure Mode |
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
| FM-00 | null | Insufficient Data |

FM-00 deserves a note. The null state — when the input lacks enough signal to diagnose anything — was not an afterthought. It was a deliberate architectural decision. The Judge prompt explicitly tells the model: *choosing FM-00 when the input is vague noise is considered a high-accuracy result, not a failure.* This matters because forcing a classification on insufficient evidence is exactly the behavior we're trying to diagnose in other systems. We couldn't let the diagnostic tool exhibit the same failure.

---

## The Anchor Case Study

The taxonomy was built around one real project: an enterprise AI agent designed to pre-screen job candidates by gathering answers to targeted questions about skill gaps.

The engineering team recommended a deterministic question set — generated per job role at the time of posting, consistent and auditable. The customer rejected this. They wanted free-flowing dynamic question generation from a general LLM. The engineering team objected. The customer insisted. No binding architectural decision was made.

The agent launched. It became inconsistent across candidates. The questions it generated varied unpredictably. No recruiter could explain why one candidate was asked about Python and another about communication skills for the same role. The system was unauditable by design — the customer's design. The project was eventually abandoned.

Four failure modes were active simultaneously: No Decision Ownership, Architecture-MVP Mismatch, Business Objective Misalignment, and Shadow Logic. The root cause wasn't the model. It wasn't the code. It was the absence of a single person authorized to say: this is how we are building it, and this is not.

This case study became Test Case 1 in the validation suite.

---

## How We Built It: Spec First, Code Second

This project was built for the Devpost Spec-Driven Development Learning Hackathon, which required planning artifacts before code. We took that seriously.

Four documents before a single line of application code:

1. **scope.md** — what the tool does, who it's for, what it explicitly does NOT do, MVP constraints
2. **prd.md** — user stories, functional requirements, acceptance criteria, non-functional requirements
3. **tech-spec.md** — architecture decisions, taxonomy schema, system prompt design, test suite
4. **build-checklist.md** — 8 phases, every task tracked, nothing built out of order

The spec-driven approach paid off in a specific way: every architecture decision was made before touching code, which meant zero mid-build pivots on fundamentals. The one exception was moving from a browser-direct API call to a FastAPI proxy backend — a decision that improved security (API key server-side, never in the browser) and was simple to make because the rest of the spec was solid.

The planning artifacts are in the public repo. They're worth reading independently of the application.

### Architecture

```
Browser (index.html)
    ↓ POST /api/analyze
FastAPI (main.py)
    ↓ Anthropic API
Claude (Judge LLM)
    ↓ Markdown report
FastAPI → Browser → marked.js render
```

The taxonomy loads at server startup. The Judge receives only `id`, `name`, and `definition` from each failure mode — remediation text is stored locally and appended by FM-ID after the Judge returns its classification. Token efficiency and a clean separation between diagnosis and prescription.

---

## The Test That Surprised Us

The first version of Test Case 1 failed.

The input we fed the Judge was a clean spec description — the what of the project, not the governance failure that caused it to collapse. The Judge returned FM-01 (Context Gap in Prompting) as the primary. That's a valid finding. It's not the important one.

We rewrote the input to include the governance context explicitly — the engineering team's recommendation, the customer rejection, the absence of a binding decision, the project abandonment. With that input, the Judge returned FM-15 (No Decision Ownership) as primary, FM-17 (Shadow Logic) and FM-14 (Evaluation Blindness) as secondaries, and a causal chain that accurately described what happened.

**The lesson: the quality of the autopsy is directly proportional to the quality of the failure description.**

This is actually the most important thing we learned from building this tool. It's also true of every consulting engagement I've ever been on. Garbage in, garbage out. But structured failure context in, precise diagnosis out. The input shape matters as much as the taxonomy.

---

## What the Judge Gets Right That Most Tools Miss

**The MD5 moment.**

Test Case 2 was a prompt/output pair where the AI responded to "build me a secure login system" by implementing MD5 password hashing. The Judge correctly identified this as FM-12 (Hallucinated Assumptions) as the secondary failure mode.

MD5 is a real algorithm. The Judge wasn't catching a hallucinated library. It was catching a *hallucinated capability claim* — the AI asserting that MD5 provides security in a context where it demonstrably doesn't. The taxonomy definition says "AI invented a library, API, or integration that does not exist" — but the detection signal is broader: confident assertion of something that doesn't hold up to verification.

This is a more dangerous failure than a fake library because it passes a surface-level sanity check. A developer who doesn't know cryptography reads "hashed using MD5 for security" and ships it. The Judge caught it. That matters.

**The FM-00 behavior.**

We pasted "Hello world" as the test input. The Judge returned FM-00 (Insufficient Data), verdict Inconclusive, and provided specific guidance on what additional context would be needed to run a real diagnosis. It did not hallucinate a failure mode for a programming greeting. That's the behavior we designed for, and it held.

---

## What It Doesn't Do

This section matters as much as any feature description.

AI Autopsy does not fix or rewrite your code. It diagnoses the upstream decision that caused the code to be wrong in the first place.

It does not work on single sentences. Context-free inputs return Inconclusive, by design.

It does not connect to GitHub, Jira, or any external system. Input is plain text, maximum 5,000 characters. For long specs, paste the executive summary or the requirements section that describes the decision that went wrong.

It does not make hiring, legal, or compliance decisions. The candidate screening case study is an example of a *failed* use of AI in hiring, not a template for one.

It does not retain any input. The backend is stateless. Every session is independent.

Knowing what a tool doesn't do is how you know the builder thought clearly about what it does.

---

## Lessons Learned

**Spec-driven development is not overhead. It's the work.**

The four planning documents took longer to write than the application. The build went clean because of it. There were no debates mid-build about what the tool should do or how it should handle edge cases — those decisions were already made and documented. The checklist approach meant every phase had clear exit criteria. Phase 3 was complete when the console.log stub fired correctly in the browser. Phase 6 was complete when all five test cases passed.

**A diagnostic tool is only as good as its vocabulary.**

Generic taxonomies produce generic diagnoses. The 17-mode taxonomy built from enterprise consulting experience produced diagnoses that matched real outcomes — including the anchor case study, which was a real project with a real failure. The failure vocabulary is the product. Everything else is scaffolding.

**Input quality is a UX problem.**

The most important thing we can do for v2 is help users write better inputs. The tool is only as useful as the failure description it receives. A guided input mode — where the tool asks structured questions before running the diagnosis — would significantly improve output quality. That's a v2 feature, but it's the right next thing.

**The null state earns trust.**

Every system that never says "I don't know" is a system you can't trust. FM-00 was the right design decision. The inconclusive verdict is not a failure mode of the tool. It's evidence that the tool reasons rather than guesses.

---

## What's Next

**v2 priorities, in order:**

1. **Guided input mode** — structured prompts before diagnosis. "Describe the moment the project started going wrong." "Who made the final architecture decision?" Better inputs, better outputs.

2. **Batch analysis** — run multiple failed builds through the same taxonomy and surface patterns across a team or organization. Where is your organization failing repeatedly? That pattern is more valuable than any individual diagnosis.

3. **Taxonomy extensibility** — let teams add their own failure modes based on their organizational patterns. The JSON is already version-controlled and human-editable. The interface for extending it should be as clean as the rest of the tool.

4. **GitHub integration** — analyze PRs, issue threads, and commit histories as input types. The failure signal is often in the conversation around the code, not the code itself.

5. **A public failure library** — anonymized case studies mapped to the taxonomy, searchable by failure mode. This would be the most valuable thing we could put on the internet for AI practitioners.

---

## The Real Finding

We set out to build a diagnostic tool for failed AI builds. What we actually built is a structured vocabulary for talking about AI failure.

That's the thing the industry is missing. Not better models. Not more compute. A shared language for the ways AI projects go wrong at the decision layer — before the first token is generated.

The 17 failure modes are not comprehensive. They're a starting point. If you've worked on AI projects that failed, you've probably seen FM-06 (Uncontrolled Evolution) and FM-15 (No Decision Ownership) more than any others. If you work in enterprise, FM-10 (Integration Reality Gap) shows up constantly. If you vibe-code, FM-01 (Context Gap) and FM-13 (No Verification Loop) are your regulars.

The tool is live. The taxonomy is open. The repo is public.

If you've got a failed AI build sitting in a drawer, paste it in. The verdict might surprise you.

---

*AI Autopsy was built for the Devpost Spec-Driven Development Learning Hackathon. The full source, planning docs, and taxonomy are at [github.com/lvjr3383/Agents/tree/main/ai-autopsy](https://github.com/lvjr3383/Agents/tree/main/ai-autopsy).*

*Built with FastAPI, Claude (Anthropic), and a taxonomy that took longer to write than the code.*
