#!/usr/bin/env python3
import os, json, sys, time, argparse, re
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "Data")
PROFILE_FP = os.path.join(DATA_DIR, "profile.json")
TOP_JOBS_FP = os.path.join(DATA_DIR, "top_jobs.json")
DREAM_JOBS_FP = os.path.join(DATA_DIR, "dream_jobs.json")

def read_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def divider(label: str = "") -> None:
    print("\n" + ("-"*12) + (f" {label} " if label else "") + ("-"*12))

def slow_print(text: str, delay: float = 0.02) -> None:
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print()

def tokens(s: str) -> List[str]:
    return re.findall(r"[A-Za-z]+", s.lower())

def tool_load_profile(_: Dict[str, Any]) -> Dict[str, Any]:
    return {"profile": read_json(PROFILE_FP)}

def tool_load_top_jobs(_: Dict[str, Any]) -> Dict[str, Any]:
    return {"jobs": read_json(TOP_JOBS_FP)}

def tool_load_dream_jobs(_: Dict[str, Any]) -> Dict[str, Any]:
    return {"jobs": read_json(DREAM_JOBS_FP)}

SENIOR_MARKERS = {
    "principal": 1.0, "architect": 1.0, "director": 0.95,
    "lead": 0.85, "senior": 0.9, "manager": 0.8, "consultant": 0.75
}
CONSULTING_CO = ["deloitte","ibm","boston consulting group","bcg","accenture","pwc","ey","kpmg"]
WEIGHTS = dict(title=0.30, skills=0.45, location=0.10, seniority=0.10, company=0.05)

def tool_rank_jobs(payload: Dict[str, Any]) -> Dict[str, Any]:
    signals = payload["signals"]
    jobs = payload["jobs"]
    threshold = payload.get("threshold", 0.0)
    top_n = payload.get("top_n", None)
    kw = set(signals.get("keywords", []))
    flat_skills = set(signals.get("flat_skills", []))
    titles_positive = [t.lower() for t in signals.get("titles_positive", [])]
    preferred_locs = [p.lower() for p in signals.get("preferred_locations", [])]
    seniority_hint = signals.get("seniority","")
    def score(job: Dict[str, Any]) -> Tuple[float, Dict[str,float], List[str]]:
        title = job.get("title","")
        loc = (job.get("location","") + " " + job.get("work_model","")).lower()
        company = job.get("company","").lower()
        basis = [b.lower() for b in job.get("match_basis", [])]
        ttl = " ".join(tokens(title))
        exact = any(pt.replace(" ","") in ttl.replace(" ","") for pt in titles_positive)
        overlap = sum(1 for t in tokens(title) if any(p in t for p in tokens(" ".join(titles_positive))) or t in kw)
        title_s = 1.0 if exact else min(1.0, 0.15*overlap)
        hits = [b for b in basis if b in flat_skills or b in kw or 'salesforce' in b or 'crm' in b or 'sap' in b]
        skills_s = min(1.0, 0.25*len(hits))
        if any(x in b for x in ['salesforce', 'crm', 'sap'] for b in basis):
            skills_s += 0.3
        if any(x in b for x in ['ai', 'genai', 'machine learning'] for b in basis):
            skills_s += 0.2
        skills_s = min(1.0, skills_s)
        location_s = 1.0 if ("remote" in loc or any(p in loc for p in preferred_locs)) else 0.5
        marker = 0.0
        for m,v in SENIOR_MARKERS.items():
            if m in title.lower(): marker = max(marker, v)
        seniority_s = marker if marker>0 else (0.6 if seniority_hint else 0.5)
        company_s = 1.0 if any(c in company for c in CONSULTING_CO) else 0.6
        total = (WEIGHTS["title"]*title_s + WEIGHTS["skills"]*skills_s + WEIGHTS["location"]*location_s + WEIGHTS["seniority"]*seniority_s + WEIGHTS["company"]*company_s) * 100.0
        badges = []
        if title_s>=0.8: badges.append("Title match")
        if skills_s>=0.5: badges.append("Skills aligned")
        if location_s>=0.9: badges.append("Preferred location/remote")
        if seniority_s>=0.8: badges.append("Seniority fit")
        if company_s>=0.9: badges.append("Consulting fit")
        if not badges and hits: badges.extend(hits[:2])
        comps = dict(title=round(title_s*100,1), skills=round(skills_s*100,1), location=round(location_s*100,1), seniority=round(seniority_s*100,1), company=round(company_s*100,1))
        return round(total,1), comps, badges
    def rank(bucket: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = []
        for j in bucket:
            total, comps, badges = score(j)
            if total >= threshold:
                rows.append({"job": j, "score": total, "components": comps, "badges": badges})
        rows.sort(key=lambda r: r["score"], reverse=True)
        return rows[:top_n] if top_n else rows
    return {
        "ranked_local": rank(jobs.get("top_local_jobs", [])),
        "ranked_other": rank(jobs.get("top_other_jobs", [])),
        "dream_ranked_local": rank(jobs.get("dream_local_jobs", [])) if "dream_local_jobs" in jobs else [],
        "dream_ranked_other": rank(jobs.get("dream_other_jobs", [])) if "dream_other_jobs" in jobs else []
    }

TOOLS = {"load_profile": tool_load_profile, "load_top_jobs": tool_load_top_jobs, "load_dream_jobs": tool_load_dream_jobs, "rank_jobs": tool_rank_jobs}

class Message:
    def __init__(self, intent: str, payload: Optional[Dict[str,Any]]=None):
        self.intent = intent
        self.payload = payload or {}

class Agent:
    NAME = "agent"
    CAPABILITIES: List[str] = []
    def handle(self, msg: Message, bb: Dict[str,Any]) -> Optional[Message]:
        raise NotImplementedError

class Jobby(Agent):
    NAME = "Jobby"
    CAPABILITIES = ["intro_consent", "show_ready_prompt", "dream_consent", "job_selection", "display_rankings", "closing"]
    def handle(self, msg: Message, bb: Dict[str,Any]) -> Optional[Message]:
        if msg.intent == "intro_consent":
            divider("Jobby")
            profile = read_json(PROFILE_FP)
            owner_name = profile.get("owner", {}).get("full_name", "Jack")
            print(f"Welcome {owner_name} to JackedIn, I'm Jobby, your AI-powered career assistant.")
            print("I'll analyze your LinkedIn profile to deliver tailored job matches based on your current and past work experience and skills.")
            print("May I proceed?")
            ans = input("Proceed? (yes/no): ").strip().lower()
            bb["consent"] = ans in {"y","yes"}
            return Message("consent_result")
        if msg.intent == "show_ready_prompt":
            if not bb.get("signals"):
                print("No signals yet; something went wrong.")
                return None
            ans = input("\nProfile scan complete. Job matches are ready. Ready to see them? (yes/no): ").strip().lower()
            bb["show_matches"] = ans in {"y","yes"}
            return Message("ready_result")
        if msg.intent == "dream_consent":
            ans = input("\nThe adventure isn't over yet! Want to explore exciting future possibilities based on your educational pursuits and certifications? (yes/no): ").strip().lower()
            bb["show_dream"] = ans in {"y","yes"}
            return Message("dream_result")
        if msg.intent == "job_selection":
            ans = input("\nExploring potential opportunities is thrilling! I can assess how you fit these roles-want to try? (yes/no): ").strip().lower()
            bb["assess_job"] = ans in {"y","yes"}
            if bb["assess_job"]:
                print("Please enter the job ID (e.g., J1, D1) from Top Jobs or Dream Jobs to assess:")
                job_id = input().strip().upper()
                bb["selected_job_id"] = job_id
                bb["assessed_jobs"] = bb.get("assessed_jobs", []) + [job_id]
                return Message("assess_job")
            return Message("closing")
        if msg.intent == "display_rankings":
            if not (bb.get("top_ranked") or bb.get("dream_ranked")):
                print("No rankings available; something went wrong.")
                return None
            self._render_rankings(bb)
            return None
        if msg.intent == "closing":
            divider("Jobby Closing")
            assessed_jobs = bb.get("assessed_jobs", [])
            if assessed_jobs:
                print("Based on your assessments, here are upskill recommendations to boost your fit:")
                print("- Technical: Earn Salesforce Einstein AI cert, master LLM prompting on Coursera, deepen Google Cloud AI skills.")
                print("- Non-technical: Enhance leadership via University of Michigan's Leading People course, improve project management with Google's PM cert, strengthen stakeholder engagement on Coursera.")
            else:
                print("No jobs assessed, but you can explore more opportunities!")
            print("Come back anytime-JackedIn will have fresh job recommendations and assessments to elevate your career!")
            return None
        return None
    def _render_rankings(self, bb: Dict[str,Any]) -> None:
        def render_bucket(title: str, rows: List[Dict[str,Any]]):
            divider(f"{title} ({len(rows)})")
            if not rows:
                print("No jobs passed the threshold.")
                return
            for i, r in enumerate(rows, 1):
                j = r["job"]
                verified = "✓ verified" if "verification" in j.get("title","").lower() else ""
                print(f"{i}. {j['job_id']} | {j['company']} - {j['title']} (Fit {r['score']:.1f}%)" + (f" [{verified}]" if verified else ""))
                print(f" Location: {j.get('location','n/a')} | Work: {j.get('work_model','n/a')}")
                print(f" Why: {', '.join(r['badges']) if r['badges'] else 'General fit'}")
        if bb.get("display_type") == "top":
            render_bucket("Top Local Jobs", bb.get("top_ranked", {}).get("ranked_local", []))
            print()
            print()
            print()
            print()
            render_bucket("Top US Jobs", bb.get("top_ranked", {}).get("ranked_other", []))
        elif bb.get("display_type") == "dream":
            render_bucket("Dream Local Jobs", bb.get("dream_ranked", {}).get("dream_ranked_local", []))
            print()
            print()
            print()
            print()
            render_bucket("Dream US Jobs", bb.get("dream_ranked", {}).get("dream_ranked_other", []))

class Scanny(Agent):
    NAME = "Scanny"
    CAPABILITIES = ["scan_profile"]
    def handle(self, msg: Message, bb: Dict[str,Any]) -> Optional[Message]:
        if msg.intent != "scan_profile":
            return None
        divider("Scanny")
        slow_print("Scanning your profile for best job matches…", 0.02)
        time.sleep(0.3)
        profile = TOOLS["load_profile"]({})["profile"]
        owner = profile.get("owner", {})
        derived = profile.get("derived", {})
        skills = profile.get("skills", {})
        top_titles = [t for t in derived.get("target_titles_positive", []) if not t.lower().startswith("ai ") and not t.lower().startswith("head of genai")]
        flat = set()
        for v in skills.values():
            if isinstance(v, list): flat.update(s.lower() for s in v)
        city = owner.get("location", {}).get("city", "") or "Unknown"
        state = owner.get("location", {}).get("state", "") or "Unknown"
        signals = {
            "owner_name": owner.get("full_name","Unknown"),
            "seniority": (derived.get("seniority") or "").lower(),
            "titles_positive": top_titles,
            "keywords": [k.lower() for k in derived.get("keywords", []) if k.lower() not in ["ai strategy", "llm", "rag"]],
            "flat_skills": sorted(list(flat)),
            "preferred_locations": [f"{city}, {state}", "Remote", "United States"],
            "ideal_industries": derived.get("ideal_industries", []),
        }
        slow_print("Jobs closely related to your work experience are ready.", 0.02)
        bb["signals"] = signals
        return Message("scan_done")

class Topsy(Agent):
    NAME = "Topsy"
    CAPABILITIES = ["rank_jobs"]
    def handle(self, msg: Message, bb: Dict[str,Any]) -> Optional[Message]:
        if msg.intent != "rank_jobs":
            return None
        divider("Topsy")
        jobs = TOOLS["load_top_jobs"]({})["jobs"]
        ranked = TOOLS["rank_jobs"]({
            "signals": bb["signals"],
            "jobs": jobs,
            "threshold": bb.get("threshold", 0.0),
            "top_n": bb.get("top_n")
        })
        bb["top_ranked"] = ranked
        print("Recommendations ready ✓")
        return Message("rank_done")

class Dreamy(Agent):
    NAME = "Dreamy"
    CAPABILITIES = ["rank_dream_jobs"]
    def handle(self, msg: Message, bb: Dict[str,Any]) -> Optional[Message]:
        if msg.intent != "rank_dream_jobs":
            return None
        divider("Dreamy")
        jobs = TOOLS["load_dream_jobs"]({})["jobs"]
        ranked = TOOLS["rank_jobs"]({
            "signals": bb["signals"],
            "jobs": jobs,
            "threshold": bb.get("threshold", 0.0),
            "top_n": bb.get("top_n")
        })
        bb["dream_ranked"] = ranked
        print("Dream recommendations ready ✓")
        return Message("dream_rank_done")

class Fitty(Agent):
    NAME = "Fitty"
    CAPABILITIES = ["assess_job"]
    def handle(self, msg: Message, bb: Dict[str,Any]) -> Optional[Message]:
        if msg.intent != "assess_job":
            return None
        divider("Fitty")
        job_id = bb.get("selected_job_id")
        if not job_id:
            print("No job selected; something went wrong.")
            return Message("assess_done")
        top_jobs = TOOLS["load_top_jobs"]({})["jobs"]
        dream_jobs = TOOLS["load_dream_jobs"]({})["jobs"]
        all_jobs = top_jobs.get("top_local_jobs", []) + top_jobs.get("top_other_jobs", []) + dream_jobs.get("dream_local_jobs", []) + dream_jobs.get("dream_other_jobs", [])
        selected_job = next((j for j in all_jobs if j.get("job_id") == job_id), None)
        if not selected_job:
            print(f"Job {job_id} not found.")
            return Message("assess_done")
        profile = TOOLS["load_profile"]({})["profile"]
        experience = profile.get("experience", [])
        skills = profile.get("skills", {})
        certifications = profile.get("licenses_and_certifications", [])
        basis = selected_job.get("match_basis", [])
        good = []
        bad = []
        ugly = []
        flat_skills = set()
        for v in skills.values():
            if isinstance(v, list): flat_skills.update(s.lower() for s in v)
        flat_certs = [cert["name"].lower() for cert in certifications]
        flat_exp = [exp["description"].lower() for exp in experience]
        for b in basis:
            if any(b.lower() in s for s in flat_skills):
                good.append(f"Your {b.lower()} skills align strongly with the role's requirements, supported by recent work at {experience[0]['company'] if experience else 'your current employer'}.")
            elif any(b.lower() in c for c in flat_certs):
                good.append(f"Your {b.lower()} certification matches a core requirement for this role, boosting your fit.")
            elif any(b.lower() in e for e in flat_exp):
                good.append(f"Your experience in {b.lower()} from past roles at {experience[-1]['company'] if experience else 'a previous employer'} is highly relevant.")
            else:
                bad.append(f"Your profile lacks direct experience or certification in {b.lower()}, a core requirement for this role.")
        if len(good) > len(basis) * 0.7:
            ugly.append("Your profile is well-aligned with minimal gaps, making you a strong candidate for this role.")
        else:
            ugly.append("Significant skill or certification gaps may reduce your competitiveness for this position.")
        slow_print(f"Assessing your profile fit to {job_id}: {selected_job['company']} - {selected_job['title']}...")
        print("Good: " + (" ".join(good) if good else "No strong alignments found with the role's requirements."))
        print("Bad: " + (" ".join(bad) if bad else "No major gaps in required skills or experience."))
        print("Ugly: " + (" ".join(ugly) if ugly else "No critical issues; your profile is competitive."))
        ans = input("\nAssess another job? (job ID or no): ").strip().upper()
        if ans != "NO":
            bb["selected_job_id"] = ans
            return Message("assess_job")
        else:
            return Message("closing")

AGENTS = [Jobby(), Scanny(), Topsy(), Dreamy(), Fitty()]
CAPABILITY_MAP = {cap: a for a in AGENTS for cap in a.CAPABILITIES}

def dispatch(msg: Message, bb: Dict[str,Any]) -> Optional[Message]:
    agent = CAPABILITY_MAP.get(msg.intent)
    if not agent:
        print(f"[Dispatcher] No agent can handle intent: {msg.intent}")
        return None
    return agent.handle(msg, bb)

def run_flow(args: argparse.Namespace) -> None:
    for p in [PROFILE_FP, TOP_JOBS_FP, DREAM_JOBS_FP]:
        if not os.path.exists(p):
            divider("Setup")
            print("Missing required file:", p)
            sys.exit(1)
    bb = {
        "threshold": float(args.threshold) if args.threshold is not None else 0.0,
        "top_n": args.top,
        "no_explain": args.no_explain,
        "local_only": args.local_only,
        "other_only": args.other_only,
        "display_type": None,
        "assessed_jobs": [],
        "top_jobs_rendered": False
    }
    dispatch(Message("intro_consent"), bb)
    if not bb.get("consent"):
        print("No problem. Exiting.")
        return
    next_msg = dispatch(Message("scan_profile"), bb)
    if next_msg.intent != "scan_done" or not bb.get("signals"):
        print("Scan failed; something went wrong.")
        return
    dispatch(Message("show_ready_prompt"), bb)
    if not bb.get("show_matches"):
        dispatch(Message("closing"), bb)
        return
    dispatch(Message("rank_jobs"), bb)
    bb["display_type"] = "top"
    dispatch(Message("display_rankings"), bb)
    dispatch(Message("dream_consent"), bb)
    if not bb.get("show_dream"):
        dispatch(Message("closing"), bb)
        return
    dispatch(Message("rank_dream_jobs"), bb)
    bb["display_type"] = "dream"
    dispatch(Message("display_rankings"), bb)
    dispatch(Message("job_selection"), bb)
    if bb.get("assess_job"):
        next_msg = dispatch(Message("assess_job"), bb)
        while next_msg and next_msg.intent == "assess_job":
            next_msg = dispatch(Message("assess_job"), bb)
    dispatch(Message("closing"), bb)

def parse_args(argv: List[str]) -> argparse.Namespace:
    import argparse
    p = argparse.ArgumentParser(description="JackedIn Agentic CLI (ranked recommendations with % and explainability)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--local-only", action="store_true", help="Only show Top Local Jobs")
    g.add_argument("--other-only", action="store_true", help="Only show Top US Jobs")
    p.add_argument("--top", type=int, help="Top N per bucket")
    p.add_argument("--no-explain", action="store_true", help="Hide component breakdowns")
    p.add_argument("--threshold", type=float, help="Only show jobs >= threshold (0-100)")
    return p.parse_args(argv)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    run_flow(args)