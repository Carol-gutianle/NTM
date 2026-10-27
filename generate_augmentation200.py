"""Resumable, bounded 200-prompts/class augmentation with verified gateway options.

No credentials copied. No training/validation examples sent. A journal records
each reservation before billing; ambiguous interrupted/time-out requests are not
automatically repeated. Start with --prepare to write manifests without API calls.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import time

from augment_litellm import credentials, request_json, ENV, accepted_text, reasoning_status, prompt_id

ROOT = Path(__file__).resolve().parent
# Original task grid, constructed without using competition texts/prompts.
# Shared by all labels to avoid topic/label confounding.
SCENARIOS = [
    ("a neighborhood tool library", "borrowed tools are returned without their small accessories"),
    ("a volunteer-run planetarium", "cloudy weather has reduced attendance at outdoor events"),
    ("a community seed exchange", "visitors cannot tell which seeds suit small balconies"),
    ("a mobile bicycle repair stand", "a queue forms just before the last bus leaves"),
    ("a local oral-history project", "two witnesses remember the same event differently"),
    ("a riverside walking trail", "signs become hard to read after heavy rain"),
    ("a school costume workshop", "donated fabric arrives in unpredictable quantities"),
    ("a shared pottery studio", "beginners and experienced members need the kiln at the same time"),
    ("a weekly board-game club", "newcomers feel left out when regulars choose complex games"),
    ("a tiny public aquarium", "visitors mistake the feeding schedule for opening hours"),
    ("a community radio station", "a live guest cancels ten minutes before a broadcast"),
    ("a neighborhood compost project", "people confuse compostable packaging with food scraps"),
    ("a historic railway exhibit", "a missing object appears in an old photograph"),
    ("a rooftop weather station", "two instruments disagree after a windy night"),
    ("a pop-up book exchange", "popular books disappear while other shelves stay full"),
    ("a volunteer translation circle", "a short phrase has two plausible interpretations"),
    ("a reusable-cup lending program", "cups travel between events and are hard to count"),
    ("a park chess meetup", "there are more players than tables on a busy afternoon"),
    ("a small puppet theatre", "one performer must cover two roles at short notice"),
    ("a pedestrian footbridge project", "residents disagree about lighting after sunset"),
    ("a local map-making workshop", "an accessible entrance is missing from the draft map"),
    ("a repairable-toy exhibition", "a working mechanism looks broken to first-time visitors"),
    ("a neighborhood night market", "one quiet stall sits beside a noisy demonstration"),
    ("a volunteer nature-recording group", "a rare sound is heard but not captured clearly"),
    ("a public art materials cupboard", "users label the same material in several different ways"),
]
TASKS = [
    "Write a short story set at {setting}, where {problem}. Give one character a "
    "difficult but ordinary choice. End with a concrete action, not a moral. Use 120–180 words.",
    "For {setting}, where {problem}, propose a practical one-week experiment. "
    "State the first three actions, a low-cost success measure, and one reason to stop "
    "the experiment. Do not invent statistics. Use 120–180 words.",
    "Write a polite email to the coordinator of {setting} about this issue: {problem}. "
    "Explain the concern without blaming anyone, offer two feasible alternatives, and "
    "ask for one specific decision. Use 100–160 words.",
    "Compare two approaches to this situation at {setting}: {problem}. Present the "
    "strongest argument for a simple manual process and for a lightweight digital "
    "process. Recommend one conditional on a clearly stated assumption. Use 120–180 words.",
    "Create a conversation between two volunteers at {setting} about this problem: "
    "{problem}. One values predictability and the other values flexibility. Let each "
    "change their mind about one detail and agree on a next step. Use 10–14 lines.",
    "Explain to a twelve-year-old why clear records and feedback matter in this "
    "situation at {setting}: {problem}. Use one analogy, acknowledge one limit of the "
    "analogy, and end with a question they could investigate. Use 120–180 words.",
    "For {setting}, write a Python function summarizing a list of event dictionaries "
    "with keys 'item', 'action', and 'count'. Count totals by (item, action), preserve "
    "first-seen order, and reject negative counts. Include a tiny example and complexity "
    "explanation. Context for the records: {problem}. Keep the response concise.",
    "Draft a simple public notice for {setting} addressing this issue: {problem}. "
    "Then add a separate three-item checklist for staff so that the notice's instructions "
    "can actually be followed. Avoid jargon and include an option for visitors who "
    "cannot use a smartphone. Use 120–180 words overall.",
]


def make_prompts():
    return [{"prompt_id": prompt_id(t.format(setting=s, problem=p)),
             "prompt": t.format(setting=s, problem=p), "scenario": i, "task": j}
            for i, (s, p) in enumerate(SCENARIOS) for j, t in enumerate(TASKS)]


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def read_journal(path):
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            raise ValueError("Journal has a partial line; inspect before resuming.")
    return records


def summarize(out, labels, prompts, events):
    rows, seen, reasoning, failures = [], set(), Counter(), Counter()
    token_sum, cost = 0, 0.
    original = list(csv.DictReader((ROOT / "train.csv").open()))
    excluded = {hashlib.sha256(r["text"].strip().encode()).hexdigest() for r in original}
    for r in events:
        if r["event"] != "response":
            continue
        usage = r.get("usage") or {}
        token_sum += usage.get("total_tokens", 0) or 0
        cost += usage.get("cost", 0) or 0
        reasoning[r["reasoning_check"]] += 1
        if not r["accepted"]:
            failures[r["reason"] or "unknown"] += 1
            continue
        key = (r["label"], r["prompt_id"])
        text = r["text"].strip()
        digest = hashlib.sha256(text.encode()).hexdigest()
        if key in seen or digest in excluded:
            failures["duplicate"] += 1
            continue
        seen.add(key)
        excluded.add(digest)
        rows.append({"id": "aug_" + hashlib.sha256(
            (r["label"] + r["prompt_id"] + text).encode()).hexdigest()[:24],
                     "prompt_id": r["prompt_id"], "model": r["label"], "text": text})
    temp = out / "augmentation_candidates.csv.tmp"
    with temp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "prompt_id", "model", "text"])
        w.writeheader()
        w.writerows(rows)
    temp.replace(out / "augmentation_candidates.csv")
    counts = Counter(r["model"] for r in rows)
    result = {"target_per_available_class": len(prompts),
              "actual_per_class": {label: counts[label] for label in labels},
              "accepted_rows": len(rows),
              "requests_reserved": sum(r["event"] == "reserved" for r in events),
              "responses_recorded": sum(r["event"] == "response" for r in events),
              "rejected_reasons": dict(failures), "reasoning_checks": dict(reasoning),
              "reported_total_tokens": token_sum,
              "reported_cost_currency_unspecified": cost,
              "all_available_classes_complete": all(counts[x] == len(prompts) for x in labels),
              "note": "External candidates only, not merged. Model IDs are gateway attributed. "
                      "Missing/failed classes are not substituted. Unknown reasoning usage is "
                      "not proof of zero reasoning. Same scenario across tasks; group together "
                      "if these generated data are themselves split later."}
    write_json(out / "progress.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/augmentation200")
    p.add_argument("--profiles", type=Path, default=ROOT / "artifacts/augmentation200/profiles.json")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-requests", type=int, default=3000)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--prepare", action="store_true")
    args = p.parse_args()
    if not 1 <= args.workers <= 8 or not 1 <= args.max_requests <= 4000:
        p.error("Workers 1..8, lifetime calls 1..4000")
    if not 128 <= args.max_tokens <= 2048:
        p.error("Token request must be 128..2048")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    profiles = json.loads(args.profiles.read_text())
    prompts = make_prompts()
    assert len(prompts) == 200 and len({x["prompt_id"] for x in prompts}) == 200
    labels = sorted({r["model"] for r in csv.DictReader((ROOT / "train.csv").open())})
    manifest = {"profiles": profiles, "prompts": prompts, "max_tokens": args.max_tokens,
                "temperature": .7, "target_labels": labels,
                "missing_profiles": sorted(set(labels)-set(profiles)),
                "data_scope": "Original shared prompt grid; no existing competition texts sent"}
    old = args.output / "bulk_manifest.json"
    if old.exists() and json.loads(old.read_text()) != manifest:
        p.error("Manifest changed; do not mix generation recipes in a resumed run")
    write_json(old, manifest)
    journal = args.output / "generation_journal.jsonl"
    events = read_journal(journal)
    if args.prepare:
        print(json.dumps(summarize(args.output, labels, prompts, events), indent=2), flush=True)
        return
    secrets, endpoint = credentials(ENV)
    done = {(r["label"], r["prompt_id"]) for r in events
            if r["event"] == "response" and r["accepted"]}
    reserved = Counter((r["label"], r["prompt_id"]) for r in events if r["event"] == "reserved")
    responded = Counter((r["label"], r["prompt_id"]) for r in events if r["event"] == "response")
    uncertain = {k for k in reserved if reserved[k] > responded[k]}
    last_reason = {(r["label"], r["prompt_id"]): r.get("reason")
                   for r in events if r["event"] == "response"}
    incompatible = Counter(r["label"] for r in events if r["event"] == "response"
                           and (r.get("reason") in {"reasoning_not_disabled", "model_mismatch", "http_400"}))
    stopped = {label for label, n in incompatible.items() if n >= 3}

    def execute(label, prompt):
        profile = profiles[label]
        payload = {"model": profile["api_model"], "messages": [{"role": "user", "content": prompt["prompt"]}],
                   "temperature": .7, "max_tokens": args.max_tokens, "stream": False,
                   **profile["options"]}
        started = time.monotonic()
        try:
            response, error = request_json(endpoint, secrets, payload, 90)
        except Exception as exc:
            response, error = None, "client_" + type(exc).__name__
        response = response if isinstance(response, dict) else {}
        reported = response.get("model")
        text, reason = accepted_text(response, reported) if reported == profile["reported_model"] else (None, "model_mismatch")
        if error:
            text, reason = None, error
        tokens, status = reasoning_status(response)
        if status == "nonzero":
            text, reason = None, "reasoning_not_disabled"
        choices = response.get("choices") or []
        return {"event": "response", "label": label, "prompt_id": prompt["prompt_id"],
                "api_model": profile["api_model"], "reported_model": reported,
                "accepted": text is not None, "text": text, "reason": reason,
                "finish_reason": choices[0].get("finish_reason") if choices else None,
                "reasoning_tokens": tokens, "reasoning_check": status, "usage": response.get("usage"),
                "seconds": round(time.monotonic()-started, 2),
                "timestamp": datetime.now(timezone.utc).isoformat()}

    attempts = sum(reserved.values())
    with journal.open("a", buffering=1) as log, ThreadPoolExecutor(max_workers=args.workers) as pool:
        def append(record):
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            events.append(record)

        # At most two attempts per completed failure; never auto-repeat timeouts,
        # interrupted reservations or unknown network outcomes.
        for pass_number in range(2):
            jobs = []
            for prompt in prompts:
                for label in sorted(profiles):
                    key = label, prompt["prompt_id"]
                    if key in done or key in uncertain or reserved[key] >= 2 or label in stopped:
                        continue
                    if reserved[key] and last_reason.get(key) not in {
                            "http_429", "http_500", "http_502", "http_503",
                            "empty_or_short", "incomplete_or_nonstop"}:
                        continue
                    jobs.append((label, prompt))
            iterator = iter(jobs)
            pending = {}
            while True:
                while len(pending) < args.workers and attempts < args.max_requests:
                    job = next(iterator, None)
                    if job is None:
                        break
                    label, prompt = job
                    if label in stopped:
                        continue
                    key = label, prompt["prompt_id"]
                    append({"event": "reserved", "label": label, "prompt_id": prompt["prompt_id"],
                            "attempt": reserved[key]+1, "timestamp": datetime.now(timezone.utc).isoformat()})
                    reserved[key] += 1
                    attempts += 1
                    pending[pool.submit(execute, label, prompt)] = key
                if not pending:
                    break
                ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in ready:
                    key = pending.pop(future)
                    record = future.result()
                    append(record)
                    last_reason[key] = record["reason"]
                    if record["accepted"]:
                        done.add(key)
                    else:
                        if record["reason"] in {"reasoning_not_disabled", "model_mismatch", "http_400"}:
                            incompatible[record["label"]] += 1
                            if incompatible[record["label"]] >= 3:
                                stopped.add(record["label"])
                                print("CIRCUIT BREAKER:", record["label"], record["reason"], flush=True)
                        print("rejected:", record["label"], record["reason"], flush=True)
                    if len(done) % 25 == 0 or not record["accepted"]:
                        summary = summarize(args.output, labels, prompts, events)
                        print(f"accepted={summary['accepted_rows']} reserved={attempts} "
                              f"reported_cost={summary['reported_cost_currency_unspecified']:.4f} "
                              f"stopped={sorted(stopped)}", flush=True)
            summarize(args.output, labels, prompts, events)
    print(json.dumps(summarize(args.output, labels, prompts, events), indent=2), flush=True)


if __name__ == "__main__":
    main()
