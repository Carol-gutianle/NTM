"""Bounded external-data pilot using the user-designated RubricLab gateway.

Never rewrites original answers with their old labels. Never sends train/val/test
texts or credentials to any host other than the configured gateway. Responses
are provisional gateway-attributed data, not independent provider verification.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

ROOT = Path(__file__).resolve().parent
ENV = Path(os.environ.get("RUBRICLAB_ENV", str(ROOT / ".env")))
# Explicit punctuation aliases, not fuzzy matching or version substitution.
# Backend identity still requires audit, as with all gateway IDs.
MODEL_ALIASES = {
    "claude-opus-4-6": "claude-opus-4.6",
    "claude-opus-4-8": "claude-opus-4.8",
}


def resolve_models(labels, advertised):
    return {label: label if label in advertised else MODEL_ALIASES[label]
            for label in labels
            if label in advertised or MODEL_ALIASES.get(label) in advertised}


# Original prompts, not adaptations of competition training/validation prompts.
PROMPTS = [
    "Write a 120–180 word story about a night-shift museum cleaner who discovers "
    "that a small model lighthouse flashes only when somebody tells the truth. "
    "Include a difficult choice and end with an action rather than an explanation.",
    "Explain to a curious teenager why two rooms at the same temperature can feel "
    "different when one has a stone floor and the other has a wooden floor. "
    "Use one everyday analogy and describe one safe comparison they could try. "
    "Keep your answer between 120 and 180 words.",
    "A neighborhood repair cafe has six volunteers, twelve broken lamps, and only "
    "two workbenches. Propose a practical plan for a two-hour event. Include intake, "
    "safety, how to handle an item that cannot be repaired, and one success measure. "
    "Use 120–180 words.",
    "Write a Python function that groups a list of (name, score) pairs by name, "
    "returning each name's highest score. Preserve the order in which names first "
    "appear and do not mutate the input. Include a short example with a tie and "
    "explain the time complexity. Keep the response concise.",
    "A town proposes replacing a small underused parking lot with a community "
    "garden. Give the strongest practical argument for each side, then recommend "
    "a reversible six-month experiment and two measurements that would guide a "
    "decision. Use 120–180 words; do not invent statistics.",
    "Write a polite email to a colleague explaining that a prototype demonstration "
    "must move from Tuesday to Friday because a sensor shipment is late. Offer an "
    "alternative activity for Tuesday and ask for one specific confirmation. "
    "Keep it between 100 and 150 words and avoid exaggerated apologies.",
]


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def credentials(path):
    # Parse only needed keys; do not execute .env as shell code.
    found = {}
    for line in path.read_text().splitlines():
        if line.startswith("export "):
            line = line[7:]
        key, sep, value = line.partition("=")
        key = key.strip()
        if sep and key in {"OPENAI_API_URL", "OPENAI_API_KEY", "LLM_COOKIE"}:
            parts = shlex.split(value, comments=True)
            if parts:
                found[key] = parts[0]
    if not found.get("OPENAI_API_KEY") or not found.get("OPENAI_API_URL"):
        raise ValueError("Required RubricLab API URL/key not configured")
    url = found["OPENAI_API_URL"].rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password or parsed.query:
        raise ValueError("Unsupported gateway URL structure")
    if not parsed.path.endswith("/chat/completions"):
        raise ValueError("Expected a configured chat/completions endpoint")
    return found, url


def request_json(url, secrets, payload=None, timeout=90):
    headers = {"Authorization": "Bearer " + secrets["OPENAI_API_KEY"],
               "Content-Type": "application/json"}
    if secrets.get("LLM_COOKIE"):
        headers["Cookie"] = secrets["LLM_COOKIE"]
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(url, data=data, headers=headers, method="GET" if data is None else "POST")
    # No retries: predictable billing and no duplicate generation after a timeout.
    try:
        with build_opener(NoRedirect()).open(req, timeout=timeout) as response:
            return json.loads(response.read()), None
    except HTTPError as exc:
        # Do not echo bodies: gateways can include credentials/request details.
        return None, f"http_{exc.code}"
    except URLError:
        return None, "network_error"
    except TimeoutError:
        return None, "timeout"
    except (ValueError, OSError):
        return None, "invalid_response_or_io"


def prompt_id(prompt):
    return "external_" + hashlib.sha256(prompt.encode()).hexdigest()[:20]


def generation_payload(label, prompt, max_tokens, reasoning_effort="none"):
    return {
        "model": label, "messages": [{"role": "user", "content": prompt}],
        "temperature": .7, "max_tokens": max_tokens, "stream": False,
        "reasoning_effort": reasoning_effort,
    }


def reasoning_status(response):
    """Unknown usage is not evidence that reasoning was disabled."""
    usage = response.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    count = details.get("reasoning_tokens")
    if count is None:
        count = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
    if not isinstance(count, (int, float)) or count < 0:
        return None, "unknown"
    return count, "nonzero" if count > 0 else "reported_zero"


def accepted_text(response, requested):
    if not isinstance(response, dict):
        return None, "invalid_response"
    if response.get("model") != requested:
        return None, "reported_model_mismatch_or_missing"
    choices = response.get("choices") or []
    if not choices:
        return None, "no_choices"
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        return None, "incomplete_or_nonstop"
    text = (choice.get("message") or {}).get("content")
    if not isinstance(text, str) or len(text.strip()) < 80:
        return None, "empty_or_short"
    return text.strip(), None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-file", type=Path, default=ENV)
    p.add_argument("--output", type=Path, default=ROOT / "artifacts" /
                   f"augmentation_pilot_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}")
    p.add_argument("--discover", action="store_true", help="GET model listings only; no generation")
    p.add_argument("--prompts", type=int, default=2, choices=range(1, len(PROMPTS)+1))
    p.add_argument("--max-requests", type=int, default=36)
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--reasoning-effort", choices=["none", "disable"], default="none",
                   help="Explicit no-reasoning request; gateway support must be verified. No silent fallback.")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--timeout", type=int, default=90)
    args = p.parse_args()
    if not 1 <= args.workers <= 4 or not 1 <= args.max_requests <= 108:
        p.error("Pilot limited to 1..4 workers and 1..108 calls")
    if not 64 <= args.max_tokens <= 2048 or not 1 <= args.timeout <= 180:
        p.error("Tokens must be 64..2048; timeout 1..180")
    secrets, endpoint = credentials(args.env_file)
    with (ROOT / "train.csv").open() as f:
        original = list(csv.DictReader(f))
    labels = sorted({r["model"] for r in original})
    existing_hashes = {hashlib.sha256(r["text"].strip().encode()).hexdigest() for r in original}
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Output nonempty; choose a new directory (never repeat/overwrite silently)")
    args.output.mkdir(parents=True, exist_ok=True)
    base = endpoint[:-len("/chat/completions")]
    parsed = urlsplit(base)
    urls = list(dict.fromkeys([base + "/models",
        urlunsplit((parsed.scheme, parsed.netloc, "/v1/models", "", ""))]))
    advertised, discoveries = set(), []
    for url in urls:
        response, error = request_json(url, secrets, timeout=min(args.timeout, 30))
        data = response.get("data", []) if isinstance(response, dict) else []
        names = [r.get("id") for r in data if isinstance(r, dict) and isinstance(r.get("id"), str)]
        advertised.update(names)
        discoveries.append({"route": urlsplit(url).path, "error": error, "model_count": len(names)})
        if names:
            break
    model_map = resolve_models(labels, advertised)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gateway_fingerprint": hashlib.sha256(endpoint.encode()).hexdigest(),
        "discoveries": discoveries,
        "available_target_labels": sorted(model_map),
        "missing_target_labels": sorted(set(labels) - set(model_map)),
        "label_to_api_model": model_map,
        "model_listing_available": bool(advertised),
        "requested_prompts": args.prompts, "max_requests": args.max_requests,
        "max_tokens_per_request": args.max_tokens, "temperature": .7,
        "reasoning_effort_requested": args.reasoning_effort,
        "note": "Provisional source attribution from gateway model IDs, NOT independent backend "
                "verification. No competition data sent. No merge into original train/test. "
                "Exact-model mismatches/truncated outputs quarantined. "
                "Human audit and competition external-data rule check required before use.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    if args.discover:
        return
    if not advertised:
        p.error("Model listing unavailable; refusing unverified generation requests.")
    selected = [label for label in labels if label in model_map]
    prompts = PROMPTS[:args.prompts]
    # Whole balanced grid, or fail: don't silently sample only early labels.
    jobs = [(label, prompt) for prompt in prompts for label in selected]
    if not jobs:
        p.error("No exact target model IDs advertised. No imitation/alias relabeling attempted.")
    if len(jobs) > args.max_requests:
        p.error(f"Balanced grid needs {len(jobs)} calls; limit is {args.max_requests}")
    (args.output / "prompts.json").write_text(json.dumps(
        [{"prompt_id": prompt_id(x), "prompt": x} for x in prompts], ensure_ascii=False, indent=2))

    def generate(job):
        label, prompt = job
        api_model = model_map[label]
        started = time.monotonic()
        response, error = request_json(endpoint, secrets, generation_payload(
            api_model, prompt, args.max_tokens, args.reasoning_effort), timeout=args.timeout)
        text, rejected = accepted_text(response, api_model) if error is None else (None, error)
        response = response if isinstance(response, dict) else {}
        reasoning_tokens, reasoning_check = reasoning_status(response)
        if rejected is None and reasoning_check == "nonzero":
            text, rejected = None, "reasoning_not_disabled"
        record = {"target_label": label, "requested_model": api_model, "reported_model": response.get("model"),
                  "prompt_id": prompt_id(prompt), "prompt": prompt,
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "seconds": round(time.monotonic()-started, 2),
                  "usage": response.get("usage"), "accepted": text is not None,
                  "reasoning_effort_requested": args.reasoning_effort,
                  "reasoning_tokens": reasoning_tokens, "reasoning_check": reasoning_check,
                  "reason": rejected}
        choices = response.get("choices") or []
        record["finish_reason"] = (choices[0].get("finish_reason")
                                   if choices and isinstance(choices[0], dict) else None)
        # Keep quarantined content for local inspection, never as labeled train CSV.
        if text is None:
            choices = response.get("choices") or []
            record["text"] = ((choices[0].get("message") or {}).get("content")
                              if choices and isinstance(choices[0], dict) else None)
        else:
            record["text"] = text
        return record

    rows, counts, seen = [], {}, set(existing_hashes)
    reported_tokens, reported_cost, over_limit = 0, 0., 0
    reasoning_checks = {"reported_zero": 0, "nonzero": 0, "unknown": 0}
    with (args.output / "responses.jsonl").open("w") as log, ThreadPoolExecutor(
            max_workers=args.workers) as pool:
        for record in pool.map(generate, jobs):
            reasoning_checks[record["reasoning_check"]] += 1
            usage = record.get("usage") or {}
            reported_tokens += usage.get("total_tokens", 0) or 0
            reported_cost += usage.get("cost", 0) or 0
            over_limit += int((usage.get("completion_tokens", 0) or 0) > args.max_tokens)
            text = record["text"]
            if record["accepted"]:
                digest = hashlib.sha256(text.encode()).hexdigest()
                if digest in seen:
                    record.update(accepted=False, reason="duplicate_text")
                else:
                    seen.add(digest)
                    uid = hashlib.sha256((record["target_label"] + record["prompt_id"] + text).encode()).hexdigest()[:24]
                    rows.append({"id": "aug_" + uid, "prompt_id": record["prompt_id"],
                                 "model": record["target_label"], "text": text})
            key = record["target_label"]
            counts[key] = counts.get(key, 0) + int(record["accepted"])
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            print(f"{key}: {'accepted' if record['accepted'] else record['reason']}; "
                  f"reasoning={record['reasoning_check']}", flush=True)
    with (args.output / "augmentation_candidates.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prompt_id", "model", "text"])
        writer.writeheader()
        writer.writerows(rows)
    manifest.update(attempted_requests=len(jobs), accepted_rows=len(rows),
                    accepted_per_class=counts,
                    reported_total_tokens=reported_tokens,
                    reported_cost_sum_currency_unspecified=reported_cost,
                    reasoning_checks=reasoning_checks,
                    responses_reporting_tokens_over_requested_limit=over_limit,
                    all_18_classes_fully_covered=all(counts.get(x, 0) == len(prompts) for x in labels))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {len(rows)}/{len(jobs)} provisional candidates: {args.output}", flush=True)


if __name__ == "__main__":
    main()
