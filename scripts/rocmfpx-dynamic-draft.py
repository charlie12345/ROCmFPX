#!/usr/bin/env python3
"""Adaptive request wrapper for ROCmFPX dynamic drafting.

This client keeps llama-server simple: the server starts with a safe speculative
cap, and this wrapper injects per-request speculative settings based on prompt
length plus optional feedback from prior draft acceptance.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROFILE_PATH = SCRIPT_DIR / "rocmfpx-draft-profile.py"
PROFILE_SPEC = importlib.util.spec_from_file_location("rocmfpx_draft_profile", PROFILE_PATH)
if PROFILE_SPEC is None or PROFILE_SPEC.loader is None:
    raise RuntimeError(f"failed to load {PROFILE_PATH}")
rocmfpx_draft_profile = importlib.util.module_from_spec(PROFILE_SPEC)
PROFILE_SPEC.loader.exec_module(rocmfpx_draft_profile)

choose_profile = rocmfpx_draft_profile.choose_profile
tokenize_count = rocmfpx_draft_profile.tokenize_count


DEFAULT_STATE = {
    "requests": 0,
    "draft_n": 0,
    "draft_n_accepted": 0,
    "acceptance_ema": None,
    "last_update": None,
}


def load_json_arg(value: str | None, path: str | None) -> dict[str, Any]:
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    if value:
        return json.loads(value)
    return {}


def load_state(path: str | None) -> dict[str, Any]:
    if not path:
        return copy.deepcopy(DEFAULT_STATE)
    state_path = Path(path)
    if not state_path.exists():
        return copy.deepcopy(DEFAULT_STATE)
    data = json.loads(state_path.read_text(encoding="utf-8"))
    state = copy.deepcopy(DEFAULT_STATE)
    state.update(data)
    return state


def save_state(path: str | None, state: dict[str, Any]) -> None:
    if not path:
        return
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract_text(payload: dict[str, Any], endpoint: str) -> str:
    if endpoint.endswith("/v1/chat/completions") or "messages" in payload:
        parts: list[str] = []
        for message in payload.get("messages", []):
            content = message.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return str(payload.get("prompt", ""))


def infer_prompt_tokens(args: argparse.Namespace, payload: dict[str, Any]) -> int:
    if args.prompt_tokens is not None:
        return args.prompt_tokens
    text = extract_text(payload, args.endpoint)
    if not text:
        return 0
    if args.no_tokenize:
        return max(1, len(text) // 4)
    return tokenize_count(args.base_url, text, args.api_key)


def adapt_policy(policy: dict[str, Any], state: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    result = dict(policy)
    n_max = int(result.get("speculative.n_max", 0))
    if n_max <= 0:
        return result

    acceptance = state.get("acceptance_ema")
    if not isinstance(acceptance, (int, float)):
        return result

    if acceptance < args.low_acceptance:
        n_max = max(args.min_n_max, n_max - 1)
        result["speculative.p_min"] = min(1.0, max(float(result.get("speculative.p_min", 0.0)), 0.25))
    elif acceptance > args.high_acceptance:
        n_max = min(args.max_n_max, n_max + 1)
        result["speculative.p_min"] = max(0.0, min(float(result.get("speculative.p_min", 0.0)), 0.25))

    result["speculative.n_max"] = n_max
    result["speculative.n_min"] = min(int(result.get("speculative.n_min", 0)), n_max)
    return result


def find_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find_key(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_key(value, key)
            if found is not None:
                return found
    return None


def update_state_from_response(state: dict[str, Any], response: dict[str, Any], alpha: float) -> dict[str, Any]:
    draft_n = find_key(response, "draft_n")
    draft_n_accepted = find_key(response, "draft_n_accepted")
    if not isinstance(draft_n, (int, float)) or not isinstance(draft_n_accepted, (int, float)) or draft_n <= 0:
        return state

    accepted = max(0.0, min(float(draft_n_accepted), float(draft_n)))
    rate = accepted / float(draft_n)
    old = state.get("acceptance_ema")
    if isinstance(old, (int, float)):
        rate = alpha * rate + (1.0 - alpha) * float(old)

    state["requests"] = int(state.get("requests", 0)) + 1
    state["draft_n"] = int(state.get("draft_n", 0)) + int(draft_n)
    state["draft_n_accepted"] = int(state.get("draft_n_accepted", 0)) + int(draft_n_accepted)
    state["acceptance_ema"] = rate
    state["last_update"] = int(time.time())
    return state


def send_request(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    request = urllib.request.Request(
        args.base_url.rstrip("/") + args.endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {body}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a request with ROCmFPX Dynamic Drafting fields.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18180", help="llama-server base URL")
    parser.add_argument("--endpoint", default="/completion", help="API endpoint, e.g. /completion or /v1/chat/completions")
    parser.add_argument("--api-key", help="Bearer token for llama-server")
    parser.add_argument("--json", help="Request payload JSON")
    parser.add_argument("--json-file", help="Request payload JSON file")
    parser.add_argument("--prompt-tokens", type=int, help="Known prompt token count")
    parser.add_argument("--profile", default="fp3-mtp", choices=("fp3-mtp", "fp4-general", "dense-coder"))
    parser.add_argument("--state-file", help="Persist draft acceptance feedback here")
    parser.add_argument("--no-tokenize", action="store_true", help="Estimate token count instead of calling /tokenize")
    parser.add_argument("--dry-run", action="store_true", help="Print adjusted payload and do not send")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--min-n-max", type=int, default=1)
    parser.add_argument("--max-n-max", type=int, default=8)
    parser.add_argument("--low-acceptance", type=float, default=0.45)
    parser.add_argument("--high-acceptance", type=float, default=0.80)
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    args = parser.parse_args()

    payload = load_json_arg(args.json, args.json_file)
    state = load_state(args.state_file)
    prompt_tokens = infer_prompt_tokens(args, payload)
    policy = choose_profile(prompt_tokens, args.profile)
    policy = adapt_policy(policy, state, args)

    adjusted = dict(payload)
    adjusted.update(policy)

    if args.dry_run:
        result: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "profile": args.profile,
            "state": state,
            "request": adjusted,
        }
    else:
        result = send_request(args, adjusted)
        state = update_state_from_response(state, result, args.ema_alpha)
        save_state(args.state_file, state)

    if args.pretty:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
