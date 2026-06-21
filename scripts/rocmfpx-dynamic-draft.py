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
import random
import re
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
    "throughput_ema": None,
    "last_n_max": None,
    "last_p_min": None,
    "n_max_stats": {},
    "position_stats": {},
    "last_update": None,
}

MODE_CHOICES = ("auto", "completion", "chat", "coding", "json", "tool")

MODE_POLICY_OVERRIDES = {
    # Coding prompts tend to have long deterministic runs where a deeper draft
    # is often useful. JSON/tool calls value structural correctness more than
    # the last bit of speculative depth, so start with tighter caps.
    "coding": {"n_delta": 1, "p_min_floor": 0.0, "p_split": 0.15},
    "json": {"n_delta": -1, "p_min_floor": 0.25, "p_split": 0.05},
    "tool": {"n_delta": -1, "p_min_floor": 0.25, "p_split": 0.05},
    "chat": {"n_delta": 0, "p_min_floor": 0.0, "p_split": None},
    "completion": {"n_delta": 0, "p_min_floor": 0.0, "p_split": None},
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


def state_scope_key(args: argparse.Namespace, payload: dict[str, Any], mode: str) -> str:
    if args.state_scope and args.state_scope != "auto":
        return args.state_scope
    model = args.model_key or str(payload.get("model") or payload.get("cache_prompt") or "default")
    backend = args.backend or "default"
    ctx = args.ctx_size if args.ctx_size is not None else "default"
    return f"{args.profile}:{mode}:{backend}:{model}:ctx{ctx}"


def get_scoped_state(root: dict[str, Any], scope: str) -> dict[str, Any]:
    scopes = root.get("scopes")
    if isinstance(scopes, dict):
        scoped = scopes.get(scope)
        state = copy.deepcopy(DEFAULT_STATE)
        if isinstance(scoped, dict):
            state.update(scoped)
        return state

    # Backward compatibility: an older flat state file is treated as the active
    # scope until it is saved again.
    state = copy.deepcopy(DEFAULT_STATE)
    state.update({key: value for key, value in root.items() if key in DEFAULT_STATE})
    return state


def set_scoped_state(root: dict[str, Any], scope: str, state: dict[str, Any]) -> dict[str, Any]:
    result = dict(root)
    scopes = result.get("scopes")
    if not isinstance(scopes, dict):
        scopes = {}
    scopes[scope] = state
    result["scopes"] = scopes
    result["version"] = 2
    return result


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


def infer_mode(args: argparse.Namespace, payload: dict[str, Any]) -> str:
    if args.mode != "auto":
        return args.mode
    if payload.get("tools") or payload.get("tool_choice"):
        return "tool"
    response_format = payload.get("response_format")
    if isinstance(response_format, dict):
        fmt_type = str(response_format.get("type", "")).lower()
        if "json" in fmt_type:
            return "json"
    text = extract_text(payload, args.endpoint).lower()
    if any(marker in text for marker in ("json", "schema", "tool call", "function call")):
        return "json"
    if any(marker in text for marker in ("write code", "debug", "function", "class ", "python", "typescript", "rust", "c++")):
        return "coding"
    if args.endpoint.endswith("/v1/chat/completions") or "messages" in payload:
        return "chat"
    return "completion"


def infer_prompt_tokens(args: argparse.Namespace, payload: dict[str, Any]) -> int:
    if args.prompt_tokens is not None:
        return args.prompt_tokens
    text = extract_text(payload, args.endpoint)
    if not text:
        return 0
    if args.no_tokenize:
        return max(1, len(text) // 4)
    return tokenize_count(args.base_url, text, args.api_key)


def apply_mode_policy(policy: dict[str, Any], mode: str, args: argparse.Namespace) -> dict[str, Any]:
    if not args.mode_overrides:
        return dict(policy)
    result = dict(policy)
    override = MODE_POLICY_OVERRIDES.get(mode)
    if not override:
        return result
    n_max = int(result.get("speculative.n_max", 0))
    if n_max > 0:
        n_max = max(args.min_n_max, min(args.max_n_max, n_max + int(override["n_delta"])))
        result["speculative.n_max"] = n_max
        result["speculative.n_min"] = min(int(result.get("speculative.n_min", 0)), n_max)
    p_min_floor = float(override["p_min_floor"])
    result["speculative.p_min"] = min(1.0, max(float(result.get("speculative.p_min", 0.0)), p_min_floor))
    if override["p_split"] is not None:
        result["speculative.p_split"] = float(override["p_split"])
    return result


def best_n_max_from_stats(base_n_max: int, state: dict[str, Any], args: argparse.Namespace) -> int:
    stats = state.get("n_max_stats")
    if not isinstance(stats, dict):
        return base_n_max
    best_n_max = base_n_max
    best_tps = -1.0
    for key, value in stats.items():
        if not isinstance(value, dict):
            continue
        try:
            candidate_n_max = int(key)
        except ValueError:
            continue
        if candidate_n_max < args.min_n_max or candidate_n_max > args.max_n_max:
            continue
        if abs(candidate_n_max - base_n_max) > args.max_profile_shift:
            continue
        tps = value.get("throughput_ema")
        acceptance = value.get("acceptance_ema")
        count = value.get("count", 0)
        if (
            isinstance(tps, (int, float))
            and isinstance(acceptance, (int, float))
            and isinstance(count, int)
            and count >= args.min_stats_count
            and float(acceptance) >= args.low_acceptance
            and float(tps) > best_tps
        ):
            best_n_max = candidate_n_max
            best_tps = float(tps)
    return best_n_max


def maybe_explore_n_max(n_max: int, state: dict[str, Any], args: argparse.Namespace) -> int:
    if args.explore_rate <= 0.0 or random.random() >= args.explore_rate:
        return n_max
    candidates = [n for n in (n_max - 1, n_max + 1) if args.min_n_max <= n <= args.max_n_max]
    if not candidates:
        return n_max
    stats = state.get("n_max_stats")
    if isinstance(stats, dict):
        candidates.sort(key=lambda item: int(stats.get(str(item), {}).get("count", 0) if isinstance(stats.get(str(item)), dict) else 0))
    return random.choice(candidates[:1])


def position_limited_n_max(n_max: int, state: dict[str, Any], args: argparse.Namespace) -> int:
    stats = state.get("position_stats")
    if not isinstance(stats, dict):
        return n_max
    limited = n_max
    for pos in range(1, n_max + 1):
        bucket = stats.get(str(pos))
        if not isinstance(bucket, dict):
            continue
        count = bucket.get("count", 0)
        rate = bucket.get("acceptance_ema")
        if (
            isinstance(count, int)
            and count >= args.min_position_stats_count
            and isinstance(rate, (int, float))
            and float(rate) < args.min_position_acceptance
        ):
            limited = max(args.min_n_max, pos - 1)
            break
    return limited


def adapt_policy(policy: dict[str, Any], state: dict[str, Any], args: argparse.Namespace, mode: str) -> dict[str, Any]:
    policy = apply_mode_policy(policy, mode, args)
    result = dict(policy)
    base_n_max = int(result.get("speculative.n_max", 0))
    n_max = base_n_max
    if n_max <= 0:
        return result

    # Prefer the fastest nearby n_max seen for this exact workload scope, use
    # occasional neighbor exploration, then clamp with acceptance/position data.
    n_max = best_n_max_from_stats(base_n_max, state, args)
    n_max = maybe_explore_n_max(n_max, state, args)
    n_max = position_limited_n_max(n_max, state, args)

    acceptance = state.get("acceptance_ema")
    if not isinstance(acceptance, (int, float)):
        result["speculative.n_max"] = n_max
        result["speculative.n_min"] = min(int(result.get("speculative.n_min", 0)), n_max)
        return result

    if acceptance < args.low_acceptance:
        n_max = max(args.min_n_max, n_max - 1)
        result["speculative.p_min"] = min(1.0, max(float(result.get("speculative.p_min", 0.0)), 0.25))
        result["speculative.p_split"] = min(float(result.get("speculative.p_split", 0.10)), 0.10)
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


def ema(old: Any, sample: float, alpha: float) -> float:
    if isinstance(old, (int, float)):
        return alpha * sample + (1.0 - alpha) * float(old)
    return sample


def update_state_from_response(state: dict[str, Any], response: dict[str, Any], alpha: float) -> dict[str, Any]:
    draft_n = find_key(response, "draft_n")
    draft_n_accepted = find_key(response, "draft_n_accepted")
    if not isinstance(draft_n, (int, float)) or not isinstance(draft_n_accepted, (int, float)) or draft_n <= 0:
        return state

    accepted = max(0.0, min(float(draft_n_accepted), float(draft_n)))
    rate = accepted / float(draft_n)
    rate = ema(state.get("acceptance_ema"), rate, alpha)

    throughput = find_key(response, "predicted_per_second")
    if isinstance(throughput, (int, float)):
        state["throughput_ema"] = ema(state.get("throughput_ema"), float(throughput), alpha)

    n_max = find_key(response, "speculative.n_max")
    p_min = find_key(response, "speculative.p_min")
    if isinstance(n_max, (int, float)):
        n_max_int = int(n_max)
        state["last_n_max"] = n_max_int
        stats = state.get("n_max_stats")
        if not isinstance(stats, dict):
            stats = {}
        bucket = stats.get(str(n_max_int))
        if not isinstance(bucket, dict):
            bucket = {"count": 0}
        bucket["count"] = int(bucket.get("count", 0)) + 1
        bucket["acceptance_ema"] = ema(bucket.get("acceptance_ema"), accepted / float(draft_n), alpha)
        if isinstance(throughput, (int, float)):
            bucket["throughput_ema"] = ema(bucket.get("throughput_ema"), float(throughput), alpha)
        stats[str(n_max_int)] = bucket
        state["n_max_stats"] = stats
    if isinstance(p_min, (int, float)):
        state["last_p_min"] = float(p_min)

    position_rates = (
        find_key(response, "acceptance_per_position")
        or find_key(response, "draft_acceptance_per_position")
        or find_key(response, "draft_position_acceptance")
    )
    if isinstance(position_rates, list):
        position_stats = state.get("position_stats")
        if not isinstance(position_stats, dict):
            position_stats = {}
        for index, value in enumerate(position_rates, start=1):
            if not isinstance(value, (int, float)):
                continue
            bucket = position_stats.get(str(index))
            if not isinstance(bucket, dict):
                bucket = {"count": 0}
            bucket["count"] = int(bucket.get("count", 0)) + 1
            bucket["acceptance_ema"] = ema(bucket.get("acceptance_ema"), max(0.0, min(float(value), 1.0)), alpha)
            position_stats[str(index)] = bucket
        state["position_stats"] = position_stats

    state["requests"] = int(state.get("requests", 0)) + 1
    state["draft_n"] = int(state.get("draft_n", 0)) + int(draft_n)
    state["draft_n_accepted"] = int(state.get("draft_n_accepted", 0)) + int(draft_n_accepted)
    state["acceptance_ema"] = rate
    state["last_update"] = int(time.time())
    return state


THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)


def strip_thinking_text(text: str) -> str:
    text = THINK_RE.sub("", text)
    return text.lstrip()


def strip_thinking_response(obj: Any) -> Any:
    if isinstance(obj, dict):
        cleaned: dict[str, Any] = {}
        for key, value in obj.items():
            if key in {"reasoning", "reasoning_content"}:
                continue
            if key in {"content", "text"} and isinstance(value, str):
                cleaned[key] = strip_thinking_text(value)
            else:
                cleaned[key] = strip_thinking_response(value)
        return cleaned
    if isinstance(obj, list):
        return [strip_thinking_response(value) for value in obj]
    return obj


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
    parser.add_argument("--mode", default="auto", choices=MODE_CHOICES,
                        help="Workload mode for scoped feedback and conservative JSON/tool caps")
    parser.add_argument("--state-scope", default="auto",
                        help="State scope key; default includes profile, inferred mode, backend, model, and ctx")
    parser.add_argument("--backend", help="Backend name to include in the state scope, e.g. ROCm0 or Vulkan0")
    parser.add_argument("--model-key", help="Model identifier to include in the state scope")
    parser.add_argument("--ctx-size", type=int, help="Context size to include in the state scope")
    parser.add_argument("--mode-overrides", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply mode-specific n_max/p_min/p_split adjustments")
    parser.add_argument("--state-file", help="Persist draft acceptance feedback here")
    parser.add_argument("--no-tokenize", action="store_true", help="Estimate token count instead of calling /tokenize")
    parser.add_argument("--dry-run", action="store_true", help="Print adjusted payload and do not send")
    parser.add_argument("--strip-thinking", action=argparse.BooleanOptionalAction, default=True,
                        help="Strip <think>...</think> blocks and reasoning fields from responses")
    parser.add_argument("--chat-reasoning-format", default="deepseek",
                        choices=("none", "auto", "deepseek", "deepseek-legacy"),
                        help="Reasoning parser format to request for OpenAI-compatible chat calls")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--min-n-max", type=int, default=1)
    parser.add_argument("--max-n-max", type=int, default=8)
    parser.add_argument("--max-profile-shift", type=int, default=2)
    parser.add_argument("--min-stats-count", type=int, default=1)
    parser.add_argument("--explore-rate", type=float, default=0.05,
                        help="Probability of testing a neighboring n_max for throughput learning")
    parser.add_argument("--min-position-acceptance", type=float, default=0.35,
                        help="Per-position acceptance floor when server telemetry is available")
    parser.add_argument("--min-position-stats-count", type=int, default=3)
    parser.add_argument("--low-acceptance", type=float, default=0.45)
    parser.add_argument("--high-acceptance", type=float, default=0.80)
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    args = parser.parse_args()

    payload = load_json_arg(args.json, args.json_file)
    root_state = load_state(args.state_file)
    mode = infer_mode(args, payload)
    scope = state_scope_key(args, payload, mode)
    state = get_scoped_state(root_state, scope)
    prompt_tokens = infer_prompt_tokens(args, payload)
    policy = choose_profile(prompt_tokens, args.profile)
    policy = adapt_policy(policy, state, args, mode)

    adjusted = dict(payload)
    adjusted.update(policy)
    if args.endpoint.endswith("/v1/chat/completions") or "messages" in adjusted:
        adjusted.setdefault("reasoning_format", args.chat_reasoning_format)

    if args.dry_run:
        result: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "profile": args.profile,
            "mode": mode,
            "state_scope": scope,
            "state": state,
            "request": adjusted,
        }
    else:
        result = send_request(args, adjusted)
        state = update_state_from_response(state, result, args.ema_alpha)
        root_state = set_scoped_state(root_state, scope, state)
        save_state(args.state_file, root_state)
        if args.strip_thinking:
            result = strip_thinking_response(result)

    if args.pretty:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
