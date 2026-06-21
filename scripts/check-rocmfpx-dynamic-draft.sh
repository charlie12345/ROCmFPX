#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

payload='{"prompt":"hello","n_predict":16}'

short_json="$("$SCRIPT_DIR/rocmfpx-dynamic-draft.py" \
    --json "$payload" \
    --prompt-tokens 4096 \
    --profile fp3-mtp \
    --explore-rate 0 \
    --dry-run)"

python3 - "$short_json" <<'PY'
import json
import sys

data = json.loads(sys.argv[1])
req = data["request"]
assert data["prompt_tokens"] == 4096
assert req["speculative.n_max"] == 4
assert req["speculative.n_min"] == 0
assert req["speculative.p_min"] == 0.75
assert req["speculative.p_split"] == 0.10
PY

state="$tmpdir/state.json"
cat > "$state" <<'JSON'
{
  "requests": 3,
  "draft_n": 100,
  "draft_n_accepted": 25,
  "acceptance_ema": 0.25,
  "last_update": 1
}
JSON

backoff_json="$("$SCRIPT_DIR/rocmfpx-dynamic-draft.py" \
    --json "$payload" \
    --prompt-tokens 4096 \
    --profile dense-coder \
    --state-file "$state" \
    --explore-rate 0 \
    --dry-run)"

python3 - "$backoff_json" <<'PY'
import json
import sys

data = json.loads(sys.argv[1])
req = data["request"]
assert req["speculative.n_max"] == 5
assert req["speculative.n_min"] == 0
assert req["speculative.p_min"] == 0.25
PY

cat > "$state" <<'JSON'
{
  "requests": 3,
  "draft_n": 100,
  "draft_n_accepted": 90,
  "acceptance_ema": 0.90,
  "last_update": 1
}
JSON

raise_json="$("$SCRIPT_DIR/rocmfpx-dynamic-draft.py" \
    --json "$payload" \
    --prompt-tokens 120000 \
    --profile fp3-mtp \
    --state-file "$state" \
    --dry-run \
    --explore-rate 0 \
    --max-n-max 4)"

python3 - "$raise_json" <<'PY'
import json
import sys

data = json.loads(sys.argv[1])
req = data["request"]
assert req["speculative.n_max"] == 2
assert req["speculative.n_min"] == 0
assert req["speculative.p_min"] == 0.0
PY

json_mode_payload='{"messages":[{"role":"user","content":"Return JSON matching this schema."}],"response_format":{"type":"json_object"},"max_tokens":64}'
json_mode_json="$("$SCRIPT_DIR/rocmfpx-dynamic-draft.py" \
    --json "$json_mode_payload" \
    --prompt-tokens 4096 \
    --profile dense-coder \
    --explore-rate 0 \
    --dry-run)"

python3 - "$json_mode_json" <<'PY'
import json
import sys

data = json.loads(sys.argv[1])
req = data["request"]
assert data["mode"] == "json"
assert "dense-coder:json:" in data["state_scope"]
assert req["speculative.n_max"] == 5
assert req["speculative.p_min"] == 0.25
assert req["speculative.p_split"] == 0.05
PY

strip_json="$(python3 - <<'PY'
import importlib.util
import json
import pathlib

path = pathlib.Path("scripts/rocmfpx-dynamic-draft.py")
spec = importlib.util.spec_from_file_location("dd", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

response = {
    "choices": [
        {
            "message": {
                "content": "<think>\nprivate\n</think>\n\n{\"ok\":true}",
                "reasoning_content": "private",
            }
        }
    ],
    "timings": {
        "draft_n": 10,
        "draft_n_accepted": 8,
        "predicted_per_second": 50.0,
    },
    "generation_settings": {
        "speculative.n_max": 6,
        "speculative.p_min": 0.0,
    },
}
print(json.dumps(mod.strip_thinking_response(response), sort_keys=True))
PY
)"

python3 - "$strip_json" <<'PY'
import json
import sys

data = json.loads(sys.argv[1])
msg = data["choices"][0]["message"]
assert msg["content"] == '{"ok":true}'
assert "reasoning_content" not in msg
PY

state_update_json="$(python3 - <<'PY'
import importlib.util
import json
import pathlib

path = pathlib.Path("scripts/rocmfpx-dynamic-draft.py")
spec = importlib.util.spec_from_file_location("dd", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

state = {}
response = {
    "timings": {
        "draft_n": 100,
        "draft_n_accepted": 90,
        "predicted_per_second": 42.0,
        "acceptance_per_position": [0.95, 0.82, 0.31],
    },
    "generation_settings": {
        "speculative.n_max": 6,
        "speculative.p_min": 0.0,
    },
}
print(json.dumps(mod.update_state_from_response(state, response, 0.35), sort_keys=True))
PY
)"

python3 - "$state_update_json" <<'PY'
import json
import sys

data = json.loads(sys.argv[1])
assert data["acceptance_ema"] == 0.9
assert data["throughput_ema"] == 42.0
assert data["last_n_max"] == 6
assert data["n_max_stats"]["6"]["acceptance_ema"] == 0.9
assert data["n_max_stats"]["6"]["throughput_ema"] == 42.0
assert data["position_stats"]["3"]["acceptance_ema"] == 0.31
PY

position_state_json="$(python3 - <<'PY'
import argparse
import importlib.util
import json
import pathlib

path = pathlib.Path("scripts/rocmfpx-dynamic-draft.py")
spec = importlib.util.spec_from_file_location("dd", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

args = argparse.Namespace(
    min_n_max=1,
    max_n_max=8,
    max_profile_shift=2,
    min_stats_count=1,
    low_acceptance=0.45,
    high_acceptance=0.80,
    explore_rate=0.0,
    min_position_acceptance=0.35,
    min_position_stats_count=3,
    mode_overrides=True,
)
state = {
    "position_stats": {
        "3": {"count": 3, "acceptance_ema": 0.30}
    }
}
policy = {
    "speculative.n_max": 6,
    "speculative.n_min": 0,
    "speculative.p_min": 0.0,
    "speculative.p_split": 0.20,
}
print(json.dumps(mod.adapt_policy(policy, state, args, "coding"), sort_keys=True))
PY
)"

python3 - "$position_state_json" <<'PY'
import json
import sys

data = json.loads(sys.argv[1])
assert data["speculative.n_max"] == 2
PY

python3 -m py_compile \
    "$SCRIPT_DIR/rocmfpx-draft-profile.py" \
    "$SCRIPT_DIR/rocmfpx-dynamic-draft.py"

bash -n \
    "$SCRIPT_DIR/check-rocmfpx-dynamic-draft.sh" \
    "$SCRIPT_DIR/run-rocmfpx-mtp-server.sh"

echo "ROCmFPX dynamic drafting smoke passed"
