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

python3 -m py_compile \
    "$SCRIPT_DIR/rocmfpx-draft-profile.py" \
    "$SCRIPT_DIR/rocmfpx-dynamic-draft.py"

bash -n \
    "$SCRIPT_DIR/check-rocmfpx-dynamic-draft.sh" \
    "$SCRIPT_DIR/run-rocmfpx-mtp-server.sh"

echo "ROCmFPX dynamic drafting smoke passed"
