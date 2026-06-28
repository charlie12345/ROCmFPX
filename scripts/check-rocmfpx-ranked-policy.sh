#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
FIXTURE_DIR="$ROOT/tests/fixtures/rocmfpx-ranked-policy"
TMP_DIR="${TMPDIR:-/tmp}/rocmfpx-ranked-policy-check.$$"

cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "$TMP_DIR"

(
    cd "$ROOT"
    python3 scripts/rocmfpx-ranked-policy.py \
        --rank-csv tests/fixtures/rocmfpx-ranked-policy/attention-rank.sample.csv \
        --leave-count 2 \
        --output "$TMP_DIR/leave2.tensor-type.txt"
)

diff -u "$FIXTURE_DIR/leave2.tensor-type.expected.txt" "$TMP_DIR/leave2.tensor-type.txt"

(
    cd "$ROOT"
    python3 scripts/rocmfpx-ranked-policy.py \
        --rank-csv tests/fixtures/rocmfpx-ranked-policy/attention-rank.sample.csv \
        --leave-count 2 \
        --base-tensor-type-file tests/fixtures/rocmfpx-ranked-policy/base.tensor-type.sample.txt \
        --output "$TMP_DIR/leave2-with-base.tensor-type.txt"
)

diff -u "$FIXTURE_DIR/leave2-with-base.tensor-type.expected.txt" "$TMP_DIR/leave2-with-base.tensor-type.txt"

if (
    cd "$ROOT"
    python3 scripts/rocmfpx-ranked-policy.py \
        --rank-csv tests/fixtures/rocmfpx-ranked-policy/attention-rank.sample.csv \
        --leave-count 0 \
        --output "$TMP_DIR/leave0.tensor-type.txt" \
        >"$TMP_DIR/leave0.stdout" 2>"$TMP_DIR/leave0.stderr"
); then
    echo "expected --leave-count 0 to fail" >&2
    exit 1
fi

grep -F -- "--leave-count must be positive" "$TMP_DIR/leave0.stderr" >/dev/null

echo "ROCmFPX ranked policy check passed"
