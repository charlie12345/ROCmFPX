#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:-master}"
SYNC_BRANCH="${SYNC_BRANCH:-}"

cd "$ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to sync a dirty worktree." >&2
    exit 1
fi
if ! git remote get-url "$UPSTREAM_REMOTE" >/dev/null 2>&1; then
    echo "Missing remote '$UPSTREAM_REMOTE'." >&2
    echo "Add it with: git remote add upstream https://github.com/ggml-org/llama.cpp.git" >&2
    exit 1
fi

git fetch --prune "$UPSTREAM_REMOTE" "$UPSTREAM_BRANCH"
upstream_ref="$UPSTREAM_REMOTE/$UPSTREAM_BRANCH"
upstream_sha="$(git rev-parse "$upstream_ref")"

if git merge-base --is-ancestor "$upstream_sha" HEAD; then
    echo "Already contains $upstream_ref at $upstream_sha"
    exit 0
fi

if [[ -z "$SYNC_BRANCH" ]]; then
    SYNC_BRANCH="automation/upstream-${upstream_sha:0:12}"
fi
git switch -c "$SYNC_BRANCH"

if ! git merge --no-ff --no-edit "$upstream_ref"; then
    echo "Upstream merge needs human conflict resolution." >&2
    echo "Conflicted paths:" >&2
    git diff --name-only --diff-filter=U >&2
    exit 2
fi

"$ROOT/scripts/rocmfpx/validate-sync.sh"

echo "Prepared $SYNC_BRANCH at $(git rev-parse HEAD)"
echo "Review ROCmFPX performance gates before merging or pushing to protected main."
