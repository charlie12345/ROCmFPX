#!/usr/bin/env bash
set -euo pipefail

repo=ROCmFPX/ROCmFPX
mode=dry-run

usage() {
    echo "usage: $0 [--dry-run|--check|--apply] [--repo ROCmFPX/ROCmFPX]" >&2
}

while (($#)); do
    case "$1" in
        --dry-run)
            mode=dry-run
            shift
            ;;
        --check)
            mode=check
            shift
            ;;
        --apply)
            mode=apply
            shift
            ;;
        --repo)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            repo=$2
            shift 2
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

[[ "$repo" == ROCmFPX/ROCmFPX ]] || {
    echo "refusing workflow changes outside ROCmFPX/ROCmFPX" >&2
    exit 2
}

command -v gh >/dev/null || {
    echo "gh is required" >&2
    exit 2
}

name=$(gh repo view "$repo" --json nameWithOwner --jq .nameWithOwner)
private=$(gh repo view "$repo" --json isPrivate --jq .isPrivate)
[[ "$name" == "$repo" && "$private" == true ]] || {
    echo "refusing non-private or unexpected repository: $name" >&2
    exit 2
}

declare -A keep=(
    [.github/workflows/check-rocmfpx.yml]=1
    [.github/workflows/rocmfpx-gfx1151.yml]=1
    [.github/workflows/rocmfpx-release.yml]=1
    [.github/workflows/rocmfpx-upstream-sync.yml]=1
)
declare -A seen=()

changes=0
while IFS=$'\t' read -r id path state; do
    seen["$path"]=1
    desired=disabled_manually
    action=disable
    if [[ ${keep[$path]:-0} == 1 ]]; then
        desired=active
        action=enable
    fi

    if [[ "$state" == "$desired" ]]; then
        printf 'keep %-17s %s\n' "$state" "$path"
        continue
    fi

    changes=$((changes + 1))
    printf '%s %-14s %s (%s -> %s)\n' "$mode" "$action" "$path" "$state" "$desired"
    if [[ "$mode" == apply ]]; then
        gh api --method PUT "repos/$repo/actions/workflows/$id/$action" >/dev/null
    fi
done < <(gh api --paginate "repos/$repo/actions/workflows?per_page=100" --jq '.workflows[] | [.id, .path, .state] | @tsv')

for path in "${!keep[@]}"; do
    if [[ ${seen[$path]:-0} == 0 ]]; then
        changes=$((changes + 1))
        printf '%s missing        %s\n' "$mode" "$path"
    fi
done

printf 'workflow policy changes: %d\n' "$changes"
if [[ "$mode" == check && $changes -ne 0 ]]; then
    exit 1
fi
