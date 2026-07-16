#!/usr/bin/env python3

import csv
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "docs" / "recipes" / "release-recipe-map.tsv"
CATALOG_PATH = ROOT / "docs" / "recipes" / "README.md"

FIELDS = [
    "hf_repo",
    "artifact",
    "implementation_family",
    "topology",
    "release_recipe_label",
    "internal_recipe_id",
]

CONTRACTS = {
    "qwen35": "dense",
    "qwen35moe": "moe",
    "step35": "moe",
}

RECIPE_ID_RE = re.compile(
    r"^[a-z0-9-]+\.[a-z0-9]+\.(dense|moe)\.[a-z0-9-]+\.v[0-9]+$"
)


def fail(message: str) -> None:
    raise SystemExit(f"release recipe map check failed: {message}")


with MAP_PATH.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    if reader.fieldnames != FIELDS:
        fail(f"unexpected columns: {reader.fieldnames!r}")
    rows = list(reader)

if not rows:
    fail("map is empty")

catalog = CATALOG_PATH.read_text(encoding="utf-8")
seen = set()
repo_counts = Counter()

for line_number, row in enumerate(rows, start=2):
    if any(not row[field].strip() for field in FIELDS):
        fail(f"line {line_number} contains an empty field")

    key = (row["hf_repo"], row["artifact"])
    if key in seen:
        fail(f"duplicate artifact mapping on line {line_number}: {key!r}")
    seen.add(key)
    repo_counts[row["hf_repo"]] += 1

    implementation = row["implementation_family"]
    topology = row["topology"]
    expected_topology = CONTRACTS.get(implementation)
    if expected_topology is None:
        fail(f"line {line_number} has unknown implementation {implementation!r}")
    if topology != expected_topology:
        fail(
            f"line {line_number} maps {implementation!r} to {topology!r}; "
            f"expected {expected_topology!r}"
        )

    recipe_id = row["internal_recipe_id"]
    match = RECIPE_ID_RE.fullmatch(recipe_id)
    if match is None:
        fail(f"line {line_number} has malformed recipe ID {recipe_id!r}")
    if match.group(1) != topology:
        fail(f"line {line_number} recipe ID topology disagrees with its row")

    if row["artifact"] not in catalog:
        fail(f"line {line_number} artifact is missing from the Markdown catalog")
    if recipe_id not in catalog:
        fail(f"line {line_number} recipe ID is missing from the Markdown catalog")

for mixed_repo in (
    "jcbtc/chadrock-35b-ace-saber-rocmfp4-mtp",
    "jcbtc/chadrock3.6-27b-coder-rocmfp4-mtp",
):
    if repo_counts[mixed_repo] != 2:
        fail(f"mixed repository {mixed_repo!r} must map exactly two artifacts")

print(f"release recipe map check passed ({len(rows)} artifacts)")
