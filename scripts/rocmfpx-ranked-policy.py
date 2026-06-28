#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def regex_exact(name: str) -> str:
    return f"^{re.escape(name)}$"


def read_ranked_names(path: Path, count: int, column: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"rank CSV is missing required column: {column}")

        for row in reader:
            name = (row.get(column) or "").strip()
            if not name:
                raise ValueError(f"rank CSV has an empty {column} value")
            if name in seen:
                raise ValueError(f"rank CSV repeats tensor name: {name}")

            seen.add(name)
            names.append(name)
            if len(names) == count:
                break

    if len(names) != count:
        raise ValueError(f"rank CSV only had {len(names)} rows, wanted {count}")

    return names


def build_policy_lines(
    rank_csv: Path,
    leave_count: int,
    name_column: str,
    base_type: str,
    restore_type: str,
    base_tensor_type_file: Path | None,
) -> list[str]:
    leave_names = read_ranked_names(rank_csv, leave_count, name_column)

    lines = [f"{regex_exact(name)}={base_type}" for name in leave_names]
    if base_tensor_type_file is not None:
        lines.extend(base_tensor_type_file.read_text(encoding="utf-8").splitlines())
    else:
        lines.extend([
            r"^blk\.[0-9]+\.ffn_up\.weight$=" + restore_type,
            r"^blk\.[0-9]+\.ffn_gate\.weight$=" + restore_type,
            r"^blk\.[0-9]+\.ffn_down\.weight$=" + restore_type,
            r"^blk\.[0-9]+\.attn_qkv\.weight$=" + restore_type,
            r"^blk\.[0-9]+\.attn_output\.weight$=" + restore_type,
        ])

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a llama-quantize --tensor-type-file for ROCmFPX ranked attention leave-N policies.",
    )
    parser.add_argument("--rank-csv", type=Path, required=True)
    parser.add_argument("--leave-count", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--name-column", default="name")
    parser.add_argument("--base-type", default="q6_0_rocmfpx")
    parser.add_argument("--restore-type", default="q6_k")
    parser.add_argument("--base-tensor-type-file", type=Path)
    args = parser.parse_args()

    if args.leave_count <= 0:
        parser.error("--leave-count must be positive")
    if args.base_tensor_type_file is not None and not args.base_tensor_type_file.is_file():
        parser.error(f"--base-tensor-type-file does not exist: {args.base_tensor_type_file}")

    text = "\n".join(build_policy_lines(
        args.rank_csv,
        args.leave_count,
        args.name_column,
        args.base_type,
        args.restore_type,
        args.base_tensor_type_file,
    )) + "\n"

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
