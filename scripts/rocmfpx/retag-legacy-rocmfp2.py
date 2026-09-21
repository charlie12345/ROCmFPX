#!/usr/bin/env python3
"""Retag known dual-scale S40 ROCmFP2 tensors from legacy type 107."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

import gguf  # noqa: E402


LEGACY_TYPE = gguf.GGMLQuantizationType.Q2_0_ROCMFPX_LEGACY_AMBIGUOUS
S40_TYPE = gguf.GGMLQuantizationType.Q2_0_ROCMFPX
S40_LAYOUT = "s40-dual-scale-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit or retag legacy ROCmFP2 GGUF tensor type 107. Type 107 is "
            "ambiguous, so the tool never attempts to infer the layout from bytes."
        ),
    )
    parser.add_argument("model", type=Path, help="GGUF model or shard to inspect")
    parser.add_argument(
        "--layout",
        choices=(S40_LAYOUT,),
        help="layout confirmed from model provenance; required with --apply",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="retag matching tensor headers in place (the default is read-only)",
    )
    return parser.parse_args()


def inspect_model(path: Path, writable: bool) -> tuple[gguf.GGUFReader, list[gguf.ReaderTensor], int]:
    if not path.is_file():
        raise ValueError(f"not a regular file: {path}")

    reader = gguf.GGUFReader(path, "r+" if writable else "r")
    legacy = [tensor for tensor in reader.tensors if tensor.tensor_type == LEGACY_TYPE]
    canonical_count = sum(tensor.tensor_type == S40_TYPE for tensor in reader.tensors)
    return reader, legacy, canonical_count


def retag(reader: gguf.GGUFReader, tensors: list[gguf.ReaderTensor]) -> None:
    fields = []
    for tensor in tensors:
        raw_type = tensor.field.parts[4]
        if raw_type.size != 1 or int(raw_type[0]) != int(LEGACY_TYPE):
            raise RuntimeError(f"tensor type field changed while inspecting {tensor.name}")
        fields.append(raw_type)

    for raw_type in fields:
        raw_type[0] = int(S40_TYPE)
    reader.data.flush()


def main() -> int:
    args = parse_args()
    if args.apply and args.layout != S40_LAYOUT:
        raise ValueError(f"--apply requires --layout {S40_LAYOUT}")

    reader, legacy, canonical_count = inspect_model(args.model, args.apply)
    print(f"model: {args.model}")
    print(f"legacy ambiguous type 107 tensors: {len(legacy)}")
    print(f"canonical dual-scale S40 type 111 tensors: {canonical_count}")

    if not legacy:
        print("no legacy type 107 tensors found; no changes needed")
        return 0

    for tensor in legacy[:10]:
        print(f"  {tensor.name}")
    if len(legacy) > 10:
        print(f"  ... and {len(legacy) - 10} more")

    if not args.apply:
        print("read-only audit complete; no bytes changed")
        print(f"to retag a provenance-confirmed S40 file, add --layout {S40_LAYOUT} --apply")
        return 0

    legacy_count = len(legacy)
    retag(reader, legacy)
    del legacy
    del reader

    verify_reader, remaining, new_canonical_count = inspect_model(args.model, False)
    del verify_reader
    if remaining or new_canonical_count != canonical_count + legacy_count:
        raise RuntimeError("post-write verification failed")

    print(f"retagged {legacy_count} tensor headers from type 107 to type 111")
    print("tensor payload bytes and encoded size were not changed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
