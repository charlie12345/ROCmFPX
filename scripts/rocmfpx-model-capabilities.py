#!/usr/bin/env python3
"""Inspect lightweight ROCmFPX model capabilities from GGUF metadata bytes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


MTP_MARKERS = (
    b"nextn_predict_layers",
    b"mtp.pre_projection",
    b"mtp.post_projection",
    b".nextn.",
    b"nextn.",
)

# GGUF metadata keys emitted only by MoE architectures (e.g. "<arch>.expert_count").
# Dense models don't write these, so their presence reliably flags a MoE model.
MOE_MARKERS = (
    b"expert_count",
    b"expert_used_count",
)

DIFFUSION_MARKERS = (
    b"diffusion_gemma",
    b"diffusion-gemma",
    b"diffusion.kv_cache",
    b"diffusion-kv-cache",
    b"llada",
)

QAT_MARKERS = (
    b"quantization-aware",
    b"quantization_aware",
    b"quantization.aware",
)

DFLASH_MARKERS = (
    b"dflash",
    b"ddflash",
)

AGENT_MARKERS = (
    b"rocmfpx_agent",
    b"rocmfp4_coherent",
)

DIFFUSION_NAME_MARKERS = DIFFUSION_MARKERS + (b"diffusion",)
QAT_NAME_MARKERS = QAT_MARKERS + (b"qat",)
DFLASH_NAME_MARKERS = DFLASH_MARKERS + (b"dynamic-detailing", b"dynamic_detailing")
AGENT_NAME_MARKERS = AGENT_MARKERS + (b"agent", b"coherent")


def marker_hits(data: bytes, markers: tuple[bytes, ...]) -> list[str]:
    return [marker.decode("ascii", errors="ignore") for marker in markers if marker in data]


def filename_hits(path: Path, markers: tuple[bytes, ...]) -> list[str]:
    name = path.name.lower()
    return [marker.decode("ascii", errors="ignore") for marker in markers if marker.decode("ascii", errors="ignore") in name]


def infer_model_kind(path: Path, supports_mtp: bool, supports_diffusion: bool, is_qat: bool, is_agent: bool) -> str:
    name = path.name.lower()
    if supports_diffusion:
        return "diffusion"
    if supports_mtp:
        return "mtp"
    if is_agent:
        return "agent"
    if is_qat or "qat" in name:
        return "qat"
    return "regular"


def is_rocmfpx_family_name(path: Path) -> bool:
    name = path.name.lower()
    return any(marker in name for marker in ("rocmfp4", "rocmfpx", "rocm-fp4", "rocm-fpx", "coherent", "agentic"))


def infer_serving_profile(path: Path, model_kind: str, supports_mtp: bool, is_agent: bool) -> dict[str, object]:
    name = path.name.lower()
    if "nemotron-3-nano-30b" in name and "rocmfpx_agent" in name:
        return {
            "name": "nemotron-30b-a3b-rocmfpx-agent",
            "reason": "Nemotron 3 Nano 30B A3B ROCmFPX agent GGUF has no MTP layers; ROCm f16 KV is the fastest validated profile.",
            "supports_mtp": False,
            "server_args": [
                "-dev", "ROCm0",
                "-ngl", "999",
                "-fa", "on",
                "--load-mode", "mmap",
                "-ctk", "f16",
                "-ctv", "f16",
                "-c", "131072",
                "-b", "512",
                "-ub", "512",
                "-t", "16",
                "-tb", "32",
                "--jinja",
                "--reasoning", "off",
                "--reasoning-format", "none",
                "--reasoning-budget", "-1",
                "--no-context-shift",
                "--no-mmproj",
                "--metrics",
            ],
        }

    if model_kind == "diffusion":
        return {
            "name": "generic-diffusion-rocmfpx",
            "reason": "Diffusion/Block-diffusion GGUF detected; benchmark ROCm vs Vulkan before changing denoising steps.",
            "supports_mtp": False,
            "server_args": [
                "-dev", "ROCm0",
                "-ngl", "999",
                "-fa", "on",
                "--load-mode", "mmap",
                "-ctk", "f16",
                "-ctv", "f16",
                "-c", "32768",
                "-b", "512",
                "-ub", "512",
                "-t", "16",
                "-tb", "32",
                "--jinja",
                "--no-mmproj",
                "--metrics",
            ],
        }

    if supports_mtp and "qwen" in name and (is_agent or is_rocmfpx_family_name(path)):
        return {
            "name": "qwen-mtp-rocmfpx-prompt-fast",
            "reason": "Qwen MTP ROCmFP4/ROCmFPX models on Strix have much faster prompt processing on Vulkan with q4 K/V and a 4096/512 batch shape; p-min 0.75 avoids low-confidence draft overhead during decode.",
            "supports_mtp": True,
            "server_args": [
                "-dev", "Vulkan0",
                "--spec-draft-device", "Vulkan0",
                "-ngl", "999",
                "--spec-draft-ngl", "all",
                "-fa", "on",
                "--load-mode", "mmap",
                "-ctk", "q4_0",
                "-ctv", "q4_0",
                "--spec-draft-type-k", "q4_0",
                "--spec-draft-type-v", "q4_0",
                "-c", "262144",
                "-b", "4096",
                "-ub", "512",
                "-t", "16",
                "-tb", "32",
                "--spec-draft-threads", "16",
                "--spec-draft-threads-batch", "32",
                "--jinja",
                "--reasoning", "off",
                "--reasoning-format", "none",
                "--reasoning-budget", "-1",
                "--no-context-shift",
                "--spec-type", "draft-mtp",
                "--spec-draft-n-max", "4",
                "--spec-draft-n-min", "0",
                "--spec-draft-p-min", "0.75",
                "--spec-draft-p-split", "0.10",
                "--no-spec-draft-backend-sampling",
                "--spec-draft-poll", "1",
                "--spec-draft-poll-batch", "1",
                "--no-mmproj",
                "--metrics",
            ],
        }

    if supports_mtp and "qwen" in name:
        return {
            "name": "qwen-dense-mtp",
            "reason": "Qwen MTP-capable models validated fastest with bounded draft depth and a confidence gate; use q8 target/q4 draft KV where memory allows.",
            "supports_mtp": True,
            "server_args": [
                "-dev", "ROCm0",
                "--spec-draft-device", "ROCm0",
                "-ngl", "999",
                "--spec-draft-ngl", "all",
                "-fa", "on",
                "--load-mode", "mmap",
                "-ctk", "q8_0",
                "-ctv", "q8_0",
                "--spec-draft-type-k", "q4_0",
                "--spec-draft-type-v", "q4_0",
                "-c", "262144",
                "-b", "512",
                "-ub", "512",
                "-t", "16",
                "-tb", "32",
                "--spec-draft-threads", "16",
                "--spec-draft-threads-batch", "32",
                "--jinja",
                "--reasoning", "off",
                "--reasoning-format", "none",
                "--reasoning-budget", "-1",
                "--no-context-shift",
                "--spec-type", "draft-mtp",
                "--spec-draft-n-max", "4",
                "--spec-draft-n-min", "0",
                "--spec-draft-p-min", "0.75",
                "--spec-draft-p-split", "0.10",
                "--no-spec-draft-backend-sampling",
                "--spec-draft-poll", "1",
                "--spec-draft-poll-batch", "1",
                "--no-mmproj",
                "--metrics",
            ],
        }

    if supports_mtp:
        return {
            "name": "generic-mtp",
            "reason": "Model exposes MTP markers; use conservative draft-mtp defaults and benchmark n-max for this model.",
            "supports_mtp": True,
            "server_args": [
                "--spec-type", "draft-mtp",
                "--spec-draft-ngl", "all",
                "--spec-draft-n-max", "4",
                "--spec-draft-n-min", "0",
                "--spec-draft-p-min", "0.0",
                "--spec-draft-p-split", "0.10",
            ],
        }

    if model_kind == "qat":
        return {
            "name": "generic-qat-gguf",
            "reason": "QAT GGUF detected; keep native tensor quantization, use GPU offload/repack/FlashAttention, and avoid ROCmFPX re-quantizing unless an A/B quality test passes.",
            "supports_mtp": False,
            "server_args": [
                "-dev", "ROCm0",
                "-ngl", "999",
                "-fa", "on",
                "--load-mode", "mmap",
                "-ctk", "f16",
                "-ctv", "f16",
                "-c", "32768",
                "-b", "2048",
                "-ub", "512",
                "-t", "16",
                "-tb", "32",
                "--jinja",
                "--no-mmproj",
                "--metrics",
            ],
        }

    if model_kind == "agent":
        return {
            "name": "generic-agent-rocmfpx",
            "reason": "Agent-oriented ROCmFPX/ROCmFP4 GGUF detected; keep K/V cache at f16 or q8_0 unless a tool/JSON smoke test passes with lower cache precision.",
            "supports_mtp": False,
            "server_args": [
                "-dev", "ROCm0",
                "-ngl", "999",
                "-fa", "on",
                "--load-mode", "mmap",
                "-ctk", "f16",
                "-ctv", "f16",
                "-c", "32768",
                "-b", "512",
                "-ub", "512",
                "-t", "16",
                "-tb", "32",
                "--jinja",
                "--reasoning", "off",
                "--reasoning-format", "none",
                "--reasoning-budget", "-1",
                "--no-mmproj",
                "--metrics",
            ],
        }

    return {
        "name": "generic-regular-gguf",
        "reason": "Regular non-MTP GGUF detected; use normal GPU offload and benchmark ROCm/Vulkan before adding MTP, diffusion, or agent-specific flags.",
        "supports_mtp": False,
        "server_args": [
            "-dev", "ROCm0",
            "-ngl", "999",
            "-fa", "on",
            "--load-mode", "mmap",
            "-ctk", "f16",
            "-ctv", "f16",
            "-c", "32768",
            "-b", "2048",
            "-ub", "512",
            "-t", "16",
            "-tb", "32",
            "--jinja",
            "--no-mmproj",
            "--metrics",
        ],
    }


def read_probe(path: Path, limit: int) -> bytes:
    with path.open("rb") as handle:
        return handle.read(limit)


def detect_capabilities(path: Path, probe_bytes: int) -> dict[str, object]:
    data = read_probe(path, probe_bytes)
    lower_data = data.lower()
    markers = marker_hits(lower_data, MTP_MARKERS) + filename_hits(path, MTP_MARKERS)
    diffusion_markers = marker_hits(lower_data, DIFFUSION_MARKERS) + filename_hits(path, DIFFUSION_NAME_MARKERS)
    qat_markers = marker_hits(lower_data, QAT_MARKERS) + filename_hits(path, QAT_NAME_MARKERS)
    dflash_markers = marker_hits(lower_data, DFLASH_MARKERS) + filename_hits(path, DFLASH_NAME_MARKERS)
    agent_markers = marker_hits(lower_data, AGENT_MARKERS) + filename_hits(path, AGENT_NAME_MARKERS)
    moe_markers = marker_hits(lower_data, MOE_MARKERS)
    supports_mtp = bool(markers)
    is_moe = bool(moe_markers)
    supports_diffusion = bool(diffusion_markers)
    supports_dflash = bool(dflash_markers)
    is_qat = bool(qat_markers)
    is_agent = bool(agent_markers)
    model_kind = infer_model_kind(path, supports_mtp, supports_diffusion, is_qat, is_agent)
    return {
        "model": str(path),
        "model_kind": model_kind,
        "probe_bytes": len(data),
        "supports_mtp": supports_mtp,
        "is_moe": is_moe,
        "moe_markers": moe_markers,
        "supports_diffusion": supports_diffusion,
        "supports_dflash": supports_dflash,
        "is_qat": is_qat,
        "is_agent": is_agent,
        "mtp_markers": markers,
        "diffusion_markers": diffusion_markers,
        "dflash_markers": dflash_markers,
        "qat_markers": qat_markers,
        "agent_markers": agent_markers,
        "serving_profile": infer_serving_profile(path, model_kind, supports_mtp, is_agent),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect ROCmFPX-serving model capabilities.")
    parser.add_argument("model", help="GGUF model path, usually the first split for split GGUFs")
    parser.add_argument("--probe-mib", type=int, default=64, help="MiB to scan from the start of the GGUF")
    parser.add_argument("--has-mtp", action="store_true", help="Exit 0 only when MTP markers are present")
    parser.add_argument("--is-moe", action="store_true", help="Exit 0 only when the model is a MoE (has expert_count metadata)")
    parser.add_argument("--server-args", action="store_true", help="Print recommended server args, one shell-quoted line")
    parser.add_argument("--quiet", action="store_true", help="Suppress JSON output")
    args = parser.parse_args()

    model = Path(args.model)
    if not model.is_file():
        print(f"model does not exist: {model}", file=sys.stderr)
        return 2

    caps = detect_capabilities(model, max(args.probe_mib, 1) * 1024 * 1024)
    if args.server_args:
        print(" ".join(json.dumps(arg) for arg in caps["serving_profile"]["server_args"]))
    elif not args.quiet:
        print(json.dumps(caps, indent=2, sort_keys=True))

    if args.has_mtp:
        return 0 if caps["supports_mtp"] else 1
    if args.is_moe:
        return 0 if caps["is_moe"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
