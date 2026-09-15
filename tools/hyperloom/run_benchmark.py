#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
from pathlib import Path


QUALITY_PROMPT = "Begin with ROCMFPX_QUALITY. State that seven times six is forty-two, then define a stack in one sentence."
MTP_PROMPT = "Explain virtual memory briefly, then write a concise C++ function that reverses a singly linked list."
PERF_RE = re.compile(r"Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def env_path(name: str, default: Path | None = None) -> Path:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser().resolve()
    if default is not None:
        return default.resolve()
    raise RuntimeError(f"{name} is required")


def run(command: list[str], timeout: int, log_path: Path, input_text: str | None = None) -> str:
    env = os.environ.copy()
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env=env,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit {completed.returncode}: {shlex.join(command)}")
    return completed.stdout


def configure_and_build(repo: Path, build_dir: Path, result_dir: Path) -> None:
    if os.environ.get("ROCMFPX_SKIP_BUILD") == "1":
        return

    cmake_args = [
        "cmake",
        "-S",
        str(repo),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_HIP=ON",
        "-DGGML_VULKAN=OFF",
        "-DGGML_NATIVE=OFF",
        "-DGGML_CCACHE=OFF",
        "-DAMDGPU_TARGETS=gfx1151",
    ]
    rocm_path = os.environ.get("ROCM_PATH")
    if rocm_path:
        cmake_args.append(f"-DCMAKE_PREFIX_PATH={rocm_path}")
        cmake_args.append(f"-DCMAKE_HIP_COMPILER={rocm_path}/bin/amdclang++")
    cmake_args.extend(shlex.split(os.environ.get("ROCMFPX_CMAKE_ARGS", "")))
    run(cmake_args, 600, result_dir / "configure.log")

    jobs = int(os.environ.get("ROCMFPX_BUILD_JOBS", min(16, len(os.sched_getaffinity(0)))))
    build_args = [
        "cmake",
        "--build",
        str(build_dir),
        "--target",
        "llama-bench",
        "llama-cli",
        "-j",
        str(jobs),
    ]
    run(build_args, 3600, result_dir / "build.log")


def binary_path(build_dir: Path, name: str) -> Path:
    override = os.environ.get(f"ROCMFPX_{name.upper().replace('-', '_')}")
    path = Path(override).expanduser().resolve() if override else build_dir / "bin" / name
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f"missing executable: {path}")
    return path


def common_args(model: Path, backend: str) -> list[str]:
    return [
        "-m",
        str(model),
        "-dev",
        backend,
        "-ngl",
        "999",
        "-fa",
        "on",
    ]


def speculation_args(backend: str) -> list[str]:
    return [
        "--spec-type",
        "draft-mtp",
        "--spec-draft-device",
        backend,
        "--spec-draft-ngl",
        "all",
        "--spec-draft-type-k",
        "f16",
        "--spec-draft-type-v",
        "f16",
        "--spec-draft-n-max",
        os.environ.get("ROCMFPX_MTP_N_MAX", "6"),
        "--spec-draft-n-min",
        "0",
        "--spec-draft-p-min",
        os.environ.get("ROCMFPX_MTP_P_MIN", "0.6"),
        "--spec-draft-backend-sampling",
    ]


def quality_output(cli: Path, model: Path, backend: str, result_dir: Path, use_mtp: bool) -> str:
    command = [str(cli), *common_args(model, backend), "-c", "1024", "-b", "512", "-ub", "512"]
    if use_mtp:
        command.extend(speculation_args(backend))
    command.extend(
        [
            "-st",
            "--temp",
            "0",
            "--seed",
            "1",
            "-n",
            "64",
            "--reasoning",
            "off",
            "--no-display-prompt",
            "--no-perf",
            "--simple-io",
            "--log-disable",
            "-p",
            QUALITY_PROMPT,
        ]
    )
    output = run(command, 300, result_dir / "quality.log", input_text="")
    normalized = ANSI_RE.sub("", output).replace("\r\n", "\n")
    marker = f"> {QUALITY_PROMPT}"
    if marker not in normalized:
        raise RuntimeError("quality output does not contain the prompt marker")
    generated = normalized.split(marker, 1)[1]
    generated = re.split(r"\n\[ Prompt:", generated, maxsplit=1)[0]
    return generated.strip()


def load_json_lines(output: str) -> list[dict]:
    rows: list[dict] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
        elif isinstance(value, list):
            rows.extend(item for item in value if isinstance(item, dict))
    return rows


def bench_metrics(bench: Path, model: Path, backend: str, result_dir: Path) -> dict[str, float]:
    repetitions = os.environ.get("ROCMFPX_REPETITIONS", "3")
    prompt_tokens = os.environ.get("ROCMFPX_PROMPT_TOKENS", "512")
    generation_tokens = os.environ.get("ROCMFPX_GENERATION_TOKENS", "128")
    command = [
        str(bench),
        *common_args(model, backend),
        "-b",
        os.environ.get("ROCMFPX_BATCH", "2048"),
        "-ub",
        os.environ.get("ROCMFPX_UBATCH", "512"),
        "-p",
        prompt_tokens,
        "-n",
        generation_tokens,
        "-r",
        repetitions,
        "-o",
        "jsonl",
    ]
    output = run(command, 1200, result_dir / "llama-bench.log")
    rows = load_json_lines(output)
    prompt = [float(row["avg_ts"]) for row in rows if int(row.get("n_prompt", 0)) > 0]
    decode = [float(row["avg_ts"]) for row in rows if int(row.get("n_gen", 0)) > 0]
    if not prompt or not decode:
        raise RuntimeError("llama-bench did not emit prompt and generation JSON rows")
    return {
        "prefill_tokens_per_second": prompt[-1],
        "decode_tokens_per_second": decode[-1],
    }


def mtp_metrics(cli: Path, model: Path, backend: str, result_dir: Path) -> dict[str, float]:
    repetitions = int(os.environ.get("ROCMFPX_REPETITIONS", "3"))
    prompt_rates: list[float] = []
    decode_rates: list[float] = []
    for index in range(repetitions):
        command = [
            str(cli),
            *common_args(model, backend),
            "-c",
            "4096",
            "-b",
            os.environ.get("ROCMFPX_BATCH", "2048"),
            "-ub",
            os.environ.get("ROCMFPX_UBATCH", "512"),
            *speculation_args(backend),
            "-st",
            "--simple-io",
            "--temp",
            "0",
            "--seed",
            "1",
            "-n",
            os.environ.get("ROCMFPX_MTP_TOKENS", "256"),
            "--reasoning",
            "off",
            "-p",
            MTP_PROMPT,
        ]
        output = run(command, 600, result_dir / f"mtp-{index + 1}.log", input_text="")
        matches = PERF_RE.findall(ANSI_RE.sub("", output))
        if not matches:
            raise RuntimeError("llama-cli did not emit an MTP performance line")
        prompt_rate, decode_rate = matches[-1]
        prompt_rates.append(float(prompt_rate))
        decode_rates.append(float(decode_rate))
    return {
        "prefill_tokens_per_second": statistics.median(prompt_rates),
        "decode_tokens_per_second": statistics.median(decode_rates),
        "decode_min_tokens_per_second": min(decode_rates),
        "decode_max_tokens_per_second": max(decode_rates),
    }


def git_head(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip()


def write_result(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    rocm_path = os.environ.get("ROCM_PATH", "").strip()
    if rocm_path:
        os.environ["PATH"] = f"{rocm_path}/bin:{os.environ.get('PATH', '')}"
        current_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{rocm_path}/lib{':' + current_library_path if current_library_path else ''}"
    repo = env_path("FRAMEWORK_REPO_PATH", Path(__file__).resolve().parents[2])
    model = env_path("MODEL_PATH")
    result_dir = env_path("RESULT_DIR")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_filename = os.environ.get("RESULT_FILENAME", "inferencex_result")
    result_path = result_dir / f"{result_filename}.json"
    build_dir = env_path("ROCMFPX_BUILD_DIR", repo / "build-hyperloom")
    backend = os.environ.get("ROCMFPX_BACKEND", "ROCm0")
    objective = os.environ.get("ROCMFPX_OBJECTIVE", "decode").lower()
    if objective not in {"decode", "prefill", "mtp"}:
        raise RuntimeError("ROCMFPX_OBJECTIVE must be decode, prefill, or mtp")
    if not model.is_file():
        raise RuntimeError(f"missing model: {model}")

    try:
        configure_and_build(repo, build_dir, result_dir)
        bench = binary_path(build_dir, "llama-bench")
        cli = binary_path(build_dir, "llama-cli")
        quality = quality_output(cli, model, backend, result_dir, objective == "mtp")
        if not quality:
            raise RuntimeError("quality command produced no captured output")
        quality_hash = hashlib.sha256(quality.encode("utf-8")).hexdigest()
        reference_path = env_path("ROCMFPX_QUALITY_REFERENCE")
        create_reference = os.environ.get("ROCMFPX_CREATE_REFERENCE") == "1"

        if create_reference:
            if repo == reference_path or repo in reference_path.parents:
                raise RuntimeError("quality reference must be outside the source worktree")
            reference = {
                "backend": backend,
                "git_head": git_head(repo),
                "model": str(model),
                "model_size": model.stat().st_size,
                "objective": objective,
                "output_sha256": quality_hash,
                "prompt": QUALITY_PROMPT,
            }
        else:
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
        quality_passed = bool(quality) and quality_hash == reference.get("output_sha256")
        quality_passed = quality_passed and model.stat().st_size == int(reference.get("model_size", -1))

        metrics = (
            mtp_metrics(cli, model, backend, result_dir)
            if objective == "mtp"
            else bench_metrics(bench, model, backend, result_dir)
        )
        if create_reference:
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            reference_path.write_text(
                json.dumps(reference, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        throughput = metrics[f"{objective if objective != 'mtp' else 'decode'}_tokens_per_second"]
        report = {
            "framework": "custom",
            "workload_kind": "scriptable",
            "throughput_unit": "tokens/s",
            "output_throughput": throughput,
            "quality_gate": {
                "passed": quality_passed,
                "expected_output_sha256": reference.get("output_sha256"),
                "output_sha256": quality_hash,
                "model_size_unchanged": model.stat().st_size == int(reference.get("model_size", -1)),
            },
            "rocmfpx": {
                "backend": backend,
                "git_head": git_head(repo),
                "model_size": model.stat().st_size,
                "objective": objective,
                **metrics,
            },
        }
        write_result(result_path, report)
        return 0 if quality_passed else 3
    except Exception as error:
        write_result(
            result_path,
            {
                "framework": "custom",
                "workload_kind": "scriptable",
                "throughput_unit": "tokens/s",
                "output_throughput": 0.0,
                "quality_gate": {"passed": False, "error": str(error)},
            },
        )
        print(f"ROCmFPX Hyperloom benchmark failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
