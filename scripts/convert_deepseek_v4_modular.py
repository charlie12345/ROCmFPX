#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conversion import ModelType, get_model_architecture, get_model_class
from conversion.base import ModelBase, gguf


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DeepSeek V4 checkpoints using the modular ROCmFPX converter.")
    parser.add_argument("model", type=Path)
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--outtype", choices=["f16", "bf16", "auto"], default="f16")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-temp-file", action="store_true")
    parser.add_argument("--deepseek4-include-mtp", action="store_true")
    parser.add_argument("--deepseek4-max-layers", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    ftype_map = {
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "auto": gguf.LlamaFileType.GUESSED,
    }

    hparams = ModelBase.load_hparams(args.model, False)
    arch = get_model_architecture(hparams, ModelType.TEXT)
    logging.getLogger("hf-to-gguf").info("Model architecture: %s", arch)
    model_class = get_model_class(arch, mmproj=False)

    model = model_class(
        args.model,
        ftype_map[args.outtype],
        args.outfile,
        use_temp_file=args.use_temp_file,
        dry_run=args.dry_run,
        deepseek4_include_mtp=args.deepseek4_include_mtp,
        deepseek4_max_layers=args.deepseek4_max_layers,
    )
    model.write()


if __name__ == "__main__":
    main()
