"""Failing-first test suite locking the public QWEN4EXP conversion contract.

Contracts locked:
1. convert_hf_to_gguf.py CLI entrypoint exits 0 on --help
2. QWEN4EXP MODEL_TENSOR enum members and canonical C++ tensor name mappings
3. GGUFWriter hyper-connection/PLE metadata methods and UINT64 array roundtrip (> 2**32)
4. Monolithic ModelBase text and mmproj architecture alias registrations
5. TensorNameMap HF-to-GGUF tensor name mappings for hyper-connection, indexer, and PLE
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_import_path() -> None:
    """Ensure repo root and gguf-py package are in sys.path."""
    gguf_path = str(REPO_ROOT / "gguf-py")
    repo_str = str(REPO_ROOT)
    if gguf_path not in sys.path:
        sys.path.insert(0, gguf_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


# ---------------------------------------------------------------------------
# Test 1: CLI entrypoint exit code
# ---------------------------------------------------------------------------
def test_convert_hf_to_gguf_cli_help():
    # Given: Path to the public convert_hf_to_gguf.py script
    script_path = REPO_ROOT / "convert_hf_to_gguf.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'gguf-py'}:{REPO_ROOT}"

    # When: Running `python3 convert_hf_to_gguf.py --help`
    proc = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    # Then: Process should exit cleanly with returncode 0
    assert proc.returncode == 0, (
        f"convert_hf_to_gguf.py --help failed with code {proc.returncode}:\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


# ---------------------------------------------------------------------------
# Test 2: MODEL_TENSOR members and canonical C++ mappings for QWEN4EXP
# ---------------------------------------------------------------------------
def test_qwen4exp_model_tensor_enum_and_canonical_names():
    # Given: gguf constants and canonical C++ tensor naming contract (src/llama-arch.cpp)
    _ensure_import_path()
    import gguf

    expected_canonical_mappings = {
        # Indexer tensors
        "INDEXER_Q_PROJ": "blk.{bid}.indexer.q_proj",
        "INDEXER_K_PROJ": "blk.{bid}.indexer.k_proj",
        "INDEXER_Q_NORM": "blk.{bid}.indexer.q_norm",
        "INDEXER_K_NORM": "blk.{bid}.indexer.k_norm",
        # Hyper-connection head tensors
        "HC_HEAD_NORM": "output_hc_norm",
        "HC_HEAD_DOWN": "output_hc_down",
        "HC_HEAD_UP": "output_hc_up",
        # Hyper-connection attention tensors
        "HC_ATTN_NORM": "blk.{bid}.hc_attn_norm",
        "HC_ATTN_DOWN": "blk.{bid}.hc_attn_down",
        "HC_ATTN_UP": "blk.{bid}.hc_attn_up",
        "HC_ATTN_INJECT": "blk.{bid}.hc_attn_inject",
        # Hyper-connection FFN tensors
        "HC_FFN_NORM": "blk.{bid}.hc_ffn_norm",
        "HC_FFN_DOWN": "blk.{bid}.hc_ffn_down",
        "HC_FFN_UP": "blk.{bid}.hc_ffn_up",
        "HC_FFN_INJECT": "blk.{bid}.hc_ffn_inject",
        # PLE tensors
        "PLE_KEY": "blk.{bid}.ple_key",
        "PLE_VALUE": "blk.{bid}.ple_value",
        "PLE_NORM_KEY": "blk.{bid}.ple_norm_key",
        "PLE_NORM_QUERY": "blk.{bid}.ple_norm_query",
        "PLE_NORM_CONV": "blk.{bid}.ple_norm_conv",
        "PLE_CONV1D": "blk.{bid}.ple_conv1d",
    }

    # When: Querying MODEL_TENSOR and TENSOR_NAMES
    # Then: MODEL_ARCH.QWEN4EXP must exist
    assert hasattr(gguf.MODEL_ARCH, "QWEN4EXP")
    qwen4exp_arch = gguf.MODEL_ARCH.QWEN4EXP
    arch_tensors = gguf.MODEL_TENSORS[qwen4exp_arch]

    # And every required tensor member must exist, map to the canonical name, and belong to QWEN4EXP
    for enum_name, canonical_name in expected_canonical_mappings.items():
        assert hasattr(gguf.MODEL_TENSOR, enum_name), f"Missing MODEL_TENSOR.{enum_name}"
        tensor_enum = getattr(gguf.MODEL_TENSOR, enum_name)
        assert tensor_enum in gguf.TENSOR_NAMES, f"MODEL_TENSOR.{enum_name} missing from TENSOR_NAMES"
        assert gguf.TENSOR_NAMES[tensor_enum] == canonical_name, (
            f"MODEL_TENSOR.{enum_name} mapped to {gguf.TENSOR_NAMES[tensor_enum]!r}, expected {canonical_name!r}"
        )
        assert tensor_enum in arch_tensors, f"MODEL_TENSOR.{enum_name} missing from MODEL_TENSORS[QWEN4EXP]"


# ---------------------------------------------------------------------------
# Test 3: GGUFWriter low-rank/PLE methods and UINT64 > 2**32 roundtrip
# ---------------------------------------------------------------------------
def test_gguf_writer_ple_methods_and_uint64_roundtrip(tmp_path: Path):
    # Given: GGUFWriter instance and large 45-bit/64-bit integer values (> 2**32)
    _ensure_import_path()
    import gguf

    required_methods = [
        "add_hyper_connection_low_rank",
        "add_ple_layers",
        "add_ple_ngram_size",
        "add_ple_heads_per_ngram",
        "add_ple_conv_kernel",
        "add_ple_layer_multipliers",
        "add_ple_head_offsets",
        "add_ple_head_vocab_sizes",
        "add_ple_eos_token_id",
        "add_ple_image_token_id",
    ]

    out_file = tmp_path / "test_qwen4exp_ple.gguf"
    writer = gguf.GGUFWriter(path=out_file, arch="qwen4exp")

    # When: Checking method existence on GGUFWriter
    # Then: Every required low-rank / PLE method must be callable
    for method_name in required_methods:
        assert hasattr(writer, method_name) and callable(getattr(writer, method_name)), (
            f"GGUFWriter missing required method: {method_name}"
        )

    # When: Writing UINT64 array metadata containing values > 2**32
    large_uint64_values = [
        2**33 + 12345,
        2**45 + 67890,
        (1 << 48) - 1,
    ]
    writer.add_ple_layer_multipliers(large_uint64_values)
    writer.add_ple_head_offsets([2**34 + 1, 2**34 + 2])
    writer.add_ple_head_vocab_sizes([2**35 + 100])
    writer.add_hyper_connection_low_rank(64)
    writer.add_ple_layers([0, 1])
    writer.add_ple_ngram_size(3)
    writer.add_ple_heads_per_ngram(4)
    writer.add_ple_conv_kernel(3)
    writer.add_ple_eos_token_id(151643)
    writer.add_ple_image_token_id(151655)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    writer.close()

    # And When: Reading back with GGUFReader
    reader = gguf.GGUFReader(out_file)
    field = reader.get_field("qwen4exp.ple.layer_multipliers")

    # Then: UINT64 array values > 2**32 must be preserved exactly
    assert field is not None, "Metadata key 'qwen4exp.ple.layer_multipliers' not found in written GGUF"
    assert field.types == [gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT64]
    read_values = [int(v) for v in field.contents()]
    assert read_values == large_uint64_values


# ---------------------------------------------------------------------------
# Test 4: ModelBase registers Qwen4Exp text and mmproj aliases
# ---------------------------------------------------------------------------
def test_modelbase_qwen4exp_registrations():
    # Given: convert_hf_to_gguf module with monolithic ModelBase registry
    _ensure_import_path()
    import convert_hf_to_gguf
    import gguf

    model_base = convert_hf_to_gguf.ModelBase
    model_type = convert_hf_to_gguf.ModelType

    # When: Resolving text model architectures
    # Then: Both Qwen4ExpForConditionalGeneration and Qwen4ExpForCausalLM are registered
    text_causal_cls = model_base.from_model_architecture("Qwen4ExpForCausalLM", model_type.TEXT)
    text_cond_cls = model_base.from_model_architecture("Qwen4ExpForConditionalGeneration", model_type.TEXT)

    assert issubclass(text_causal_cls, convert_hf_to_gguf.TextModel)
    assert text_causal_cls.model_arch == gguf.MODEL_ARCH.QWEN4EXP

    assert issubclass(text_cond_cls, convert_hf_to_gguf.TextModel)
    assert text_cond_cls.model_arch == gguf.MODEL_ARCH.QWEN4EXP

    # And When: Resolving conditional mmproj architecture
    # Then: Qwen4ExpForConditionalGeneration is registered under MMPROJ
    mmproj_cls = model_base.from_model_architecture("Qwen4ExpForConditionalGeneration", model_type.MMPROJ)
    assert issubclass(mmproj_cls, convert_hf_to_gguf.MmprojModel)
    assert mmproj_cls.model_arch == gguf.MODEL_ARCH.MMPROJ


# ---------------------------------------------------------------------------
# Test 5: TensorNameMap HF to GGUF mapping for HC / Indexer / PLE
# ---------------------------------------------------------------------------
def test_tensor_name_map_qwen4exp():
    # Given: TensorNameMap initialized for QWEN4EXP architecture
    _ensure_import_path()
    import gguf

    tensor_map = gguf.TensorNameMap(gguf.MODEL_ARCH.QWEN4EXP, n_blocks=4)
    suffixes = (".weight", ".bias")

    representative_hf_mappings = {
        # Hyper-connection attention
        "model.layers.0.hc_attn_norm.weight": "blk.0.hc_attn_norm.weight",
        "model.layers.0.hc_attn_down.weight": "blk.0.hc_attn_down.weight",
        "model.layers.0.hc_attn_up.weight": "blk.0.hc_attn_up.weight",
        "model.layers.0.hc_attn_inject.weight": "blk.0.hc_attn_inject.weight",
        # Hyper-connection FFN
        "model.layers.1.hc_ffn_norm.weight": "blk.1.hc_ffn_norm.weight",
        "model.layers.1.hc_ffn_down.weight": "blk.1.hc_ffn_down.weight",
        "model.layers.1.hc_ffn_up.weight": "blk.1.hc_ffn_up.weight",
        "model.layers.1.hc_ffn_inject.weight": "blk.1.hc_ffn_inject.weight",
        # Hyper-connection Head / Output norm
        "model.norm.hc_head_norm.weight": "output_hc_norm.weight",
        "model.norm.hc_head_down.weight": "output_hc_down.weight",
        "model.norm.hc_head_up.weight": "output_hc_up.weight",
        # Indexer
        "model.layers.2.indexer.q_norm.weight": "blk.2.indexer.q_norm.weight",
        "model.layers.2.indexer.k_norm.weight": "blk.2.indexer.k_norm.weight",
        "model.layers.2.indexer.q_proj.weight": "blk.2.indexer.q_proj.weight",
        "model.layers.2.indexer.k_proj.weight": "blk.2.indexer.k_proj.weight",
        # PLE
        "model.layers.3.ple.key.weight": "blk.3.ple_key.weight",
        "model.layers.3.ple.value.weight": "blk.3.ple_value.weight",
        "model.layers.3.ple.norm_key.weight": "blk.3.ple_norm_key.weight",
        "model.layers.3.ple.norm_query.weight": "blk.3.ple_norm_query.weight",
        "model.layers.3.ple.norm_conv.weight": "blk.3.ple_norm_conv.weight",
        "model.layers.3.ple.conv1d.weight": "blk.3.ple_conv1d.weight",
    }

    # When: Mapping representative HF tensor names
    # Then: Every representative tensor maps to the expected canonical GGUF name
    for hf_name, expected_gguf_name in representative_hf_mappings.items():
        mapped_name = tensor_map.get_name(hf_name, try_suffixes=suffixes)
        assert mapped_name == expected_gguf_name, (
            f"HF tensor {hf_name!r} mapped to {mapped_name!r}, expected {expected_gguf_name!r}"
        )


# ---------------------------------------------------------------------------
# Test 6: End-to-end HF to GGUF conversion (Variant A: no PLE, Variant B: PLE)
# ---------------------------------------------------------------------------
def _build_synthetic_qwen4exp_hf_model(
    model_dir: Path,
    ple_enabled: bool,
    hidden_size: int = 16,
) -> dict[str, Any]:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)
    tok.decoder = decoders.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)
    trainer = trainers.BpeTrainer(vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator([], trainer=trainer)
    tok.save(str(model_dir / "tokenizer.json"))

    tok_cfg = {
        "tokenizer_class": "Qwen2TokenizerFast",
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "added_tokens_decoder": {},
    }
    (model_dir / "tokenizer_config.json").write_text(json.dumps(tok_cfg))

    vocab_size = tok.get_vocab_size() + 10
    intermediate_size = 32
    moe_intermediate_size = 16
    num_experts = 2
    num_experts_per_tok = 1
    num_hidden_layers = 1
    num_attention_heads = 2
    num_key_value_heads = 2
    head_dim = 8
    max_position_embeddings = 128
    hc_count = 2
    hc_lowrank = 4
    indexer_n_heads = 2
    indexer_head_dim = 4
    indexer_budget = 2
    indexer_compress_ratio = 2
    eos_token_id = 151643
    image_token_id = 151655

    config: dict[str, Any] = {
        "architectures": ["Qwen4ExpForCausalLM"],
        "model_type": "qwen4exp",
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "moe_intermediate_size": moe_intermediate_size,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "num_hidden_layers": num_hidden_layers,
        "rms_norm_eps": 1e-6,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "max_position_embeddings": max_position_embeddings,
        "hc_count": hc_count,
        "hc_lowrank": hc_lowrank,
        "indexer_n_heads": indexer_n_heads,
        "indexer_head_dim": indexer_head_dim,
        "indexer_budget": indexer_budget,
        "indexer_compress_ratio": indexer_compress_ratio,
        "layer_types": ["full_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 4,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_value_head_dim": 4,
        "eos_token_id": eos_token_id,
        "ple_layer_ids": [],
    }

    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(vocab_size, hidden_size, dtype=torch.float32),
        "lm_head.weight": torch.randn(vocab_size, hidden_size, dtype=torch.float32),
        "model.norm.hc_head_norm.weight": torch.randn(hidden_size, dtype=torch.float32),
        "model.norm.hc_head_down.weight": torch.randn(hc_lowrank, hidden_size, dtype=torch.float32),
        "model.norm.hc_head_up.weight": torch.randn(hidden_size, hc_lowrank, dtype=torch.float32),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(hidden_size, hidden_size, dtype=torch.float32),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(hidden_size, hidden_size, dtype=torch.float32),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(hidden_size, hidden_size, dtype=torch.float32),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(hidden_size, hidden_size, dtype=torch.float32),
        "model.layers.0.mlp.gate.weight": torch.randn(num_experts, hidden_size, dtype=torch.float32),
        "model.layers.0.mlp.experts.gate_up_proj.weight": torch.randn(
            num_experts, intermediate_size, hidden_size, dtype=torch.float32
        ),
        "model.layers.0.mlp.experts.down_proj.weight": torch.randn(
            num_experts, hidden_size, moe_intermediate_size, dtype=torch.float32
        ),
        "model.layers.0.hc_attn_norm.weight": torch.randn(hidden_size, dtype=torch.float32),
        "model.layers.0.hc_attn_down.weight": torch.randn(hc_lowrank, hidden_size, dtype=torch.float32),
        "model.layers.0.hc_attn_up.weight": torch.randn(hidden_size, hc_lowrank, dtype=torch.float32),
        "model.layers.0.hc_attn_inject.weight": torch.randn(hidden_size, hc_lowrank, dtype=torch.float32),
        "model.layers.0.hc_ffn_norm.weight": torch.randn(hidden_size, dtype=torch.float32),
        "model.layers.0.hc_ffn_down.weight": torch.randn(hc_lowrank, hidden_size, dtype=torch.float32),
        "model.layers.0.hc_ffn_up.weight": torch.randn(hidden_size, hc_lowrank, dtype=torch.float32),
        "model.layers.0.hc_ffn_inject.weight": torch.randn(hidden_size, hc_lowrank, dtype=torch.float32),
        "model.layers.0.indexer.index_qk_proj.weight": torch.randn(hidden_size, hidden_size, dtype=torch.float32),
        "model.layers.0.indexer.q_norm.weight": torch.randn(indexer_head_dim * indexer_n_heads, dtype=torch.float32),
        "model.layers.0.indexer.k_norm.weight": torch.randn(indexer_head_dim * indexer_n_heads, dtype=torch.float32),
    }

    expected_info: dict[str, Any] = {
        "config": config,
        "ple_enabled": ple_enabled,
    }

    if ple_enabled:
        config["ple_layer_ids"] = [1]
        config["ngram_size"] = 3
        config["heads_per_ngram"] = 2
        config["ple_conv_kernel_size"] = 3
        config["split_ngram_parts"] = 2
        config["image_token_id"] = image_token_id

        shard0 = torch.randn(4, hidden_size, dtype=torch.float32)
        shard1 = torch.randn(4, hidden_size, dtype=torch.float32)
        expected_ple_table = torch.cat([shard0, shard1], dim=0)

        mult_vals = [2**33 + 12345, 2**45 + 67890]
        offset_vals = [2**34 + 1, 2**34 + 2]
        vocab_vals = [2**35 + 100, 2**35 + 200]

        tensors.update({
            "model.layers.0.ple.key.weight": torch.randn(hidden_size, hidden_size, dtype=torch.float32),
            "model.layers.0.ple.value.weight": torch.randn(hidden_size, hidden_size, dtype=torch.float32),
            "model.layers.0.ple.norm_key.weight": torch.randn(hidden_size, dtype=torch.float32),
            "model.layers.0.ple.norm_query.weight": torch.randn(hidden_size, dtype=torch.float32),
            "model.layers.0.ple.norm_conv.weight": torch.randn(hidden_size, dtype=torch.float32),
            "model.layers.0.ple.conv1d.weight": torch.randn(hidden_size, 1, 3, dtype=torch.float32),
            "model.layers.0.ple_embedding.layer_multipliers": torch.tensor(mult_vals, dtype=torch.int64),
            "model.layers.0.ple_embedding.ngram_heads_offsets": torch.tensor(offset_vals, dtype=torch.int64),
            "model.layers.0.ple_embedding.ngram_heads_vocab_sizes": torch.tensor(vocab_vals, dtype=torch.int64),
            "model.layers.0.ngram_embedding.shard_0.weight": shard0,
            "model.layers.0.ngram_embedding.shard_1.weight": shard1,
        })

        expected_info["expected_ple_table"] = expected_ple_table
        expected_info["mult_vals"] = mult_vals
        expected_info["offset_vals"] = offset_vals
        expected_info["vocab_vals"] = vocab_vals

    (model_dir / "config.json").write_text(json.dumps(config))
    save_file(tensors, str(model_dir / "model.safetensors"))
    return expected_info


@pytest.mark.parametrize("ple_enabled", [False, True], ids=["variant_a_no_ple", "variant_b_with_ple"])
def test_qwen4exp_e2e_conversion(tmp_path: Path, ple_enabled: bool):
    # Given: A synthetic miniature Qwen4Exp checkpoint with config, tokenizer, and safetensors
    _ensure_import_path()
    import gguf

    model_dir = tmp_path / ("model_ple" if ple_enabled else "model_no_ple")
    model_dir.mkdir()
    out_gguf = tmp_path / ("out_ple.gguf" if ple_enabled else "out_no_ple.gguf")

    expected_info = _build_synthetic_qwen4exp_hf_model(model_dir, ple_enabled=ple_enabled)
    config = expected_info["config"]

    script_path = REPO_ROOT / "convert_hf_to_gguf.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'gguf-py'}:{REPO_ROOT}"

    # When: Executing convert_hf_to_gguf.py on the synthetic checkpoint
    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            str(model_dir),
            "--outfile",
            str(out_gguf),
            "--outtype",
            "f32",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )

    # Then: 1. Conversion completes with returncode 0
    assert proc.returncode == 0, (
        f"convert_hf_to_gguf.py failed with code {proc.returncode}:\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    # And Then: 2. Output GGUF file is valid, loadable, and general.architecture is qwen4exp
    assert out_gguf.is_file(), f"Output GGUF file not found at {out_gguf}"
    reader = gguf.GGUFReader(out_gguf)
    arch_field = reader.get_field("general.architecture")
    assert arch_field is not None, "Missing general.architecture field in GGUF"
    assert arch_field.contents() == "qwen4exp"

    # And Then: 3. Critical metadata keys are present and correct
    hc_count_field = reader.get_field("qwen4exp.hyper_connection.count")
    assert hc_count_field is not None and hc_count_field.contents() == config["hc_count"]

    hc_lowrank_field = reader.get_field("qwen4exp.hyper_connection.low_rank")
    assert hc_lowrank_field is not None and hc_lowrank_field.contents() == config["hc_lowrank"]

    idx_heads_field = reader.get_field("qwen4exp.attention.indexer.head_count")
    assert idx_heads_field is not None and idx_heads_field.contents() == config["indexer_n_heads"]

    idx_key_field = reader.get_field("qwen4exp.attention.indexer.key_length")
    assert idx_key_field is not None and idx_key_field.contents() == config["indexer_head_dim"]

    idx_budget_field = reader.get_field("qwen4exp.attention.indexer.top_k")
    assert idx_budget_field is not None and idx_budget_field.contents() == config["indexer_budget"]

    compress_ratios_field = reader.get_field("qwen4exp.attention.compress_ratios")
    assert compress_ratios_field is not None
    assert [int(x) for x in compress_ratios_field.contents()] == [config["indexer_compress_ratio"]]

    # And Then: 4. Canonical tensor names in GGUF match expected naming
    gguf_tensor_names = {t.name for t in reader.tensors}

    expected_canonical_tensors = {
        "output.weight",
        "token_embd.weight",
        "blk.0.indexer.q_proj.weight",
        "blk.0.indexer.k_proj.weight",
        "blk.0.indexer.q_norm.weight",
        "blk.0.indexer.k_norm.weight",
        "output_hc_norm.weight",
        "output_hc_down.weight",
        "output_hc_up.weight",
        "blk.0.hc_attn_norm.weight",
        "blk.0.hc_attn_down.weight",
        "blk.0.hc_attn_up.weight",
        "blk.0.hc_attn_inject.weight",
        "blk.0.hc_ffn_norm.weight",
        "blk.0.hc_ffn_down.weight",
        "blk.0.hc_ffn_up.weight",
        "blk.0.hc_ffn_inject.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate_inp.weight",
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
    }

    for expected_name in expected_canonical_tensors:
        assert expected_name in gguf_tensor_names, f"Expected tensor {expected_name!r} missing from GGUF"

    # And Then: 5. Exercise variant-specific behaviors
    if not ple_enabled:
        assert reader.get_field("qwen4exp.ple.layers") is None
        assert reader.get_field("qwen4exp.ple.ngram_size") is None
        assert "per_layer_token_embd.weight" not in gguf_tensor_names
        assert "blk.0.ple_key.weight" not in gguf_tensor_names
    else:
        ple_layers_field = reader.get_field("qwen4exp.ple.layers")
        assert ple_layers_field is not None
        assert [int(x) for x in ple_layers_field.contents()] == [0]

        ngram_size_field = reader.get_field("qwen4exp.ple.ngram_size")
        assert ngram_size_field is not None and ngram_size_field.contents() == config["ngram_size"]

        heads_ngram_field = reader.get_field("qwen4exp.ple.heads_per_ngram")
        assert heads_ngram_field is not None and heads_ngram_field.contents() == config["heads_per_ngram"]

        conv_kernel_field = reader.get_field("qwen4exp.ple.conv_kernel")
        assert conv_kernel_field is not None and conv_kernel_field.contents() == config["ple_conv_kernel_size"]

        eos_field = reader.get_field("qwen4exp.ple.eos_token_id")
        assert eos_field is not None and eos_field.contents() == config["eos_token_id"]

        image_field = reader.get_field("qwen4exp.ple.image_token_id")
        assert image_field is not None and image_field.contents() == config["image_token_id"]

        mult_field = reader.get_field("qwen4exp.ple.layer_multipliers")
        assert mult_field is not None
        assert mult_field.types == [gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT64]
        assert [int(x) for x in mult_field.contents()] == expected_info["mult_vals"]

        offsets_field = reader.get_field("qwen4exp.ple.head_offsets")
        assert offsets_field is not None
        assert offsets_field.types == [gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT64]
        assert [int(x) for x in offsets_field.contents()] == expected_info["offset_vals"]

        vocab_field = reader.get_field("qwen4exp.ple.head_vocab_sizes")
        assert vocab_field is not None
        assert vocab_field.types == [gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT64]
        assert [int(x) for x in vocab_field.contents()] == expected_info["vocab_vals"]

        temp_memmap_file = out_gguf.parent / f".{out_gguf.stem}.ple.tmp"
        assert not temp_memmap_file.exists(), (
            f"Temporary memmap file {temp_memmap_file} was not removed after conversion"
        )

        expected_ple_tensors = {
            "per_layer_token_embd.weight",
            "blk.0.ple_key.weight",
            "blk.0.ple_value.weight",
            "blk.0.ple_norm_key.weight",
            "blk.0.ple_norm_query.weight",
            "blk.0.ple_norm_conv.weight",
            "blk.0.ple_conv1d.weight",
        }
        for ple_tensor_name in expected_ple_tensors:
            assert ple_tensor_name in gguf_tensor_names, (
                f"Expected PLE tensor {ple_tensor_name!r} missing from GGUF"
            )

        ple_tensor = next(t for t in reader.tensors if t.name == "per_layer_token_embd.weight")
        expected_table_np = expected_info["expected_ple_table"].numpy()
        assert np.allclose(ple_tensor.data, expected_table_np), (
            "Merged PLE table in GGUF does not match expected concatenation of shards"
        )


# ---------------------------------------------------------------------------
# Test 7: llama-quantize must protect the PLE n-gram lookup table
# ---------------------------------------------------------------------------
def test_qwen4exp_rocmfp4_fast_protects_ple_table(tmp_path: Path):
    # Given: A miniature qwen4exp GGUF converted from the synthetic checkpoint
    _ensure_import_path()
    import gguf

    quantize_bin = REPO_ROOT / "build-strix-rocmfp4" / "bin" / "llama-quantize"
    if not quantize_bin.is_file():
        pytest.skip(f"llama-quantize binary not built at {quantize_bin}")

    model_dir = tmp_path / "model_quant_src"
    model_dir.mkdir()
    f16_gguf = tmp_path / "qwen4exp-f16.gguf"
    quant_gguf = tmp_path / "qwen4exp-rocmfp4-fast.gguf"

    _build_synthetic_qwen4exp_hf_model(model_dir, ple_enabled=True, hidden_size=64)

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'gguf-py'}:{REPO_ROOT}"

    convert_proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "convert_hf_to_gguf.py"),
            str(model_dir),
            "--outfile", str(f16_gguf),
            "--outtype", "f16",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert convert_proc.returncode == 0, (
        f"conversion failed with code {convert_proc.returncode}:\n{convert_proc.stderr}"
    )

    # When: Quantizing to Q4_0_ROCMFP4_FAST
    quant_proc = subprocess.run(
        [str(quantize_bin), str(f16_gguf), str(quant_gguf), "Q4_0_ROCMFP4_FAST"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert quant_proc.returncode == 0, (
        f"llama-quantize failed with code {quant_proc.returncode}:\n"
        f"STDOUT:\n{quant_proc.stdout}\nSTDERR:\n{quant_proc.stderr}"
    )

    # Then: The PLE n-gram lookup table survives at Q8_0 precision instead of
    # inheriting the low-bit base type (it is gathered via hash lookups, not GEMMs)
    quant_reader = gguf.GGUFReader(quant_gguf)
    ple_tensor = next(
        (t for t in quant_reader.tensors if t.name == "per_layer_token_embd.weight"), None
    )
    assert ple_tensor is not None, "per_layer_token_embd.weight missing after quantization"
    observed_type = int(ple_tensor.tensor_type)
    expected_type = int(gguf.GGMLQuantizationType.Q8_0)
    assert observed_type == expected_type, (
        f"PLE lookup table was quantized to type code {observed_type} "
        f"(tensor_type={ple_tensor.tensor_type!r}); expected Q8_0 ({expected_type})"
    )


def test_qwen4exp_rocmfp4_tensor_type_override_pins_ple_table(tmp_path: Path):
    _ensure_import_path()
    import gguf

    quantize_bin = REPO_ROOT / "build-strix-rocmfp4" / "bin" / "llama-quantize"
    if not quantize_bin.is_file():
        pytest.skip(f"llama-quantize binary not built at {quantize_bin}")

    model_dir = tmp_path / "model_pin_src"
    model_dir.mkdir()
    f16_gguf = tmp_path / "qwen4exp-pin-f16.gguf"
    quant_gguf = tmp_path / "qwen4exp-pin.gguf"

    _build_synthetic_qwen4exp_hf_model(model_dir, ple_enabled=True, hidden_size=64)

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'gguf-py'}:{REPO_ROOT}"

    convert_proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "convert_hf_to_gguf.py"),
            str(model_dir),
            "--outfile", str(f16_gguf),
            "--outtype", "f16",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert convert_proc.returncode == 0, (
        f"conversion failed with code {convert_proc.returncode}:\n{convert_proc.stderr}"
    )

    quant_proc = subprocess.run(
        [
            str(quantize_bin),
            "--tensor-type", "per_layer_token_embd.weight=f16",
            str(f16_gguf),
            str(quant_gguf),
            "Q4_0_ROCMFP4_FAST",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert quant_proc.returncode == 0, (
        f"llama-quantize failed with code {quant_proc.returncode}:\n"
        f"STDOUT:\n{quant_proc.stdout}\nSTDERR:\n{quant_proc.stderr}"
    )

    quant_reader = gguf.GGUFReader(quant_gguf)
    ple_tensor = next(
        (t for t in quant_reader.tensors if t.name == "per_layer_token_embd.weight"), None
    )
    assert ple_tensor is not None, "per_layer_token_embd.weight missing after quantization"
    observed_type = int(ple_tensor.tensor_type)
    expected_type = int(gguf.GGMLQuantizationType.F16)
    assert observed_type == expected_type, (
        f"--tensor-type override ignored: PLE table got type code {observed_type} "
        f"(tensor_type={ple_tensor.tensor_type!r}); expected F16 ({expected_type})"
    )

