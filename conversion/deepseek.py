from __future__ import annotations

import json
import re

from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TYPE_CHECKING

import numpy as np
import torch

TORCH_FLOAT8_E8M0FNU = getattr(torch, "float8_e8m0fnu", None)

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, MmprojModel, ModelBase, TextModel, gguf, logger

try:
    from .qwen import QwenModel
except ModuleNotFoundError:
    class QwenModel:
        @staticmethod
        def token_bytes_to_string(b: bytes) -> str:
            from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode  # type: ignore[import-not-found]
            byte_encoder = bytes_to_unicode()
            return ''.join([byte_encoder[ord(char)] for char in b.decode('latin-1')])

        @staticmethod
        def bpe(mergeable_ranks: dict[bytes, int], token: bytes, max_rank: int | None = None) -> list[bytes]:
            parts = [bytes([b]) for b in token]
            while True:
                min_idx = None
                min_rank = None
                for i, pair in enumerate(zip(parts[:-1], parts[1:])):
                    rank = mergeable_ranks.get(pair[0] + pair[1])
                    if rank is not None and (min_rank is None or rank < min_rank):
                        min_idx = i
                        min_rank = rank
                if min_rank is None or (max_rank is not None and min_rank >= max_rank):
                    break
                assert min_idx is not None
                parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
            return parts


@ModelBase.register("DeepseekOCRForCausalLM")
class DeepseekOCRVisionModel(MmprojModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_projector_type = gguf.VisionProjectorType.DEEPSEEKOCR

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_clip_projector_type(self.clip_projector_type)
        # default values below are taken from HF tranformers code
        self.gguf_writer.add_vision_attention_layernorm_eps(hparams.get("layer_norm_eps", 1e-6))
        self.gguf_writer.add_vision_use_gelu(True)
        # calculate proj_scale_factor (used by tinygemma3 test model)
        image_seq_length = self.preprocessor_config.get("image_seq_length", 256)
        n_per_side = int(image_seq_length ** 0.5)
        image_size = self.hparams["image_size"]
        patch_size = self.hparams["patch_size"]
        proj_scale_factor = (image_size // patch_size) // n_per_side
        if proj_scale_factor > 0 and proj_scale_factor != 4:
            # we only need to write this if it's not the default value
            # in this case, we are converting a test model
            self.gguf_writer.add_vision_projector_scale_factor(proj_scale_factor)
        # @bluebread: there's no window_size in config but just add it here anyway
        self.gguf_writer.add_vision_window_size(self.hparams.get("window_size", 14))

        # SAM configuration
        sam_hparams = hparams['sam']
        self.gguf_writer.add_vision_sam_layers_count(sam_hparams['layers'])
        self.gguf_writer.add_vision_sam_embedding_length(sam_hparams['width'])
        self.gguf_writer.add_vision_sam_head_count(sam_hparams['heads'])

    def get_vision_config(self) -> dict[str, Any]:
        vision_config: dict[str, Any] | None = self.global_config.get("vision_config")

        if not vision_config:
            raise ValueError("DeepseekOCR model requires 'vision_config' in the model configuration, but it was not found")

        vision_config['sam'] = vision_config['width']['sam_vit_b']
        if vision_config['width'].get('clip-l-14-224') is not None:
            vision_config.update(vision_config['width']['clip-l-14-224'])
        if isinstance(vision_config['width'], int):
            vision_config['hidden_size'] = vision_config['width']
        if vision_config.get('heads') is not None:
            vision_config['num_heads'] = vision_config['heads']
            vision_config['intermediate_size'] = vision_config['heads'] * 4

        return vision_config

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        for nq_name in ('.embeddings.', 'pos_embed', '.rel_pos_h', '.rel_pos_w', '.neck.', '.net_'):
            if nq_name in name:
                return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith("view_seperator"):
            data_torch = data_torch.unsqueeze(0)
        yield from super().modify_tensors(data_torch, name, bid)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # Only process vision-related tensors, skip language model tensors
        # Vision components: sam_model, vision_model, projector, image_newline, view_seperator
        # Language model components to skip: lm_head, embed_tokens, layers, norm
        if name.startswith(("lm_head.", "model.embed_tokens.", "model.layers.", "model.norm.")):
            return None

        if name.endswith("pos_embed") or name.endswith("rel_pos_h") or name.endswith("rel_pos_w"):
            name += ".weight"

        return super().filter_tensors((name, gen))


@ModelBase.register("DeepseekOCR2ForCausalLM")
class DeepseekOCR2VisionModel(DeepseekOCRVisionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_projector_type = gguf.VisionProjectorType.DEEPSEEKOCR2

    def set_gguf_parameters(self):
        # the vision tower's qwen2 encoder is built from fixed defaults,
        # see build_qwen2_decoder_as_encoder() in deepencoderv2.py
        if self.hparams.get("patch_size") is None:
            self.hparams["patch_size"] = 16
        if self.hparams.get("intermediate_size") is None:
            self.hparams["intermediate_size"] = 4864
        if self.hparams.get("num_attention_heads") is None:
            self.hparams["num_attention_heads"] = 14
        super().set_gguf_parameters()
        # qwen2 encoder is GQA: 14 Q heads, 2 KV heads
        self.gguf_writer.add_vision_head_count_kv(2)

    def get_vision_config(self) -> dict[str, Any]:
        vision_config = super().get_vision_config()
        vision_config['hidden_size'] = vision_config['width']['qwen2-0-5b']['dim']
        if vision_config.get('layers') is None:
            vision_config['layers'] = 24
        return vision_config


@ModelBase.register("DeepseekForCausalLM")
class DeepseekModel(TextModel):
    model_arch = gguf.MODEL_ARCH.DEEPSEEK

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        if (rope_dim := hparams.get("head_dim")) is None:
            rope_dim = hparams["hidden_size"] // hparams["num_attention_heads"]

        self.gguf_writer.add_rope_dimension_count(rope_dim)
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_weights_scale(1.0)
        self.gguf_writer.add_expert_count(hparams["n_routed_experts"])
        self.gguf_writer.add_expert_shared_count(hparams["n_shared_experts"])

    _experts: list[dict[str, Tensor]] | None = None

    @staticmethod
    def permute(weights: Tensor, n_head: int, n_head_kv: int | None):
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")

        if name.endswith(("q_proj.weight", "q_proj.bias")):
            data_torch = DeepseekModel.permute(data_torch, n_head, n_head)
        if name.endswith(("k_proj.weight", "k_proj.bias")):
            data_torch = DeepseekModel.permute(data_torch, n_head, n_kv_head)

        # process the experts separately
        if name.find("mlp.experts") != -1:
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    yield from super().modify_tensors(data_torch, merged_name, bid)
                return
            else:
                return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@ModelBase.register(
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
    "KimiVLForConditionalGeneration",
    "KimiK25ForConditionalGeneration",
    "YoutuForCausalLM",
    "YoutuVLForConditionalGeneration",
)
class DeepseekV2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.DEEPSEEK2

    # TODO @ngxson : remove this when we support MTP for deepseek models
    skip_mtp = True

    merge_expert = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hparams: dict = ModelBase.load_hparams(self.dir_model, is_mistral_format=False)
        self.origin_hf_arch = hparams.get('architectures', [None])[0]

        # special handling for Deepseek OCR
        if self.origin_hf_arch in ("DeepseekOCRForCausalLM", "DeepseekOCR2ForCausalLM"):
            self.model_arch = gguf.MODEL_ARCH.DEEPSEEK2OCR
            self.gguf_writer.arch = gguf.MODEL_ARCH_NAMES[self.model_arch]
            self.gguf_writer.add_architecture()
            # default jinja template
            self.gguf_writer.add_chat_template("{% for m in messages %}{{m['content']}}{% endfor %}")

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item
        # DeepSeek-OCR vision encoder (SAM + DeepSeek-OCR-2 qwen2 tower)
        if "sam_model" in name or "qwen2_model" in name:
            return None
        return super().filter_tensors(item)

    def set_vocab(self):
        try:
            self._set_vocab_gpt2()
            return
        except Exception:
            pass

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)
        tokpre = self.get_vocab_base_pre(tokenizer)

        if tokpre == "kimi-k2":
            # Build merges list using the approach similar to HunYuanMoE
            merges = []
            vocab = {}
            mergeable_ranks = tokenizer.model._mergeable_ranks  # ty: ignore[unresolved-attribute]
            for token, rank in mergeable_ranks.items():
                vocab[QwenModel.token_bytes_to_string(token)] = rank
                if len(token) == 1:
                    continue
                merged = QwenModel.bpe(mergeable_ranks, token, max_rank=rank)
                if len(merged) == 2:
                    merges.append(' '.join(map(QwenModel.token_bytes_to_string, merged)))

            # Build token list
            vocab_size = self.hparams["vocab_size"]
            special_tokens = tokenizer.special_tokens  # ty: ignore[unresolved-attribute]
            reverse_vocab = {id_ : encoded_tok for encoded_tok, id_ in {**vocab, **special_tokens}.items()}
            tokens: list[str] = []
            toktypes: list[int] = []

            for i in range(vocab_size):
                if i not in reverse_vocab:
                    tokens.append(f"[PAD{i}]")
                    toktypes.append(gguf.TokenType.UNUSED)
                else:
                    token = reverse_vocab[i]
                    tokens.append(token)
                    if i in special_tokens.values():
                        toktypes.append(gguf.TokenType.CONTROL)
                    else:
                        toktypes.append(gguf.TokenType.NORMAL)

            self.gguf_writer.add_tokenizer_model("gpt2")
            self.gguf_writer.add_tokenizer_pre(tokpre)
            self.gguf_writer.add_token_list(tokens)
            self.gguf_writer.add_token_types(toktypes)
            self.gguf_writer.add_token_merges(merges)

            special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False)
            special_vocab.add_to_gguf(self.gguf_writer)
        else:
            raise NotImplementedError(f"Deepseek pre-tokenizer {tokpre!r} is not supported yet!")

    def set_gguf_parameters(self):
        is_ocr = (self.model_arch == gguf.MODEL_ARCH.DEEPSEEK2OCR)

        if is_ocr:
            self.hparams['rope_theta'] = self.hparams.get('rope_theta', 10000.0)
        else:
            # note: deepseek2 using MLA converts into MQA (ie: GQA with 1 group)
            self.hparams["num_key_value_heads"] = 1

        self.hparams['rms_norm_eps'] = self.hparams.get('rms_norm_eps', 1e-6)

        super().set_gguf_parameters()
        hparams = self.hparams

        # first_k_dense_replace: number of leading layers using dense FFN instead of MoE
        # For non-MoE models (like Youtu), set to n_layer to use dense FFN for all layers
        # For MoE models (like DeepSeek-V2), this is the number of leading non-MoE layers
        has_moe = hparams.get("n_routed_experts") is not None
        first_k_dense_replace = hparams.get("first_k_dense_replace")
        if first_k_dense_replace is None:
            # Default: if no MoE, all layers are dense; if MoE, none are dense
            first_k_dense_replace = hparams["num_hidden_layers"] if not has_moe else 0
        self.gguf_writer.add_leading_dense_block_count(first_k_dense_replace)
        kv_lora_rank = hparams.get("kv_lora_rank", 512)
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        if "q_lora_rank" in hparams and hparams["q_lora_rank"] is not None:
            self.gguf_writer.add_q_lora_rank(hparams["q_lora_rank"])

        # note: deepseek2 using MLA converts into MQA with larger heads, then decompresses to MHA
        if not is_ocr:
            self.gguf_writer.add_kv_lora_rank(kv_lora_rank)
            self.gguf_writer.add_key_length(kv_lora_rank + hparams["qk_rope_head_dim"])
            self.gguf_writer.add_value_length(kv_lora_rank)
            self.gguf_writer.add_key_length_mla(hparams["qk_nope_head_dim"] + hparams["qk_rope_head_dim"])
            self.gguf_writer.add_value_length_mla(hparams["v_head_dim"])

        # MoE parameters (required by C++ code for DEEPSEEK2 arch)
        # For non-MoE models like Youtu, use intermediate_size as expert_feed_forward_length
        moe_intermediate_size = self.find_hparam(["moe_intermediate_size", "intermediate_size"], optional=False)
        self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)

        if (n_routed_experts := hparams.get("n_routed_experts")) is not None:
            self.gguf_writer.add_expert_count(n_routed_experts)

        # expert_shared_count is required by C++ code, default to 0 for non-MoE models
        n_shared_experts = hparams.get("n_shared_experts", 0)
        self.gguf_writer.add_expert_shared_count(n_shared_experts)

        # When not set, C++ code will use scale_w = false to skip the no-op scaling
        if (routed_scaling_factor := hparams.get("routed_scaling_factor")) is not None:
            self.gguf_writer.add_expert_weights_scale(routed_scaling_factor)

        if (norm_topk_prob := hparams.get("norm_topk_prob")) is not None and norm_topk_prob:
            self.gguf_writer.add_expert_weights_norm(norm_topk_prob)

        self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])

        if (rope_mscale_all := self.rope_parameters.get("mscale_all_dim")) is not None:
            # [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            # note: for legacy reasons, this is not consistent with the other usages of self.gguf_writer.add_rope_scaling_yarn_log_mul
            # ref https://github.com/ggml-org/llama.cpp/pull/17945
            self.gguf_writer.add_rope_scaling_yarn_log_mul(0.1 * rope_mscale_all)

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # skip lm_head.weight if tie_word_embeddings is True
        if self.hparams.get("tie_word_embeddings", False):
            if name == "lm_head.weight" or name == "model.lm_head.weight":
                logger.info("Skipping tied output layer 'lm_head.weight' (will use token_embd.weight)")
                return

        # skip Multi-Token Prediction (MTP) layers
        if self.skip_mtp:
            block_count = self.hparams["num_hidden_layers"]
            match = re.match(r"model.layers.(\d+)", name)
            if match and int(match.group(1)) >= block_count:
                return

        # process the experts separately
        if self.merge_expert and name.find("mlp.experts") != -1:
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    yield from super().modify_tensors(data_torch, merged_name, bid)
                return
            else:
                return

        # note: MLA with the absorption optimization, needs these two split and k_b_proj transposed
        if name.endswith("kv_b_proj.weight"):
            name_kb = name.replace("kv_b_proj", "k_b_proj")
            name_vb = name.replace("kv_b_proj", "v_b_proj")

            n_head_kv = self.hparams["num_key_value_heads"]
            v_head_dim = self.hparams["v_head_dim"]
            qk_nope_head_dim = self.hparams["qk_nope_head_dim"]

            assert data_torch.shape[0] == n_head_kv * (v_head_dim + qk_nope_head_dim)

            kv_b = data_torch.view(n_head_kv, v_head_dim + qk_nope_head_dim, data_torch.shape[-1])
            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
            k_b = k_b.transpose(1, 2)

            yield from super().modify_tensors(k_b, name_kb, bid)
            yield from super().modify_tensors(v_b, name_vb, bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@ModelBase.register("DeepseekV32ForCausalLM")
class DeepseekV32Model(DeepseekV2Model):
    model_arch = gguf.MODEL_ARCH.DEEPSEEK32
    skip_mtp = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_count = self.hparams["num_hidden_layers"] + self.hparams.get("num_nextn_predict_layers", 0)
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def set_vocab(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        assert getattr(tokenizer, "add_bos_token", False), "Change value of add_bos_token to true in tokenizer_config.json file."
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        # NextN/MTP prediction layers
        if (num_nextn_predict_layers := self.hparams.get("num_nextn_predict_layers")) is not None:
            self.gguf_writer.add_nextn_predict_layers(num_nextn_predict_layers)

        # DSA indexer parameters
        self.gguf_writer.add_indexer_head_count(self.hparams["index_n_heads"])
        self.gguf_writer.add_indexer_key_length(self.hparams["index_head_dim"])
        self.gguf_writer.add_indexer_top_k(self.hparams["index_topk"])


@ModelBase.register("DeepseekV4ForCausalLM")
class DeepseekV4Model(TextModel):
    model_arch = gguf.MODEL_ARCH.DEEPSEEK4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with open(self.dir_model / "config.json", "r", encoding="utf-8") as config_file:
            raw_hparams = json.load(config_file)
        for key in (
            "compress_ratios",
            "compress_rope_theta",
            "hc_eps",
            "hc_mult",
            "hc_sinkhorn_iters",
            "index_head_dim",
            "index_n_heads",
            "index_topk",
            "num_hash_layers",
            "num_nextn_predict_layers",
            "o_groups",
            "o_lora_rank",
            "quantization_config",
        ):
            if key in raw_hparams:
                self.hparams[key] = raw_hparams[key]

        self._deepseek4_original_block_count = self.block_count
        self._deepseek4_main_block_count = self.block_count
        if self.deepseek4_max_layers is not None:
            if self.deepseek4_max_layers <= 0:
                raise ValueError("--deepseek4-max-layers must be positive")
            if self.deepseek4_max_layers > self.block_count:
                raise ValueError(
                    f"--deepseek4-max-layers={self.deepseek4_max_layers} exceeds model layer count {self.block_count}"
                )
            self._deepseek4_main_block_count = self.deepseek4_max_layers
            logger.warning(
                "DeepSeek V4 fixture export: writing only the first %d/%d transformer layers",
                self._deepseek4_main_block_count,
                self._deepseek4_original_block_count,
            )

        self._deepseek4_mtp_layers = (
            int(self.hparams.get("num_nextn_predict_layers", 0))
            if self.deepseek4_include_mtp
            else 0
        )
        if self.deepseek4_include_mtp and self._deepseek4_mtp_layers <= 0:
            raise ValueError("--deepseek4-include-mtp requested but the checkpoint contains no MTP layers")

        self.block_count = self._deepseek4_main_block_count + self._deepseek4_mtp_layers
        self.hparams["num_hidden_layers"] = self.block_count
        self.hparams["n_layers"] = self.block_count
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def set_vocab(self):
        self._set_vocab_gpt2()
        tokenizer_config_path = self.dir_model / "tokenizer_config.json"
        source_has_template = (
            (self.dir_model / "chat_template.jinja").is_file()
            or (self.dir_model / "chat_template.json").is_file()
        )
        if tokenizer_config_path.is_file():
            with open(tokenizer_config_path, "r", encoding="utf-8") as tokenizer_config_file:
                source_has_template |= json.load(tokenizer_config_file).get("chat_template") is not None
        if source_has_template:
            return

        fallback_template = Path(__file__).parent.parent / "models/templates/deepseek-ai-DeepSeek-V4.jinja"
        if not fallback_template.is_file():
            raise FileNotFoundError(f"missing DeepSeek V4 chat template fallback: {fallback_template}")
        self.gguf_writer.add_chat_template(fallback_template.read_text(encoding="utf-8"))

    def set_gguf_parameters(self):
        self.hparams["num_key_value_heads"] = self.hparams.get("num_key_value_heads", 1)
        super().set_gguf_parameters()
        hparams = self.hparams

        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])
        self.gguf_writer.add_q_lora_rank(hparams["q_lora_rank"])
        self.gguf_writer.add_attention_output_lora_rank(hparams["o_lora_rank"])
        self.gguf_writer.add_attention_output_group_count(hparams["o_groups"])
        compress_ratios = hparams["compress_ratios"]
        selected_compress_ratios = compress_ratios[:self._deepseek4_main_block_count]
        if self._deepseek4_mtp_layers:
            selected_compress_ratios += compress_ratios[
                self._deepseek4_original_block_count:
                self._deepseek4_original_block_count + self._deepseek4_mtp_layers
            ]
        self.gguf_writer.add_attention_compress_ratios(selected_compress_ratios)
        self.gguf_writer.add_attention_compress_rope_freq_base(hparams["compress_rope_theta"])
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_count(hparams["n_routed_experts"])
        self.gguf_writer.add_expert_shared_count(hparams["n_shared_experts"])
        self.gguf_writer.add_expert_weights_scale(hparams.get("routed_scaling_factor", 1.0))
        self.gguf_writer.add_hash_layer_count(min(hparams["num_hash_layers"], self._deepseek4_main_block_count))
        if self._deepseek4_mtp_layers:
            self.gguf_writer.add_nextn_predict_layers(self._deepseek4_mtp_layers)

        if (norm_topk_prob := hparams.get("norm_topk_prob")) is not None:
            self.gguf_writer.add_expert_weights_norm(norm_topk_prob)
        if (swiglu_limit := hparams.get("swiglu_limit")) is not None and float(swiglu_limit) > 0.0:
            self.gguf_writer.add_swiglu_clamp_exp([float(swiglu_limit)] * self.block_count)
        if (sliding_window := hparams.get("sliding_window")) is not None:
            self.gguf_writer.add_sliding_window(sliding_window)

        self.gguf_writer.add_indexer_head_count(hparams["index_n_heads"])
        self.gguf_writer.add_indexer_key_length(hparams["index_head_dim"])
        self.gguf_writer.add_indexer_top_k(hparams["index_topk"])
        self.gguf_writer.add_hyper_connection_count(hparams["hc_mult"])
        self.gguf_writer.add_hyper_connection_sinkhorn_iters(hparams["hc_sinkhorn_iters"])
        self.gguf_writer.add_hyper_connection_eps(hparams["hc_eps"])

    @staticmethod
    def _strip_model_prefix(name: str) -> str:
        return name.removeprefix("model.")

    def _normalize_tensor_name(self, name: str) -> str:
        stripped = self._strip_model_prefix(name)
        match = re.match(r"mtp\.(\d+)\.(.+)", stripped)
        if match is None:
            return stripped

        mtp_idx, rest = int(match.group(1)), match.group(2)
        bid = self._deepseek4_main_block_count + mtp_idx
        special = {
            "e_proj.weight":  "nextn.e_proj.weight",
            "h_proj.weight":  "nextn.h_proj.weight",
            "enorm.weight":   "nextn.enorm.weight",
            "hnorm.weight":   "nextn.hnorm.weight",
            "norm.weight":    "nextn.shared_head_norm.weight",
            "hc_head_base":   "nextn.hc_head_base.weight",
            "hc_head_fn":     "nextn.hc_head_fn.weight",
            "hc_head_scale":  "nextn.hc_head_scale.weight",
        }
        return f"layers.{bid}.{special.get(rest, rest)}"

    def _skip_tensor(self, name: str) -> bool:
        stripped = self._strip_model_prefix(name)
        if stripped.startswith("mtp."):
            if not self._deepseek4_mtp_layers:
                return True
            match = re.match(r"mtp\.(\d+)\.", stripped)
            return match is None or int(match.group(1)) >= self._deepseek4_mtp_layers
        if self.deepseek4_max_layers is None:
            return False
        match = re.match(r"layers\.(\d+)\.", stripped)
        return match is not None and int(match.group(1)) >= self._deepseek4_main_block_count

    @staticmethod
    def _scale_to_float(scale: Tensor) -> Tensor:
        if TORCH_FLOAT8_E8M0FNU is not None and scale.dtype == TORCH_FLOAT8_E8M0FNU:
            return scale.float()
        if scale.dtype in (torch.uint8, torch.int8):
            exponent = scale.view(torch.uint8).to(torch.int32)
            bits = torch.where(
                exponent == 0,
                torch.full_like(exponent, 0x00400000),
                exponent << 23,
            )
            return bits.view(torch.float32)
        return scale.float()

    @staticmethod
    def _scale_to_e8m0_bytes(scale: Tensor) -> Tensor:
        if TORCH_FLOAT8_E8M0FNU is not None and scale.dtype == TORCH_FLOAT8_E8M0FNU:
            return scale.view(torch.uint8)
        if scale.dtype in (torch.uint8, torch.int8):
            return scale.view(torch.uint8)

        scale = scale.float()
        exponent = torch.where(
            scale > 0,
            torch.floor(torch.log2(scale)).to(torch.int32) + 127,
            torch.zeros_like(scale, dtype=torch.int32),
        )
        return torch.clamp(exponent, 0, 255).to(torch.uint8)

    @classmethod
    def _dequant_fp8_weight(cls, weight: Tensor, scale: Tensor, block_size: Sequence[int]) -> Tensor:
        if len(block_size) != 2:
            raise ValueError(f"DeepSeek V4 expects 2D FP8 block scales, got block size {block_size}")

        block_out, block_in = block_size
        out_dim, in_dim = weight.shape
        if out_dim % block_out != 0 or in_dim % block_in != 0:
            raise ValueError(f"FP8 tensor shape {tuple(weight.shape)} is not divisible by block size {block_size}")

        scale = cls._scale_to_float(scale)
        expected_scale = (out_dim // block_out, in_dim // block_in)
        if tuple(scale.shape) != expected_scale:
            raise ValueError(f"FP8 scale shape {tuple(scale.shape)} does not match expected {expected_scale}")

        weight = weight.reshape(out_dim // block_out, block_out, in_dim // block_in, block_in)
        return (weight.float() * scale[:, None, :, None]).reshape(out_dim, in_dim)

    @classmethod
    def _pack_fp4_as_mxfp4(cls, weight: Tensor, scale: Tensor) -> np.ndarray:
        weight = weight.view(torch.uint8)
        out_dim, packed_in_dim = weight.shape
        in_dim = packed_in_dim * 2
        if in_dim % 32 != 0:
            raise ValueError(f"FP4 packed tensor shape {tuple(weight.shape)} does not contain 32-value blocks")

        n_blocks = in_dim // 32
        scale_e = cls._scale_to_e8m0_bytes(scale)
        if tuple(scale_e.shape) != (out_dim, n_blocks):
            raise ValueError(f"FP4 scale shape {tuple(scale_e.shape)} does not match expected {(out_dim, n_blocks)}")

        packed = weight.reshape(out_dim, n_blocks, 16)
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        values = torch.stack((low, high), dim=-1).reshape(out_dim, n_blocks, 32)
        quants = values[:, :, :16] | (values[:, :, 16:] << 4)
        return torch.cat((scale_e.unsqueeze(-1), quants), dim=-1).reshape(out_dim, n_blocks * 17).numpy()

    def _write_tid2eid_tensors(self) -> set[str]:
        consumed: set[str] = set()
        for name in list(self.model_tensors):
            if self._skip_tensor(name):
                consumed.add(name)
                continue
            stripped = self._normalize_tensor_name(name)
            if re.match(r"layers\.\d+\.ffn\.gate\.tid2eid$", stripped) is None:
                continue

            data = LazyTorchTensor.to_eager(self.model_tensors[name]()).to(torch.int32).numpy()
            self.gguf_writer.add_tensor(self.map_tensor_name(stripped), data)
            consumed.add(name)
        return consumed

    def _pad_fable_expert_tensor(self, weight: Tensor, wid: str) -> Tensor:
        n_embd = int(self.hparams["hidden_size"])
        n_ff_exp = int(self.hparams["moe_intermediate_size"])
        target_shape = {
            "w1": (n_ff_exp, n_embd),
            "w2": (n_embd, n_ff_exp),
            "w3": (n_ff_exp, n_embd),
        }[wid]

        if tuple(weight.shape) == target_shape:
            return weight

        if weight.ndim != 2 or weight.shape[0] > target_shape[0] or weight.shape[1] > target_shape[1]:
            raise ValueError(
                f"DeepSeek V4 expert {wid} shape {tuple(weight.shape)} cannot be padded to {target_shape}"
            )

        # DeepSeek V4 Fable is distributed with half-width routed experts. Zero padding is
        # only a compatibility fallback so the current runtime graph can load the model.
        padded = torch.zeros(target_shape, dtype=weight.dtype, device=weight.device)
        padded[:weight.shape[0], :weight.shape[1]] = weight
        logger.warning(
            "DeepSeek V4: zero-padding routed expert %s from %s to %s for loader compatibility",
            wid,
            tuple(weight.shape),
            target_shape,
        )
        return padded

    def _write_expert_tensors(self) -> set[str]:
        n_experts = self.hparams["n_routed_experts"]
        consumed: set[str] = set()
        groups: dict[tuple[int, str], dict[int, tuple[str, str | None]]] = {}

        for name in list(self.model_tensors):
            if self._skip_tensor(name):
                consumed.add(name)
                continue
            stripped = self._normalize_tensor_name(name)
            match = re.match(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight$", stripped)
            if match is None:
                continue

            bid, xid, wid = int(match.group(1)), int(match.group(2)), match.group(3)
            scale_name = f"{name.removesuffix('.weight')}.scale"
            model_scale_name = scale_name if scale_name in self.model_tensors else f"model.{scale_name}"
            if model_scale_name not in self.model_tensors:
                model_scale_name = None

            groups.setdefault((bid, wid), {})[xid] = (name, model_scale_name)
            consumed.add(name)
            if model_scale_name is not None:
                consumed.add(model_scale_name)

        for (bid, wid), group in sorted(groups.items()):
            missing = sorted(set(range(n_experts)).difference(group))
            if missing:
                raise ValueError(f"Missing DeepSeek V4 expert tensors for layer {bid} {wid}: {missing[:8]}")

            experts: list[np.ndarray] = []
            has_scales = any(scale_name is not None for _, scale_name in group.values())
            if has_scales and not all(scale_name is not None for _, scale_name in group.values()):
                raise ValueError(f"DeepSeek V4 expert tensors mix scaled and unscaled weights in layer {bid} {wid}")

            for xid in range(n_experts):
                weight_name, scale_name = group[xid]
                weight = LazyTorchTensor.to_eager(self.model_tensors[weight_name]())
                if scale_name is None:
                    weight = self._pad_fable_expert_tensor(weight, wid)
                    experts.append(weight.to(torch.float16).numpy())
                else:
                    scale = LazyTorchTensor.to_eager(self.model_tensors[scale_name]())
                    experts.append(self._pack_fp4_as_mxfp4(weight, scale))

            merged = np.stack(experts, axis=0)
            new_name = self.map_tensor_name(f"layers.{bid}.ffn.experts.{wid}.weight")
            if has_scales:
                logger.info("DeepSeek V4: preserving blk.%d %s routed experts as MXFP4", bid, wid)
                self.gguf_writer.add_tensor(new_name, merged, raw_dtype=gguf.GGMLQuantizationType.MXFP4)
            else:
                logger.info("DeepSeek V4: merging blk.%d %s routed experts as F16", bid, wid)
                self.gguf_writer.add_tensor(new_name, merged)

        return consumed

    def prepare_tensors(self):
        block_size = (self.hparams.get("quantization_config") or {}).get("weight_block_size", [128, 128])
        consumed = self._write_tid2eid_tensors()
        consumed.update(self._write_expert_tensors())

        for name in list(self.model_tensors):
            if name in consumed or self._skip_tensor(name):
                consumed.add(name)
                continue
            stripped = self._normalize_tensor_name(name)
            if not stripped.endswith(".scale"):
                continue

            weight_name = f"{name.removesuffix('.scale')}.weight"
            model_weight_name = weight_name if weight_name in self.model_tensors else f"model.{weight_name}"
            if model_weight_name not in self.model_tensors:
                raise ValueError(f"Missing DeepSeek V4 FP8 weight tensor for scale {stripped}")

            weight = self.model_tensors[model_weight_name]
            scale = self.model_tensors[name]
            self.model_tensors[model_weight_name] = (
                lambda weight=weight, scale=scale, block_size=block_size: self._dequant_fp8_weight(
                    LazyTorchTensor.to_eager(weight()),
                    LazyTorchTensor.to_eager(scale()),
                    block_size,
                )
            )
            self._fp8_dequantized.add(model_weight_name)
            consumed.add(name)

        for name in consumed:
            self.model_tensors.pop(name, None)

        super().prepare_tensors()

    def map_tensor_name(self, name: str, try_suffixes: Sequence[str] = (".weight", ".bias")) -> str:
        name = self._normalize_tensor_name(name)
        top_level: dict[str, tuple[gguf.MODEL_TENSOR, str]] = {
            "embed.weight":  (gguf.MODEL_TENSOR.TOKEN_EMBD, ".weight"),
            "norm.weight":   (gguf.MODEL_TENSOR.OUTPUT_NORM, ".weight"),
            "head.weight":   (gguf.MODEL_TENSOR.OUTPUT, ".weight"),
            "hc_head_base":  (gguf.MODEL_TENSOR.OUTPUT_HC_BASE, ".weight"),
            "hc_head_fn":    (gguf.MODEL_TENSOR.OUTPUT_HC_FN, ".weight"),
            "hc_head_scale": (gguf.MODEL_TENSOR.OUTPUT_HC_SCALE, ".weight"),
        }
        if name in top_level:
            tensor, suffix = top_level[name]
            return self.format_tensor_name(tensor, suffix=suffix)

        match = re.match(r"layers\.(\d+)\.(.+)", name)
        if match is None:
            return super().map_tensor_name(name, try_suffixes)

        bid, rest = int(match.group(1)), match.group(2)
        layer_level: dict[str, tuple[gguf.MODEL_TENSOR, str]] = {
            "hc_attn_base":                         (gguf.MODEL_TENSOR.HC_ATTN_BASE, ".weight"),
            "hc_attn_fn":                           (gguf.MODEL_TENSOR.HC_ATTN_FN, ".weight"),
            "hc_attn_scale":                        (gguf.MODEL_TENSOR.HC_ATTN_SCALE, ".weight"),
            "hc_ffn_base":                          (gguf.MODEL_TENSOR.HC_FFN_BASE, ".weight"),
            "hc_ffn_fn":                            (gguf.MODEL_TENSOR.HC_FFN_FN, ".weight"),
            "hc_ffn_scale":                         (gguf.MODEL_TENSOR.HC_FFN_SCALE, ".weight"),
            "attn.attn_sink":                       (gguf.MODEL_TENSOR.ATTN_SINKS, ".weight"),
            "attn.wq_a.weight":                     (gguf.MODEL_TENSOR.ATTN_Q_A, ".weight"),
            "attn.wq_b.weight":                     (gguf.MODEL_TENSOR.ATTN_Q_B, ".weight"),
            "attn.q_norm.weight":                   (gguf.MODEL_TENSOR.ATTN_Q_A_NORM, ".weight"),
            "attn.wkv.weight":                      (gguf.MODEL_TENSOR.ATTN_KV, ".weight"),
            "attn.kv_norm.weight":                  (gguf.MODEL_TENSOR.ATTN_KV_A_NORM, ".weight"),
            "attn.wo_a.weight":                     (gguf.MODEL_TENSOR.ATTN_OUT_A, ".weight"),
            "attn.wo_b.weight":                     (gguf.MODEL_TENSOR.ATTN_OUT_B, ".weight"),
            "attn.compressor.ape":                  (gguf.MODEL_TENSOR.ATTN_COMPRESSOR_APE, ".weight"),
            "attn.compressor.wkv.weight":           (gguf.MODEL_TENSOR.ATTN_COMPRESSOR_KV, ".weight"),
            "attn.compressor.wgate.weight":         (gguf.MODEL_TENSOR.ATTN_COMPRESSOR_GATE, ".weight"),
            "attn.compressor.norm.weight":          (gguf.MODEL_TENSOR.ATTN_COMPRESSOR_NORM, ".weight"),
            "attn.indexer.wq_b.weight":             (gguf.MODEL_TENSOR.INDEXER_ATTN_Q_B, ".weight"),
            "attn.indexer.weights_proj.weight":     (gguf.MODEL_TENSOR.INDEXER_PROJ, ".weight"),
            "attn.indexer.compressor.ape":          (gguf.MODEL_TENSOR.INDEXER_COMPRESSOR_APE, ".weight"),
            "attn.indexer.compressor.wkv.weight":   (gguf.MODEL_TENSOR.INDEXER_COMPRESSOR_KV, ".weight"),
            "attn.indexer.compressor.wgate.weight": (gguf.MODEL_TENSOR.INDEXER_COMPRESSOR_GATE, ".weight"),
            "attn.indexer.compressor.norm.weight":  (gguf.MODEL_TENSOR.INDEXER_COMPRESSOR_NORM, ".weight"),
            "attn_norm.weight":                     (gguf.MODEL_TENSOR.ATTN_NORM, ".weight"),
            "ffn_norm.weight":                      (gguf.MODEL_TENSOR.FFN_NORM, ".weight"),
            "ffn.shared_experts.w1.weight":         (gguf.MODEL_TENSOR.FFN_GATE_SHEXP, ".weight"),
            "ffn.shared_experts.w3.weight":         (gguf.MODEL_TENSOR.FFN_UP_SHEXP, ".weight"),
            "ffn.shared_experts.w2.weight":         (gguf.MODEL_TENSOR.FFN_DOWN_SHEXP, ".weight"),
            "ffn.gate.weight":                      (gguf.MODEL_TENSOR.FFN_GATE_INP, ".weight"),
            "ffn.gate.bias":                        (gguf.MODEL_TENSOR.FFN_EXP_PROBS_B, ".bias"),
            "ffn.gate.tid2eid":                     (gguf.MODEL_TENSOR.FFN_GATE_TID2EID, ".weight"),
            "ffn.experts.w1.weight":                (gguf.MODEL_TENSOR.FFN_GATE_EXP, ".weight"),
            "ffn.experts.w3.weight":                (gguf.MODEL_TENSOR.FFN_UP_EXP, ".weight"),
            "ffn.experts.w2.weight":                (gguf.MODEL_TENSOR.FFN_DOWN_EXP, ".weight"),
            "nextn.e_proj.weight":                  (gguf.MODEL_TENSOR.NEXTN_E_PROJ, ".weight"),
            "nextn.h_proj.weight":                  (gguf.MODEL_TENSOR.NEXTN_H_PROJ, ".weight"),
            "nextn.enorm.weight":                   (gguf.MODEL_TENSOR.NEXTN_ENORM, ".weight"),
            "nextn.hnorm.weight":                   (gguf.MODEL_TENSOR.NEXTN_HNORM, ".weight"),
            "nextn.shared_head_norm.weight":        (gguf.MODEL_TENSOR.NEXTN_SHARED_HEAD_NORM, ".weight"),
            "nextn.hc_head_base.weight":            (gguf.MODEL_TENSOR.NEXTN_HC_HEAD_BASE, ".weight"),
            "nextn.hc_head_fn.weight":              (gguf.MODEL_TENSOR.NEXTN_HC_HEAD_FN, ".weight"),
            "nextn.hc_head_scale.weight":           (gguf.MODEL_TENSOR.NEXTN_HC_HEAD_SCALE, ".weight"),
        }
        if rest in layer_level:
            tensor, suffix = layer_level[rest]
            return self.format_tensor_name(tensor, bid, suffix=suffix)
        return super().map_tensor_name(name, try_suffixes)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid
        if self._skip_tensor(name):
            return
        yield self.map_tensor_name(self._normalize_tensor_name(name)), data_torch
