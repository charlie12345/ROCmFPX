#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

MTP_MODEL="$TMP_DIR/fake-mtp.gguf"
PLAIN_MODEL="$TMP_DIR/fake-plain.gguf"
DIFFUSION_MODEL="$TMP_DIR/fake-diffusion-gemma.gguf"
QAT_MODEL="$TMP_DIR/fake-gemma-qat.gguf"
AGENT_MODEL="$TMP_DIR/fake-rocmfpx-agent.gguf"
DFLASH_MODEL="$TMP_DIR/fake-ddflash.gguf"

printf 'GGUF qwen35.nextn_predict_layers blk.0.ffn_down.weight\n' > "$MTP_MODEL"
printf 'GGUF blk.0.ffn_down.weight blk.0.attn_q.weight\n' > "$PLAIN_MODEL"
printf 'GGUF diffusion_gemma diffusion-kv-cache blk.0.ffn_down.weight\n' > "$DIFFUSION_MODEL"
printf 'GGUF quantization-aware qat blk.0.ffn_down.weight\n' > "$QAT_MODEL"
printf 'GGUF rocmfpx_agent coherent blk.0.attn_q.weight\n' > "$AGENT_MODEL"
printf 'GGUF dflash target_layers blk.0.attn_q.weight\n' > "$DFLASH_MODEL"

"$SCRIPT_DIR/rocmfpx-model-capabilities.py" "$MTP_MODEL" --has-mtp --quiet

if "$SCRIPT_DIR/rocmfpx-model-capabilities.py" "$PLAIN_MODEL" --has-mtp --quiet; then
    echo "plain synthetic GGUF was incorrectly marked MTP-capable" >&2
    exit 1
fi

mtp_profile="$("$SCRIPT_DIR/rocmfpx-model-capabilities.py" "$MTP_MODEL")"
plain_profile="$("$SCRIPT_DIR/rocmfpx-model-capabilities.py" "$PLAIN_MODEL")"
diffusion_profile="$("$SCRIPT_DIR/rocmfpx-model-capabilities.py" "$DIFFUSION_MODEL")"
qat_profile="$("$SCRIPT_DIR/rocmfpx-model-capabilities.py" "$QAT_MODEL")"
agent_profile="$("$SCRIPT_DIR/rocmfpx-model-capabilities.py" "$AGENT_MODEL")"
dflash_profile="$("$SCRIPT_DIR/rocmfpx-model-capabilities.py" "$DFLASH_MODEL")"

python3 - "$mtp_profile" "$plain_profile" "$diffusion_profile" "$qat_profile" "$agent_profile" "$dflash_profile" <<'PY'
import json
import sys

mtp = json.loads(sys.argv[1])
plain = json.loads(sys.argv[2])
diffusion = json.loads(sys.argv[3])
qat = json.loads(sys.argv[4])
agent = json.loads(sys.argv[5])
dflash = json.loads(sys.argv[6])

assert mtp["supports_mtp"] is True
assert plain["supports_mtp"] is False
assert mtp["serving_profile"]["supports_mtp"] is True
assert plain["serving_profile"]["supports_mtp"] is False
assert mtp["model_kind"] == "mtp"
assert plain["model_kind"] == "regular"
assert diffusion["model_kind"] == "diffusion"
assert diffusion["supports_diffusion"] is True
assert diffusion["serving_profile"]["name"] == "generic-diffusion-rocmfpx"
assert qat["model_kind"] == "qat"
assert qat["is_qat"] is True
assert qat["serving_profile"]["name"] == "generic-qat-gguf"
assert agent["model_kind"] == "agent"
assert agent["is_agent"] is True
assert agent["serving_profile"]["name"] == "generic-agent-rocmfpx"
assert dflash["supports_dflash"] is True
assert dflash["dflash_markers"]
PY

preflight_json="$(WRAPPER_OUT="$TMP_DIR/fake-mtp-server" BIN=/bin/true MODEL="$MTP_MODEL" "$SCRIPT_DIR/rocmfpx-production-preflight.sh")"

python3 - "$preflight_json" "$MTP_MODEL" "$TMP_DIR/fake-mtp-server" <<'PY'
import json
import os
import pathlib
import sys

data = json.loads(sys.argv[1])
model = sys.argv[2]
wrapper = pathlib.Path(sys.argv[3])
cmd = data["launch_command"]

assert data["status"] == "pass"
assert cmd[:3] == ["/bin/true", "-m", model]
assert "--spec-type" in cmd
assert "draft-mtp" in cmd
assert data["wrapper_out"] == str(wrapper)
assert wrapper.is_file()
assert os.access(wrapper, os.X_OK)
text = wrapper.read_text(encoding="utf-8")
assert text.startswith("#!/usr/bin/env bash\n")
assert 'exec /bin/true -m ' in text
assert ' "$@"\n' in text
PY

if BIN=/bin/true MODEL="$PLAIN_MODEL" REQUIRE_MTP=1 "$SCRIPT_DIR/rocmfpx-production-preflight.sh" >/dev/null 2>&1; then
    echo "production preflight allowed REQUIRE_MTP=1 for a non-MTP model" >&2
    exit 1
fi

echo "ROCmFPX model capability detection passed"
