import math
import os
import re
import socket
import tempfile
from pathlib import Path

import pytest

from utils import *


LONG_PROMPT = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers. "
    "He met many creatures along the way including dragons and fairies "
    "and wizards who helped him on his noble quest to save the kingdom."
)

MODEL_DRAFT_FILE_URL = "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories15M-q4_0.gguf"
MODEL_TARGET_FILE_URL = "https://huggingface.co/ggml-org/test-model-stories260K/resolve/main/stories260K-f32.gguf"
MODEL_TARGET_DRAFT_PAIR_FILE_URL = "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories15M.gguf"

server = ServerPreset.tinyllama2()


# This module uses two explicit tiny local files. Do not invoke the parent
# conftest's all-preset Hugging Face preload, which is unrelated to this test
# and prevents offline/no-HTTPS server builds from reaching the assertions.
@pytest.fixture(scope="module", autouse=True)
def do_something():
    yield


class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0

    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content


def configure_disk_server(cache_dir, limit_mib=64, draft=False):
    global server
    server = ServerPreset.tinyllama2()
    server.model_file = download_file(
        MODEL_TARGET_DRAFT_PAIR_FILE_URL if draft else MODEL_TARGET_FILE_URL
    )
    server.model_hf_repo = None
    server.model_hf_file = None
    # Keep the test isolated from a developer shell that already exports a
    # llama-server API key.
    server.api_key = os.environ.get("LLAMA_API_KEY")
    with socket.socket() as sock:
        sock.bind((server.server_host, 0))
        server.server_port = sock.getsockname()[1]
    server.n_slots = 2
    server.n_ctx = 512
    server.n_gpu_layer = 0
    server.n_gpu_layer_draft = 0 if draft else None
    server.n_predict = 1
    server.temperature = 0.0
    server.server_slots = True
    server.cache_ram = 0
    server.cache_disk = cache_dir
    server.cache_disk_limit = limit_mib
    server.kv_unified = True
    server.debug = True
    if draft:
        server.model_draft = download_file(MODEL_DRAFT_FILE_URL)
        server.spec_draft_n_min = 1
        server.spec_draft_n_max = 4
        server.fa = "off"
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    return server


def complete(prompt, id_slot=None):
    data = {
        "prompt": prompt,
        "cache_prompt": True,
        "n_predict": 1,
        "temperature": 0.0,
    }
    if id_slot is not None:
        data["id_slot"] = id_slot
    headers = {"Authorization": f"Bearer {server.api_key}"} if server.api_key else None
    res = server.make_request("POST", "/completion", data=data, headers=headers)
    assert res.status_code == 200
    return res


def prime_and_displace(prompt=LONG_PROMPT):
    original = complete(prompt, 0)
    complete("The quick brown fox checks a different cache slot.", 1)
    return original


def test_disk_only_parse_restore_and_owned_cleanup(tmp_path):
    configure_disk_server(str(tmp_path), limit_mib=64)
    server.start()
    log = LogReader(server.log_path)

    startup = log.drain()
    assert "prompt cache RAM disabled: limit_mib=0" in startup
    assert "prompt cache SSD enabled:" in startup
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" in startup

    original = prime_and_displace()
    saved = log.drain()
    assert re.search(r"prompt cache disk save: .*target_bytes=[1-9][0-9]* draft_bytes=0", saved)
    assert re.search(r"cache state: 0 prompts,", saved)

    restored = complete(LONG_PROMPT)
    loaded = log.drain()
    assert "prompt cache disk load:" in loaded
    assert "draft_bytes=0" in loaded
    assert restored.body["timings"]["cache_n"] > 0
    assert restored.body["timings"]["prompt_n"] < original.body["timings"]["prompt_n"]

    # A successful load remains a reusable MRU entry. Saving the restored idle
    # slot should touch it, then a second restore should hit the same entry.
    first_entry = re.search(r"prompt cache disk load: entry=([0-9]+)", loaded)
    assert first_entry is not None
    complete("A third prompt displaces the restored slot safely.", 1)
    touched = log.drain()
    assert f"prompt cache disk touch: entry={first_entry.group(1)}" in touched
    assert "safe_to_clear=true" in touched

    restored_again = complete(LONG_PROMPT)
    loaded_again = log.drain()
    assert f"prompt cache disk load: entry={first_entry.group(1)}" in loaded_again
    assert "prompt cache disk load accepted:" in loaded_again
    assert restored_again.body["timings"]["cache_n"] > 0

    namespace = tmp_path / ".llama-prompt-cache-v1"
    owned = list(namespace.glob("run-*"))
    assert len(owned) == 1
    assert not list(owned[0].glob("*.tmp"))

    server.stop()
    assert not list(namespace.glob("run-*"))
    assert not list(namespace.glob(".deleting-run-*"))


def test_disk_lru_enforces_mib_limit(tmp_path):
    # Measure this fixture's streamed state size first, then restart with a
    # model-independent limit that fits a few entries and must evict under load.
    measure_dir = tmp_path / "measure"
    configure_disk_server(str(measure_dir), limit_mib=64)
    server.start()
    log = LogReader(server.log_path)
    log.drain()
    prime_and_displace()
    measured_log = log.drain()
    match = re.search(r"prompt cache disk save: .*total_bytes=([1-9][0-9]*)", measured_log)
    assert match is not None
    entry_bytes = int(match.group(1))
    server.stop()

    limit_mib = max(1, math.ceil((entry_bytes * 3) / (1024 * 1024)))
    cache_dir = tmp_path / "bounded"
    configure_disk_server(str(cache_dir), limit_mib=limit_mib)
    server.start()
    log = LogReader(server.log_path)
    log.drain()

    # Equal-length, non-prefix prompts produce similarly sized independent LRU
    # entries. Twenty-four turns is deliberately above the measured capacity.
    for i in range(24):
        complete((f"Cache lane {i:02d} unique marker. " * 18), i % 2)

    bounded_log = log.drain()
    assert "prompt cache disk eviction:" in bounded_log
    state_lines = [line for line in bounded_log.splitlines() if "prompt cache disk state:" in line]
    assert state_lines
    final_state = state_lines[-1]
    bytes_match = re.search(r" bytes=([0-9]+) limit_bytes=([0-9]+) ", final_state)
    assert bytes_match is not None
    assert int(bytes_match.group(1)) <= int(bytes_match.group(2)) == limit_mib * 1024 * 1024

    run_dirs = list((cache_dir / ".llama-prompt-cache-v1").glob("run-*"))
    assert len(run_dirs) == 1
    payload_bytes = sum(path.stat().st_size for path in run_dirs[0].glob("state-*.bin"))
    assert payload_bytes <= limit_mib * 1024 * 1024


def test_disk_cache_round_trips_target_and_draft(tmp_path):
    configure_disk_server(str(tmp_path), limit_mib=64, draft=True)
    server.start()
    log = LogReader(server.log_path)
    log.drain()

    original = prime_and_displace()
    saved = log.drain()
    save_match = re.search(
        r"prompt cache disk save: .*target_bytes=([1-9][0-9]*) draft_bytes=([1-9][0-9]*)",
        saved,
    )
    assert save_match is not None

    restored = complete(LONG_PROMPT)
    loaded = log.drain()
    load_match = re.search(
        r"prompt cache disk load: .*target_bytes=([1-9][0-9]*) draft_bytes=([1-9][0-9]*)",
        loaded,
    )
    assert load_match is not None
    assert "prompt cache cold fallback:" not in loaded
    assert restored.body["timings"]["cache_n"] > 0
    assert restored.body["timings"]["prompt_n"] < original.body["timings"]["prompt_n"]


def test_disk_cache_rejects_partial_target_draft_pair(tmp_path):
    configure_disk_server(str(tmp_path), limit_mib=64, draft=True)
    server.start()
    log = LogReader(server.log_path)
    log.drain()

    original = prime_and_displace()
    saved = log.drain()
    assert re.search(
        r"prompt cache disk save: .*target_bytes=[1-9][0-9]* draft_bytes=[1-9][0-9]*",
        saved,
    )
    assert re.search(r"cache state: 0 prompts,", saved)

    run_dirs = list((tmp_path / ".llama-prompt-cache-v1").glob("run-*"))
    assert len(run_dirs) == 1
    draft_files = list(run_dirs[0].glob("state-*-draft.bin"))
    assert draft_files
    draft_files[0].unlink()

    restored = complete(LONG_PROMPT)
    rejected = log.drain()
    assert "prompt cache disk load failed:" in rejected
    assert "component=draft" in rejected
    assert "prompt cache cold fallback:" in rejected
    assert "target_and_draft_cleared=true" in rejected
    assert restored.body["timings"]["cache_n"] == 0
    assert restored.body["timings"]["prompt_n"] == original.body["timings"]["prompt_n"]

    # A rejected pair must not poison the slot or the server.
    healthy = complete("The server remains healthy after a rejected cache pair.")
    assert healthy.status_code == 200


def test_disk_save_failure_opens_breaker_and_preserves_idle_slot(tmp_path):
    configure_disk_server(str(tmp_path), limit_mib=64)
    server.start()
    log = LogReader(server.log_path)
    log.drain()

    original = complete(LONG_PROMPT, 0)
    run_dirs = list((tmp_path / ".llama-prompt-cache-v1").glob("run-*"))
    assert len(run_dirs) == 1

    # Remove directory write permission so the first target temp cannot be
    # created. The slot must remain live because no durable cache exists.
    run_dirs[0].chmod(0o500)
    try:
        complete("This request forces an idle-slot save failure.", 1)
    finally:
        run_dirs[0].chmod(0o700)

    failed = log.drain()
    assert "prompt cache disk writes disabled:" in failed
    assert "reason=target-save" in failed
    assert "preserving idle slot because prompt cache save was not safe" in failed
    assert "safe_to_clear=true" not in failed

    reused_live_slot = complete(LONG_PROMPT, 0)
    after_breaker = log.drain()
    assert "reason=circuit-open" in after_breaker
    assert reused_live_slot.body["timings"]["cache_n"] > 0
    assert reused_live_slot.body["timings"]["prompt_n"] < original.body["timings"]["prompt_n"]


def test_failed_corrupt_entry_removal_keeps_conservative_accounting(tmp_path):
    configure_disk_server(str(tmp_path), limit_mib=64)
    server.start()
    log = LogReader(server.log_path)
    log.drain()

    original = prime_and_displace()
    saved = log.drain()
    size_match = re.search(r"prompt cache disk save: .*total_bytes=([1-9][0-9]*)", saved)
    assert size_match is not None
    accounted = int(size_match.group(1))

    run_dirs = list((tmp_path / ".llama-prompt-cache-v1").glob("run-*"))
    assert len(run_dirs) == 1
    target_files = list(run_dirs[0].glob("state-*-target.bin"))
    assert target_files
    with target_files[0].open("r+b") as f:
        f.truncate(max(1, accounted // 2))

    # Prevent quarantine cleanup. The entry must remain fully accounted and
    # unusable rather than being reported as freed after unlink failure.
    run_dirs[0].chmod(0o500)
    try:
        restored = complete(LONG_PROMPT)
    finally:
        run_dirs[0].chmod(0o700)

    rejected = log.drain()
    assert "reason=size-mismatch" in rejected
    assert "prompt cache disk removal failed:" in rejected
    assert f"accounted_bytes={accounted}" in rejected
    assert re.search(rf"prompt cache disk state: entries=1 unusable=1 bytes={accounted} ", rejected)
    assert "save_disabled=true" in rejected
    assert restored.body["timings"]["cache_n"] == 0
    assert restored.body["timings"]["prompt_n"] == original.body["timings"]["prompt_n"]
