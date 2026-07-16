# User Isolation and Scheduling Design

## Status

Implemented. Five commits in `master`:

- `common : add namespace_prefix to kv_ssd_init for u/ isolation`
- `server : thread user_id field from request body to server_task`
- `server : per-user concurrency cap and slot affinity`
- `server : page manager routes user_id to u/ namespace, no cross-user lookup`
- `server : return HTTP 429 when per-user concurrency cap is hit`

This document is the design rationale; the source of truth for behaviour
is the code itself.

## Problem

Original isolation is content-derived: `conv_hash` is FNV-1a of the first
1024 task tokens. This has four problems:

1. **Identity instability.** Edit the system prompt and the first 1024
   tokens change, `conv_hash` rotates, and all warm checkpoints for that
   "user" become unreachable except via the fuzzy continuation fallback.
2. **No operator-declared boundary.** The server operator cannot enforce
   "this is user X" - the boundary is whatever the prompt looks like today.
3. **PII in the hash key.** Tokens may contain names, emails, identifiers.
   The hash itself is not reversible but the *content* drives the routing
   key.
4. **No scheduling isolation.** A single noisy tenant can fill every slot
   (`n_parallel`). The DeepSeek model exposes per-`user_id` concurrency
   caps for exactly this reason.

## Goal

Introduce a first-class `user_id` that:

- Replaces or augments the content-derived `conv_hash` for KV cache
  routing.
- Carries a per-`user_id` concurrency cap (scheduling isolation).
- Carries a per-`user_id` log key for content safety auditing.
- Is wired through the OpenAI and Anthropic request surfaces.
- Falls back gracefully to content-derived routing when `user_id` is
  absent.
- Is backwards compatible. Existing deployments with no `user_id` keep
  working via `conv_hash`.

## The Three Isolation Dimensions

| Dimension | DeepSeek term | Current state | Target state |
|---|---|---|---|
| Identity | `user_id` (regex `[a-zA-Z0-9\-_]+`, max 512) | None (content-derived only) | First-class request field |
| KVCache | "KVCache Isolation" | `conv_hash` per conversation dir | `user_id` dir; `conv_hash` is the fuzzy fallback |
| Scheduling | "Scheduling Isolation" | None, global `n_parallel` | `user_id` slot cap, operator-configurable |
| Content safety | "Content Safety Isolation" | None | Per-`user_id` log key in server logs |
| Concurrency | Account-level | `n_parallel` only | `n_parallel` (account) + per-`user_id` cap |

## Request Surface

### OpenAI Chat Completions

Top-level field on the request body. OpenAI's own `user` parameter
naming convention would be tempting but conflicts with OpenAI's own
field semantics, so we use a llama-specific key:

```json
{
  "model": "...",
  "messages": [...],
  "llama_user_id": "tenant-42-user-7"
}
```

OpenAI SDK callers pass it through `extra_body` as DeepSeek documents:

```python
client.chat.completions.create(
    model="...",
    messages=[...],
    extra_body={"llama_user_id": "tenant-42-user-7"},
)
```

### Anthropic Messages

Anthropic supports `metadata` as a free-form object. We extract
`metadata.user_id` (the existing extraction at
`tools/server/server-chat.cpp:543` was orphaned - parsed but never
used). The `metadata.user_id` field is promoted from an orphan to the
canonical Anthropic path. The field is preserved verbatim from the
client, so Anthropic SDK callers use it as-is.

### Validation

```
regex:    ^[a-zA-Z0-9\-_]+$
max len:  512
empty:    treat as anonymous bucket (still isolated, still subject to caps)
```

Reject with HTTP 400 on invalid input (via
`server_task::validate_user_id`). Reject with HTTP 429 when the
`user_id` is at its concurrency cap.

## KVCache Routing

### Precedence

When `user_id` is present and valid:

1. Route KV cache to `{SSD_PATH}/u/{fnv1a(user_id)}/`
2. Cold-start lookup uses `user_id` as the directory key
3. Content-derived continuation matching is **disabled** within the
   `user_id` directory (no fuzzy cross-`user_id` lookups - privacy
   guarantee)
4. The page manager only inspects the requesting user's own cache

When `user_id` is absent or empty:

1. Route to the anonymous bucket keyed by `conv_hash`:
   `{SSD_PATH}/{fnv1a(conv_hash)}/`
2. Cold-start lookup uses `conv_hash`
3. Content-derived continuation matching is enabled across the
   anonymous bucket and on the global fallback path

Rationale: explicit `user_id` is a privacy declaration. We must not
silently mix KV state across declared identities. The anonymous bucket
keeps the content-derived behaviour for callers who have not opted in.

### Disk Layout

```
{SSD_PATH}/
  {016x hex of conv_hash}/   # anonymous bucket (one dir per conv_hash)
  u/
    {016x hex of fnv1a(user_id)}/   # user-scoped bucket (one dir per user_id)
```

The `u/` namespace is a child of `{SSD_PATH}` so the global scanner
functions (`kv_ssd_find_continuation` and friends) only walk the
top-level `016x hex` directories and naturally skip the `u/`
subtree.

### Compatibility With Existing Deployments

`conv_hash` (content-derived) is preserved as a separate code path.
It is still used:

- When `user_id` is not provided.
- As the directory naming scheme for the anonymous bucket.

`user_id` and `conv_hash` are not mixed. A request either has a
`user_id` and uses the user-scoped directory, or it does not and uses
the `conv_hash` directory. Existing anonymous checkpoints are
unaffected by the introduction of the `u/` namespace.

## Scheduling Isolation

### Configuration

```
--max-concurrent-per-user <N>   # 0 = unlimited (default)
```

Setting this enables per-`user_id` slot accounting. When a request
arrives with a `user_id`, the slot allocator checks the current count
of in-flight requests for that `user_id`. If the count is at or above
`N`, the request is rejected with HTTP 429
(`error.type = "rate_limit_error"`).

The anonymous bucket is subject to the same cap under the bucket name
`_anonymous`, so a deployment that did not previously have a cap does
not get one for free by opting out of `user_id`.

### Slot Allocator Changes

`server_context::get_available_slot`
(`tools/server/server-context.cpp:1332`) is `user_id`-aware:

1. Per-user cap check first. If the requesting `user_id` is at the
   cap, return `nullptr` and log `SRV_INF`. The HTTP handler emits 429.
2. Free slots that already belong to the requesting `user_id` are
   preferred (cache locality - the slot may hold a warm prompt from
   the same user).
3. An empty `slot.user_id` (cleared on release) is fair game for any
   user.
4. Among the surviving candidates, LCP similarity (existing logic)
   picks the best warm-prompt match; LRU is the fallback.
5. After assignment, `slot.user_id` is set and the per-user counter
   is incremented.

### Per-Slot State

`server_slot` gained:

```cpp
std::string user_id;  // empty if anonymous
```

Set when the slot is assigned. Cleared when the slot is released.
Used for both the routing check and the per-user concurrency
counter.

### Concurrency Counter

`server_context_impl` gained a
`mutable std::unordered_map<std::string, int> user_counts_` guarded
by `queue_tasks.mutex_tasks`. Increment on slot assign, decrement
on release. The `_anonymous` key holds the count for requests
without a `user_id`. The total across all keys never exceeds
`n_parallel`.

`user_counts_` is `mutable` so the const `is_user_at_cap` method can
read it under the mutex without a `const_cast`. The mutex is also
`mutable` (and public on `server_queue`) for the same reason. The
"mutable mutex" idiom: the mutex is the synchronization point, not
the data structure, so a const accessor is sound.

## HTTP 429 Path

Two layers of cap enforcement, in increasing authority:

1. **Synchronous fast-fail** in `handle_completions_impl`. After
   `task.user_id` is resolved (from `metadata.user_id` or
   `llama_user_id`), `is_user_at_cap` is called. If true, return
   HTTP 429 with a `rate_limit_error` envelope and a
   `Retry-After`-style message. The task is not queued.
2. **Authoritative check** in `get_available_slot`. Even if a
   request passes the synchronous check, the actual allocation is
   re-checked under the slot allocator's lock. If the cap is hit
   there (e.g. a race), the task is deferred. The next attempt
   will see the cap again and re-defer.

The race window between (1) and (2) is benign: if the cap frees
up between the two checks, the task proceeds normally. If not, the
task is deferred and (1) is hit again on the next request from
the same user.

## Content Safety

Per-`user_id` log keys are achieved by including `task.user_id` in
the existing structured log fields. The content itself is not
logged - this is a routing key for log filtering, not a content
audit trail.

If deeper content safety tooling is needed later (per-user
allow/deny lists, toxicity scoring integration) that is a separate
feature. This design only adds the log key.

## CLI Parameters

| Flag | Default | Effect |
|---|---|---|
| `--max-concurrent-per-user N` | 0 (unlimited) | Per-`user_id` slot cap. Enforced for both explicit and anonymous buckets. |

The SSD cache parameters (`--cache-ssd-path`,
`--cache-ssd-max-conversations`, etc.) are unchanged. The
`user_id` scheme is automatic when SSD caching is enabled.

## Backwards Compatibility

- No `user_id` provided: behaviour identical to current. Anonymous
  bucket used. `conv_hash` and continuation matching still work.
- Existing SSD cache directories: untouched. `user_id` traffic lands
  in a separate `u/` namespace.
- `metadata.user_id` parsing: the previously-orphaned extraction at
  `server-chat.cpp:543` is now actually routed into the slot
  allocator and the page manager.
- Slot allocation: unchanged when `--max-concurrent-per-user` is 0.

## Files Touched

| File | Change |
|---|---|
| `common/common.h` | Add `max_concurrent_per_user` to `common_params` |
| `common/arg.cpp` | Parse `--max-concurrent-per-user` |
| `common/kv-ssd-cache.h` | `kv_ssd_init` accepts optional `namespace_prefix` |
| `common/kv-ssd-cache.cpp` | Pass `namespace_prefix` through to `kv_ssd_open` |
| `tools/server/server-task.{h,cpp}` | `server_task::user_id` field and `validate_user_id` |
| `tools/server/server-chat.cpp` | Extract `metadata.user_id` (Anthropic) into the task |
| `tools/server/server-context-page-manager.{h,cpp}` | Route `user_id` to a `u/` namespace; disable cross-user continuation matching |
| `tools/server/server-context.cpp` | Per-user cap check; per-slot `user_id`; cap-aware LCP/LRU loops; HTTP 429 fast-fail in `handle_completions_impl` |
| `tools/server/server-context.h` | (no API change required; const method on impl) |
| `tools/server/server-queue.h` | `mutex_tasks` is `mutable` and public for the const cap check |
| `tools/server/server-common.{h,cpp}` | `ERROR_TYPE_RATE_LIMIT` mapped to HTTP 429 |

## Open Questions

- **Default cap for the anonymous bucket.** If
  `--max-concurrent-per-user` is set, do we apply it to
  `_anonymous`? Decision: yes, for fairness. Operators who want
  anonymous to be uncapped can set the cap to `n_parallel` and
  the anonymous bucket will hit it the same way.
- **Per-model caps.** DeepSeek does per-model concurrency. We have
  one model per server, so this is not relevant today. If
  multi-model routing lands later it will need to be revisited.
- **`Retry-After` semantics.** Streaming requests mid-flight when
  the cap drops the next request. Not a real concern for the cap
  itself (it only blocks new requests) but worth a test.

## References

- DeepSeek docs: <https://api-docs.deepseek.com/quick_start/rate_limit>
- Current `conv_hash` definition: `common/kv-ssd-cache.cpp:41`
- Page manager: `tools/server/server-context-page-manager.{h,cpp}`
- `metadata.user_id` extraction: `tools/server/server-chat.cpp`
- Slot allocation: `tools/server/server-context.cpp` (around line 1332)
