# qwen3_5: TF_QWEN_KV_HOST — pinned-host KV pages for Qwen3.8-27B (16 GB card, 262,144-token window)

## What this does

Adds `TF_QWEN_KV_HOST=1` (with `TF_QWEN_KV_HOST_GIB=n` as an optional cap): the
qwen3_5 CUDA pool's KV pages — and run pages — are allocated from pinned host memory
(`cuMemHostAlloc` PORTABLE|DEVICEMAP, the same primitive GLM's host pages use since
#1795) and mapped into the device address space. One page base serves pages and runs,
so the attention kernels, prefill tiles, graph capture, prompt-cache copies and page
tables are untouched.

Purpose: capacity, not speed. A 13.7 GiB EXL3 Qwen3.8-27B checkpoint leaves ~2 GiB
on a 16 GB card — no live stream of any size fits stock (the plan refuses at 64K fp8
with "the budget leaves 0.00 GiB"). With the pages host-funded, the full 262,144-token
window exists and the card carries only weights + context. Decode pays one PCIe read
of the window's KV per token (dense full attention re-reads every position; measured
48.5 GiB/s sequential from mapped host memory on an RTX 5080), while prefill streams
KV writes and stays compute-bound.

This follows the ledger's row 7 ("Host KV tier": q3.5 yes, q27 yes, glm done #1795):
the q27 cell was the standing invitation; this is the implementation.

## What changed

- NEW `zig/src/families/qwen3_5/cuda_kv_host.zig`: the exact-one switch through the
  profile registry, `TF_QWEN_KV_HOST_GIB` via env, and host-byte sizing (full window
  + run, or the cap).
- `zig/src/core/kv_pool.zig`: `Budget.pages_room` and `Plan.host`; a host-funded plan
  charges no device room for pages (the engine's own included) and counts pages from
  the host bytes instead. Also fixes a real planning bug this work exposed: the paged
  plan's `most` bound omitted the engine's own run pages, so a full-window stream was
  refused near the end of its window even when the pool had room.
- `zig/src/native/cuda_kv_pool.zig` / `cuda_streams.zig`: the family exports
  `kvHostRoom()`; the plan and the preflight `fits()` pass it through; the NoStream
  text reports host funding.
- `zig/src/families/qwen3_5/cuda_pages.zig`: `Pages.init` allocates the arena as a
  device buffer or a mapped host buffer; single `base` for pages and runs; the host
  arena zero-fills through a borrowed device view on the engine stream; frees the
  right buffer either way.
- `zig/src/families/qwen3_5/cuda_native.zig`: refuses `TF_QWEN_KV_HOST` without
  paging (`QwenKvHostNeedsPages`, text in `cuda_native_explain.zig`); warns that the
  default prompt cache would pin more host memory beside the pages and should be
  bounded (`--prompt-cache-gib` / `TF_QWEN_KV_SHARE`).
- `zig/src/core/profile.zig`: registers the switch (qwen families, exact-one).

## Measured: Phase 2 beats the reference recipe at depth

Phase 2 (second commit) keeps only the KV pages in pinned host memory; every
stream's state run and verify log moved back to the device (one slot per lane
in a device arena the plan charges). Measured on the identical card and pack
(RTX 5080 16 GB, driver 580.105.08, `--context 262144 --kv-cache fp8`,
server-reported decode rate, 64-token replies):

| Context depth | TF_QWEN_KV_HOST Phase 2 | ExLlamaV3 whole-layer host KV (reference) |
| --- | --- | --- |
| 0 | 43.7 tok/s | 48.3 tok/s |
| 8,192 | **37.6** | 37.5 |
| 32,768 | **27.3** | 23.2 |
| 131,072 | **14.2** | 9.8 |
| 258,048 | **8.5** | 5.63 |

TensorFold wins from 32K outward (+51% at the 258K mark); the reference
recipe keeps a small lead only at the empty-window extreme. Phase 1 (all
pages host, runs included) measured 0.4 tok/s on the same setup — the state
run's per-step host round trips were the entire gap, which is why Phase 2's
placement fix was the decisive change.

Both lanes serve the full 262,144-token live window on a card that stock
engines refuse any stream on. Reference recipe:
https://github.com/MaverickKB/qwen38-27b-262k-16gb-exl3 (48.3 tok/s at 0
depth → 5.63 at 258K, whole-layer host KV on stock ExLlamaV3).

For reference, the point of doing this inside TensorFold is fp8 KV (half the
PCIe bytes of the ExLlamaV3 recipe's q4), native paged attention, and the
prompt cache / prefix reuse stack.

## Testing

- `zig build` all steps (the two CUDA test exes included).
- Server validation on the SAGE pack at `--context 262144 --kv-cache fp8` with
  `TF_QWEN_KV_PAGES=256 TF_QWEN_KV_PAGES_ALL=1 TF_QWEN_KV_HOST=1`, plus sparkDash
  DecodeBench waves at 0/8K/32K/131K/258K depth. Numbers above.

## Notes

- No issue exists for this; the work came out of serving this pack on a 16 GB card
  (the GLM host-KV row pointed the way). Happy to split or reshape per reviewer
  preference — e.g. the `most`-bound fix can land separately if that is cleaner.
- VMM host memory (`cuMemCreate` HOST_NUMA + map) was probed and rejected: launches
  succeed then fault on first synchronize on this driver/GPU (580.105.08 / sm_120),
  taking other CUDA clients down. `cuMemHostAlloc` mapped is the safe path.
