# Qwen3.8-27B SAGE REXL3 3.64bpw at 262,144-token context on a 16 GB GPU (TensorFold lane)

The same trick as the [ExLlamaV3 recipe](https://github.com/MaverickKB/qwen38-27b-262k-16gb-exl3),
implemented natively in [TensorFold](https://github.com/ashhart/TensorFold): the KV pool's
pages live in pinned host memory (`cuMemHostAlloc` mapped into the device's address space)
while the EXL3 weights stay fully device-resident. One page base serves every kernel, so
attention, prefill, graph capture and prompt-cache paths are untouched.

This repo carries the patch (`qwen27-kv-host.patch`, 7 files, +62/−15) and the serving/
benchmark recipe. The patch is also open as a PR to upstream TensorFold (link below).

## The switch

```sh
TF_QWEN_KV_PAGES=256 \
TF_QWEN_KV_PAGES_ALL=1 \
TF_QWEN_KV_HOST=1 \
tensorfold serve /path/to/Qwen3.8-27B-SAGE-REXL3-3.64bpw \
  --context 262144 --kv-cache fp8 --prompt-cache-gib 4 --no-thinking
```

- `TF_QWEN_KV_PAGES=256` pages the pool (required; pages are the unit the host tier holds)
- `TF_QWEN_KV_PAGES_ALL=1` puts a single-stream server's engine sequence on pages too
- `TF_QWEN_KV_HOST=1` funds the pages from pinned host memory
- `TF_QWEN_KV_HOST_GIB=n` caps the host footprint (default: a full window + run pages)
- `--kv-cache fp8` halves the KV bytes, which halves the PCIe traffic at depth

A `--prompt-cache-gib` (or `TF_QWEN_KV_SHARE`) bound is required alongside the host tier:
the default prompt cache would pin several more GiB beside the KV pages.

## What the patch changes

| File | Change |
| --- | --- |
| `zig/src/families/qwen3_5/cuda_kv_host.zig` | NEW: the `TF_QWEN_KV_HOST` / `_GIB` switch and host-page sizing |
| `zig/src/core/kv_pool.zig` | `Budget.pages_room` + `Plan.host`; host-funded plans stop charging device room for pages; fixes the paged plan's own-run-page undercount that refused a full window near its end |
| `zig/src/native/cuda_kv_pool.zig`, `cuda_streams.zig` | plumb the family's host-room into the plan; NoStream text reports host funding |
| `zig/src/families/qwen3_5/cuda_pages.zig` | the pool arena becomes device-or-host (`HostBuffer.allocMapped`), single base for pages and runs, zero-fill through the device view |
| `zig/src/families/qwen3_5/cuda_native.zig` | refuses the switch without paging; exports the host room; warns about the prompt-cache double-pin |
| `zig/src/families/qwen3_5/cuda_native_explain.zig`, `zig/src/core/profile.zig` | refusal text and the reviewed-switch registration |

Precedent: GLM's `TF_GLM_KV_HOST` (#1795) already proves pinned-host KV pages in this
codebase. Qwen3.8's dense full attention reads every position every token, so this tier
is a capacity mode: decode pays one PCIe read of the window's KV per token, prefill stays
compute-bound. Measured device-read bandwidth from mapped host memory on an RTX 5080:
48.5 GiB/s sequential.

## Results (honest)

The patched engine **serves the full 262,144-token window on the 16 GB card**
(`model 12.27 GiB; 1 stream … 1075 pages … 8.53 GiB … enough for 1 at a
262144-token window`, TTFT 93 ms, chat completions verified, own glue kernels).
Decode through this Phase-1 shape runs ~0.4 tok/s: the state run and the
verify-log rollback sit in host pages beside the KV, so every step pays
host-page round trips beyond the KV reads.

The ExLlamaV3 reference recipe with the same primitive — whole-layer host KV,
state and logs device-resident — measures 48.3 tok/s at 0 depth, 23.2 at 32K,
9.8 at 131K, 5.63 at 258K on the identical card and pack
(https://github.com/MaverickKB/qwen38-27b-262k-16gb-exl3). The bandwidth is
there; the gap is which tensors live on which side. Phase 2 (host the KV
pages only, keep the run, verify log and a hot prefix on the device) is the
follow-up that restores the fast path inside TensorFold.

## Credits

- Engine: [TensorFold](https://github.com/ashhart/TensorFold) by ashhart; GLM host-KV mechanism by MiaAI-Lab and contributors (#1795)
- Model pack: vcruz305 (SAGE mixed-K EXL3); base model: Qwen (Apache-2.0)
- Benchmark harness: sparkDash DecodeBench (MiaAI-Lab)

License: MIT for this repo's recipe materials; the patch carries TensorFold's license
(see the PR).
