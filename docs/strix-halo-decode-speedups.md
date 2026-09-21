# Strix Halo decode speedups (halo-box/strix-llama.cpp)

Target: gfx1151 only, ROCm/HIP path. Decode = batch-1 token generation
for qwen4exp (Qwen3.8-Flash-Next: GDN + MoE 512x10 + hyper-connections +
QSA sparse attention + PLE).

## Baseline comparison

Compared local HEAD against halo-box/strix-llama.cpp `0636c9ae`
("pwilkin/strix-halo official merge (#63)", 2026-09):

* `git diff HEAD 0636c9ae -- ggml/src/ggml-cuda`: +7246/-2797, 65 files
* Local already has the pwilkin/strix-halo base (#1): `mmb.cu`,
  `qsa.cu`, `hc-cn/mix.cu`, `ple-conv.cu`, `gdn-conv.cu`,
  `norm-gated.cu`, older `gated_delta_net.cu`.
* Missing locally: all of halo's later HIP decode work listed below.
* Do NOT blindly sync `src/`: halo deleted `llama-kv-cache-kpool.*`,
  `llama-lazy-reader.h`, `glm5next.cpp` in favor of `ple-disk` and
  different KV handling. Port `ggml-cuda` kernels plus minimal hooks.
* Do NOT port `top-k.cu`: local radix top-k (1049 lines) is ahead of
  halo (341 lines). Keep it.

## Ranked findings for qwen4exp decode

1. MMVQ RDNA3.5 kernels (`mmvq.cu` +2006 in halo). Dedicated
   `MMVQ_PARAMETERS_RDNA3_5` table (`calc_nwarps = 1`),
   `mul_mat_vec_q4_columns*` doing 4 columns per thread for
   Q1_0/Q2_0/Q5_1/Q4_K/Q5_K/Q6_K/IQ3_S and Q8_0. Decode is almost
   entirely MMVQ at batch 1. PORTED (this commit, core only).
2. MMVF single-column work (`mmvf.cu` +223). 4x software-prefetch K
   loop for ncols_dst==1 f32 matvecs (MoE routers, HC projs,
   bit-identical) + `mul_mat_vec_bf16_wave` for K<=512 BF16 rows
   (93 -> 67 us on 8x2048 K=512 per halo comment). PORTED.
3. Grouped decode matvecs (`GGML_CUDA_DISABLE_MMV_GROUP` in
   `ggml-cuda.cu`). Consecutive single-column matvecs sharing one
   activation (gate/up pairs, HC projs) launch as one kernel.
   PORTED (needs the FQ core below; commit `27f4573c9`).
   Fused-quantize core (`mul_mat_vec_q_fq_*`, `fq_try` entry hook,
   prologue API): PORTED (commit `ec359ce3d`).
4. GDN decode fusion (`GGML_CUDA_DISABLE_GDN_GATE`). Whole
   conv->norm->gate->recurrence->state-copy chain as one kernel, plus
   DPP reductions and 16/32-warp configs in `gated_delta_net.cu`.
   DEFERRED.
5. MoE decode fusion (`GGML_CUDA_DISABLE_WEIGHTED_DOWN`). Fused
   routing + weighted expert reduction + shared-expert gate.
   PARTLY PORTED: one-token IQ4_NL/Q8_0 down-proj + weighted sum
   (commit below). Shared-expert merge stays in the local
   moe-weighted-reduction path.
6. Fused activation quantization (`quantize_mmq_q8_1_swiglu` in
   `quantize.cu`). SiLU(gate)*up fused into the Q8_1 quantize.
   DEFERRED.
7. Flash-attention decode (`fattn.cu` + new `fattn-tile-rdna3-5.cu`).
   Q8_0 KV tile decode path (D=64/128/256, GQA>=2) + WMMA D=256
   routing fixes. PORTED (commit `ce215c979`), including the QSA
   split (item 11) it depends on.
8. MMQ tile tuning (`mmq-config-rdna3-5.cuh`, `mmq.cuh` prefetch,
   `mmq-vec-dot.cuh` split-j). Mostly prefill/spec-verify.
   PORTED (commit `ba40f6862`). Kept the local 512-expert compact
   MoE selection; halo's 256-expert routed-compact selection,
   swiglu/pair decls, and whitespace-only hunks stay deferred.
9. Compact MUL_MAT_ID 512x10 (`mmid.cu`). Spec-verify win.
   PORTED (commit `19a7318df`); file is now identical to halo.
10. hyperconn vs hc-*: halo renamed `hc-cn/mix.cu` to `hyperconn.*`
    with BF16-only streams + `mmb_enabled()` gated to RDNA3.5.
    Reconcile with the Windows build before touching. DEFERRED.
11. QSA split (`qsa.cu` -> `qsa-decode.cu` + `qsa-prefill.cu`).
    PORTED as part of item 7 (commit `ce215c979`).

Also deferred from inside the ported files (same files, later steps):

* `mmvq.cu`: IQ3_S rows/grid/LDS probe kernels + case-1 dispatch
  (`GGML_IQ3_ROWS/LDS/GRID`, ncols_x==2560 only), `moe_launch`
  IQ4_NL rpb change (kept local rpb=4), gdn_gate + weighted MoE
  kernels, `mul_mat_q_pair` decl.
* `mmvf.cu`: `tokens_in_block` plumbing (grouped path), `exact_batch`
  / `GGML_HINT_EXACT_BATCH` plumbing (absent locally).
* `common.cuh`: hipCUB enable (QSA top-k beyond 2k tokens).
* `vecdotq.cuh`: fully in sync with halo after this commit.

## This commit

* `ggml/src/ggml-cuda/mmvq.cu`: RDNA3.5 table split, `calc_nwarps=1`,
  `q4_columns` kernels + case-4 dispatch, `unroll_q8` removal
  (superseded by the new dispatch), Q5_1 volatile accumulate
  (ROCm 7.14 rounding workaround), IQ3_S fused-gate pair dot,
  generic `warp_reduce_sum` (halo dropped the local DPP special
  case with the redesign).
* `ggml/src/ggml-cuda/mmvf.cu`: HIP 4x K-loop prefetch, bf16_wave
  kernel + dispatch (BF16, ncols_dst==1, no fusion, block 256,
  K<=512, single sample).
* `ggml/src/ggml-cuda/vecdotq.cuh`: `vec_dot_iq3_s_q8_1_pair`
  (bit-identical shared-load pair dot for fused gate/up).
* Inserted blocks are byte-identical to halo `0636c9ae`; behavior
  changes are limited to kernel selection on RDNA3.5.

## Porting notes (halo kernel gates vs local graph/model)

* 2026-09-18: halo's QSA gates require `op_params[4] == 0` (their nodes
  never set it). Our qwen4exp model wrote the selected-key count there
  via `set_n_kv_max`, so maskless prefill strips matched no kernel and
  hit halo's maskless abort in `fattn.cu`. Fixed model-side
  (`set_n_kv_max(cur, 0)`); the kernels never read p4. Nothing else in
  the tree reads p4 for these nodes (HIP tile `use_sparse` is always
  false, the NVIDIA sparse check is compiled out).
* Follow-up (done): removed the `qsa_pack_keys/values` pre-pass
  (`pack.inc` deleted) and the src6/7 attaches; halo kernels pack
  in-kernel. Nodes now carry null src4/6/7 and p4 == 0, matching
  halo's gate contract for both prefill and decode.

* Divergence from halo: `launch_fattn_tile_case` gates `use_q8_0_KV`
  on RDNA3.5 like `q8_0_KV_supported()` does. Without it a Q8_0
  decode on other AMD archs would run the NO_DEVICE_CODE trap.
  No-op on gfx1151.

## Verify

Build `build-rocm10-gfx1151` per `build-windows.ps1`, then with the
same Qwen3.8 quant before/after:

* `llama-bench -p 512 -n 128` (pp + tg)
* `llama-perplexity` for correctness (watched: Q5_1 rounding,
  IQ3_S pair path, Q8_0 4-column path)
* `test-backend-ops` MMVQ cases if available

Suggested order for the deferred items: MMQ config (8) -> FQ (6) ->
MMV_GROUP (3) -> GDN fused (4) -> WEIGHTED_DOWN (5) -> FATTN tile
(7) -> MMID_512 (9) -> hyperconn reconcile (10) -> QSA split (11).

## HC dispatch after the mainline integration (2026-09-20)

The Qwen4exp graph now uses gated `DSV4_HC_PRE` and identity `DSV4_HC_POST`. The gfx1151 backend routes eligible contiguous gated PRE inputs through the local HC mix kernel. The combine-plus-normalization matcher accepts identity POST followed by grouped RMSNorm and gamma multiplication, while retaining the decomposed graph pattern. Nonidentity POST, unsupported layouts, and other architectures retain their existing paths.

The combined residual and normalized output must occupy different buffers, including when the residual has no later consumer. An allocation dependency enforces that requirement. BF16 cache reservation and normalized-stream marking use token count rather than grouped row count (`hc * tokens`). The Windows HC16 exclusions remain in place.

Component measurements compare the integrated baseline `64dddeb35` with these working-tree changes, using separate copies of the HIP DLLs. Both execute the new combine-plus-normalization graph with F32 inputs, embedding width 2560, four HC streams, and all tensors on ROCm0 (Radeon 8060S, gfx1151). Build: Release, VS 2022, TheRock at `C:/TheRock/build`, Clang 23. HIP graph capture is enabled; results are medians of five trials of 30 executions after five warmups. Driver version was unavailable. Model quantization, KV settings, context occupancy, lazy reads, speculation, and sampling do not apply to this synthetic graph.

| Tokens | Integrated baseline (us) | Restored fusion (us) |
| --- | ---: | ---: |
| 1 | 16.37 | 10.82 |
| 128 | 94.53 | 28.33 |
| 512 | 356.40 | 406.72 |
| 4096 | 3620.78 | 3136.78 |

The restored path produces the same outputs as the old fused graph in this benchmark; the unfused baseline differs by at most `3.58e-7`. At 512 tokens, the restored path is slower in isolation. It preserves the existing BF16 cache production for subsequent matrix operations, whose benefit this graph does not measure. No token-specific performance cutoff was added based on this isolated comparison. Separate PRE measurements show similar large-batch timings and noisy small-batch timings, so they do not establish an additional throughput gain.

Validation: the full Windows regression suite and all 14 source guards pass. ROCm backend checks against CPU pass 27/27 `HC_F32_CONSUMER`, 22/22 `DSV4_HC_PRE`, and 6/6 `DSV4_HC_POST` cases. Vulkan passes 27/27 consumer, 10/10 supported PRE, and 6/6 POST cases; 12 PRE cases with unsupported stream counts are skipped. The added cases cover the 32-token HC mix boundary, the 512-token BF16-cache boundary, F32 and Q8 consumers, and nonidentity POST fallback. Logs and temporary benchmark harnesses are under the ignored `build-review-integration/` directory. These checks do not substitute for full-model generation, MTP, long-context, or delivered-throughput comparisons.
