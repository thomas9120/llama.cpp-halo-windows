# Dense decode speedups for Strix Halo (gfx1151, ROCm/HIP)

Scope: dense-model token generation (batch 1) on this fork. Target hardware is
Strix Halo gfx1151 only, ROCm/HIP path on Windows (TheRock 10). No other
backends, no other GPUs.

Example models: the request says "qwen3.8 27b". The tree has no arch by that
name. The closest matches are:

* `qwen3` 32B dense (`src/models/qwen3.cpp`, 64 layers, `LLM_TYPE_32B`).
* `qwen35` 27B (`src/models/qwen35.cpp`, 64 layers, `LLM_TYPE_27B`,
  hidden 5120, FFN 17408, GQA 24 Q / 4 KV heads, head dim 256, vocab 248320,
  hybrid Gated DeltaNet + full attention in a 3+1 pattern).

Both are "dense" in the sense that matters for decode: every token streams the
full FFN weights. Qwen3.5-27B additionally has 48 GDN layers and 16 full
attention layers. This doc covers both. Where they differ it says so.

This doc records research only. It makes no code changes.

Related: `docs/strix-halo-decode-speedups.md` (qwen4exp/MoE HIP ports, many
already landed). Dense decode reuses the same MMVQ/MMVF/grouped-matvec
machinery, so that doc is required background.

## 1. Roofline: why dense decode is bandwidth bound

Decode at batch 1 does one matrix-vector product per weight matrix per layer.
Arithmetic intensity is ~1 FLOP/byte, so tokens/sec is set by bytes read per
token divided by sustained memory bandwidth:

```
tok/s ~= sustained_BW / bytes_per_token
bytes_per_token ~= weight_bytes + KV_bytes_read + small overhead
```

Strix Halo: 256-bit LPDDR5X-8000, theoretical peak ~256 GB/s unified.
Sustained depends on backend, kernels, clocks, and carve-out. Published
llama.cpp data points on gfx1151 (Linux, ROCm 7.x, see issues #24438, #13565):

* MoE Q8_0 active ~3 GB/token: HIP ~33-46 t/s vs Vulkan ~49-52 t/s on the
  same box. Computed effective BW ~100-150 GB/s, i.e. 40-60% of peak.
* AMD engineer repro (b9592): HIP reaches 88-94% of Vulkan on tg, gap is
  6-12%, not 60%. Compute-bound prefill is slightly faster on HIP. So the
  gap is specific to the bandwidth-bound batch-1 path and sensitive to build
  flags and ROCm version.
* rocprofv3 `--kernel-trace` on Q4_K_M tg: `mul_mat_vec_q` is 91.7% of GPU
  time, `mul_mat_vec_f` 4.8%, `quantize_q8_1` 1.2%, norms/softmax/rope/copy
  each <1%. Launch geometry on gfx1151 is one wave32 per block, LDS=0.

Dense math for orientation (measure, do not trust these estimates):

* Qwen3.5-27B Q4 (~0.55-0.6 byte/param with K-quants): ~15-17 GB/token just
  for weights. At 100 GB/s effective that is ~6 t/s; at 150 GB/s ~10 t/s.
  Smaller quants scale tok/s almost linearly until kernels fall off peak.
* KV read per token grows with context. Per-token KV footprint:
  `layers * 2 * n_kv_heads * head_dim * bytes_per_elem`.
  Qwen3.5-27B F16: full-attn layers only (16 of 64) dominate; GDN state is
  recurrent and small per step. Qwen3-32B dense: all 64 layers are full
  attention, so KV BW matters much sooner. At 32k+ context KV traffic is a
  visible fraction of weight traffic.
* lm-head: Qwen3 vocab ~151k, Qwen3.5 vocab 248320 x 5120 = ~1.27B params.
  That is ~5% of a 27B model in one GEMV with nrows=248k. It has different
  kernel geometry from FFN mats and deserves its own measurement.

Takeaway: the ranked list below is ordered by bytes-per-token reduction first
(quant, KV, speculation), then by achieved-BW increase (MMVQ/MMVF/fusion),
then by fixed-overhead removal (graphs, launches, sampler).

## 2. How to measure before changing anything

Use the same GGUF before/after every experiment. Suggested baseline:

```
llama-bench -m <qwen35-27b-Q4_K_M.gguf> -p 512 -n 128 -fa auto -r 5
llama-bench -m <qwen35-27b-Q4_K_M.gguf> -p 0 -n 128 -fa auto -r 5   # pure tg
llama-perplexity -m <same.gguf> -f <wiki.test.raw> -c 8192 -ub 8   # correctness
test-backend-ops perf -o MUL_MAT        # isolate GEMV vs GEMM paths
test-backend-ops perf -o FATTN          # isolate attention kernels
```

Record: pp, tg, NGL/offload, `-fa`, `-ctk/-ctv`, threads, HIP graphs on/off,
`ROCBLAS_USE_HIPBLASLT`, driver/TheRock version, clocks, power profile.
Keep PM4 graph replay enabled (default) so decode uses the replay path; see
`docs/development/qwen4exp-decode-indexer.md` for the replay qualification
method (bit-identical logits, paired token match).

Profiling notes specific to gfx1151:

* rocprofv3 hardware byte counters (FETCH_SIZE/WRITE_SIZE) return 0/bogus on
  gfx1151 in ROCm 7.2.1 for the short high-frequency tg kernels. Use
  `--kernel-trace` timing instead, or llama.cpp built-in op timing.
* The hot kernels are `mul_mat_vec_q<T>` + `quantize_q8_1` pairs per matvec.
  Dispatch counts scale with layers x mats per layer x tokens.
* Compare HIP vs Vulkan on the same box only as a ceiling check, not as a
  release metric. This fork ships HIP only.

## 3. Ranked opportunities (dense decode, gfx1151 only)

### 3.1 Weight quant selection for decode (largest lever, no code)

What: bytes-per-token is set by the weight quant. A 27B dense model at Q3/IQ3
moves ~30% fewer bytes than at Q4; Q5/Q6/Q8 move proportionally more.

Why dense: dense reads all params per token. MoE reads only active experts,
so quant choice matters more for dense tok/s than for MoE tok/s.

Status: no code needed. But per-quant kernel quality differs on RDNA3.5:

* `ggml/src/ggml-cuda/mmvq.cu` has dedicated 4-columns-per-thread decode
  kernels (`mul_mat_vec_q4_columns_rdna3_5`) for Q1_0/Q2_0/Q5_1/Q4_K/Q5_K/
  Q6_K/IQ3_S and Q8_0 (see `is_rdna3_5_q4_columns_type`). Other types use the
  generic path.
* `calc_nwarps` for `MMVQ_PARAMETERS_RDNA3_5` returns 1 for all types. The
  AMD repro in #24438 tried nwarps=2/8 for Q8_0/Q4_K: +1.7% for Q8_0 at 2,
  regressions elsewhere, type-dependent and contradictory. Current table
  looks close to optimal; per-type tuning needs per-quant A/B, not a global
  bump.

Action:

1. Benchmark the same model in Q4_0, Q4_K_M, Q4_K_S, IQ4_NL, IQ4_XS, Q5_K_M,
   Q6_K, Q8_0, IQ3_S, MXFP4 (where available) with `-p 0 -n 128`.
2. Compute effective BW = tok/s x weight_bytes to separate "smaller file"
   from "faster kernel". Prefer the quant with best tok/s at acceptable
   perplexity, not the smallest file.
3. Watch Q5_1 rounding (ROCm 7.14 workaround in mmvq.cu), IQ3_S pair path,
   and Q8_0 4-column path with `llama-perplexity`.

Expected: linear-ish with size, plus/minus 5-15% kernel efficiency spread.
Risk: quality. Gate every candidate on perplexity + a short coherent sample.

### 3.2 Gate/up fusion + grouped decode matvecs (already partly landed)

What: each dense FFN does gate and up projections off the same activation
(`[n_embd] x [n_embd, n_ff]` twice), then SiLU-mul, then down projection.
Without fusion that is 2x activation reads + 2x activation quantizes + extra
launches per layer x 64 layers.

Status in this tree (`ggml/src/ggml-cuda/ggml-cuda.cu`):

* Plain `{MUL_MAT, MUL_MAT, GLU}` fusion into `mul_mat_vec_q/f` with gate
  fusion args: present (search `fused_mul_mat_vec`, `should_fuse_mul_mat`).
* Grouped decode (`GGML_CUDA_DISABLE_MMV_GROUP`, `ggml_cuda_mmv_group`):
  PORTED per `docs/strix-halo-decode-speedups.md` item 3.
* Fused-quantize core (`mul_mat_vec_q_fq_*`, `fq_try` hook): PORTED.
* Fused SiLU(gate)*up inside the Q8_1 quantize
  (`quantize_mmq_q8_1_swiglu` in halo `quantize.cu`): DEFERRED.
* `mmvf.cu` `tokens_in_block` / `exact_batch` plumbing: deferred (grouped
  path only).

Action for dense:

1. Verify dense FFN actually hits the fused path: run with and without
   `GGML_CUDA_DISABLE_MMV_GROUP=1` and `GGML_CUDA_DISABLE_FUSION=1` on a
   Qwen3/Qwen3.5 dense GGUF and compare tg. If no delta, the graph shape
   (bias/scale nodes, QK-norm placement) is blocking fusion, not the kernel.
2. If fused, port the deferred `quantize_mmq_q8_1_swiglu` next. It removes
   one full `n_ff`-element SiLU+mul+quantize pass per layer (17408 elems x
   64 layers per token for Qwen3.5-27B). Small per layer, large in aggregate.
3. Keep the local compact MoE selection; it does not affect dense.

Expected: low single-digit % on dense (activation traffic is small vs weight
traffic), plus fewer launches (helps graph replay, section 3.7).
Risk: low. Fusion kernels are bit-sensitive; qualify with perplexity.

### 3.3 Attention decode: KV quant + tile kernel + `-fa` choice

What: at batch 1 with short context, attention is <5% of GPU time. At long
context it grows linearly with KV length while FFN stays flat. Qwen3-32B
(64 full-attn layers) hits this sooner than Qwen3.5-27B (16 full-attn
layers).

Status:

* Q8_0 KV tile decode (`fattn-tile-rdna3-5.cu`, D=64/128/256, GQA>=2):
  PORTED. Gate requires batch 1 and Q8_0 K+V; local divergence gates
  `use_q8_0_KV` on RDNA3.5 so other archs hit no trap.
* KV cache types allowed: F32/F16/BF16/Q8_0/Q4_0/Q4_1/IQ4_NL/Q5_0/Q5_1
  (`common/arg.cpp`). Quantized KV needs Flash Attention support or it falls
  back.
* Known tradeoff: q8_0/q4_0 KV saves VRAM but can decode slower than f16 on
  some backends/contexts from dequant overhead (llama.cpp #27796, RDNA4/HIP
  reports of 20-37% slower at long ctx). Must A/B per context length.

Action:

1. Sweep `-ctk/-ctv f16` vs `q8_0` vs `q4_0` at 4k, 32k, 128k ctx on the dense
   model, with `-fa auto` and `-fa 1` / `-fa 0`. Record tg and perplexity.
2. Confirm the tile kernel is selected (log/dispatch counters). If Q8_0 KV
   never dispatches the tile path, the gate (`op_params`, GQA ratio, D) is
   wrong for Qwen3/Qwen3.5 heads (Qwen3.5 full-attn head dim 256, GQA 6:1).
3. Do not enable `GGML_HIP_ROCWMMA_FATTN=ON` blindly: issue #24437 reports up
   to -41% prefill regression at long ctx on gfx1151. A/B prefill too.

Expected: negligible at short ctx; 10-30% at long ctx if KV BW bound.
Risk: quality (KV quant) and prefill regressions (ROCWMMA). Gate on both.

### 3.4 GDN fusion for Qwen3.5 hybrid layers (deferred halo item 4)

What: 48 of 64 Qwen3.5-27B layers are Gated DeltaNet recurrent layers, not
full attention. Halo fuses the whole
conv->norm->gate->recurrence->state-copy chain into one kernel
(`GGML_CUDA_DISABLE_GDN_GATE`), plus DPP reductions and 16/32-warp configs
in `gated_delta_net.cu`.

Status: DEFERRED in this fork. The MoE/qwen4exp ports did not bring it over.

Why it matters for "dense" Qwen3.5: even though GDN state per step is small,
48 layers x several small kernels each is ~200-300 launches per token of
pure overhead plus repeated state traffic. Fusion converts that to ~48
launches and keeps state on-chip.

Action: port halo `gated_delta_net.cu` GDN decode fusion + DPP reductions,
gated to RDNA3.5, after items 3.1-3.3 are measured. Qualify with the
recurrent-state lifecycle checks (append/rollback/restore/copy, bit-identical
logits) used for the qwen4exp indexer promotion.

Expected: single-digit % on tg at short ctx (overhead removal), more at
large batch or with graphs disabled. Risk: medium. Recurrent kernels are
hard to debug; keep the unfused path behind the disable env.

Does not apply to pure Qwen3 dense (no GDN layers). Skip for Qwen3-32B.

### 3.5 Speculative decoding (biggest algorithmic lever)

What: verify K draft tokens in one target forward pass to amortize weight
reads. Effective tok/s ~= accepted_tokens / target_time. Needs acceptance
rate x K > 1 after draft cost.

Options in this tree:

* Qwen3.5 MTP heads: Qwen3.5 ships MTP layers (dense attention-only). Use
  `--spec-type draft-mtp` (or `mtp` sidecar) instead of a separate draft
  model. Upstream PR #20700 added Qwen3.5 MTP support; `FastMTP` trims the
  MTP lm-head vocab 248k->32k to cut draft overhead. This fork already has
  adaptive draft sizing (`--spec-draft-adaptive`, see README): each draft is
  sized from measured acceptance instead of always `--spec-draft-n-max`.
* Qwen3 dense (no MTP): use a small draft model from the same family
  (e.g. Qwen3-1.7B drafting for Qwen3-32B) or prompt-lookup
  (`--spec-type lookup`). Draft must be cheap on the same iGPU and share the
  quant/KV budget.
* Verify path uses MMQ/BLAS at batch K+1, not MMVQ. So MMQ tile tuning
  (halo item 8, PORTED) and hipBLASLt GEMM quality (section 3.6) set the
  verify cost. Compact MUL_MAT_ID 512x10 (PORTED) is MoE-only; dense verify
  is plain MUL_MAT.

Action:

1. For Qwen3.5-27B: benchmark `--spec-type draft-mtp` with adaptive sizing
   on/off, draft widths 2/3/4, at short and long ctx. Report acceptance +
   net tok/s. Try `--spec-draft-n-max` 3 first (typical ~75% acceptance,
   1.4-2.2x in community reports).
2. For Qwen3-32B: benchmark a 0.6B/1.7B/4B Qwen3 draft at Q4, same widths.
   Include draft VRAM in the budget.
3. Confirm HIP graphs stay replayable with speculation on (section 3.7).
   Speculative batches change shape; graph re-capture cost must not erase
   the gain at low acceptance.

Expected: 1.3-2x when acceptance is high; net negative when it is not.
The adaptive flag exists to avoid the negative case. Risk: VRAM (two
models or MTP heads + extra KV), quality-neutral if verified correctly.

### 3.6 Prefill/verify GEMM path: hipBLASLt + MMQ tuning

What: decode uses MMVQ, but prefill and speculative verify use GEMM/MMQ.
On gfx1151 the rocBLAS Tensile kernels for these shapes are weak; routing
through hipBLASLt (TensileLT) fixed pp 348->883 t/s on Llama-2-7B in issue
#13565 (`ROCBLAS_USE_HIPBLASLT=1`). Some shapes still fall back with
`hipBlasLT failed, falling back to tensile`.

Status: MMQ tile tuning + K prefetch (`mmq-config-rdna3-5.cuh`, `mmq.cuh`,
`mmq-vec-dot.cuh` split-j): PORTED. Halo's 256-expert routed-compact MoE
selection and swiglu/pair decls were deliberately left out; they do not
affect dense.

Action:

1. A/B `ROCBLAS_USE_HIPBLASLT=1` vs unset for pp512 and for speculative
   verify throughput on the dense model. Keep it if verify/pp wins and tg
   is neutral (tg should be unaffected since it is MMVQ).
2. Run `test-backend-ops perf -o MUL_MAT` across the dense FFN shapes
   (e.g. 5120x17408, 17408x5120, 5120x248320 for lm-head) to find shapes
   that fall back to Tensile. Those are candidates for MMQ-tile work, not
   MMVQ work.
3. Keep `GGML_CUDA_FORCE_MMQ` / `GGML_CUDA_FORCE_CUBLAS` experiments
   explicit and short; they cut across all shapes.

Expected: mostly TTFT/verify, not tg. Risk: low for tg; verify correctness
on the fallback shapes.

### 3.7 Launch overhead: HIP graphs, scheduler, threads

What: 64 layers x ~10-15 kernels = 600-900 launches per token. At ~10 t/s
that is 6k-9k launches/sec; CPU launch + scheduler overhead is visible once
MMVQ is near peak.

Status:

* `GGML_HIP_GRAPHS` defaults ON; `GGML_CUDA_GRAPHS` defaults ON
  (`ggml/CMakeLists.txt`, `CMakeLists.txt`). Upstream PR #22254 enabled HIP
  graphs by default to cut decode launch overhead. This fork relies on PM4
  graph replay on gfx1151 (see qwen4exp doc).
* Scheduler has a UMA ring tweak (`GGML_SCHED_UMA_RING` in
  `ggml/src/ggml-backend.cpp`) for integrated GPUs.
* `build-windows.ps1` builds `gfx1151`-only (`-DGPU_TARGETS=gfx1151`),
  Release, HIP ON. Good: no fat-binary dispatch cost.

Action:

1. A/B HIP graphs on/off on the dense model tg. If off wins, some node is
   breaking capture and forcing re-capture; find it via the graph-compat log
   (`disabling CUDA graphs due to unsupported node type`) rather than
   leaving graphs off.
2. Sweep `-t` (CPU threads) at full GPU offload (`-ngl 999`). Decode with
   full offload wants few threads (scheduler/sampler only); too many causes
   contention. Also sweep `-ub` (physical batch) for the server case.
3. Keep all weights on the iGPU (`-ngl` full). Partial offload on a unified
   APU still costs copies/syncs. If the 27B Q4 + ctx + KV does not fit the
   carve-out, reduce ctx or KV bytes (section 3.3) before splitting layers
   to CPU.

Expected: 5-15% if graphs are currently breaking; near zero if replay is
already clean. Risk: low. Graphs are a correctness hazard only when new
kernel types are added without capture support; qualify with long runs.

### 3.8 Small kernels: norms, QK-norm, rope, embedding gather, sampler

What: individually <1%, collectively ~5-8% of GPU time plus disproportionate
launch count. Qwen3/Qwen3.5 add QK RMSNorms per full-attn layer (2 extra
norms x 16-64 layers per token). Qwen3.5 vocab 248k makes lm-head softmax
and sampling heavier than on 151k-vocab models.

Status:

* RMS norm + fused norm-mul kernels exist (`norm.cu`,
  `ggml_cuda_op_rms_norm_fused`). Rope fused kernel exists
  (`ggml_cuda_op_rope_fused`); fused path handles norm/neoX rope modes only.
* Embedding gather uses getrows kernels (`getrows.cu`); this fork already
  prefetches mmap rows before sparse gathers (commit `48d8f1a67`). Single
  row gather per token is negligible after that.
* Top-k: local radix top-k (1049 lines) is ahead of halo (341 lines); kept
  per the decode-speedups doc. Do not "sync" it backwards.

Action:

1. Profile the non-GEMV tail at batch 1 (`--kernel-trace`): norm/rope/GLU/
   softmax/copy shares. Only fuse what shows up.
2. Candidates in priority order: (a) QK-norm fold into QK-matvec epilogue or
   rope fusion where the fused kernel already supports the mode; (b) norm
   fused with preceding residual add (already partly supported via
   `rms_norm_fused`); (c) sampler-side: keep the radix top-k, avoid heavy
   sampler chains (dry/xtc/typ_p) in benchmarks.
3. lm-head: measure the 5120x248320 GEMV + softmax + top-k slice alone at
   batch 1. If it is >5% of step time, consider F16 vs Q4 lm-head output
   weight tradeoff and FastMTP-style vocab trimming for the draft path only.
   Do not trim the target vocab.

Expected: 2-5% total. Risk: low-medium. Norm/rope fusion is precision
sensitive; qualify with perplexity and exact-logit checks.

### 3.9 Memory placement, context, batch, server flags

* Full offload first (`-ngl 999` / `-ngl -1` depending on CLI). Partial
  offload on Strix Halo trades a little VRAM for a large sync cost per
  layer. Measure split points only if full offload OOMs.
* Context (`-c`) sets KV allocation and attention cost. Do not benchmark
  dense tg at 128k ctx and conclude the kernels are slow; sweep 4k/32k/128k
  and report all three. Enable KV defrag defaults; report OOM vs slow.
* Server: single slot for peak tok/s. Parallel slots share the same ~150
  GB/s sustained BW, so per-slot tok/s falls roughly 1/N. For throughput
  (tokens/sec across slots) batching helps; for latency it hurts.
* Keep `LLAMA_HIP_UMA`-style env experiments explicit. The tree keys UMA
  behavior off detection + `GGML_SCHED_UMA_RING`, not a single documented
  flag; check `ggml-cuda.cu` UMA paths and `ggml-backend.cpp` ring before
  copying Linux env recipes to Windows/TheRock.
* Windows specifics: keep `C:\TheRock\build\bin` on PATH at runtime,
  `gfx1151`-only build, Release, TheRock 10.0.0 pinned (see
  `docs/windows-rocm10-release.md`). Re-check the runtime DLL/.kpack layout
  after any SDK bump; a wrong rocBLAS data file silently costs pp.

### 3.10 Compiler and codegen traps (gfx1151-specific)

Two findings from hand-written gfx1151 decode kernels (Atlas, quoted in
#24438) to check before writing new kernels:

1. Scalar-issue trap: the HIP compiler emits long scalar `v_fmac` chains for
   dequant-multiply loops unless the kernel is tiled/unrolled with
   `__restrict__` pointers and a VGPR budget that vectorizes. Signature:
   high VALU utilization with low VMEM stall on a kernel that should be
   bandwidth bound. Fix lives in kernel structure, not launch geometry.
2. Constant-cache broadcast serialization: LUT indexed divergently per lane
   serializes on RDNA3.5 (one broadcast/cycle/wave). Moving the LUT to
   banked LDS fixed one w4a16 GEMV from ~40% to bandwidth bound. Q8_0
   dequant is scale-mul rather than LUT so the exact mechanism differs, but
   routing/gather paths have the same divergent-index shape.

Action: when a new MMVQ variant underperforms, check VALU vs MemUnitStalled
before touching `nwarps` or tile sizes. Prefer reading the emitted ISA for
the dequant loop over guessing parameters. The nwarps=1 RDNA3.5 table is
already the measured optimum across quants (section 3.1); do not globally
raise it.

### 3.11 What NOT to port for dense

From `docs/strix-halo-decode-speedups.md`, these halo items do not help
dense Qwen3/Qwen3.5 decode and should stay out:

* Compact MUL_MAT_ID 512x10 (`mmid.cu`): MoE/spec-verify only. Already
  identical to halo; leave it.
* MoE weighted down-proj fusion (`GGML_CUDA_DISABLE_WEIGHTED_DOWN`): MoE
  only. Partly ported; nothing to do for dense FFN.
* hyperconn rename (`hc-cn/mix.cu` -> `hyperconn.*`): qwen4exp/DeepSeek-V4
  only. Needs Windows-build reconcile first; no dense benefit.
* QSA split (`qsa-decode.cu`/`qsa-prefill.cu`), top-k QSA beyond 2k
  (hipCUB in `common.cuh`): sparse-attention models only.
* `top-k.cu` sync: local radix top-k is ahead; keep it.
* GCN/CDNA/RDNA4/NVIDIA tuning tables: gfx1151-only fork; do not widen
  kernel gates.

## 4. Suggested order of work

1. Baseline + quant sweep (3.1) with perplexity gates. Cheapest, largest
   dense swing. Decides which MMVQ path to tune.
2. Verify fusion hits (3.2): `DISABLE_MMV_GROUP` / `DISABLE_FUSION` A/B,
   then port `quantize_mmq_q8_1_swiglu` if the pair path is hot.
3. Attention/KV sweep (3.3) at 4k/32k/128k. Picks `-fa` and `-ctk/-ctv`
   defaults for the release notes.
4. Speculation (3.5): MTP for Qwen3.5, draft model for Qwen3, adaptive
   sizing on. Largest upside, needs acceptance data to justify VRAM cost.
5. GDN fusion (3.4) for Qwen3.5 only, after 1-4 are stable.
6. Graphs/threads/offload hygiene (3.7) + small-kernel tail (3.8) + GEMM
   verify path (3.6). Each is small alone; together they set the ceiling.
7. System/build lock (3.9): TheRock pin, `gfx1151`-only, hipBLASLt default,
   clocks/power, single- vs multi-slot server guidance.

## 5. Acceptance checklist for any dense change

* Same GGUF, same `-p/-n/-fa/-ctk/-ctv/-t/-ngl`, same TheRock build, `r>=3`.
  Report pp and tg separately; `-p 0 -n 128` isolates tg.
* `llama-perplexity` before/after (watch Q5_1, IQ3_S pair, Q8_0 4-column,
  KV-quant, and any norm/rope fusion).
* Long-context spot check when touching attention or KV (perplexity at the
  ctx the change targets, plus a coherent sample to EOS).
* Speculation: report acceptance rate + net tok/s, not just draft width.
* Keep RDNA3.5 gates narrow (`GGML_CUDA_CC_IS_RDNA3_5`). No arch-widening
  in a gfx1151-only fork.
* Keep HIP graph replay working; a "faster kernel" that breaks capture and
  forces eager launch every token is a net loss at batch 1.

## 6. Sources and local references

* Local: `docs/strix-halo-decode-speedups.md` (ranked halo ports, ported vs
  deferred), `ggml/src/ggml-cuda/mmvq.cu` (RDNA3.5 table, q4_columns),
  `ggml/src/ggml-cuda/mmvf.cu` (single-column prefetch/bf16_wave),
  `ggml/src/ggml-cuda/ggml-cuda.cu` (fusion, MMV_GROUP, WEIGHTED_DOWN,
  graph capture), `ggml/src/ggml-cuda/fattn.cu` + `fattn-tile-rdna3-5.cu`
  (Q8_0 KV tile), `ggml/src/ggml-cuda/quantize.cu` (MMQ quantize),
  `ggml/src/ggml-cuda/norm.cu`, `getrows.cu`, `top-k.cu`,
  `src/models/qwen3.cpp`, `src/models/qwen35.cpp`,
  `docs/development/qwen4exp-decode-indexer.md` (replay qualification
  method), `docs/windows-rocm10-release.md`, `build-windows.ps1`.
* Upstream issues: ggml-org/llama.cpp #24438 (HIP ~40% BW report, AMD repro
  with `--kernel-trace` breakdown, Atlas codegen notes), #24437 (ROCWMMA
  prefill regression), #13565 (HIP vs Vulkan on Strix Halo, hipBLASLt
  348->883 pp fix), #27796 (quantized-KV decode slowdown reports).
* Speculation: `docs/speculative.md`, upstream PR #20700 (Qwen3.5 MTP),
  README adaptive draft sizing in this fork.
* Hardware: Strix Halo 256-bit LPDDR5X-8000 ~256 GB/s peak, RDNA3.5 40 CU
  iGPU; sustained depends on backend (see measurements above, not the
  peak number).

## 7. Pros and cons per item (reviewed against the tree)

Each item in section 3 was checked against the code cited. Four findings
change how to read the list; they come first, then pros/cons per item.

### 7.0 Corrections to sections 3.1-3.4

A. The 4-column kernels do not run at batch-1 decode. Both
   `mul_mat_vec_q4_columns_rdna3_5` (types in
   `is_rdna3_5_q4_columns_type`, mmvq.cu:584) and the Q8_0-only
   `mul_mat_vec_q4_columns` live in `case 4:` of `switch (ncols_dst)`
   (mmvq.cu:2368-2398). Batch-1 tg takes `case 1`. Both also require
   `no_fusion` and `ids == nullptr`. Their real audience is batch-4
   traffic: speculative verify at draft width 3, or multi-slot batches.
   So the 3.1 kernel-quality discussion and the "watch the Q8_0 4-column
   path" advice do not apply to `-p 0 -n 128` tg runs. Side note: Q8_0
   is not in `is_rdna3_5_q4_columns_type`; it has its own kernel and
   dispatch. The `calc_nwarps` = 1 table for RDNA3.5 checks out
   (mmvq.cu:443-446 plus static_asserts).

B. The FQ-cap math confirms the 3.2 swiglu target. `mul_mat_vec_q_fq_try`
   (mmvq.cu:2599) caps the y buffer at `(ne10/32)*36 <= 16384` bytes.
   Gate/up (n_embd 5120, ~5.6 KB) fit; the down projection (n_ff 17408,
   544 blocks, ~19 KB) does not, so every down matvec falls back to a
   separate `quantize_row_q8_1_cuda` launch (mmvq.cu:2923-2933). That
   fallback launch is exactly where halo's `quantize_mmq_q8_1_swiglu`
   (confirmed absent from `quantize.cu`) hooks: one 17k-element
   SiLU+mul+quantize pass per layer x 64.

C. IQ4_NL KV has no FA kernel on HIP. `ggml_cuda_fattn_kv_type_supported`
   (fattn.cu:505) accepts F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 only;
   IQ4_NL returns `BEST_FATTN_KERNEL_NONE` (fattn.cu:620). The 3.3 sweep
   list needs that caveat. The Q8_0 tile path (fattn.cu:521-531) also
   needs `gqa_opt_applies` at dispatch (fattn.cu:663: GQA >= 2, mask
   present, zero bias, padded KV), which can silently not engage; check
   dispatch counters, do not assume.

D. The 3.4 premise is partly stale. The fused GDN op
   (`GGML_OP_GATED_DELTA_NET`, AR+CH) is already supported on HIP
   (ggml-cuda.cu:2411, :6342), defaults on with auto-probes
   (llama-context.cpp:251-252, :595-596), and a local GDN+snapshot-`cpy`
   fusion already exists (`ggml_cuda_try_gdn_cache_fusion`,
   ggml-cuda.cu:2818, :4031). Genuinely missing vs halo is only:
   conv/norm/gate pre-fusion (qwen35.cpp:429-430 still emits two separate
   `build_gdn_l2_norm` per layer), DPP reductions, and 16/32-warp
   configs. Re-measure launch counts before porting; the "200-300
   launches/token -> 48" claim predates the local fusion.

### 7.1 Weight quant sweep (3.1)

Pros: zero code, fully reversible, largest byte-reduction lever. Dense
reads all params per token, so tok/s scales near-linearly with file size.
Effective-BW arithmetic separates "smaller file" from "faster kernel".
No fork-maintenance cost; results go straight into release notes.

Cons: quality is the real price and perplexity alone is a thin gate;
quant noise in a hybrid GDN model can hurt reasoning in ways wiki
perplexity misses, so each candidate also needs a coherent sample. It is
a per-GGUF activity (~10 quants x bench + perplexity at ~6-10 t/s), not
a durable fork improvement. Per finding A the kernel-quality subplot
mostly does not apply at batch 1, so this is really "smallest
acceptable file" plus a secondary +-5-15% kernel spread. Rankings shift
again once speculation (batch-4 verify) is in play.

### 7.2 Gate/up fusion + swiglu quantize (3.2)

Pros: scaffolding already ported (`should_fuse_mul_mat`, MMV_GROUP,
`mul_mat_vec_q_fq`), so remaining work is bounded. Per finding B the
deferred `quantize_mmq_q8_1_swiglu` has a proven specific target: the
down-proj fallback quantize on every layer. Fewer launches help graph
replay (3.7). Low risk, rollback via existing `DISABLE_*` flags.

Cons: low single-digit % at best; activation traffic is small next to
~15-17 GB/token of weights. The 3.1 and 3.2 A/Bs interact (fusion on
skips the `no_fusion`-gated kernels), so they cannot be evaluated
independently. Fused kernels are bit-sensitive; each needs perplexity +
exact-logit qualification. Ceiling polish, not a roofline change.

### 7.3 KV quant + tile kernel + `-fa` (3.3)

Pros: the expensive part (Q8_0 tile kernel, D in {64,128,256}, batch 1,
GQA >= 2) is already ported and both models qualify on paper (Qwen3.5:
D=256, GQA 6:1; Qwen3-32B: D=128). At long ctx this is the only item
that cuts the growing term; at 32k+ ctx on a 64-full-attention-layer
model, KV traffic is a visible share of weight traffic. Config-only to
try; also buys VRAM headroom for ctx or for fitting at all.

Cons: highest quality risk on the list; K/V quant error is systematic
and compounds over ctx, so gate on long-ctx perplexity, not short.
Dequant overhead can make quantized KV slower than f16 at short/mid ctx
(the cited 20-37% regressions). The sweep matrix (types x ctx lengths x
`-fa` modes on a 27B) is expensive and the answer will be "depends on
ctx", i.e. release-note complexity. Only symmetric Q8_0 K+V gets the
tile kernel; per finding C, IQ4_NL KV has no FA kernel on HIP, and the
tile gate conditions can silently disable the fast path. Do not touch
`GGML_HIP_ROCWMMA_FATTN` without a prefill A/B (up to -41% reported).

### 7.4 GDN fusion, Qwen3.5 only (3.4)

Pros: 48/64 layers makes GDN the largest remaining non-GEMV kernel group
on Qwen3.5. Fusing conv/norm/gate into the recurrence keeps state
on-chip and cuts launches, helping most when the GPU is underfed
(graphs off, larger batches). GDN support plus a qualification pattern
(state lifecycle checks, bit-identical logits) already exist, and the
unfused path can stay behind the disable env.

Cons: per finding D the prize is smaller than stated; fused GDN AR/CH
and state-copy fusion are already in this tree, so only the
conv/norm/gate pre-fusion + DPP + warp configs remain, and the
launch-count claim must be re-profiled first. Highest effort and debug
cost on the list (recurrent kernels, rollback/restore) for a
doc-estimated single-digit %. Zero benefit for Qwen3-32B. Halo's version
needs Windows/MSVC reconciliation.

### 7.5 Speculative decoding (3.5)

Pros: the only item that beats the bandwidth roofline instead of
approaching it; amortizing ~15 GB/token of weight reads across accepted
tokens is a genuine 1.3-2x at high acceptance. Cheap to try here:
`draft-mtp` exists, qwen35.cpp already loads nextn/MTP weights, and the
local `--spec-draft-adaptive` mitigates the main failure mode (wide
drafts into low acceptance). Quality-neutral by construction. No kernel
work needed for the experiment.

Cons: acceptance is workload-dependent; report acceptance rate, not just
tok/s. On a unified APU the costs land on the scarcest resource: MTP
heads + deeper draft KV compete with weights/KV in one carve-out.
Verify runs at batch K+1, off the tuned batch-1 MMVQ path onto the
weaker GEMM/MMQ path, so 3.6 is a prerequisite, not a follow-up. Graph
re-capture on changing batch shapes can erase the gain. The draft
lm-head cost is currently unavoidable (full 248k-vocab GEMV per draft
step); FastMTP-style vocab trimming is not in this tree. Qwen3-32B
(no MTP) additionally pays a second draft model in VRAM.

### 7.6 hipBLASLt + MMQ tuning (3.6)

Pros: env-var A/B (`ROCBLAS_USE_HIPBLASLT`), zero code risk to tg, large
historical upside (348->883 pp in the cited report). Serves TTFT and,
via 3.5, verify throughput. MMQ RDNA3.5 tile config already ported, and
`test-backend-ops` gives a clean per-shape fallback map.

Cons: does nothing for the stated goal (batch-1 tg is MMVQ, untouched).
Wins are shape- and version-dependent with silent fallbacks, so any
default needs re-qualification per TheRock bump; a wrong rocBLAS data
file on Windows silently costs pp. `FORCE_MMQ`/`FORCE_CUBLAS`
experiments cut across all shapes; keep them short and explicit.

### 7.7 Graphs, scheduler, threads, offload (3.7)

Pros: nearly free to test with conditional upside: a node breaking PM4
capture, found via the graph-compat log, is a 5-15% fix hiding in plain
sight. The hygiene items (full offload, few `-t` at full GPU offload,
KV-bytes before CPU-splitting) are the highest value-per-word release
notes; they stop users benchmarking their own misconfiguration.

Cons: pays ~zero if replay is already clean; validation, not capability.
`-t`/`-ub` optima are per-model and do not generalize. Capture debugging
on Windows/PM4 is painful, and the item never closes: every kernel from
3.2/3.4/3.8 can silently re-break capture. Full offload may simply not
fit (27B Q4 + 128k KV in a shared carve-out).

### 7.8 Small kernels (3.8)

Pros: fusion machinery already exists (`rms_norm_fused`,
`rms_norm_fused_add`, `rope_fused`), so leftovers are plumbing, not new
kernels; the local radix top-k is correctly ahead of halo (nothing to
do). The lm-head slice (5120x248320 GEMV + softmax + top-k, ~5% of
params in one oddly-shaped GEMV) is a genuinely unmeasured single item
that could reveal an outlier. Collectively ~5-8% GPU time plus
launch count.

Cons: worst effort/reward ratio unless profiling shows an outlier first:
several touches + a large A/B matrix for an expected 2-5%. Norm/rope sit
on the exact-logit path and fused variants are generally not
bit-identical, so each needs perplexity + exact-logit qualification.
QK-norm-into-QK-epilogue depends on limited fused-kernel mode support.
Sampler-chain advice is user config, not a fork win.

### 7.9 Memory, context, batch, server flags (3.9)

Pros: free, no code, biggest variance-reducer in practice; most "slow
backend" reports are really "dense tg at 128k ctx". Single-slot
guidance on a shared ~150 GB/s bus prevents bogus multi-slot
expectations. The build half (TheRock pin, gfx1151-only) is done;
locking it in is cheap.

Cons: documentation, not engineering; closes no gap, only stops false
ones. Carve-out is a BIOS/driver variable the fork cannot control, so
numbers will not reproduce across machines. No single UMA flag exists
(detection + `GGML_SCHED_UMA_RING`), so Linux env recipes stay a
footgun the doc can only warn about.

### 7.10 Codegen traps (3.10)

Pros: cheap to check, potentially large per kernel, and it generalizes
to every new kernel from 3.2/3.4/3.8. The trap is confirmed live in this
tree (ROCm 7.14 Q5_1 workaround, mmvq.cu:816). "Check VALU vs
MemUnitStalled before touching nwarps" protects the already-optimal
nwarps=1 table from cargo-cult tuning.

Cons: the counters needed for this diagnosis are the ones reported
bogus on gfx1151. ISA inspection on Windows/TheRock is awkward, fixes
are LLVM-version-dependent (rot per SDK bump), and there is no action
item until a new kernel underperforms. Risk of premature optimization on
kernels already near bandwidth.

### 7.11 What NOT to port (3.11)

Pros: negative-value-work prevention is real value in a fork sustained
by agent labor. Narrow RDNA3.5 gates, keeping the ahead-of-halo top-k,
and recording "decided, do not re-litigate" all compound. Scoping
matches the acceptance checklist.

Cons: a veto list looks like progress while the real bottlenecks
(bytes-per-token policy, VRAM, acceptance data) are measurement and
judgment calls. Vetoes go stale: if qwen4exp-family or
sparse-attention support lands later, the hyperconn/QSA rejections need
re-checking, and a doc-only veto has no enforcement against a future
port sweep. Even "MoE-only" items deserve a second look once dense +
spec-verify changes which batches actually run (see finding A).

### 7.12 Adjusted reading order

Section 4 is mostly sound, with two adjustments from the findings above:
(1) precede everything with a `--kernel-trace` profile establishing
which path batch-1 actually takes (case-1 MMVQ? FQ hit or fallback? tile
kernel engaged? GDN launch count?) - the section 2 bench commands do not
include this and several items depend on it; (2) evaluate 3.1 and 3.2 as
a coupled pair (fusion state changes which kernels run), and treat 3.6
as a prerequisite input to 3.5 rather than later cleanup. Highest
expected-value sequence: profile -> quant sweep (3.1) -> verify fusion +
port swiglu-quantize (3.2) -> KV/`-fa` sweep at three ctx lengths (3.3)
-> MTP speculation with adaptive sizing (3.5) -> graphs/threads hygiene
(3.7) -> GDN pre-fusion only if the profile shows the launches (3.4) ->
small kernels opportunistically (3.8).
