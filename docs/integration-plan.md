# Mainline and Unsloth integration plan

Investigation date: 2026-09-20 (America/New_York).

Status: investigation only. No inference code was changed, no integration was applied to the working checkout, and no commit, push, or PR was created. The current source guards passed. Builds and GPU/model tests were not run for this document.

## Recommendation

Merge a pinned **canonical `ggml-org/llama.cpp` revision into this fork**, preserving the Windows, Strix Halo, Qwen4exp, loader, and scheduler work. Then consider individual Unsloth additions after that baseline passes validation. This follows the user's clarification that mainline is preferable if Unsloth substantially increases integration complexity.

Use canonical `b11065`, commit `ce8caa6e60a03093351d6016a818720e0d46f0fb`, as the proposed first target. It was the canonical master tip and newest published build tag when checked. Refresh the investigation before substituting a newer revision; do not make the implementation depend on a moving `master`.

A direct pull from `unslothai/llama.cpp:master` would not deliver the expected newer inference features. Its inference source is still the canonical `b10632` tree from August 26. Unsloth builds its newer binaries from a separately selected canonical release tag plus an ordered set of pinned PRs. Those PRs are not all merged into Unsloth master. The latest published mix inspected was `b11030-mix-5ff778e`, based on canonical `b11030` with 13 pins.

The practical choices are:

| Route | Result | Assessment |
| --- | --- | --- |
| Merge Unsloth master | Primarily imports Unsloth release automation and repository housekeeping | Small textual conflict count, but does not provide the desired runtime update |
| Merge canonical b11065 into this fork | All canonical changes through the pinned revision, with the existing local improvements retained | Recommended; four source/test conflict files in the dry run |
| Canonical b11065 plus selected Unsloth deltas | Mainline update plus specific additional functionality | Recommended follow-up when a delta addresses an actual need |
| Reproduce the full Unsloth b11030 mix, then reconcile this fork | All 13 shipped Unsloth features plus local improvements | Larger scope, especially MTP, GLM-5-Next, Inkling, quantization, and shared loader/backend changes |
| Start from Unsloth or mainline and replay the local fork | Reconstructs a substantial intertwined local patch set | Avoid; increases opportunities to omit Windows fixes and complicates history |

For the recommended route, completion means mainline integration with local behavior preserved. It does **not** mean all 13 Unsloth features are present. Section 6 accounts for every pin so any omitted feature is an explicit scope decision.

## 1. Verified revisions and provenance

| Item | Verified value |
| --- | --- |
| Current branch | `upstream-sync` |
| Local HEAD | `a5618a6db45429427d23bfc7705550b4829bffe6` |
| Initial working tree | Clean |
| `origin` | `https://github.com/thomas9120/llama.cpp-halo-windows.git` |
| Existing `upstream` | `https://github.com/pwilkin/llama.cpp.git` |
| `halo-box` | `https://github.com/halo-box/strix-llama.cpp.git` |
| Unsloth default branch | `master` |
| Unsloth master | `40a6d6d92fd4c6189928ba6f0252375573902533` |
| Unsloth source base / common ancestor with this fork | `11cd98842874cc1b87ac274bd2d5cceb38102bb2` (`b10632`) |
| Canonical master / b11065 | `ce8caa6e60a03093351d6016a818720e0d46f0fb` |
| Common ancestor of local HEAD and canonical master | `718f7b4175bf8b6af6f5eac09fee10754b3ecddd`, September 12 |
| Latest inspected Unsloth release | `b11030-mix-5ff778e`, published September 18 at 15:36:14 UTC |
| That release's canonical base | `bdcbaaf6e7520b68c8c60ff724c67409970d70e1` (`b11030`) |
| Mix commit recorded in its manifest | `6ba30d05b140ebb0baeded27d7d9b843c5b71ff1` |
| Mix `ggml` tree recorded in its manifest | `677df46a6800d1070bb787aaded017229583ce9b` |

The release manifest's ordered 13 PR URLs exactly matched `scripts/unsloth/pr-set.json` at the inspected Unsloth master. The mix source is published as a release asset; do not assume the release tag or its GitHub-generated source archive resolves to the composed inference tree. Use the named `llama.cpp-source-b11030-mix-5ff778e.tar.gz` asset and manifest if exact release reproduction is needed. The manifest's synthetic commit was recorded, not fetched or validated as an accessible Git commit in this investigation.

Primary references:

- [Unsloth source-base declaration](https://github.com/unslothai/llama.cpp/blob/40a6d6d92fd4c6189928ba6f0252375573902533/scripts/unsloth/upstream-sync.json).
- [Pinned PR set](https://github.com/unslothai/llama.cpp/blob/40a6d6d92fd4c6189928ba6f0252375573902533/scripts/unsloth/pr-set.json).
- [Mix-build workflow](https://github.com/unslothai/llama.cpp/blob/40a6d6d92fd4c6189928ba6f0252375573902533/.github/workflows/unsloth-prebuilt.yml).
- [Inspected release](https://github.com/unslothai/llama.cpp/releases/tag/b11030-mix-5ff778e) and [release manifest](https://github.com/unslothai/llama.cpp/releases/download/b11030-mix-5ff778e/llama-prebuilt-manifest.json).
- [Canonical b11065](https://github.com/ggml-org/llama.cpp/releases/tag/b11065).

## 2. Measured integration surface

These results use complete ancestry available through the local repository plus freshly fetched public heads and exact pins. A separate bare analysis repository under the ignored `build-integration-investigation/` directory borrowed local objects. `git merge-tree --write-tree --name-only` generated diagnostic trees there without changing this checkout's index, branch, or files. Exit code 1 represented conflicts, not a completed merge.

| Comparison | Local-only commits | Incoming-only commits | Textual conflict files |
| --- | ---: | ---: | --- |
| Local HEAD vs Unsloth master | 382 | 282 | 3: one workflow, `AGENTS.md`, `CODEOWNERS` |
| Local HEAD vs canonical b11030 | 87 | 103 | 24: 21 workflows and 3 source files |
| Local HEAD vs canonical b11065 | 87 | 138 | 29: 25 workflows and 4 source/test files |

Commit counts describe ancestry, not the number of distinct features or patches that must be ported. Cherry-picked and squashed equivalents can be counted on both sides.

Further measurements:

- The delta from Unsloth's canonical base to its master touches 102 files, with 12,604 insertions and 8,275 deletions, in automation/helpers and repository housekeeping. There is no delta under `src`, `ggml`, `common`, `conversion`, or `include`.
- The canonical delta from the September 12 merge base to b11065 touches 367 files, with 26,758 insertions and 15,926 deletions. The local delta from the same base touches 211 files, with 19,400 insertions and 9,213 deletions. This supports merging onto the local fork rather than rebuilding its changes from scratch.
- Stopping at b11030 removes one source/test conflict, but leaves the central Qwen and attention reconciliation work. It also omits 35 subsequent canonical commits. It is useful as the exact Unsloth-release reference, not a compelling shortcut for the mainline route.
- Many shared loader, scheduler, graph, enum, and CUDA/HIP files merge textually. They still require semantic review. Four conflicts is not a claim that only four files need inspection.

No cumulative 13-pin integration was resolved, built, or benchmarked. Individual pin conflict probes described below were against the original local HEAD, not against a finished mainline merge. Their counts cannot be added together to predict the final conflict count.

## 3. Local behavior that must survive

Use this as the preservation ledger during review. A replacement is acceptable only when it provides equivalent correctness and measured behavior on this fork's workloads.

| Area | Current implementation / evidence | Preservation requirement |
| --- | --- | --- |
| Windows build and distribution | `build-windows.ps1`, `build-windows-vulkan.ps1`, `test-windows.ps1`, `scripts/package-windows-rocm10.ps1`, [build guide](build.md), [release guide](windows-rocm10-release.md) | Retain VS 2022 x64 selection, configurable SDK/build paths, gfx1151 targeting, portable build behavior, runtime DLLs and kernel data. Keep the fork's release workflow. |
| Windows HC16 exclusions | `ggml/src/ggml-cuda/mmb.cu`, `hc-mix.cu`, `ggml-cuda.cu` | Keep all three exclusions. New HC graph paths must not silently reintroduce the corrupting reduced-precision behavior. |
| Positioned lazy PLE reads | `src/llama-lazy-reader.h`, `src/llama-model.cpp`, Qwen PLE input code; `43ab114c3` | Preserve overlapped positioned Windows reads, concurrent gathers, speculative prefetch, and mmap fallback. `on-direct` is buffered on Windows and still uses the OS cache. |
| Mmap gather readahead | `src/llama-mmap.*`, `llama_model::prefetch_rows`, Gemma/Qwen callers; `48afa37d4` | Already incorporates the behavior of Unsloth #137. Keep the direct-reader branch and mmap prefetch branch distinct. |
| Sparse QSA graph and kernels | `src/models/qwen4exp.cpp`, `qsa-decode.cu`, `qsa-decode-wmma.cuh`, `qsa-prefill.cu`; `d67d58836`, `ce215c979`, `f5182c4f6`, `4012c0301` | Preserve direct selected indices, strip handling, masking, F16 K/V requirements, and the exact graph/kernel source and parameter contract. |
| Incremental indexer and visibility | `src/llama-memory-hybrid-idx.*`, `src/qsa-prefix-state.h`, `src/prefix.h`; [indexer notes](development/qwen4exp-decode-indexer.md), `2b453d2b8` | Preserve derived-cache invalidation across rollback, restore, sequence copy/keep/clear, prompt reuse, and screenshot-induced position gaps. |
| RDNA3.5 decode and prefill tuning | `mmvq.cu`, `mmvf.cu`, `vecdotq.cuh`, `mmq-config-rdna3-5.cuh`, `mmq.cuh`, `mmid.cu`, `fattn-tile-rdna3-5.cu`, dispatch in `ggml-cuda.cu` | Preserve architecture gates, fused quantization, grouped matvecs, weighted-down reduction, 512-expert/10-active handling, and RDNA3.5-only Q8 KV tile dispatch. |
| Existing local fusions | `mmb.*`, `hc-cn.*`, `hc-mix.*`, `hc-match.inc`, `gdn-conv.*`, `ple-conv.*`, `norm-gated.*`, `moe-weighted-reduction.*` | Review recognizers when upstream changes graph shapes, node order, or op contracts. A clean merge can stop a fusion from matching. |
| Radix top-k | `ggml/src/ggml-cuda/top-k.cu` and existing backend tests | Keep local selection algorithms and lifetime coverage; do not substitute a shorter external implementation by file replacement. |
| Scheduler input lifetime | `ggml/src/ggml-backend.cpp`, `ggml-backend-sanitize.*`, `src/llama-context.cpp`, `tests/test-backend-sched-ring.cpp`; [scheduler guide](development/backend-scheduler.md) | Preserve input-ring rotation, `ggml_backend_sched_prepare_inputs`, and split UID refresh when addresses move. Keep asynchronous output readback ordering. |
| MTP and speculative execution | `common/speculative.cpp`, Qwen converter/graph/metadata, `src/llama-context.*`; `a57fec31b`, `827e49380`, `e037f59fb` | Preserve adaptive draft sizing, complete-set head-mixer selection, draft-only loading, separate graph arenas for MTP batches without output, and target-tensor reuse already supported by the local path. |
| State I/O and other local models | `src/llama-kv-cache*.cpp`, `97f8b53f7`, `da44a1594`, local GLM-5-Next and Xing4.0 files | Preserve quantized block sizes in device state I/O, K-pool sizing/reuse, GLM-5-Next support, and the Xing4.0 port. These are part of the fork even though Qwen is the priority. |
| Server allocation recovery | `tools/server/server-context.cpp`, `scripts/windows-server-recovery.cpp` | Keep partial-batch discard, per-slot recovery, prompt-cache cleanup, and Windows allocation diagnostics. |

HIP compiles shared `ggml/src/ggml-cuda/*.cu` sources through `ggml/src/ggml-hip/CMakeLists.txt`. Review CUDA changes for HIP implications, including files that Git auto-merges. The historical [decode port inventory](strix-halo-decode-speedups.md) is useful context, but its original missing/deferred lists are not the current source inventory.

## 4. Resolve the mainline conflicts as coherent changes

### 4.1 Qwen hyper-connections, MTP, and sparse attention

`src/models/qwen4exp.cpp` contains the largest semantic conflict. Three incoming changes must be reconciled together:

1. [#28896: RMSNorm/multiply fusion](https://github.com/ggml-org/llama.cpp/pull/28896), commit `41abbfd59`, loads HC and PLE norm weights as `{ n_embd, hc }` with `TENSOR_ALLOW_RESHAPE`, and multiplies before flattening the grouped activation.
2. [#28901: HC operations](https://github.com/ggml-org/llama.cpp/pull/28901), commit `37b53fd45`, uses gated HC pre-ops and HC post-ops with a null combination tensor. CPU and backend implementations/support checks change with the graph.
3. [#28770: NVIDIA sparse attention](https://github.com/ggml-org/llama.cpp/pull/28770), commit `3cf03257f`, enables mask-derived sparse attention and changes the CUDA query-tile selection contract.

Resolution requirements:

- Combine upstream norm shapes and reshape flags with local `trunk_flags`, MTP flags, optional tensors, PLE loading, and draft-only checks. Do not restore mandatory trunk weights in a draft-only export.
- Audit every norm passed to `build_hc_mix`, including dedicated MTP `nextn.hc_head_norm` and the fallback `nextn.shared_head_norm`. Both are flat in the current local loader. Moving the graph to grouped multiplication while leaving those flat can cause `ggml_can_repeat` failures. Use compatible reshaping/loading for every supported mixer path, while retaining complete-set selection of norm/down/up tensors.
- Do not mechanically reshape every NextN norm: `nextn.hnorm` also has a separate use. Trace its consumer and serialized layout.
- Preserve the local graph-expansion/lifetime requirements and adapt HC pattern matching to the final graph. Inspect both explicit HC ops and legacy patterns. Validate generic CPU and Vulkan paths as well as HIP.
- Keep the HIP direct-index path, maskless prefill handling, and strip assembly. The dry-run QSA conflict juxtaposes the locally assembled `cur` with a new generic `build_attn_mha` declaration. Keeping both would be wrong; integrate the NVIDIA improvement in the applicable generic path.
- For the current gfx1151 direct-index nodes, preserve `src[5] = indices`, null `src[4]`, `src[6]`, and `src[7]`, and `op_params[4] == 0`. These are checked by the local kernel gate. Do not propagate the NVIDIA selected-key count into those nodes.
- Keep direct sparse selection gated on Flash Attention, KV offload, and F16 K/V. Verify 1-8 query tokens, D=256, single-stream layout, selected-index bounds, masks, and strides together. Larger verification batches need a correct fallback.

The Unsloth #144 discussion independently reports the MTP norm-shape failure after the canonical fusion change. This is a concrete integration hazard, not a hypothetical reason to reject the mainline update. Its fix must be adapted to the local fallback mixer as well as the dedicated mixer.

### 4.2 Attention accumulator layout

`ggml/src/ggml-cuda/fattn-mma-f16.cuh` conflicts around `VKQ_C` sizing and AMD accumulator branches. The local branch uses a type-based RDNA3 accumulator-layout condition; upstream separates MFMA behavior following [#28576](https://github.com/ggml-org/llama.cpp/pull/28576), commit `bfdc32183`.

Reconcile the actual accumulator types, matrix fragment layout, array lengths, and later accesses for RDNA3, RDNA3.5, MFMA, and NVIDIA separately. Retain the local RDNA3 mask-initialization fix (`dc39effa6`). Do not select an entire side of this header or transplant another architecture's performance assumptions.

Review auto-merged `fattn-common.cuh`, `fattn.cu`, and the sparse helper declarations at the same time. Upstream changes the compact-mask helper to produce per-query-tile index unions and counts; all callers and declarations must agree while the HIP-specific path remains separate.

### 4.3 Model dispatch condition

`src/llama-graph.cpp` has a clamped-SwiGLU condition where the local side includes `LLM_ARCH_GLM5NEXT` and the canonical side adds `LLM_ARCH_MAPLE`. Preserve both applicable architecture cases and the existing DeepSeek4/DFLASH/HY_V4 behavior. This is a small conflict, but choosing either whole side drops a model's intended arithmetic.

### 4.4 Backend test definitions

`tests/test-backend-ops.cpp` conflicts where the local top-k lifetime test precedes a reduction test renamed upstream to `test_moe_reduce`, and where local SYCL Hadamard cases meet new F16-output cases. Retain the lifetime test and both relevant test sets; reconcile class names, constructors, and registrations. Confirm the resulting executable actually runs the requested cases.

### 4.5 Workflow deletions and cleanly merged files

The 25 workflow conflicts are modify/delete conflicts: this fork previously removed those inherited workflows. Default to retaining those intentional deletions, while reviewing whether any newly arriving workflow belongs in the Windows fork. Preserve the fork's own release workflow. Do not use a directory-wide `ours` resolution that hides additions or unrelated changes.

Manually review auto-merged scheduler, model-loader, graph, tensor-enum, converter, and backend dispatch changes against the preservation ledger. In particular, mainline reservation-failure handling must coexist with local input-ring lifetime rules and server recovery. Keep API declarations, op implementations, capability checks, and test registration synchronized.

## 5. Implementation sequence for the recommended route

### Phase A: Freeze inputs and establish a baseline

1. Recheck status, branch, remote URLs, and the applicable repository instructions. Preserve any work added after this investigation.
2. Keep the current working binaries and record their exact commit and paths. Read each `CMakeCache.txt` before deciding whether it is reusable.
3. Run the current Windows source guards, full Windows regression suite, and representative Qwen target/MTP workloads before integration. Record any existing failures rather than attributing them to the merge later.
4. Inventory the actual local target and draft GGUF metadata, tensor names, shapes, quantization, and projector pairing. Do not infer shared-sidecar support from a filename.
5. Create an isolated `codex/` integration branch/worktree from the preserved local revision. Keep logs and generated artifacts in ignored build directories.

Example preparation commands for a future implementation, not commands run by this investigation:

```powershell
git status --short
git branch --show-current
git remote -v

$localBase = 'a5618a6db45429427d23bfc7705550b4829bffe6'
$canonical = 'ce8caa6e60a03093351d6016a818720e0d46f0fb'

# Fetch by URL so the existing pwilkin remote is not repointed.
git fetch --no-tags https://github.com/ggml-org/llama.cpp.git refs/tags/b11065
if ($LASTEXITCODE -ne 0) { throw 'Canonical fetch failed' }
if ((git rev-parse 'FETCH_HEAD^{commit}') -ne $canonical) { throw 'Canonical tag changed' }

# Use a new, unused path; if local HEAD has advanced, refresh the baseline first.
git worktree add -b codex/integrate-mainline-b11065 '..\upstream-sync-mainline-b11065' $localBase
if ($LASTEXITCODE -ne 0) { throw 'Worktree creation failed' }
Set-Location '..\upstream-sync-mainline-b11065'
git merge-tree --write-tree --name-only HEAD $canonical

# Begin the actual integration only when implementation is authorized.
git -c merge.conflictStyle=zdiff3 merge --no-commit --no-ff $canonical
```

The example baseline is the investigated code revision. Include later reviewed changes, including this plan, when preparing the actual worktree. The sibling worktree path requires filesystem access outside this checkout under restricted execution environments.

### Phase B: Resolve one canonical merge

1. Resolve workflow policy and the four source/test conflict files using Section 4.
2. Inspect the entire merged diff, including paths without conflict markers. Compare critical functions to both pinned parents, not just to the conflict-marker region.
3. Run `test-windows.ps1 -SourceOnly` immediately. If a structural guard fails because code moved, prove the protected behavior survives before updating the guard.
4. Check for missing declarations, changed op contracts, duplicate enum/table entries, broken model registration, and source files missing from CMake.
5. Build HIP and Vulkan and complete the validation gates below. Correct only integration-related defects.

Keep this as a real ancestry-preserving merge when the human finalizes it. Avoid squash-syncing or rebasing the published local history: future Git merges need to know the canonical revision is already integrated. Do not create synthetic ancestry with an `ours` merge. This plan does not authorize an automated commit or submission.

### Phase C: Qualify the mainline baseline

Accept the mainline integration only after the preservation ledger and validation matrix pass. If a new fusion is incorrect on Windows/gfx1151, retain a correct architecture-gated fallback and record the deviation and evidence. A successful build alone cannot establish preservation of decode speed or model output.

If the merge has not been finalized and must be abandoned, use `git merge --abort` in the isolated integration worktree after checking its status. Keep the original checkout and previous executable directory intact. Do not replace launcher binaries during qualification.

### Phase D: Evaluate optional Unsloth deltas separately

Start from the qualified mainline baseline. Recompute each candidate's patch equivalence and dependencies at that point. For an already ported feature, bring over only missing fixes; for a new feature, port its complete converter, graph, backend, API, and build dependencies. Keep an explicit PR/SHA/disposition record.

Prioritize small correctness fixes relevant to this machine, then performance candidates backed by local measurement. Large new-model and quantization features should have independent review and validation. Do not combine another halo-box optimization sweep with the canonical sync.

## 6. Unsloth pin inventory and optional full-feature route

All pins below are from the inspected release and master manifest. All were open with heads matching their pins at inspection except #144, which was **closed without a recorded merge**. Its code remains in the required mix. Closed status alone does not establish why it was closed or whether its behavior is safe.

Sizes are each pin's delta from its merge base with the inspected canonical head. The PR descriptions sometimes cite older sizes/bases; use the fetched diff rather than those historical figures.

| PR and pinned SHA | Pin delta | Local overlap and recommended disposition |
| --- | --- | --- |
| [ggml-org #24423](https://github.com/ggml-org/llama.cpp/pull/24423), `12e0a9627d02c6395fd4bbf2aadff93d0d46a0e4` | 29 files, +3,241/-124 | DiffusionGemma is a separate optional model/tool feature. It adds shared context/model changes and diffusion sampling code under the HIP-compiled CUDA directory. Include its executables and validation if full mix parity is required. |
| [ggml-org #25731](https://github.com/ggml-org/llama.cpp/pull/25731), `946fc11d1afb1e6cd316e853f1b23487754a74b9` | 70 files, +4,120/-126 | Inkling is a broad optional port: banded attention, indexing-width changes, converter, parser, audio/vision, and shared matvec/backend code. Individual probe conflicts in `fattn.cu` and `ggml.c`; full combined behavior needs review. |
| [Unsloth #70](https://github.com/unslothai/llama.cpp/pull/70), `883f2c9ba78f3847148454adf025da29385fff3e` | 9 files, +204/-1 | Kimi-K3 vision/full-size loading additions. Separate optional feature; preserve existing Kimi text support and verify projector/converter registration. Individual probe merged textually. |
| [Unsloth #61](https://github.com/unslothai/llama.cpp/pull/61), `46cbf0e95786fe8f5b7c0e86d57aaf8f8eceea7f` | 41 files, +2,334/-25 | Adds IQ1_XS/XXS/XXXS. Optional compatibility expansion, not a demonstrated gfx1151 speedup. Touches shared type IDs, quantization, MMQ/MMVQ and RDNA3.5 configuration. Individual probe merged textually but AMD matrix loaders need hardware validation. |
| [Unsloth #95](https://github.com/unslothai/llama.cpp/pull/95), `3db8cb5b2e9bf291057b9f19960e8601a162da81` | 2 files, +112/-13 | Penalty sampling optimization, absent from the local delta and a small candidate after mainline. Preserve fallback behavior for reordered, truncated, and duplicate-token candidate arrays. Reuse `test-sampling`. |
| [ggml-org #27754](https://github.com/ggml-org/llama.cpp/pull/27754), `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9` | 42 files, +2,558/-36 | GLM-5-Next already exists locally, but the newer pin includes additional changes, including indexer softmax grid overflow handling at its tip. Compare the delta to the local port; do not import the architecture again. Probe conflicts include K-pool code, `glm5next.cpp`, graph dispatch, and arch tests. Preserve local K-pool fixes. |
| [Unsloth #137](https://github.com/unslothai/llama.cpp/pull/137), `4e1865e34ec5f6ca39403215c89129c13731be70` | 6 files, +151/-1 | Batched mmap readahead is already present in local `48afa37d4`. Direct comparison shows the mmap helper additions are retained, alongside local file-name support. Verify the callers, then mark covered rather than replaying the pin. |
| [Unsloth #158](https://github.com/unslothai/llama.cpp/pull/158), `abfc45b9cb21eae4848cb82196e659f42c9a8341` | 1 file, +25/-2 | Rejects direct HIP integrated-GPU compute from host buffers. This fork already has a scheduler input-ring solution. Prefer preserving and qualifying that solution; importing the workaround can disable the path the ring was built to support. The separate `NO_PINNED` capability-consistency hunk can be assessed independently. |
| [Unsloth #157](https://github.com/unslothai/llama.cpp/pull/157), `6c6da89266ba7839d825c9997782af4f4d26b81b` | 3 files, +10/-1 | Changes the 2D copy fast path to `cudaMemcpyDefault`, with HIP/MUSA aliases. Not present in the inspected local or canonical code. High-priority candidate if managed memory is used; test strided copies, state snapshots, and the normal non-managed path. Keep the fast path's type coverage. |
| [Unsloth #149](https://github.com/unslothai/llama.cpp/pull/149), `b65a2dce12c14a489e19a059cb6ee59112f1b733` | 1 file, +44/-2 | Parses the value of `GGML_CUDA_ENABLE_UNIFIED_MEMORY`; currently local and canonical code test presence, so `=0` still enables it. Small candidate alongside #157. This is configuration behavior, not the memory-copy corruption fix. |
| [Unsloth #144](https://github.com/unslothai/llama.cpp/pull/144), `ca1426903fabe9af26cd10c42034cb4bbd2e0e11` | 24 files, +550/-83 | Qwen MTP is partly covered locally. Missing pieces include the explicit `model_shared`/`borrow_shared_tensor` API and shape-aware CUDA graph-cache key. Do not wholesale merge this pin over local MTP, adaptive drafting, sparse decode, and graph-arena fixes. Evaluate these pieces separately; see below. |
| [Unsloth #152](https://github.com/unslothai/llama.cpp/pull/152), `b2b5ed9ff86427a530b762a45d3fdbd453bcd4e8` | 3 files, +81/-15 | Splits backend buffers into contiguous mapping runs. Local code still uses `get_mapping_range`; this is not already implemented. Optional loader change. Its clearest stated motivation is avoiding Metal residency over large gaps; measure Windows benefit before adding it. |
| [Unsloth #176](https://github.com/unslothai/llama.cpp/pull/176), `09ce1a4d2939844e211f7b4d30a296f4c1aed9a8` | 1 file, +61/-0 | Adds projector name/registry checks to existing `test-mtmd-impl.cpp`. Useful if adding the optional vision models; it is validation, not a runtime feature. |

### MTP-specific decisions

Local Qwen conversion already contains the dedicated mixer mapping and embedding/hidden combiner. Local code also handles draft-only exports and complete-set mixer fallback. The existence of those paths is not equivalent to Unsloth's general shared-sidecar borrowing API: `include/llama.h` and the loader do not currently contain that API.

If adding borrowing, treat converter metadata, target lifetime, tied-output fallback, draft scheduler devices, automatic fitting, and skipped-layer offload accounting as one compatibility change. The #144 discussion raises these cases; verify them against the chosen pin rather than assuming every historical finding remains open or has been fixed.

The shape-aware graph key changes shared CUDA/HIP code for all models. Compare it with the local separate MTP graph arenas (`827e49380`) and input-ring UID updates. Profile actual capture churn first, then validate alternating verification shapes and bounded cache growth. A graph key must not hide changed tensor addresses.

The pinned head fixes an MTP norm reshape issue, but its discussion also contains a gfx1100/Linux hang report and differing greedy-output reports under different conditions. These are regression cases to investigate, not proof of a Windows/gfx1151 failure. Published speedups on B200 or other GPUs are not expected speedups for this fork.

### If full Unsloth feature parity is selected later

1. Freeze the base tag, ordered 13 pins, source manifest, and source artifact checksum. Reconstruct the reference mix in a separate scratch checkout using the manifest order. Compare the result to the published source asset, accounting for the workflow's build-info stamping.
2. Use that reference as an oracle for feature presence and dependencies. Do not copy its whole `src/` or `ggml/` tree over this fork.
3. Integrate onto the qualified canonical/local baseline in dependency groups: small runtime fixes; MTP/loader reconciliation; residual GLM fixes; optional new models/quantization; projector coverage. Re-probe conflicts between groups. If exact b11030 reproduction is required, keep that reference separate from the newer b11065 integration.
4. Record each pin as fully imported, already covered, intentionally superseded, or deferred. Mainline squashes and local cherry-picks need content comparison; ancestry alone is insufficient.
5. For each added feature, run its synthetic model/op checks and at least one real applicable workload. An unsupported backend or zero executed cases is a documented gap, not a pass.
6. Keep Unsloth's publishing, signing, repin-bot, notification, and capacity-management workflows out of the Windows inference integration. Their repository/account assumptions do not match this fork. Some helper/workflow files carry AGPL-3.0 SPDX headers; retain applicable notices if any are deliberately reused. Do not adopt the entire automation stack just to obtain runtime patches.

## 7. Validation gates

### Build and fast regression gate

Use the existing scripts from the integration worktree, with fresh output directories for the candidate build:

```powershell
.\test-windows.ps1 -SourceOnly
.\test-windows.ps1 -Jobs 8 -BuildDir build-mainline-b11065-hip
.\build-windows-vulkan.ps1 -Jobs 8 -BuildDir build-mainline-b11065-vulkan
```

Supply `-RocmPath` and `-VulkanSdkPath` when the installed SDKs differ from the documented defaults. `test-windows.ps1` performs the HIP rebuild through `build-windows.ps1`; a separate identical HIP build is unnecessary. Run the candidate tools' `--list-devices` with the correct runtime DLL environment before interpreting GPU failures.

The Windows runner covers source guards, executable startup, lazy-reader cases, allocation recovery, QSA metadata/visibility, K-pool calculations, and selected attention checks against CPU. It temporarily clears `HIP_DEVICE_LIB_PATH` for GPU execution. Follow that setup for manual reproductions. Neither `-SourceOnly` nor `-SkipGpu` qualifies an upstream integration, and the full suite does not exercise an entire Qwen server conversation.

Reuse existing `test-backend-ops`, `test-backend-sched-ring`, `test-alloc`, `test-save-load-state`, `test-llama-archs`, and relevant server tests. Check their actual command-line help/CTest registrations in the merged tree. Build the requested targets explicitly; the normal tool build does not establish that every test target compiled. Do not add new `tests/*` files for this integration.

### Correctness and preservation matrix

| Gate | Required cases | Acceptance evidence |
| --- | --- | --- |
| Generic inference | Small dense and MoE models; CPU reference and Vulkan; HIP candidate | Successful load, finite logits, coherent output, existing arch/state tests pass |
| Qwen loading | Full target with/without MTP; existing self-contained drafts; dedicated and fallback mixer layouts; invalid draft used as target | Valid supported layouts load; incompatible inputs fail normally; tensor names/types/shapes agree with graph consumers |
| Qwen direct attention | F16 K/V, `-fa on -ctk f16 -ctv f16 -np 1`, KV offload enabled; decode/verification widths 1, 2, 4, 8, and above 8 | Correct selected-attention dispatch for eligible cases and correct fallback beyond the gate; compare backend results to CPU/reference |
| Attention fallbacks | Q8 K/V separately; Flash Attention/offload disabled where supported; nonstandard masks and layouts | No unsupported direct sparse dispatch, abort, NaNs, or lost exclusions |
| HC and fused math | Target and MTP; HC pre/post, RMSNorm/mul; quantized matvecs and MoE reduction | Existing numerical checks pass; Windows HC16 paths remain excluded; evidence that tuned eligible paths still execute |
| Context/cache lifecycle | Actually filled 8K, 32K, 64K contexts; repeated prompt; append, rollback, restore, copy, keep, clear; incomplete blocks and gaps | Visibility, derived-state invalidation, selected indices, and state roundtrips remain correct |
| Multimodal | Screenshot conversation, then follow-up text, prompt reuse and MTP where supported | Correct position-gap masking and continuation; synthetic QSA checks plus a real server case |
| Scheduler concurrency | `n_ubatch < n_batch`; one slot and four slots; supported unified/non-unified KV modes; repeated asynchronous submissions | No cross-slot response contamination or stale input reads; input-ring and graph-address refresh checks pass |
| Lazy reads | `on` and `on-direct`; cold/warm runs; startup activation and fallback; repeated/concurrent rows | Reader tests pass and output agrees; no accidental eager PLE load or lost prefetch |
| State and recovery | Quantized KV state save/load; K-pool sizing; allocation-failure injection and subsequent request | State remains usable and server recovers without partial cached prompts |
| MTP behavior | No speculation, MTP only, ngram only, and MTP+ngram; fixed and adaptive draft sizes | Correct output/state, draft/accept counts recorded, delivered throughput measured separately |

For deterministic comparisons, fix seeds, sampling, checkpoint settings, batching, and backend. Repeated non-speculative runs establish whether the selected setup is deterministic. Start MTP comparison with `--ctx-checkpoints 0`, then test normal checkpoint behavior separately. Greedy mismatches require investigation against logits/state; do not automatically attribute every difference to draft quality or harmless rounding. Across CPU/GPU backends, use the existing numerical tolerances rather than requiring universal bit identity.

If importing optional pins, add their specific cases: penalty sampler array layouts (#95), managed/unmanaged strided copies and false-valued environment settings (#157/#149), shared-sidecar fitting and device placement (#144), mapping-run gaps and tensor containment (#152), IQ1 format IDs/roundtrips/MMQ and MMVQ/legacy quant regressions (#61), and actual new-model/projector execution (#24423/#25731/#70/#27754/#176). Check test case counts and skips explicitly. Unsloth's [feature matrix](https://github.com/unslothai/llama.cpp/blob/40a6d6d92fd4c6189928ba6f0252375573902533/scripts/unsloth/feature-checks.json) itself states that its prebuild runners do not validate GPU kernels.

### Performance gate

Record the local baseline and candidate commit/tree state, executable and DLL paths, SDK/compiler/driver, model hashes and quantization, KV types, offload settings, CPU threads, batch/ubatch, slots, occupied context, lazy mode, speculative settings, and sampling parameters. Keep these fixed in paired runs.

Measure prompt processing, time to first token, delivered generation tokens/sec, total request completion time, and peak memory separately. Run repeated interleaved baseline/candidate trials, including warm short requests and actually filled long contexts. Separate cold-file-cache observations from warm results; do not label a Windows `on-direct` run as cache-bypassing.

Start with no speculation, then isolate MTP and ngram effects. Test representative coding requests long enough to include sustained generation and cache reuse. A short greedy smoke test or a higher acceptance rate is insufficient. Investigate repeatable regressions outside the baseline's measured run-to-run spread before accepting the merge; report any accepted tradeoff explicitly. Do not overwrite the previous working build until the candidate passes.

## 8. Completion checklist and investigation limits

The implementation is ready for human finalization when:

- The chosen canonical revision is completely integrated and all deviations are explained.
- Every preservation-ledger entry is reviewed against the final source and supported by the applicable tests.
- HIP and Vulkan build; full Windows regression and representative model/cache/MTP checks pass, with unavailable coverage clearly listed.
- Performance comparisons use matched settings and show acceptable delivered performance and memory usage.
- Optional Unsloth pins have explicit dispositions; there is no claim of full mix parity if any feature remains deferred.
- No conflict markers remain; `git diff --check` passes; the final diff contains only the integration and necessary adaptations.
- Source PRs/SHAs and validation results are recorded. The human handles any commit or submission under repository policy.

This investigation verified remote URLs, heads/tags, ancestry, source-base equivalence, the release manifest and pin order, exact pin deltas, individual dry-run conflicts, relevant source contracts, PR descriptions, and selected discussion findings. CodeGraph was attempted first; its results did not cover the requested C++ areas adequately, so targeted source reads and Git diffs were used.

Local analysis artifacts remain under ignored `build-integration-investigation/`: fetched history, merge-tree reports, PR metadata/discussions, and release metadata. Git emitted automatic geometric-repack warnings during some scratch fetches; the fetched pin refs and tag revisions were subsequently resolved and used successfully for diff/merge-tree analysis. These diagnostics are local aids; the pinned references and findings above make the plan independent of retaining that directory.

Not performed: a resolved canonical merge, reconstruction or byte comparison of the full published mix source, builds, GPU tests, real-model runs, or before/after benchmarks. Consequently this document recommends the least costly integration route supported by source/history evidence; it does not certify the resulting software or claim a performance improvement.
