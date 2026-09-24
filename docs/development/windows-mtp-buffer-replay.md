# Windows MTP buffer isolation and checkpoint replay

## Source revisions

This port starts from Windows fork revision `81a114322` and incorporates:

- [strix-llama.cpp PR #83](https://github.com/halo-box/strix-llama.cpp/pull/83), open when inspected, head `1a005ba1c9bdcf53ed92e8a2914344667ce4b717` against base `8c1c282ecb194e8f02613defcc4a07c22b6d1c08`.
- [strix-llama.cpp PR #34](https://github.com/halo-box/strix-llama.cpp/pull/34), merged as `99a40a3e6f210ea0ebea39571e29f7635ecc4323`, head `f8668dd354c829a92e0e4d58a6c4eaec64ec8ddd` against base `7449a0fe9710ab584c5f9a6d25e7a31eea2708b8`.

## Behavior

MMB activation scratch, producer slots, BF16 marks, converted weights, and graph-sequence tracking belong to their backend context. Scratch and temporary conversions use the active stream's pool. Target and draft contexts can no longer overwrite or release each other's MMB storage. Context destruction waits for its streams, destroys captured graphs, and then releases its MMB allocations before destroying the pools. The normal inference path adds no target/draft synchronization.

After a speculative checkpoint restore, replay accepts the already verified tokens into the restored sampler and samples only the continuation. It does not compare the replay tokens against logits that may change with batch shape. This prevents another rejection from restoring the same checkpoint repeatedly. Ordinary verification and synthetic acceptance keep their existing paths.

## Windows adaptations

- Keep the Windows HC16 exclusions in MMB, HC mixing, and graph optimization.
- Apply the upstream `hyperconn.cu` cache lookup changes to this fork's `hc-mix.cu`.
- Retain the explicit token-count argument to `ggml_cuda_mmb_cache_reserve`.
- Keep this fork's graph optimizer and cache handling. Upstream graph fingerprinting is absent here and is not introduced by this port; the existing first-split and after-compute tracking becomes context-local.
- Retain server allocation-stage diagnostics around speculative sampling and restoration.
- Adapt `MMB_CONTEXT` to the existing `HC_F32_CONSUMER` graph with HC post, Q4_0 projection, and 512 tokens. The upstream `HC_CHAIN` fixture is not present here.

## Validation

The source guards pass, including Windows HC16 exclusions. A temporary CPU harness compiles the actual server sampling block with controlled sampler responses. It passes on the fixed source and fails on the unchanged source when replay logits reject the first previously accepted token. It also checks sampler history, grammar advancement, continuation batch index, zero additional rollback, ordinary verification, synthetic acceptance, and an empty replay draft. This is a branch-level test, not a model-level generation test.

The standard `test-windows.ps1` GPU scope now runs `test-backend-ops test -b ROCm0 -o MMB_CONTEXT`. The test compares two asynchronously submitted contexts against their own serialized outputs and checks target reuse after destroying the draft context that allocated scratch first. Inputs remain allocated across graph replays. Also run it with `GGML_CUDA_DISABLE_GRAPHS=1` to check the non-captured path.

### Local results (2026-09-24)

- Windows ROCm Release build: passed, TheRock ROCm 10, Clang 23, VS 2022 tools, `gfx1151`, HIP graphs enabled. Server, CLI, benchmark, fit-params, and perplexity built in `build-rocm10-mtp-fix`.
- Windows Vulkan Release build: passed with LunarG 1.4.350.0 and VS 2022/MSVC 14.44. The long worktree build path exceeded the shader generator's Windows compiler path limit; an independent shorter `build-mtp-vk` directory resolved it. Vulkan server `--help` passed.
- `test-windows.ps1 -BuildDir build-rocm10-mtp-fix -Jobs 8 -SkipGpu`: passed all source guards, executable startup checks, lazy-reader tests, server allocation recovery, QSA metadata/CPU visibility, and K-pool checks.
- The adapted `test-backend-ops` target compiled with both ROCm/Clang and Vulkan/MSVC.
- Full `test-windows.ps1 -BuildDir build-rocm10-mtp-fix -Jobs 8`: passed, including QSA GPU visibility, F16/Q8 attention, and `MMB_CONTEXT` on Radeon 8060S (`gfx1151`).
- `MMB_CONTEXT` passed with captured graphs enabled and with `GGML_CUDA_DISABLE_GRAPHS=1`.
- The unchanged Windows backend also passed this fixture in both modes. A wider projection and original-owner-first teardown did not reproduce corruption. A HIP trace confirmed HC combine/normalize and MMB dense kernels ran. These checks provide execution coverage but do not establish a regression oracle for the upstream race on this Windows runtime.
- The baseline used the complete DLL set from the existing `build-rocm10-gfx1151` build and the same test executable in a separate directory. Its `ggml-hip.dll` SHA-256 is `d56c39e1b75e441b69e824043b3fb868103b016c3c65f520a6e0e87f7f849435`. The tested source areas have no diff between its reported `fbf3d2fcd` revision and the port base.

### Qwen model validation

The patched server completed six text requests: filled contexts of 8,192, 32,768, and 65,536 tokens, each followed by a changed suffix with prompt reuse. Each generated 128 tokens. MTP accepted draft tokens on every request, and the reused requests evaluated only 2,051 new tokens after checkpoint-based prefix reuse. Two subsequent image turns also completed, generating 95 and 161 tokens. The second reused 9,420 cached tokens. The model correctly described red/blue and green/yellow test images and compared their colors.

The text requests produced no non-consecutive-position warnings. The image turns produced these warnings and still completed normally. This supports treating the warning alone as insufficient evidence of the original crash; it does not make every occurrence harmless.

Configuration:

- Target: `Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf`; draft: `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`; projector: `mmproj-Qwen3.8-Flash-Next-F16.gguf`.
- Context allocation 98,304; batch 4,096; microbatch 2,048; 16 CPU threads; 99 GPU layers; GPU 0; F16 K/V; Flash Attention and KV offload; one slot with continuous batching and unified KV.
- Lazy mode on, DIO load mode, no repack, operation/projector offload, warmup, prompt caching, 16 context checkpoints, cache RAM 0. As with the supplied command, cache RAM 0 causes idle-slot caching to be disabled by the server.
- `draft-mtp,ngram-mod`; adaptive drafting enabled; draft max 7, probability minimum 0.75; ngram match 24, minimum 48, maximum 64.
- Temperature 1, top-k 20, top-p 0.95, min-p 0, repeat-last-n 64, request seed 4242. Template default reasoning effort was used; the supplied explicit xhigh override was not part of this smoke test.
- Installed HIP runtime `10.0.3679.0` loaded from Windows System32; hipBLAS, rocBLAS, and hipBLASLt loaded from `C:/TheRock/build/bin`. Actual module paths/versions are saved in the ignored build artifacts.

Requests used generated JavaScript and two synthetic color images. These are bounded smoke tests totaling 1,024 generated tokens, not a replay of the original coding task or a 179K-context endurance test. They were run after the user reported an unpatched run past 179K context with multiple screenshots and no failure. That result does not isolate the cause of the earlier crash.

Artifacts are under `build-rocm10-mtp-fix/windows-regression` and `build-rocm10-mtp-fix/model-validation`, including launch arguments, loaded runtime modules, responses, timings, and server logs. The test server was stopped after validation. Other open PRs remain outside this port.

### Subsequent allocation failure (2026-09-24)

A later run with adaptive drafting disabled reached about 189K cached tokens in a 262,144-token context, then failed while replaying an image chunk. The log records a failed 2,923,366,272-byte (2.72 GiB) `ROCm_Host` allocation, including failure of the CPU allocation fallback. The server returned an error and crashed during the next request. This is an allocation failure below the context limit; the log does not establish whether system commit, available RAM, or another allocation constraint caused it.

Windows Event 1000 records exception `0xc0000005` in `ggml-base.dll` at offset `0x1d1d0`. Symbolization and disassembly resolve this to the virtual-buffer dereference in `ggml_gallocr_alloc_graph`. The deployed DLL matches the preceding patched build (SHA-256 `531ab477c01a26a0b970a23cdc1c6acf61326449629f47e345ef8b9b7751ba4f`).

The allocator records the new graph's sizes before reserving its backing buffers. If reservation fails, a retry with the same graph can incorrectly reuse those recorded sizes and dereference a missing buffer. Shared buffer aliases can also retain freed pointers when reservation exits early. The correction checks for missing buffers before graph reuse and updates every alias immediately when its backing buffer changes. This does not reduce the memory required by the graph.

The existing `test-alloc` now forces allocation failures without memory pressure. Against the exact deployed DLL, the repeated-failure test exits with `0xc0000005` after a forced 16-byte allocation failure. The fixed allocator must reject repeated failures, recover when allocation becomes available, and support both destruction and retry after failure at either of two distinct buffer types with a shared alias. The Windows regression runner includes this test even with `-SkipGpu`.

[Canonical issue #23422](https://github.com/ggml-org/llama.cpp/issues/23422) reports the same missing-buffer dereference after a failed reservation, with a different initial CLIP warmup failure. [Halo PR #37](https://github.com/halo-box/strix-llama.cpp/pull/37) addresses persistent view initialization after allocation splits; it is a different failure path and is not part of this correction.

Validation of this correction on base `784d4741c` with local changes:

- ROCm/Clang Release and Vulkan/MSVC Release builds passed in `build-rocm10-gfx1151` and `build-vulkan`. Both detected the Radeon 8060S, and all 18 allocator tests passed with each build, including explicit checks that destruction releases all mock buffers.
- Full `test-windows.ps1 -Jobs 8` passed, including allocator recovery, server recovery, lazy reads, QSA CPU/GPU visibility, K-pool, MMB context overlap, and F16/Q8 attention.
- Fixed ROCm `ggml-base.dll` SHA-256: `14e78f290cba02b81e6a9584f7f41b5149b56a80aca493241b31c5b3108e4c90`.
- Crash-event details, baseline reproduction, Vulkan test output, and build logs are in the ignored `build-mtp-buffer-fix/allocator-*` artifacts. ROCm runtime logs are under `build-rocm10-gfx1151/windows-regression`.

These checks reproduce and correct the allocator failure mechanism. They do not reproduce the original system memory pressure or repeat the 189K multimodal run. The allocation-pressure trigger and model-level continuation after such a failure remain unverified.
