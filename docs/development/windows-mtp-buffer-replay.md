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

### Direct pinned-host execution failure (2026-09-25)

A different failure on base `1db3d4d29` occurred during generation after screenshot processing, without a reported allocation failure. The actual captured conversation reproduced `hipEventSynchronize: unspecified launch failure` near 92K filled context. Disabling HIP graph capture did not prevent it. Serializing GPU work and checking each node localized a repeatable failure to the embedding `GET_ROWS` operation, with valid row indices and a successful synchronization immediately before the gather. The Q8_0 embedding table and token indices were both in `ROCm_Host` buffers; the output was in a device buffer.

A standalone gather over every row of the same table dimensions passed in both host and device buffers. Moving only the embedding table to a device buffer allowed two requests to complete but failed on the next screenshot. Disabling pinned allocation entirely failed during model warmup. These experiments do not establish a bad quantization, an invalid token index, or a general inability to use pinned memory.

The mitigation makes the Windows HIP backend decline direct execution against pinned host buffers on RDNA3.5. The existing scheduler then uses compatible buffers or CPU execution. Pinned allocations remain available for transfers. The change is confined to buffer compatibility; it does not alter PLE reads, MTP acceptance, graph capture, or allocation recovery. CUDA, Linux HIP, Vulkan, and other GPU architecture families retain their existing behavior.

With this compatibility restriction enabled experimentally, the unchanged captured request completed, followed by two controlled screenshot continuations with prompt-cache reuse. At normal execution speed, the original prompt contained 91,769 tokens and generated 201 tokens at 20.05 tokens/s. The continuations processed 1,527 and 1,308 new prompt tokens and generated 512 tokens each at 16.11 and 15.83 tokens/s, reaching 95,622 cached tokens. Graph capture was enabled and diagnostic synchronization was disabled. Continuations reused a captured screenshot; generated tool calls were saved but never executed.

The model, draft, projector, lazy/DIO settings, offload, batching, checkpoint count, and sampling matched the supplied command: context 262,144; batch 4,096; microbatch 2,048; 16 threads; one unified slot; F16 K/V; Flash Attention; `draft-mtp,ngram-mod` with draft max 7 and minimum probability 0.75; adaptive drafting off; temperature 1, top-k 20, top-p 0.95, min-p 0; xhigh reasoning. The build used TheRock ROCm 10, Clang 23 and VS 2022/MSVC 14.44 on Radeon 8060S (`gfx1151`), with installed HIP runtime `10.0.3679.0`.

This is a bounded mitigation for the reproduced workload, not proof of the underlying driver failure mechanism or stability at 179K-262K filled context. Extra device buffers can increase memory use. During the normal-speed prompt replay, system commit reached about 116.1 GiB against a 127.6 GiB limit. The token rates are observations from sampled completions, not a controlled performance comparison.

Temporary request capture, per-node checks, row validation, serialization controls, and the experimental environment switch were removed from the production source. Local logs, launch metadata, private captured requests, replay scripts, and the saved diagnostic diff remain in the ignored `build-mtp-buffer-fix/screenshot-crash-20260925` directory.

The cleaned production build passed the full `test-windows.ps1 -Jobs 8` suite, including GPU checks. A fresh replay without diagnostic flags completed the captured 91,769-token prompt and generated 176 tokens at 23.74 tokens/s. Two screenshot continuations reused the cache, evaluated 1,502 and 1,307 new prompt tokens, and generated 512 tokens each at 13.72 and 18.38 tokens/s. Final cache occupancy was 95,596 tokens. All three streams ended normally; the final visible answer matched the screenshot's scene, controls, and voxel counts. The first capped continuation spent its token budget in reasoning, so it supplies execution coverage but no visible answer to assess. No allocation or GPU failure was reported. The test server was stopped while idle.

Production `ggml-hip.dll` SHA-256: `f0778135f1bf352a12c560927b3cc1dccf01e6c74c5893017115be21bf11b9ac`. The complete tested application DLL/executable set was installed into the existing local launcher directory after backing up and hashing its previous files. Installed hashes match the tested build. The runtime SDK and driver were retained. This validation does not cover adaptive drafting, a different quantization, or the longer endurance run.

### Comparison with Unsloth b11139 (2026-09-25)

The user reported a decode slowdown after the workaround and successful screenshot use with Unsloth. The installed Unsloth build identifies as `b11139-d79953103`. Its [release manifest and exact source archive](https://github.com/unslothai/llama.cpp/releases/tag/b11139-mix-a6922cc) identify source commit `d799531035c892f42c9ada5e9b0403060213b266` and include [Unsloth PR #158](https://github.com/unslothai/llama.cpp/pull/158), commit `abfc45b9cb21eae4848cb82196e659f42c9a8341`. Although that PR is open, it is pinned into the released mix. Its buffer-compatibility restriction has the same effect as this fork's safeguard on Windows gfx1151: pinned staging remains enabled and direct GPU compute on `ROCm_Host` is rejected. Unsloth's success therefore does not show that unrestricted host compute is safe.

The upstream discussion identifies scheduler races in direct host-buffer execution. This strengthens the case for a llama.cpp scheduling problem rather than establishing a general ROCm defect. This fork already contains UMA input-ring and cross-backend lifetime work related to [draft PR #27311](https://github.com/ggml-org/llama.cpp/pull/27311); importing the entire PR as a missing fix is not justified. The exact cause of the remaining screenshot failure here is still unproven.

The running Unsloth server used the same launch arguments, but loaded bundled HIP runtime `10.0.3581.0`, hipBLAS `3.7.0.0`, rocBLAS `5.7.0.0`, and hipBLASLt `1.4.1.0`. The local comparison loaded system HIP `10.0.3679.0` and TheRock hipBLAS `3.6.0.0`, rocBLAS `5.6.0.0`, and hipBLASLt `1.4.1.0`. No runtime libraries were swapped. This is a comparison of delivered builds, not an isolated source or runtime benchmark.

An identical 95,085-token screenshot history was submitted to both builds with seed 4242, the original sampling settings, a 512-token cap, and `ignore_eos=true`. Unsloth decoded at 12.91 tokens/s; the guarded local build decoded at 18.25 tokens/s and 17.78 tokens/s on a cached repeat. All requests completed without error, with MTP active. Unsloth reused 91,765 prompt tokens, the first local request processed the entire prompt, and the local repeat reused 95,081 tokens. Prompt-processing times are therefore not comparable. Despite the same seed, sampled output differed across builds, so this is a bounded workload comparison rather than identical-token timing. It does not quantify the user's reported loss against this fork's pre-workaround build or guarantee the same relative performance on other prompts. No additional production code or binary changes were made during this comparison. Artifacts are in `build-mtp-buffer-fix/screenshot-crash-20260925/unsloth-comparison`.

### Halo scratch-pool experiment (2026-09-25)

Tested an RDNA3.5-only adaptation of [Halo PR #78](https://github.com/halo-box/strix-llama.cpp/pull/78), head `2167f5f4b46b660fdec6fbe0f60207408f6fd2d8`, with the direct-host safeguard retained. The candidate rounded growing scratch allocations to powers of two. Unlike the source patch, it retried the original allocation size before the existing OOM pool flush. Baseline binaries were copied before rebuilding; no runtime DLLs were mixed and no launcher binaries were replaced.

The model remained AtomicChat AD-4.27bpw-Q4_K_M-M64 with the same shared MTP draft and projector. System commit before launch was approximately 27-28 GiB. A watchdog sampled memory every two seconds and stopped only its child server when commit headroom fell below 3 GiB or available RAM below 2 GiB. Both builds reached the commit guard while replaying the 14-image, 95,085-token history from an empty prompt cache at context 262,144, with microbatch 2,048 and again with 1,024. A baseline trial at context 131,072 and microbatch 2,048 also reached the guard. These deliberate stops are not HIP crashes or successful endurance tests. The first baseline overlapped compilation and is unsuitable for timing comparison.

For a comparison that reached decode, a fixed variant retained the text history and last screenshot while replacing the other 13 images with omission markers. Both builds used the original context 262,144 and microbatch 2,048, processed 78,808 prompt tokens, and generated 512 tokens per request:

| Request | Baseline tokens/s | PR #78 candidate tokens/s |
| --- | ---: | ---: |
| Empty prompt cache | 17.18 | 17.15 |
| Cached repeat | 15.75 | 15.81 |

Normalized content, reasoning, and tool-call deltas matched between builds for each request, as did draft counts. Peak process-private commit was approximately 79.46 GiB in both initial stages. A fixed extension added about 61.6K tokens of inert JavaScript and another screenshot; both arms reached the commit guard before completing it. Different background system commit affected where they stopped, so a later stop does not establish a memory saving.

No useful improvement was demonstrated for this workload. The candidate was saved under the ignored `build-mtp-buffer-fix/host-performance-20260925` directory and reverted. The restored source passed the full `test-windows.ps1 -Jobs 8` suite, including GPU checks. This experiment does not disprove the PR's Linux results or quantify the original slowdown caused by restricting direct host-buffer execution.

The [PR #87](https://github.com/halo-box/strix-llama.cpp/pull/87) review found that its highlighted Q5_1 expert optimization does not match this AtomicChat model's expert weights; its only Q5_1 tensor is the lazy PLE table. This fork already accepts F32 PLE convolution weights. The scheduler split-cap difference also reflects the intentional canonical change [#28387](https://github.com/ggml-org/llama.cpp/pull/28387), not simply a missing fix. Those changes were not imported.
