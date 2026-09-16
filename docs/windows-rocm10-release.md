# Windows ROCm 10 releases

The **Release Windows ROCm 10** workflow builds Windows x64 binaries for Strix Halo (`gfx1151`) using AMD's ROCm 10.0.0 SDK. It publishes a GitHub Release containing a ZIP and its SHA-256 checksum.

## Run a release

1. Commit and push the workflow, packaging script, and build script changes to this fork. The workflow must exist on the repository's default branch before GitHub shows the manual run button.
2. Open **Actions -> Release Windows ROCm 10 -> Run workflow**.
3. Select the branch containing the code to build.
4. Enter a new tag, such as `halo-rocm10-v1`. Existing tags are rejected so a release cannot accidentally point to different code.
5. Optionally select **Mark as a prerelease**, then run the workflow.

The workflow creates the tag at the selected run's exact commit and publishes the release after the build and package smoke tests succeed. It uses the built-in `GITHUB_TOKEN` with `contents: write`; no personal token is needed. Repository or organization policies must permit that access.

## ZIP contents

- `bin/llama-server.exe`, `llama-cli.exe`, `llama-bench.exe`, and `llama-fit-params.exe`.
- llama.cpp DLLs, Visual C++ runtime DLLs, and the required ROCm runtime libraries.
- rocBLAS/hipBLASLt kernel data, `.kpack` data, licenses, and a README recording the source commit.

Extract the entire archive and run the tools from `bin`. Keep `.kpack` beside `bin`. A compatible AMD graphics driver is required; the ROCm SDK does not need to be installed separately.

The build uses `build-windows.ps1 -Portable` to disable CPU tuning for the GitHub runner and request static OpenSSL libraries when available. Ordinary local builds retain native CPU tuning. GitHub's Windows runner has no AMD GPU, so the workflow runs source guards and packaged executable `--help` checks; GPU correctness and performance still need testing on Strix Halo.

The SDK download is pinned to 10.0.0 and checked against the SHA-256 of the archive used for the local build. Changing SDK versions requires checking the packaging layout and updating the runtime library list and checksum.

References: [AMD ROCm installation](https://rocm.docs.amd.com/en/latest/install/rocm.html), [manual GitHub Actions runs](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow).
