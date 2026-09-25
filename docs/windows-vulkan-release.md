# Windows Vulkan releases

The **Release Windows Vulkan** workflow builds Windows x64 binaries and publishes a GitHub Release with a ZIP and its SHA-256 checksum. It follows the [ROCm 10 release workflow](windows-rocm10-release.md), with stable releases selected by default.

## Run a release

1. Commit and push the workflow and packaging script to this fork. The workflow must exist on the repository's default branch before GitHub shows the manual run button.
2. Open **Actions -> Release Windows Vulkan -> Run workflow**.
3. Select the branch containing the code to build.
4. Enter a new tag, such as `halo-vulkan-v1`. Existing tags are rejected.
5. Leave **Mark as a prerelease** unchecked for a stable release, or select it for a test build.

The workflow creates the tag at the selected run's exact commit after the build and package smoke tests succeed. It uses the built-in `GITHUB_TOKEN` with `contents: write`; no personal token is needed. Repository or organization policies must permit that access.

## ZIP contents and requirements

- `bin/llama-server.exe`, `llama-cli.exe`, `llama-bench.exe`, `llama-fit-params.exe`, and `llama-perplexity.exe`.
- llama.cpp and Vulkan backend DLLs, plus Visual C++ and OpenMP runtime DLLs.
- Project licenses and a README recording the source commit.

Extract the entire archive and run the tools from `bin`. Install a Vulkan-capable graphics driver, which provides the system Vulkan loader (`vulkan-1.dll`). Neither the Vulkan SDK nor ROCm is required on the user's machine. Run `bin\llama-cli.exe --list-devices` to verify GPU detection.

The build uses `build-windows-vulkan.ps1 -Portable`, which disables native CPU tuning for the GitHub runner and requests static OpenSSL libraries when available. The packaging script rejects non-portable or non-Release builds. Local native builds remain unchanged.

The workflow pins LunarG Vulkan SDK 1.4.350.0 and verifies the installer SHA-256. The SDK installation also installs the loader needed for startup checks on the hosted runner. The package smoke tests run all five tools with `--help` and a PATH limited to the package and Windows system directories. These checks and the Windows source guards do not validate GPU inference; test model correctness and long-context behavior on Strix Halo before choosing code for a stable release.

## Package locally

Run both commands in the same PowerShell session:

```powershell
.\build-windows-vulkan.ps1 -BuildDir build-vulkan-release -Jobs 8 -Portable
.\scripts\package-windows-vulkan.ps1 -BuildDir build-vulkan-release
```

The ZIP and checksum are written under `build-vulkan-release/release`. Use a fresh build directory when changing toolchains. If MSVC hits a path-length limit in the nested shader build, use a shorter `-BuildDir`, such as `build-vk-rel`, for both commands. The packager refuses to overwrite an existing package directory.

References: [LunarG SDK downloads and checksums](https://vulkan.lunarg.com/sdk/home), [LunarG Windows installation instructions](https://vulkan.lunarg.com/doc/view/latest/windows/getting_started.html).
