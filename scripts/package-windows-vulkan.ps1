#Requires -Version 5.1
param(
    [string]$BuildDir = 'build-vulkan-release'
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not [System.IO.Path]::IsPathRooted($BuildDir)) {
    $BuildDir = Join-Path $RepoRoot $BuildDir
}
$BuildDir = (Resolve-Path -LiteralPath $BuildDir).Path
if (-not $env:VCToolsRedistDir) {
    throw 'Run build-windows-vulkan.ps1 first in the same PowerShell session to initialize the Visual Studio environment.'
}
$Cache = Get-Content -LiteralPath "$BuildDir/CMakeCache.txt" -Raw
foreach ($Setting in @('CMAKE_BUILD_TYPE:[^=]+=Release', 'GGML_VULKAN:[^=]+=ON', 'GGML_HIP:[^=]+=OFF', 'GGML_CUDA:[^=]+=OFF', 'GGML_NATIVE:[^=]+=OFF')) {
    if ($Cache -notmatch "(?m)^$Setting\r?$") {
        throw "Release build setting missing: $Setting. Use build-windows-vulkan.ps1 -Portable in a fresh build directory."
    }
}

$Commit = git -C $RepoRoot rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Could not determine the source commit.' }
$Name = "llama-windows-x64-vulkan-$($Commit.Substring(0, 9))"
$OutputDir = Join-Path $BuildDir 'release'
$PackageDir = Join-Path $OutputDir $Name
$BinDir = Join-Path $PackageDir 'bin'
if (Test-Path -LiteralPath $PackageDir) {
    throw "Package already exists: $PackageDir. Use a fresh build directory."
}
New-Item -ItemType Directory -Path $BinDir | Out-Null

$Tools = @('llama-server', 'llama-cli', 'llama-bench', 'llama-fit-params', 'llama-perplexity')
foreach ($Tool in $Tools) {
    Copy-Item -LiteralPath "$BuildDir/bin/$Tool.exe" -Destination $BinDir
}
Copy-Item -Path "$BuildDir/bin/*.dll" -Destination $BinDir
foreach ($Runtime in @('CRT', 'OpenMP')) {
    Copy-Item -Path "$env:VCToolsRedistDir/x64/Microsoft.VC143.$Runtime/*.dll" -Destination $BinDir -Force
}
if (-not (Test-Path -LiteralPath "$BinDir/ggml-vulkan.dll")) {
    throw 'The Vulkan backend DLL is missing.'
}
Copy-Item -LiteralPath "$RepoRoot/LICENSE" -Destination $PackageDir
Copy-Item -LiteralPath "$RepoRoot/licenses" -Destination $PackageDir -Recurse
@(
    'Windows x64 / Vulkan / Strix Halo fork',
    "Source commit: $Commit",
    '',
    'Extract the entire archive and run the executables in bin.',
    'Visual C++ and OpenMP runtimes are included. Install a Vulkan-capable graphics driver, which supplies the Vulkan loader.',
    'The Vulkan SDK is not required to run this package.',
    'Example: bin\llama-server.exe -m C:\models\model.gguf -ngl 999',
    'Check GPU detection: bin\llama-cli.exe --list-devices',
    '',
    'CI startup checks do not validate GPU inference or model correctness.'
) | Set-Content -LiteralPath "$PackageDir/README.txt" -Encoding ascii

# Check the package without using build tools or SDK DLLs on PATH.
$SavedPath = $env:PATH
try {
    $env:PATH = "$BinDir;$env:SystemRoot/System32;$env:SystemRoot"
    foreach ($Tool in $Tools) {
        & "$BinDir/$Tool.exe" --help *> "$OutputDir/$Tool-help.log"
        if ($LASTEXITCODE -ne 0) { throw "$Tool package smoke test failed." }
    }
} finally {
    $env:PATH = $SavedPath
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$ZipPath = Join-Path $OutputDir "$Name.zip"
[System.IO.Compression.ZipFile]::CreateFromDirectory($PackageDir, $ZipPath)
$Hash = (Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash.ToLowerInvariant()
"$Hash  $Name.zip" | Set-Content -LiteralPath "$ZipPath.sha256" -Encoding ascii
Write-Host "Release ZIP: $ZipPath"
