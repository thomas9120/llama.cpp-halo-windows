#Requires -Version 5.1
param(
    [string]$RocmPath = 'C:\TheRock\build',
    [string]$BuildDir = 'build-rocm10-release'
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$RocmPath = (Resolve-Path -LiteralPath $RocmPath).Path
if (-not [System.IO.Path]::IsPathRooted($BuildDir)) {
    $BuildDir = Join-Path $RepoRoot $BuildDir
}
$BuildDir = (Resolve-Path -LiteralPath $BuildDir).Path
if ((Get-Content -LiteralPath "$RocmPath/.info/version" -Raw).Trim() -ne '10.0.0') {
    throw 'This package layout requires ROCm 10.0.0.'
}
if (-not $env:VCToolsRedistDir) {
    throw 'Run build-windows.ps1 first in the same PowerShell session to initialize the Visual Studio environment.'
}

$Commit = git -C $RepoRoot rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Could not determine the source commit.' }
$Name = "llama-windows-x64-rocm-10.0.0-gfx1151-$($Commit.Substring(0, 9))"
$OutputDir = Join-Path $BuildDir 'release'
$PackageDir = Join-Path $OutputDir $Name
$BinDir = Join-Path $PackageDir 'bin'
if (Test-Path -LiteralPath $PackageDir) {
    throw "Package already exists: $PackageDir. Use a fresh build directory."
}
New-Item -ItemType Directory -Path $BinDir | Out-Null

foreach ($Tool in @('llama-server', 'llama-cli', 'llama-bench', 'llama-fit-params')) {
    Copy-Item -LiteralPath "$BuildDir/bin/$Tool.exe" -Destination $BinDir
}
Copy-Item -Path "$BuildDir/bin/*.dll" -Destination $BinDir
Copy-Item -Path "$env:VCToolsRedistDir/x64/Microsoft.VC143.CRT/*.dll" -Destination $BinDir -Force
foreach ($Dll in @(
    'amdhip64_7.dll', 'amd_comgr.dll', 'rocm_kpack.dll', 'hipblas.dll',
    'rocblas.dll', 'rocsolver.dll', 'libhipblaslt.dll', 'origami.dll',
    'hiprtc0715.dll', 'hiprtc-builtins0715.dll'
)) {
    Copy-Item -LiteralPath "$RocmPath/bin/$Dll" -Destination $BinDir
}
foreach ($Library in @('rocblas', 'hipblaslt')) {
    Copy-Item -LiteralPath "$RocmPath/bin/$Library" -Destination $BinDir -Recurse
}
New-Item -ItemType Directory -Path "$PackageDir/.kpack" | Out-Null
Copy-Item -LiteralPath "$RocmPath/.kpack/blas_lib_gfx1151.kpack" -Destination "$PackageDir/.kpack"
Copy-Item -LiteralPath "$RocmPath/share/doc" -Destination "$PackageDir/rocm-licenses" -Recurse
Copy-Item -LiteralPath "$RepoRoot/LICENSE" -Destination $PackageDir
Copy-Item -LiteralPath "$RepoRoot/licenses" -Destination $PackageDir -Recurse
@(
    'Windows x64 / Strix Halo gfx1151 / ROCm 10.0.0',
    "Source commit: $Commit",
    '',
    'Run the executables in bin. Keep bin and .kpack together in this layout.',
    'ROCm and Visual C++ runtime libraries are included. Install a compatible AMD graphics driver.',
    'Example: bin\llama-server.exe -m C:\models\model.gguf -ngl 999',
    '',
    'This package targets gfx1151. Other GPU architectures are not supported by this build.'
) | Set-Content -LiteralPath "$PackageDir/README.txt" -Encoding ascii

# Check the package without using DLLs from the SDK on PATH.
$SavedPath = $env:PATH
$SavedDeviceLibPath = $env:HIP_DEVICE_LIB_PATH
try {
    $env:PATH = "$BinDir;$env:SystemRoot/System32;$env:SystemRoot"
    $env:HIP_DEVICE_LIB_PATH = $null
    foreach ($Tool in @('llama-server', 'llama-cli', 'llama-bench', 'llama-fit-params')) {
        & "$BinDir/$Tool.exe" --help *> "$OutputDir/$Tool-help.log"
        if ($LASTEXITCODE -ne 0) { throw "$Tool package smoke test failed." }
    }
} finally {
    $env:PATH = $SavedPath
    $env:HIP_DEVICE_LIB_PATH = $SavedDeviceLibPath
}

# ZipFile includes the hidden .kpack directory, which Compress-Archive can omit.
Add-Type -AssemblyName System.IO.Compression.FileSystem
$ZipPath = Join-Path $OutputDir "$Name.zip"
[System.IO.Compression.ZipFile]::CreateFromDirectory($PackageDir, $ZipPath)
$Hash = (Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash.ToLowerInvariant()
"$Hash  $Name.zip" | Set-Content -LiteralPath "$ZipPath.sha256" -Encoding ascii
Write-Host "Release ZIP: $ZipPath"
