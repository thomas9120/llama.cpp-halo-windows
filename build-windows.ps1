#Requires -Version 5.1

<#
.SYNOPSIS
Build the Windows HIP server, CLI, and benchmark tools for gfx1151.
.EXAMPLE
.\build-windows.ps1
.EXAMPLE
.\build-windows.ps1 -RocmPath D:\TheRock\build -Jobs 8
#>
param(
    [string]$RocmPath = 'C:\TheRock\build',
    [string]$BuildDir = 'build-rocm10-gfx1151',
    [ValidateRange(1, 256)]
    [int]$Jobs = 12,
    [switch]$ConfigureOnly
)

$ErrorActionPreference = 'Stop'
$RocmPath = (Resolve-Path -LiteralPath $RocmPath).Path
if (-not [System.IO.Path]::IsPathRooted($BuildDir)) {
    $BuildDir = Join-Path $PSScriptRoot $BuildDir
}
$BuildDir = [System.IO.Path]::GetFullPath($BuildDir)
$Clang = Join-Path $RocmPath 'lib\llvm\bin\clang.exe'
$ClangCpp = Join-Path $RocmPath 'lib\llvm\bin\clang++.exe'
$DeviceLibDir = Join-Path $RocmPath 'lib\llvm\amdgcn\bitcode'
$VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

foreach ($RequiredFile in @($Clang, $ClangCpp, "$DeviceLibDir\ocml.bc", $VsWhere)) {
    if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
        throw "Required build dependency not found: $RequiredFile"
    }
}

$VsPath = & $VsWhere -latest -version '[17.0,18.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if ($LASTEXITCODE -ne 0 -or -not $VsPath) {
    throw 'Visual Studio 2022 with the x64 C++ build tools is required.'
}
& "$VsPath\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation

$env:HIP_PATH = $RocmPath
$env:ROCM_PATH = $RocmPath
$env:HIP_DEVICE_LIB_PATH = $DeviceLibDir
$env:PATH = "$RocmPath\bin;$RocmPath\lib\llvm\bin;$env:PATH"
$env:CCACHE_DIR = Join-Path $BuildDir 'ccache'
Get-Command cmake, ninja -ErrorAction Stop | Out-Null

Write-Host "ROCm SDK: $RocmPath"
Write-Host "Build directory: $BuildDir"
Write-Host "Target: gfx1151, Release, $Jobs parallel jobs"

cmake -S $PSScriptRoot -B $BuildDir -G Ninja -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_C_COMPILER=$Clang" "-DCMAKE_CXX_COMPILER=$ClangCpp" `
    "-DCMAKE_PREFIX_PATH=$RocmPath" -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 `
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
if ($LASTEXITCODE -ne 0) {
    throw "CMake configuration failed with exit code $LASTEXITCODE. Use a new -BuildDir when changing toolchains."
}
if ($ConfigureOnly) {
    return
}

cmake --build $BuildDir --target llama-server llama-cli llama-bench --parallel $Jobs
if ($LASTEXITCODE -ne 0) {
    throw "Build failed with exit code $LASTEXITCODE."
}

$BinDir = Join-Path $BuildDir 'bin'
$OpenMpRuntime = Join-Path $env:VCToolsRedistDir 'debug_nonredist\x64\Microsoft.VC143.OpenMP.LLVM\libomp140.x86_64.dll'
if (Test-Path -LiteralPath $OpenMpRuntime) {
    Copy-Item -LiteralPath $OpenMpRuntime -Destination $BinDir -Force
} else {
    Write-Warning 'OpenMP runtime not found. If needed, put libomp140.x86_64.dll beside the executables.'
}

Write-Host "Build complete: $BinDir"
Write-Host "Keep $RocmPath\bin on PATH when running the executables from a new shell."
