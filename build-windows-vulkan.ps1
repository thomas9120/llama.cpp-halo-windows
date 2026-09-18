#Requires -Version 5.1

<#
.SYNOPSIS
Build the Windows Vulkan server, CLI, benchmark, parameter-fitting, and perplexity tools.
.EXAMPLE
.\build-windows-vulkan.ps1
.EXAMPLE
.\build-windows-vulkan.ps1 -VulkanSdkPath D:\VulkanSDK\1.4.350.0 -Jobs 8
#>
param(
    [string]$VulkanSdkPath = $env:VULKAN_SDK,
    [string]$BuildDir = 'build-vulkan',
    [ValidateRange(1, 256)]
    [int]$Jobs = 12,
    [switch]$Portable,
    [switch]$ConfigureOnly
)

$ErrorActionPreference = 'Stop'
if (-not $VulkanSdkPath) {
    throw 'Install the LunarG Vulkan SDK and open a new PowerShell, or specify -VulkanSdkPath.'
}
$VulkanSdkPath = (Resolve-Path -LiteralPath $VulkanSdkPath).Path
if (-not [System.IO.Path]::IsPathRooted($BuildDir)) {
    $BuildDir = Join-Path $PSScriptRoot $BuildDir
}
$BuildDir = [System.IO.Path]::GetFullPath($BuildDir)
$VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

foreach ($RequiredFile in @("$VulkanSdkPath\Bin\glslc.exe", "$VulkanSdkPath\Lib\vulkan-1.lib", "$VulkanSdkPath\Include\vulkan\vulkan.h", "$VulkanSdkPath\Include\spirv\unified1\spirv.hpp", $VsWhere)) {
    if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
        throw "Required build dependency not found: $RequiredFile"
    }
}

$VsPath = & $VsWhere -latest -version '[17.0,18.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if ($LASTEXITCODE -ne 0 -or -not $VsPath) {
    throw 'Visual Studio 2022 with the x64 C++ build tools is required.'
}
& "$VsPath\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation

$env:VULKAN_SDK = $VulkanSdkPath
$env:PATH = "$VulkanSdkPath\Bin;$env:PATH"
$env:CCACHE_DIR = Join-Path $BuildDir 'ccache'
Get-Command cmake, ninja, cl -ErrorAction Stop | Out-Null

Write-Host "Vulkan SDK: $VulkanSdkPath"
Write-Host "Build directory: $BuildDir"
Write-Host "Target: x64 Vulkan, Release, $Jobs parallel jobs"

$NativeCpu = if ($Portable) { 'OFF' } else { 'ON' }
$StaticOpenSsl = if ($Portable) { 'ON' } else { 'OFF' }
cmake -S $PSScriptRoot -B $BuildDir -G Ninja -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl `
    "-DCMAKE_PREFIX_PATH=$VulkanSdkPath" -DGGML_VULKAN=ON -DGGML_HIP=OFF -DGGML_CUDA=OFF `
    "-DGGML_NATIVE=$NativeCpu" "-DOPENSSL_USE_STATIC_LIBS=$StaticOpenSsl" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
if ($LASTEXITCODE -ne 0) {
    throw "CMake configuration failed with exit code $LASTEXITCODE. Use a new -BuildDir when changing toolchains."
}
if ($ConfigureOnly) {
    return
}

cmake --build $BuildDir --target llama-server llama-cli llama-bench llama-fit-params llama-perplexity --parallel $Jobs
if ($LASTEXITCODE -ne 0) {
    throw "Build failed with exit code $LASTEXITCODE."
}

$BinDir = Join-Path $BuildDir 'bin'
Write-Host "Build complete: $BinDir"
Write-Host 'Run llama-cli.exe --list-devices to check Vulkan GPU detection.'
