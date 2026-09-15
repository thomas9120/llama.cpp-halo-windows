#Requires -Version 5.1

<#
.SYNOPSIS
Check this fork's Windows fixes after an upstream sync.
.EXAMPLE
.\test-windows.ps1
.EXAMPLE
.\test-windows.ps1 -SourceOnly
.EXAMPLE
.\test-windows.ps1 -SkipGpu
#>
param(
    [string]$RocmPath = 'C:\TheRock\build',
    [string]$BuildDir = 'build-rocm10-gfx1151',
    [ValidateRange(1, 256)]
    [int]$Jobs = 12,
    [switch]$SourceOnly,
    [switch]$SkipGpu
)

$ErrorActionPreference = 'Stop'
$RepoRoot = $PSScriptRoot

function Assert-Source([string]$Path, [string]$Pattern, [string]$Reason) {
    $Source = Get-Content -LiteralPath (Join-Path $RepoRoot $Path) -Raw
    if ($Source -notmatch $Pattern) {
        throw "Source guard failed: $Reason ($Path). Review the upstream change before updating this check."
    }
    Write-Host "PASS: $Reason"
}

# These guards intentionally require review when upstream rewrites the affected code.
Assert-Source 'ggml/src/ggml-cuda/mmb.cu' '#ifdef\s+_WIN32\s+bool\s+mmb_hc16\(\)\s*\{\s*return\s+false;\s*\}\s*#else' 'Windows MMB HC16 remains disabled'
Assert-Source 'ggml/src/ggml-cuda/hc-mix.cu' '#ifdef\s+_WIN32\s+static\s+const\s+bool\s+hc16\s*=\s*false;\s*#else' 'Windows HC mixing uses the safe path'
Assert-Source 'ggml/src/ggml-cuda/ggml-cuda.cu' '#ifdef\s+_WIN32\s+static\s+const\s+int\s+hc16\s*=\s*0;\s*#else' 'Windows graph optimizer keeps HC16 disabled'
Assert-Source 'src/models/qwen4exp.cpp' '(?s)const bool direct_indices\s*=[^;]*&&\s*q_cur->ne\[0\]\s*==\s*256\s*&&\s*mctx_cur->type_k\(\)\s*==\s*GGML_TYPE_F16\s*&&\s*mctx_cur->type_v\(\)\s*==\s*GGML_TYPE_F16\s*;' 'Direct sparse attention requires F16 K and V'
Assert-Source 'src/models/qwen4exp.cpp' '(?s)static bool qwen4exp_use_block_selection\([^{}]*\)\s*\{[^{}]*&&\s*mctx_attn->type_k\(\)\s*==\s*GGML_TYPE_F16\s*&&\s*mctx_attn->type_v\(\)\s*==\s*GGML_TYPE_F16\s*;' 'Q8 caches cannot select padded sparse blocks'
Assert-Source 'src/llama-model.cpp' '(?s)#ifdef\s+_WIN32\s+const HANDLE fd\s*=\s*ReOpenFile\([^;]*FILE_FLAG_OVERLAPPED\s*\|\s*FILE_FLAG_RANDOM_ACCESS\);' 'Windows loader enables positioned lazy reads'
Assert-Source 'src/models/qwen4exp.cpp' 'ple_reader\s*=\s*load_lazy_reader\(ml,\s*ple_name\.c_str\(\),\s*per_layer_tok_embd\)' 'Qwen PLE still connects to the direct reader'
Assert-Source 'src/llama-lazy-reader.h' '(?s)void prefetch\([^#]*#ifdef\s+_WIN32\s+try\s*\{[^#]*read_at\(' 'Windows prefetch performs file reads'
Assert-Source 'build-windows.ps1' '--target llama-server llama-cli llama-bench llama-fit-params\s' 'The build includes all four requested tools'

if ($SourceOnly) {
    Write-Host 'Source guards passed. Build and runtime tests were not run.'
    return
}

if (-not [System.IO.Path]::IsPathRooted($BuildDir)) {
    $BuildDir = Join-Path $RepoRoot $BuildDir
}
$BuildDir = [System.IO.Path]::GetFullPath($BuildDir)
$RocmPath = (Resolve-Path -LiteralPath $RocmPath).Path
$LogDir = Join-Path $BuildDir 'windows-regression'
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

function Invoke-Logged([string]$Name, [string]$Command, [string[]]$Arguments) {
    $Log = Join-Path $LogDir "$Name.log"
    Write-Host "Running $Name (log: $Log)"
    Get-Command $Command -ErrorAction Stop | Out-Null
    # Native diagnostic stderr must not become a terminating error in Windows PowerShell 5.1.
    $ErrorActionPreference = 'Continue'
    & $Command @Arguments *> $Log
    $Code = $LASTEXITCODE
    if ($Code -ne 0) {
        Get-Content -LiteralPath $Log -Tail 25 | Out-Host
        throw "$Name failed with exit code $Code. See $Log"
    }
    Write-Host "PASS: $Name"
}

Push-Location $RepoRoot
try {
    # Reuse the build script's VS 2022 environment and rebuild to avoid testing stale DLLs.
    & (Join-Path $RepoRoot 'build-windows.ps1') -RocmPath $RocmPath -BuildDir $BuildDir -Jobs $Jobs
    $BinDir = Join-Path $BuildDir 'bin'
    $env:PATH = "$BinDir;$env:PATH"
    foreach ($Tool in @('llama-server', 'llama-cli', 'llama-bench', 'llama-fit-params')) {
        Invoke-Logged "$Tool-help" (Join-Path $BinDir "$Tool.exe") @('--help')
    }

    $ReaderExe = Join-Path $LogDir 'test-lazy-reader.exe'
    Invoke-Logged 'reader-build' (Join-Path $RocmPath 'lib/llvm/bin/clang++.exe') @(
        '-std=c++17', '-O2', '-fms-runtime-lib=dll', '-DGGML_SHARED', '-D_CRT_SECURE_NO_WARNINGS',
        '-Isrc', '-Iinclude', '-Iggml/include', 'scripts/windows-lazy-reader.cpp',
        'src/llama-mmap.cpp', 'src/llama-impl.cpp', (Join-Path $BuildDir 'ggml/src/ggml-base.lib'),
        '-o', $ReaderExe
    )
    $RunDir = Join-Path $LogDir ('reader-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $RunDir | Out-Null
    Push-Location $RunDir
    try {
        Invoke-Logged 'reader-runtime' $ReaderExe @()
    } finally {
        Pop-Location
    }
    Remove-Item -LiteralPath $RunDir
    Get-Content -LiteralPath (Join-Path $LogDir 'reader-runtime.log') | Out-Host

    if ($SkipGpu) {
        Write-Warning 'GPU checks skipped; this is a partial validation.'
    } else {
        Invoke-Logged 'backend-build' 'cmake' @('--build', $BuildDir, '--target', 'test-backend-ops', '--parallel', "$Jobs")
        $OldFaDisable = $env:LLAMA_TEST_FA_VEC_DISABLE
        $OldDeviceLibPath = $env:HIP_DEVICE_LIB_PATH
        try {
            $env:LLAMA_TEST_FA_VEC_DISABLE = '1'
            # This compiler-only path causes HIP runtime initialization failures on this SDK/driver combination.
            $env:HIP_DEVICE_LIB_PATH = $null
            Invoke-Logged 'attention' (Join-Path $BinDir 'test-backend-ops.exe') @(
                'test', '-b', 'ROCm0', '-o', 'FLASH_ATTN_EXT', '-p', 'hsk=256,hsv=256,nh=2,'
            )
        } finally {
            $env:LLAMA_TEST_FA_VEC_DISABLE = $OldFaDisable
            $env:HIP_DEVICE_LIB_PATH = $OldDeviceLibPath
        }
        $Results = Get-Content -LiteralPath (Join-Path $LogDir 'attention.log') -Raw
        $Results = $Results -replace '\x1B\[[0-9;]*m', ''
        foreach ($Type in @('f16', 'q8_0')) {
            if ($Results -notmatch "(?m)^\s*FLASH_ATTN_EXT\(hsk=256,hsv=256,nh=2,[^\r\n]*type_K=$Type,type_V=$Type,[^\r\n]*\): OK\s*$") {
                throw "No successful $Type attention case ran. The filter or backend support may have changed."
            }
        }
        if ($Results -notmatch 'Backend ROCm0: OK') {
            throw 'ROCm0 was not tested successfully. See attention.log.'
        }
        Write-Host 'PASS: ROCm attention exercised both F16 and Q8 K/V caches'
    }
    Write-Host "Windows regression checks passed for the selected scope. Logs: $LogDir"
} finally {
    Pop-Location
}
