[CmdletBinding()]
param(
    [string]$BuildDirectory = 'D:\translation\tmp\pmpp5-solutions-code-vs2022',
    [ValidatePattern('^[0-9]+[a-z]?$')]
    [string]$CudaArchitecture = '75',
    [switch]$SkipRun,
    [switch]$SkipCTest,
    [switch]$SkipCudaRun
)

$ErrorActionPreference = 'Stop'
if ($SkipRun) {
    $SkipCTest = $true
    $SkipCudaRun = $true
}

$sourceDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$resolvedSource = (Resolve-Path -LiteralPath $sourceDirectory).Path
$resolvedBuild = [System.IO.Path]::GetFullPath($BuildDirectory)
$repositoryRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $resolvedSource '..\..'))
$repositoryBoundary = $repositoryRoot.TrimEnd('\', '/') +
    [System.IO.Path]::DirectorySeparatorChar

if ($resolvedBuild.Equals($repositoryRoot,
        [System.StringComparison]::OrdinalIgnoreCase) -or
    $resolvedBuild.StartsWith($repositoryBoundary,
        [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'BuildDirectory must be outside the repository.'
}

$nvccCommand = Get-Command nvcc -ErrorAction SilentlyContinue
$nvccPath = if ($nvccCommand) { $nvccCommand.Source } else { $null }
if (-not $nvccPath) {
    if ($env:CUDA_PATH) {
        $cudaPathCandidate = Join-Path $env:CUDA_PATH 'bin\nvcc.exe'
        if (Test-Path -LiteralPath $cudaPathCandidate) {
            $nvccPath = $cudaPathCandidate
        }
    }
    if (-not $nvccPath) {
        $fallbackCandidate =
            'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe'
        if (Test-Path -LiteralPath $fallbackCandidate) {
            $nvccPath = $fallbackCandidate
        }
    }
}

$clDirectory = $null
$isWindowsHost = $env:OS -eq 'Windows_NT'
if ($isWindowsHost) {
    $vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) {
        throw 'vswhere.exe was not found; MSVC Build Tools are required on Windows.'
    }
    $vswhereArguments = @(
        '-latest', '-products', '*',
        '-version', '[17.0,18.0)',
        '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
        '-property', 'installationPath'
    )
    $vsRoot = & $vswhere @vswhereArguments
    if ($LASTEXITCODE -ne 0) {
        throw 'vswhere.exe failed while locating MSVC Build Tools.'
    }
    if (-not $vsRoot) {
        throw 'MSVC x64 build tools were not found.'
    }
    $msvcRoot = Get-ChildItem -LiteralPath (Join-Path $vsRoot 'VC\Tools\MSVC') -Directory |
        Sort-Object @{ Expression = { [version]$_.Name }; Descending = $true } |
        Select-Object -First 1
    $clDirectory = Join-Path $msvcRoot.FullName 'bin\Hostx64\x64'
    if (-not (Test-Path -LiteralPath (Join-Path $clDirectory 'cl.exe'))) {
        throw 'The MSVC x64 host compiler was not found.'
    }
}

$cmakeArguments = @(
    '-S', $resolvedSource,
    '-B', $resolvedBuild,
    '-DPMPP_ENABLE_CUDA_WITH_CMAKE=OFF'
)
if ($isWindowsHost) {
    $cachePath = Join-Path $resolvedBuild 'CMakeCache.txt'
    if (Test-Path -LiteralPath $cachePath) {
        $cacheLines = Get-Content -LiteralPath $cachePath
        $generatorLine = $cacheLines |
            Where-Object { $_ -like 'CMAKE_GENERATOR:INTERNAL=*' } |
            Select-Object -First 1
        $platformLine = $cacheLines |
            Where-Object { $_ -like 'CMAKE_GENERATOR_PLATFORM:INTERNAL=*' } |
            Select-Object -First 1
        $instanceLine = $cacheLines |
            Where-Object { $_ -like 'CMAKE_GENERATOR_INSTANCE:INTERNAL=*' } |
            Select-Object -First 1
        $configuredGenerator = if ($generatorLine) {
            ($generatorLine -split '=', 2)[1]
        } else { '' }
        $configuredPlatform = if ($platformLine) {
            ($platformLine -split '=', 2)[1]
        } else { '' }
        $configuredInstance = if ($instanceLine) {
            ($instanceLine -split '=', 2)[1]
        } else { '' }
        if ($configuredGenerator -ne 'Visual Studio 17 2022' -or
            $configuredPlatform -ne 'x64' -or
            ($configuredInstance -and
             -not $configuredInstance.Equals($vsRoot,
                 [System.StringComparison]::OrdinalIgnoreCase))) {
            throw "BuildDirectory already uses generator '$configuredGenerator' " +
                "platform '$configuredPlatform', and instance '$configuredInstance'. " +
                'Choose a fresh external directory.'
        }
    }
    # Keep ordinary C++/MPI and nvcc host objects on one Windows ABI/toolchain.
    $cmakeArguments += @(
        '-G', 'Visual Studio 17 2022',
        '-A', 'x64',
        "-DCMAKE_GENERATOR_INSTANCE=$vsRoot"
    )
}
cmake @cmakeArguments
if ($LASTEXITCODE -ne 0) {
    throw 'CMake configure failed.'
}
cmake --build $resolvedBuild --config Release
if ($LASTEXITCODE -ne 0) {
    throw 'CMake build failed.'
}
if (-not $SkipCTest) {
    ctest --test-dir $resolvedBuild --build-config Release --output-on-failure
    if ($LASTEXITCODE -ne 0) {
        throw 'CTest failed.'
    }
}

if (-not $nvccPath) {
    Write-Warning 'nvcc was not found; CMake built CPU examples only.'
    Write-Host "Build outputs: $resolvedBuild"
    exit 0
}
Write-Host "nvcc: $nvccPath"
& $nvccPath --version
if ($LASTEXITCODE -ne 0) {
    throw 'nvcc --version failed.'
}

if (-not $clDirectory) {
    throw 'This build script requires the MSVC x64 host compiler for nvcc.'
}

$cudaOutput = Join-Path $resolvedBuild 'cuda'
New-Item -ItemType Directory -Force -Path $cudaOutput | Out-Null
$executables = @()
$sourcePrefix = $resolvedSource + [System.IO.Path]::DirectorySeparatorChar
$cudaSources = Get-ChildItem -LiteralPath $resolvedSource -Recurse -Filter 'ch*.cu' -File
foreach ($source in $cudaSources) {
    if (-not $source.FullName.StartsWith(
            $sourcePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "CUDA source is outside SourceDirectory: $($source.FullName)"
    }
    $relative = $source.FullName.Substring($sourcePrefix.Length)
    $name = [regex]::Replace($relative, '[^A-Za-z0-9]+', '_').TrimEnd('_')
    $output = Join-Path $cudaOutput ($name + '.exe')
    $nvccArguments = @(
        '-std=c++17',
        '-ccbin', $clDirectory,
        '-Xcompiler=/utf-8',
        '-Xcompiler=/Zc:preprocessor',
        "-arch=sm_$CudaArchitecture",
        '-I', $resolvedSource,
        $source.FullName,
        '-o', $output
    )
    & $nvccPath @nvccArguments
    if ($LASTEXITCODE -ne 0) {
        throw "nvcc failed for $relative"
    }
    $executables += $output
}

if (-not $SkipCudaRun) {
    foreach ($executable in $executables) {
        & $executable '--cpu-only'
        if ($LASTEXITCODE -ne 0) {
            throw "CPU reference run failed: $executable"
        }
    }
}

Write-Host "Built $($cudaSources.Count) CUDA example(s)."
Write-Host "Build outputs: $resolvedBuild"
