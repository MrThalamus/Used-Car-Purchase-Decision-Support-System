<#
.SYNOPSIS
Converts a local file to Markdown with Microsoft MarkItDown.

.EXAMPLE
.\convert-to-markdown.ps1 "C:\Documents\report.pdf"

.EXAMPLE
.\convert-to-markdown.ps1 "C:\Documents\slides.pptx" -OutputFile "C:\Documents\slides.md"
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [ValidateNotNullOrEmpty()]
    [string]$InputFile,

    [Parameter(Position = 1)]
    [string]$OutputFile,

    [switch]$UsePlugins
)

$converter = Join-Path $env:USERPROFILE ".markitdown-venv\Scripts\markitdown.exe"

if (-not (Test-Path -LiteralPath $converter)) {
    throw "MarkItDown is not installed at '$converter'. Run: py -m venv $env:USERPROFILE\.markitdown-venv; $env:USERPROFILE\.markitdown-venv\Scripts\python.exe -m pip install 'markitdown[all]'"
}

if (-not (Test-Path -LiteralPath $InputFile -PathType Leaf)) {
    throw "Input file not found: $InputFile"
}

$sourcePath = (Resolve-Path -LiteralPath $InputFile).Path
if (-not $OutputFile) {
    $OutputFile = "$sourcePath.md"
}

$arguments = @($sourcePath, "-o", $OutputFile)
if ($UsePlugins) {
    $arguments = @("--use-plugins") + $arguments
}

& $converter @arguments
if ($LASTEXITCODE -ne 0) {
    throw "MarkItDown conversion failed with exit code $LASTEXITCODE."
}

Write-Host "Markdown created: $OutputFile"
