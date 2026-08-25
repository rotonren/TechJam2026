$ErrorActionPreference = "Stop"
$releaseBase = "https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit"
$dataDir = Join-Path $PSScriptRoot "..\data"
$archive = Join-Path $dataDir "catalog.jsonl.gz"
$checksumFile = Join-Path $dataDir "SHA256SUMS"
$catalog = Join-Path $dataDir "catalog.jsonl"

New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
Invoke-WebRequest "$releaseBase/catalog.jsonl.gz" -OutFile $archive
Invoke-WebRequest "$releaseBase/SHA256SUMS" -OutFile $checksumFile

$expected = ((Get-Content $checksumFile | Where-Object { $_ -match "catalog.jsonl.gz" }) -split "\s+")[0].ToLowerInvariant()
$actual = (Get-FileHash -Algorithm SHA256 $archive).Hash.ToLowerInvariant()
if ($expected -ne $actual) {
    throw "catalog checksum mismatch"
}

$inputStream = [System.IO.File]::OpenRead($archive)
$outputStream = [System.IO.File]::Create($catalog)
$gzipStream = [System.IO.Compression.GZipStream]::new(
    $inputStream,
    [System.IO.Compression.CompressionMode]::Decompress
)
try {
    $gzipStream.CopyTo($outputStream)
} finally {
    $gzipStream.Dispose()
    $outputStream.Dispose()
    $inputStream.Dispose()
}

Write-Host "Catalog ready at $catalog"
