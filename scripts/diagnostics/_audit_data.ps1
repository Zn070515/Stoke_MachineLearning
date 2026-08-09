# DIAGNOSTIC (diagnostics): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
$base = (Resolve-Path (Join-Path $PSScriptRoot "..\..\data\a_shares")).Path
$dirs = @("daily", "news_raw", "news_silver", "news_sentiment", "guba_raw", "guba_silver", "guba_sentiment", "announcements", "comment_raw", "comment_silver", "comment_sentiment", "market_wide", "fundamentals", "minute")
foreach ($d in $dirs) {
    $p = Join-Path $base $d
    if (Test-Path $p) {
        $fs = Get-ChildItem $p -Recurse -Filter "*.parquet" -ErrorAction SilentlyContinue
        $sizeMB = ($fs | Measure-Object Length -Sum).Sum / 1MB
        Write-Output ("{0,-22} {1,6} files  {2,8:N1} MB" -f $d, $fs.Count, $sizeMB)
    } else {
        Write-Output ("{0,-22}   MISSING" -f $d)
    }
}
