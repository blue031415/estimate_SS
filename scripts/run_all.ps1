$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

$files = @(
    "src\ex1.py",
    "src\ex2_first.py",
    "src\ex2_first_origin.py",
    "src\ex2_first_overlay.py",
    "src\ex2_second.py",
    "src\ex2_second_origin.py",
    "src\ex2_second_overlay.py",
    "src\ex2_geodestic.py",
    "src\ex2_geodestic_origin.py",
    "src\ex3_second.py"
)

Push-Location $projectRoot
try {
    foreach ($file in $files) {
        Write-Host "Running $file"
        python $file
        if ($LASTEXITCODE -ne 0) {
            throw "$file failed with exit code $LASTEXITCODE"
        }
    }
}
finally {
    Pop-Location
}
