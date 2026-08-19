$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

$experiments = @(
    @("src\ex1.py", "configs\ex1.yaml"),
    @("src\ex2_first.py", "configs\ex2_first.yaml"),
    @("src\ex2_first_origin.py", "configs\ex2_first_origin.yaml"),
    @("src\ex2_first_overlay.py", "configs\ex2_first_overlay.yaml"),
    @("src\ex2_second.py", "configs\ex2_second.yaml"),
    @("src\ex2_second_origin.py", "configs\ex2_second_origin.yaml"),
    @("src\ex2_second_overlay.py", "configs\ex2_second_overlay.yaml"),
    @("src\ex2_geodestic.py", "configs\ex2_geodestic.yaml"),
    @("src\ex2_geodestic_origin.py", "configs\ex2_geodestic_origin.yaml"),
    @("src\ex3_second.py", "configs\ex3_second.yaml")
)

Push-Location $projectRoot
try {
    foreach ($experiment in $experiments) {
        $file = $experiment[0]
        $config = $experiment[1]
        Write-Host "Running $file with $config"
        python $file $config
        if ($LASTEXITCODE -ne 0) {
            throw "$file failed with exit code $LASTEXITCODE"
        }
    }
}
finally {
    Pop-Location
}
