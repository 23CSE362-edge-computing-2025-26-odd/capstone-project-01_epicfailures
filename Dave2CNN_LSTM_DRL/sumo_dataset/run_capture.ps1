# run_capture.ps1
# PowerShell wrapper for dataset capture
# Usage: .\run_capture.ps1

# Ensure SUMO_HOME is set
if (-not $env:SUMO_HOME) {
    Write-Error "❌ SUMO_HOME is not set. Please run set_sumo_home.ps1 first."
    exit 1
}

# Paths
$netFile = "net.net.xml"
$routeFile = "routes.rou.xml"
$outDir = "dataset"

# Python script name
$pyScript = "capture_sumo_dataset.py"

if (-not (Test-Path $pyScript)) {
    Write-Error "❌ $pyScript not found in current folder."
    exit 1
}

# Simulation parameters
$begin = 0
$end = 200
$step = 0.5
$useGui = $true   # set to $false if you want headless capture

# Construct command
$cmd = "python `"$pyScript`" --net $netFile --route $routeFile --out $outDir --begin $begin --end $end --step $step"

if ($useGui) {
    $cmd += " --gui"
} else {
    $cmd += " --nogui"
}

Write-Host "▶ Running: $cmd"
Invoke-Expression $cmd
