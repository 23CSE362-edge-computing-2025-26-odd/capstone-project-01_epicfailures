# launch_sumo_gui.ps1
# Launch sumo-gui with routes. Use --step-length to control simulation speed.


param(
[string]$NetFile = 'net.net.xml',
[string]$RouteFile = 'routes.rou.xml',
[double]$StepLength = 0.1
)


if (-Not (Test-Path $NetFile)) { Write-Host "ERROR: $NetFile not found" -ForegroundColor Red; exit 1 }
if (-Not (Test-Path $RouteFile)) { Write-Host "ERROR: $RouteFile not found" -ForegroundColor Red; exit 1 }


$SumoGui = Join-Path $env:SUMO_HOME 'bin\sumo-gui.exe'
if (-Not (Test-Path $SumoGui)) { Write-Host "ERROR: sumo-gui.exe missing at $SumoGui" -ForegroundColor Red; exit 1 }


Write-Host "Launching SUMO GUI..."
& "$SumoGui" -n $NetFile -r $RouteFile --start --quit-on-end false --step-length $StepLength


Read-Host -Prompt "SUMO GUI closed. Press Enter to return to the shell"