# set_sumo_home.ps1
# Run this first to set SUMO_HOME for the current PowerShell session.
# Edit the path below if your SUMO is installed somewhere else.


$SumoPath = 'C:\Program Files (x86)\Eclipse\Sumo'


if (-Not (Test-Path $SumoPath)) {
Write-Host "ERROR: Path $SumoPath not found. Please edit set_sumo_home.ps1 and set correct SUMO install path." -ForegroundColor Red
exit 1
}


$env:SUMO_HOME = $SumoPath
Write-Host "SUMO_HOME set to: $env:SUMO_HOME" -ForegroundColor Green
Write-Host "randomTrips.py exists?" (Test-Path "$env:SUMO_HOME\tools\randomTrips.py")
Write-Host "duarouter exists?" (Test-Path "$env:SUMO_HOME\bin\duarouter.exe")
Write-Host "sumo-gui exists?" (Test-Path "$env:SUMO_HOME\bin\sumo-gui.exe")


# Keep window open when double-clicked
Read-Host -Prompt "Press Enter to return to the shell"