# generate_trips.ps1
# Run after set_sumo_home.ps1
# Generates trips.trips.xml from net.net.xml


param(
[string]$NetFile = 'net.net.xml',
[int]$Begin = 0,
[int]$End = 200,
[double]$Prob = 1.0
)


if (-Not (Test-Path $NetFile)) {
Write-Host "ERROR: $NetFile not found in current directory: $(Get-Location)" -ForegroundColor Red
exit 1
}


$RandomTrips = Join-Path $env:SUMO_HOME 'tools\randomTrips.py'
if (-Not (Test-Path $RandomTrips)) {
Write-Host "ERROR: randomTrips.py not found at $RandomTrips" -ForegroundColor Red
exit 1
}


Write-Host "Running randomTrips.py -> trips.trips.xml"
python "$RandomTrips" -n $NetFile -o trips.trips.xml -b $Begin -e $End -p $Prob


if (Test-Path 'trips.trips.xml') {
Write-Host "trips.trips.xml created." -ForegroundColor Green
} else {
Write-Host "Failed to create trips.trips.xml" -ForegroundColor Red
}


Read-Host -Prompt "Press Enter to return to the shell"