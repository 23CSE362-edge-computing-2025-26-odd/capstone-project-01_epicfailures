# convert_trips.ps1
# Run after generate_trips.ps1
# Converts trips.trips.xml -> routes.rou.xml using duarouter


param(
[string]$NetFile = 'net.net.xml',
[string]$TripsFile = 'trips.trips.xml',
[string]$OutRoutes = 'routes.rou.xml'
)


if (-Not (Test-Path $TripsFile)) {
Write-Host "ERROR: $TripsFile not found in current dir." -ForegroundColor Red
exit 1
}


$Dua = Join-Path $env:SUMO_HOME 'bin\duarouter.exe'
if (-Not (Test-Path $Dua)) {
Write-Host "duarouter.exe not found at $Dua" -ForegroundColor Red
exit 1
}


Write-Host "Running duarouter -> $OutRoutes"
& "$Dua" -n $NetFile -t $TripsFile -o $OutRoutes


if (Test-Path $OutRoutes) {
Write-Host "$OutRoutes created." -ForegroundColor Green
} else {
Write-Host "duarouter failed to create $OutRoutes" -ForegroundColor Red
}


Read-Host -Prompt "Press Enter to return to the shell"  