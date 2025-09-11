Run in powerShell
-------------------


# 1. Set SUMO_HOME manually (adjust path if different)
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"

# 2. Also add SUMO tools to PYTHONPATH so traci/sumolib can import
$env:PYTHONPATH = "$env:SUMO_HOME\tools;$env:PYTHONPATH"

# 3. Verify
echo "SUMO_HOME = $env:SUMO_HOME"
Test-Path "$env:SUMO_HOME\bin\sumo.exe"

# 4. Run the capture script
python capture_sumo_dataset.py --net net.net.xml --routes routes.rou.xml --duration 20 --overwrite --verbose
