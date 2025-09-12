# diag_step.py — quick traci stepping probe
import os, sys, time
# ensure SUMO tools on path if SUMO_HOME set
if os.environ.get("SUMO_HOME"):
    sys.path.insert(0, os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci, traceback
try:
    sumo_bin = os.path.join(os.environ.get("SUMO_HOME","C:\\Program Files (x86)\\Eclipse\\Sumo"), "bin", "sumo-gui.exe")
    cmd = [sumo_bin, "-n", "net.net.xml", "-r", "routes.rou.xml", "--start"]
    print("Starting SUMO (diag) with:", " ".join(cmd))
    traci.start(cmd)
    for i in range(1, 50):
        print("Before step", i, "time =", traci.simulation.getTime())
        traci.simulationStep()
        print(" After step", i, "time =", traci.simulation.getTime(), "veh_count=", len(traci.vehicle.getIDList()))
    traci.close()
except Exception:
    traceback.print_exc()
