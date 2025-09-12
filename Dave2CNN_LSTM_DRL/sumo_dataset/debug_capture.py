# debug_capture.py
import os, time, sys
try:
    import traci
except Exception:
    SUMO_HOME = os.environ.get("SUMO_HOME", "")
    if not SUMO_HOME:
        raise RuntimeError("SUMO_HOME not set; cannot import traci.")
    sys.path.append(os.path.join(SUMO_HOME, "tools"))
    import traci

import pandas as pd

NET = "net.net.xml"
ROUTE = "routes.rou.xml"
OUTDIR = "dataset_debug"
FRAMES = os.path.join(OUTDIR, "frames")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FRAMES, exist_ok=True)

SUMO_BIN = "sumo-gui"  # change to "sumo" if you prefer headless
end_time = 30.0
step_len = 0.5

print("Starting SUMO (debug)...")
traci.start([SUMO_BIN, "-n", NET, "-r", ROUTE, "--step-length", str(step_len), "--start"])

rows = []
tcount = 0
max_steps = int(end_time / step_len)
for step in range(max_steps):
    traci.simulationStep()
    t = traci.simulation.getTime()
    vids = traci.vehicle.getIDList()
    print(f"time={t:.2f}s vehicles={len(vids)} -> {vids[:10]}")
    # if any vehicles present, print their positions
    for vid in vids[:5]:
        pos = traci.vehicle.getPosition(vid)
        speed = traci.vehicle.getSpeed(vid)
        print(f"  {vid} pos={pos} speed={speed:.2f}")
        rows.append({"time": t, "veh_id": vid, "x": pos[0], "y": pos[1], "frame_path": ""})
    if len(vids) > 0:
        print("Vehicles have spawned. Debug capture will stop after collecting initial data.")
        # break optionally to keep output short
        # break

traci.close()
df = pd.DataFrame(rows)
csvp = os.path.join(OUTDIR, "debug_traces.csv")
df.to_csv(csvp, index=False)
print("Wrote debug CSV to", csvp)
