"""
sumo_collect_fixed.py
- Uses routes.rou.xml (contains vType 'car' and route 'r0')
- Starts SUMO with the route file so 'car' exists
- Adds vehicles via TraCI and logs mobility_traces.csv
"""

import os
import csv
import traci
import random

# ---------------- Config ----------------
SUMO_BINARY = "sumo-gui"        # or "sumo-gui" for visualization
NET_FILE = "network.net.xml"
ROUTE_FILE = "routes.rou.xml"
OUTPUT_CSV = "mobility_traces.csv"

SIM_STEPS = 600             # total simulation steps (seconds)
SPAWN_PERIOD = 3            # spawn a vehicle every N steps
VEHICLE_SPEED = 13.0        # desired speed (m/s)
ROUTE_ID = "r0"
# ---------------- Helper to start SUMO ----------------
def start_sumo():
    if not os.path.exists(NET_FILE):
        raise FileNotFoundError(f"{NET_FILE} not found. Generate network with netconvert first.")
    if not os.path.exists(ROUTE_FILE):
        raise FileNotFoundError(f"{ROUTE_FILE} not found. Create it with a vType 'car' and a route 'r0'.")

    sumo_cmd = [SUMO_BINARY, "-n", NET_FILE, "-r", ROUTE_FILE, "--step-length", "1.0"]
    print("Starting SUMO with command:", " ".join(sumo_cmd))
    traci.start(sumo_cmd)


# ---------------- Main simulation & logging ----------------
def run_and_collect():
    # CSV header
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "veh_id", "x", "y", "speed", "angle"])

    # Start SUMO (with route file so vType 'car' is known)
    start_sumo()

    veh_index = 0
    for step in range(SIM_STEPS):
        # spawn a new vehicle every SPAWN_PERIOD steps
        if step % SPAWN_PERIOD == 0:
            vid = f"veh_{veh_index}"
            try:
                # refer to route r0 and vehicle type 'car' defined in routes.rou.xml
                traci.vehicle.add(vehID=vid, routeID=ROUTE_ID, typeID="car", depart=str(step))
                # optionally set speed
                traci.vehicle.setSpeed(vid, VEHICLE_SPEED)
            except Exception as e:
                print(f"vehicle.add error for {vid}:", e)
            veh_index += 1

        # advance the simulation by one step
        traci.simulationStep()

        # iterate over current vehicles and record data
        ids = traci.vehicle.getIDList()
        if ids:
            rows = []
            for vid in ids:
                try:
                    x, y = traci.vehicle.getPosition(vid)
                    speed = traci.vehicle.getSpeed(vid)
                    angle = traci.vehicle.getAngle(vid)
                    rows.append([step, vid, x, y, speed, angle])
                except Exception as e:
                    # vehicle might disappear between getIDList() and getPosition(); ignore such race
                    print("vehicle data read error:", e)

            # append to CSV
            with open(OUTPUT_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(rows)

    traci.close()
    print(f"Simulation finished. Mobility traces saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_and_collect()
