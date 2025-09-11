#!/usr/bin/env python3
"""
capture_sumo_dataset.py

Complete capture script with:
 - SUMO start (sumo or sumo-gui)
 - robust addition of SUMO tools to PYTHONPATH
 - writes dataset/traces.csv (appends by default; use --overwrite)
 - optional screenshots saved to dataset/frames using traci.gui.screenshot
   (non-blocking screenshot worker with timeout to avoid hangs)
 - verbose/debug prints

Usage examples:
  python capture_sumo_dataset.py --net net.net.xml --routes routes.rou.xml --duration 20
  python capture_sumo_dataset.py --net net.net.xml --routes routes.rou.xml --duration 20 --use-gui --save-frames --verbose
"""

import os
import sys
import csv
import argparse
from pathlib import Path
import time
import traceback
import threading

# Default SUMO_HOME fallback (only used if SUMO_HOME not set and this path exists)
_DEFAULT_SUMO_HOME_WINDOWS = r"C:\Program Files (x86)\Eclipse\Sumo"

def ensure_sumo_tools_on_path():
    """Ensure SUMO_HOME/tools is on sys.path. If SUMO_HOME not set, try fallback path."""
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        # try fallback if present
        if Path(_DEFAULT_SUMO_HOME_WINDOWS).exists():
            sumo_home = _DEFAULT_SUMO_HOME_WINDOWS
            print(f"SUMO_HOME not set — using fallback: {sumo_home}")
            os.environ["SUMO_HOME"] = sumo_home
        else:
            print("ERROR: SUMO_HOME environment variable not set and no fallback found.")
            return

    tools_path = Path(sumo_home) / "tools"
    if not tools_path.exists():
        print(f"WARNING: SUMO tools path not found at: {tools_path}")
    else:
        if str(tools_path) not in sys.path:
            sys.path.insert(0, str(tools_path))
            # Also add to PYTHONPATH env for child processes
            prev = os.environ.get("PYTHONPATH", "")
            if str(tools_path) not in prev:
                os.environ["PYTHONPATH"] = f"{tools_path}{os.pathsep}{prev}"

ensure_sumo_tools_on_path()

try:
    import traci
    import sumolib  # optional, used only to confirm importability
except Exception as e:
    print("ERROR importing SUMO python modules (traci/sumolib).")
    print("Make sure SUMO_HOME is set and SUMO_HOME/tools is on PYTHONPATH.")
    print("Exception while importing:", e)
    raise

def find_sumo_binary(use_gui=False):
    """Return path to sumo or sumo-gui binary based on SUMO_HOME"""
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise RuntimeError("SUMO_HOME not set")
    bin_dir = Path(sumo_home) / "bin"
    if not bin_dir.exists():
        raise FileNotFoundError(f"SUMO bin directory not found: {bin_dir}")

    candidates = ["sumo-gui.exe", "sumo-gui.bat", "sumo-gui"] if use_gui else ["sumo.exe", "sumo.bat", "sumo"]
    for name in candidates:
        p = bin_dir / name
        if p.exists():
            return str(p)
    # fallback: if requested sumo but only gui exists (or vice versa), return available one
    fallback_list = ["sumo-gui.exe", "sumo-gui.bat", "sumo-gui", "sumo.exe", "sumo.bat", "sumo"]
    for name in fallback_list:
        p = bin_dir / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"Could not find SUMO binary in {bin_dir}")

def write_header_if_needed(csv_path: Path, header, overwrite=False):
    if overwrite and csv_path.exists():
        csv_path.unlink()
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

def append_rows(csv_path: Path, rows):
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

# Helper for non-blocking screenshot
def try_screenshot(view_id, filename, result_container):
    try:
        traci.gui.screenshot(view_id, filename)
        result_container['ok'] = True
    except Exception as e:
        result_container['ok'] = False
        result_container['exc'] = e

def main():
    parser = argparse.ArgumentParser(description="Capture SUMO traces and optional frames")
    parser.add_argument("--net", required=True, help="SUMO network file (.net.xml)")
    parser.add_argument("--routes", required=True, help="SUMO routes file (.rou.xml)")
    parser.add_argument("--duration", type=float, default=60.0, help="simulation duration in seconds (approx)")
    parser.add_argument("--step-length", type=float, default=0.5, help="SUMO step-length (seconds)")
    parser.add_argument("--out-dir", default="dataset", help="output directory for traces and frames")
    parser.add_argument("--use-gui", action="store_true", help="use sumo-gui (required for screenshots)")
    parser.add_argument("--save-frames", action="store_true", help="save per-step screenshots (requires --use-gui)")
    parser.add_argument("--overwrite", action="store_true", help="overwrite traces.csv if exists")
    parser.add_argument("--verbose", action="store_true", help="print verbose debug messages")
    parser.add_argument("--screenshot-timeout", type=float, default=2.0, help="seconds to wait for screenshot before skipping")
    parser.add_argument("--save-every", type=int, default=1, help="save every Nth frame (1 => every frame)")
    args = parser.parse_args()

    net_path = Path(args.net)
    routes_path = Path(args.routes)
    if not net_path.exists():
        print(f"ERROR: network file {net_path} not found.")
        return
    if not routes_path.exists():
        print(f"ERROR: routes file {routes_path} not found.")
        return

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    traces_path = out_dir / "traces.csv"
    header = [
        "sim_time", "veh_id", "x", "y", "speed", "angle", "lane_id", "edge_id", "type_id"
    ]
    write_header_if_needed(traces_path, header, overwrite=args.overwrite)

    # Find sumo binary and craft command
    try:
        sumo_bin = find_sumo_binary(use_gui=args.use_gui)
    except Exception as e:
        print("ERROR finding SUMO binary:", e)
        return

    sumo_cmd = [
        sumo_bin,
        "-n", str(net_path),
        "-r", str(routes_path),
        "--step-length", str(args.step_length),
        "--start"
    ]
    print("Starting SUMO with command:", " ".join(sumo_cmd))

    try:
        traci.start(sumo_cmd)
    except Exception as e:
        print("traci.start failed. Make sure the sumo binary path is correct and traci can call it.")
        print("Exception:", e)
        return

    gui_view_id = None
    if args.save_frames:
        try:
            gui_ids = traci.gui.getIDList()
            if gui_ids:
                gui_view_id = gui_ids[0]
                if args.verbose:
                    print("GUI view ID for screenshots:", gui_view_id)
            else:
                if args.verbose:
                    print("No GUI views available; screenshots won't be saved.")
                gui_view_id = None
        except Exception as e:
            if args.verbose:
                print("Could not get GUI view list (are you running headless?). Screenshots disabled.", e)
            gui_view_id = None

    start_time_wall = time.time()
    sim_time = 0.0
    step_length = float(args.step_length)
    max_steps = int(args.duration / step_length) + 1

    print(f"Running capture for approx {args.duration}s ({max_steps} steps at step_length={step_length}s).")

    total_rows = 0
    try:
        for step in range(max_steps):
            traci.simulationStep()
            sim_time = traci.simulation.getTime()

            veh_ids = traci.vehicle.getIDList()
            rows = []
            for vid in veh_ids:
                try:
                    x, y = traci.vehicle.getPosition(vid)
                    speed = traci.vehicle.getSpeed(vid)
                    angle = traci.vehicle.getAngle(vid)
                    lane_id = traci.vehicle.getLaneID(vid)
                    edge_id = traci.vehicle.getRoadID(vid)
                    type_id = traci.vehicle.getTypeID(vid)
                except traci.TraCIException:
                    continue
                rows.append([f"{sim_time:.3f}", vid, f"{x:.3f}", f"{y:.3f}", f"{speed:.3f}", f"{angle:.3f}", lane_id, edge_id, type_id])

            if rows:
                append_rows(traces_path, rows)
                total_rows += len(rows)

            # non-blocking screenshot block (only on selected steps)
            if args.save_frames and gui_view_id is not None and (step % max(1, args.save_every) == 0):
                frame_fname = frames_dir / f"frame_{step:06d}.png"
                frame_fname = frame_fname.resolve()
                if args.verbose:
                    print(f"DEBUG: attempting screenshot step={step} -> {frame_fname}")

                result = {}
                worker = threading.Thread(target=try_screenshot, args=(gui_view_id, str(frame_fname), result))
                worker.daemon = True
                worker.start()
                worker.join(timeout=args.screenshot_timeout)

                if worker.is_alive():
                    if args.verbose:
                        print(f"WARNING: screenshot timed out at step {step}. Skipping and continuing.")
                    # do not attempt to kill; continue simulation
                else:
                    if result.get('ok'):
                        if args.verbose:
                            print(f"DEBUG: screenshot saved for step {step}")
                    else:
                        exc = result.get('exc')
                        if args.verbose:
                            print(f"WARNING: screenshot failed at step {step}. Exception: {exc}")

            if args.verbose and step % max(1, int(1/step_length)) == 0:
                print(f"[step {step}] sim_time={sim_time:.2f}s vehs={len(veh_ids)} rows_written={len(rows)}")

            # end early if no vehicles expected and none present
            try:
                min_expected = traci.simulation.getMinExpectedNumber()
                if min_expected == 0 and len(veh_ids) == 0:
                    if args.verbose:
                        print("No vehicles present and none expected. Ending early.")
                    break
            except Exception:
                # ignore and continue
                pass

        elapsed = time.time() - start_time_wall
        print(f"Capture finished. Steps run: {step+1}. Total rows written: {total_rows}. Wall time: {elapsed:.2f}s")
        print("Traces CSV:", traces_path.resolve())
        if args.save_frames:
            print("Frames directory:", frames_dir.resolve())
    except Exception:
        print("Exception during simulation loop:")
        traceback.print_exc()
    finally:
        try:
            traci.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
