#!/usr/bin/env python3
"""
capture_sumo_dataset_matplotlib.py

Runs SUMO, collects vehicle traces and optionally renders frames using matplotlib
instead of SUMO GUI screenshots (more reliable for batch runs).

Usage examples:
  # headless, no frames
  python capture_sumo_dataset_matplotlib.py --net net.net.xml --routes routes.rou.xml --duration 20

  # headless with frames (saves every frame)
  python capture_sumo_dataset_matplotlib.py --net net.net.xml --routes routes.rou.xml --duration 20 --save-frames

  # save every 5th frame, overwrite traces and frames directory
  python capture_sumo_dataset_matplotlib.py --net net.net.xml --routes routes.rou.xml --duration 20 --save-frames --save-every 5 --overwrite --verbose
"""

import os
import sys
import csv
import argparse
from pathlib import Path
import time
import traceback

# add SUMO tools if SUMO_HOME is set
if os.environ.get("SUMO_HOME"):
    tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)

try:
    import traci
except Exception as e:
    print("ERROR: cannot import traci. Make sure SUMO_HOME/tools is on PYTHONPATH and SUMO is installed.")
    raise

# matplotlib for rendering frames
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

# try to use sumolib to read network bounds (optional)
try:
    import sumolib
    _HAS_SUMOLIB = True
except Exception:
    _HAS_SUMOLIB = False

def find_sumo_binary(use_gui=False):
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise RuntimeError("SUMO_HOME not set")
    bin_dir = Path(sumo_home) / "bin"
    candidates = ["sumo-gui.exe", "sumo-gui.bat", "sumo-gui"] if use_gui else ["sumo.exe", "sumo.bat", "sumo"]
    for name in candidates:
        p = bin_dir / name
        if p.exists():
            return str(p)
    # fallback try either binary
    for name in ["sumo-gui.exe", "sumo-gui.bat", "sumo-gui", "sumo.exe", "sumo.bat", "sumo"]:
        p = bin_dir / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"SUMO binary not found in {bin_dir}")

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

def compute_plot_bounds(netfile: Path):
    """Return (xmin, xmax, ymin, ymax). Try sumolib; if not available, return None."""
    try:
        if _HAS_SUMOLIB:
            net = sumolib.net.readNet(str(netfile))
            xmin, ymin, xmax, ymax = net.getBoundary()
            return xmin, xmax, ymin, ymax
    except Exception:
        pass
    return None

def plot_frame(positions, bounds, out_path: Path, fig_size=(6,6), dpi=150):
    """
    positions: list of (x,y)
    bounds: (xmin,xmax,ymin,ymax) or None
    out_path: Path to save PNG
    """
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    plt.figure(figsize=fig_size)
    ax = plt.gca()
    # plot vehicles as points (let matplotlib pick default color/style)
    if xs and ys:
        ax.scatter(xs, ys, s=10)  # small markers
    ax.set_aspect('equal', adjustable='box')

    if bounds:
        xmin, xmax, ymin, ymax = bounds
        # add small margin
        xpad = max(1.0, (xmax - xmin) * 0.02)
        ypad = max(1.0, (ymax - ymin) * 0.02)
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)
    else:
        # autoscale
        ax.autoscale(enable=True)
    ax.axis('off')
    plt.tight_layout(pad=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Capture SUMO traces and render frames with matplotlib")
    parser.add_argument("--net", required=True, help="SUMO network file (.net.xml)")
    parser.add_argument("--routes", required=True, help="SUMO routes file (.rou.xml)")
    parser.add_argument("--duration", type=float, default=60.0, help="simulation duration in seconds (approx)")
    parser.add_argument("--step-length", type=float, default=0.5, help="SUMO step-length (seconds)")
    parser.add_argument("--out-dir", default="dataset", help="output directory for traces and frames")
    parser.add_argument("--use-gui", action="store_true", help="use sumo-gui (not required for rendering; left as option)")
    parser.add_argument("--save-frames", action="store_true", help="save per-step frames using matplotlib")
    parser.add_argument("--save-every", type=int, default=1, help="save every Nth frame (1 => every frame)")
    parser.add_argument("--overwrite", action="store_true", help="overwrite traces.csv if exists")
    parser.add_argument("--verbose", action="store_true", help="print verbose output")
    parser.add_argument("--dpi", type=int, default=150, help="dpi for saved PNG frames")
    parser.add_argument("--fig-size", type=float, nargs=2, default=(6,6), help="figure size in inches for saved PNGs")
    args = parser.parse_args()

    net_path = Path(args.net)
    routes_path = Path(args.routes)
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    traces_path = out_dir / "traces.csv"
    header = ["sim_time", "veh_id", "x", "y", "speed", "angle", "lane_id", "edge_id", "type_id"]
    write_header_if_needed(traces_path, header, overwrite=args.overwrite)

    # start SUMO via traci.start
    try:
        sumo_bin = find_sumo_binary(use_gui=args.use_gui)
    except Exception as e:
        print("ERROR finding SUMO binary:", e)
        return

    sumo_cmd = [sumo_bin, "-n", str(net_path), "-r", str(routes_path), "--step-length", str(args.step_length), "--start"]
    if args.verbose:
        print("Starting SUMO with:", " ".join(sumo_cmd))

    try:
        traci.start(sumo_cmd)
    except Exception as e:
        print("ERROR starting SUMO via traci.start:", e)
        return

    # get plot bounds (if sumolib available)
    bounds = compute_plot_bounds(net_path)
    if args.verbose:
        print("Plot bounds:", bounds)

    start_wall = time.time()
    step_len = float(args.step_length)
    max_steps = int(args.duration / step_len) + 1

    total_rows = 0
    try:
        for step in range(max_steps):
            traci.simulationStep()
            sim_time = traci.simulation.getTime()

            veh_ids = traci.vehicle.getIDList()
            rows = []
            positions = []
            for vid in veh_ids:
                try:
                    x, y = traci.vehicle.getPosition(vid)
                    speed = traci.vehicle.getSpeed(vid)
                    angle = traci.vehicle.getAngle(vid)
                    lane_id = traci.vehicle.getLaneID(vid)
                    edge_id = traci.vehicle.getRoadID(vid)
                    type_id = traci.vehicle.getTypeID(vid)
                except Exception:
                    continue
                rows.append([f"{sim_time:.3f}", vid, f"{x:.3f}", f"{y:.3f}", f"{speed:.3f}", f"{angle:.3f}", lane_id, edge_id, type_id])
                positions.append((x,y))

            if rows:
                append_rows(traces_path, rows)
                total_rows += len(rows)

            # render frame if requested and matches save_every
            if args.save_frames and (step % max(1, args.save_every) == 0):
                frame_file = frames_dir / f"frame_{step:06d}.png"
                try:
                    plot_frame(positions, bounds, frame_file, fig_size=tuple(args.fig_size), dpi=args.dpi)
                    if args.verbose:
                        print(f"[step {step}] sim_time={sim_time:.2f}s vehs={len(veh_ids)} rows={len(rows)} -> saved {frame_file.name}")
                except Exception as e:
                    print(f"WARNING: failed to render frame at step {step}: {e}")

            elif args.verbose:
                print(f"[step {step}] sim_time={sim_time:.2f}s vehs={len(veh_ids)} rows={len(rows)}")

            # end early if no vehicles expected and none present
            try:
                if traci.simulation.getMinExpectedNumber() == 0 and len(veh_ids) == 0:
                    if args.verbose:
                        print("No vehicles expected and none present. Ending early.")
                    break
            except Exception:
                pass

        elapsed = time.time() - start_wall
        print(f"Finished. Steps run: {step+1}. Total rows written: {total_rows}. Wall time: {elapsed:.2f}s")
        print("Traces CSV:", traces_path.resolve())
        if args.save_frames:
            print("Frames directory:", frames_dir.resolve())
    except Exception:
        print("Exception during loop:")
        traceback.print_exc()
    finally:
        try:
            traci.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
