#!/usr/bin/env python3
"""
cov_publisher_control.py

- Connects to CARLA (host/port configurable).
- Listens for control messages (set_target) on control port.
- When target set, finds the actor by id or by expected_pose (fallback), then publishes cov_detections to DT.
"""

import argparse, time, json, socket, threading, math, sys
import carla

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1", help="CARLA host")
parser.add_argument("--port", type=int, default=2000, help="CARLA port")
parser.add_argument("--dt-host", default="127.0.0.1", help="DT host to send detections")
parser.add_argument("--dt-port", type=int, default=7007, help="DT port to send detections")
parser.add_argument("--control-port", type=int, default=7012, help="local UDP port to listen for control")
parser.add_argument("--rate", type=float, default=10.0, help="publish rate (Hz)")
parser.add_argument("--radius", type=float, default=150.0, help="detection radius (m)")
args = parser.parse_args()

POLL_SLEEP = 1.0 / args.rate
DT_ADDR = (args.dt_host, args.dt_port)

# ---------- connect to CARLA ----------
print(f"[publisher] Connecting to CARLA {args.host}:{args.port} ...")
client = carla.Client(args.host, args.port)
client.set_timeout(10.0)
try:
    world = client.get_world()
    print("[publisher] Connected to CARLA. Map:", world.get_map().name)
    vehicles = world.get_actors().filter("vehicle.*")
    print("[publisher] initial vehicle count:", len(vehicles), "sample ids:", [v.id for v in vehicles][:12])
except Exception as e:
    print("[publisher] Failed to connect to CARLA:", e)
    sys.exit(1)

# ---------- sockets ----------
send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ctrl_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
ctrl_sock.bind(("0.0.0.0", args.control_port))
ctrl_sock.settimeout(0.5)

# ---------- control & state ----------
target_cov_id = None
target_expected_pose = None
search_radius = args.radius
target_lock = threading.Lock()

def control_loop():
    global target_cov_id, target_expected_pose
    print("[publisher] control listening on 0.0.0.0:%d" % args.control_port)
    while True:
        try:
            data, addr = ctrl_sock.recvfrom(4096)
        except Exception:
            time.sleep(0.01)
            continue
        try:
            msg = json.loads(data.decode("utf-8"))
        except Exception:
            print("[publisher] control: received invalid JSON")
            continue
        if msg.get("type") == "set_target":
            try:
                tcov = int(msg.get("cov_id"))
            except Exception:
                tcov = None
            tpose = msg.get("expected_pose", None)
            with target_lock:
                target_cov_id = tcov
                target_expected_pose = tpose
            print(f"[publisher] received set_target cov_id={target_cov_id} expected_pose={target_expected_pose}")
        elif msg.get("type") == "clear_target":
            with target_lock:
                target_cov_id = None
                target_expected_pose = None
            print("[publisher] received clear_target")

threading.Thread(target=control_loop, daemon=True).start()

def find_actor_by_id_or_pose(cov_id, expected_pose=None, search_radius=80.0):
    # try direct lookup
    try:
        actor = world.get_actors().find(cov_id)
        if actor is not None:
            return actor
    except Exception:
        pass
    # pose fallback
    if expected_pose:
        vx = float(expected_pose.get("x", 0.0))
        vy = float(expected_pose.get("y", 0.0))
        vz = float(expected_pose.get("z", 0.0))
        best = None
        best_d = float("inf")
        for v in world.get_actors().filter("vehicle.*"):
            loc = v.get_location()
            dx = loc.x - vx; dy = loc.y - vy; dz = loc.z - vz
            d = math.sqrt(dx*dx + dy*dy + dz*dz)
            if d < best_d:
                best_d = d; best = v
        if best is not None and best_d <= search_radius:
            print(f"[publisher] pose-fallback: picked actor id={best.id} at dist={best_d:.1f}m (expected id={cov_id})")
            return best
    return None

def loc_to_list(loc):
    return [float(loc.x), float(loc.y), float(loc.z)]

frame_idx = 0
print("[publisher] entering main loop; waiting for set_target ...")
try:
    while True:
        # non-blocking check of current target
        with target_lock:
            cur_target = target_cov_id
            cur_pose = target_expected_pose

        if cur_target is None:
            # no target set - sleep a little and continue
            time.sleep(0.05)
            continue

        # attempt to find actor by id or pose
        actor = find_actor_by_id_or_pose(cur_target, cur_pose, search_radius=search_radius)
        if actor is None:
            # debug: list known vehicles
            known = [a.id for a in world.get_actors().filter("vehicle.*")]
            print(f"[publisher] target {cur_target} not found. known vehicles ({len(known)}): {known[:40]}")
            time.sleep(POLL_SLEEP)
            continue

        # build detections relative to this actor
        cov_loc = actor.get_location()
        detections = []
        for a in world.get_actors().filter("vehicle.*"):
            if a.id == actor.id:
                continue
            loc = a.get_location()
            dx = loc.x - cov_loc.x; dy = loc.y - cov_loc.y; dz = loc.z - cov_loc.z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist <= search_radius:
                bbox = a.bounding_box
                at = a.get_transform()
                # approximate bbox center in world coords
                bbox_center_world = carla.Location(x=at.location.x + bbox.location.x,
                                                  y=at.location.y + bbox.location.y,
                                                  z=at.location.z + bbox.location.z)
                det = {
                    "object_id": int(a.id),
                    "type": a.type_id,
                    "world_pos": loc_to_list(bbox_center_world),
                    "bbox_extent": [float(bbox.extent.x), float(bbox.extent.y), float(bbox.extent.z)],
                    "distance": float(dist)
                }
                detections.append(det)

        msg = {
            "type":"cov_detections",
            "frame_idx": frame_idx,
            "sim_time": float(world.get_snapshot().timestamp.elapsed_seconds),
            "cov_id": int(actor.id),
            "detections": detections
        }
        try:
            send_sock.sendto(json.dumps(msg).encode("utf-8"), DT_ADDR)
            print(f"[publisher] sent cov_detections frame={frame_idx} cov_id={actor.id} -> DT {DT_ADDR} detections={len(detections)}")
        except Exception as e:
            print("[publisher] UDP send error:", e)

        frame_idx += 1
        time.sleep(POLL_SLEEP)

except KeyboardInterrupt:
    print("[publisher] stopped by user")
