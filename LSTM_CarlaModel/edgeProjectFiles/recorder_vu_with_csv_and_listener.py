#!/usr/bin/env python3
"""
recorder_vu_with_csv_and_listener.py

- Spawns a VU vehicle and N CoV vehicles (configurable).
- Record initial frames into run_output/trajectories.csv (INIT_FRAMES).
- After CSV is written, send a UDP "csv_ready" notice to DT.
- Continue dashcam recording; listen on UDP for "selected_cov_metadata"
  messages from DT and overlay them onto frames.
"""
import carla
import random, time, os, csv, json, socket, threading
import numpy as np
import cv2, math

# -------------- CONFIG --------------
OUTPUT_DIR = "run_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_FILE = os.path.join(OUTPUT_DIR, "trajectories.csv")
VIDEO_FILE = os.path.join(OUTPUT_DIR, "vu_udp_overlay.mp4")

# How many frames to collect into the CSV before notifying DT
INIT_FRAMES = 200  # change this to the number of frames you want recorded first (e.g., 60)

# number of CoV vehicles to spawn (set 0 to not spawn)
NUM_COVS = 3

# UDP / networking
DT_HOST = "127.0.0.1"
DT_CSV_READY_PORT = 7010      # VU -> DT csv ready notice
VU_HOST = "127.0.0.1"
VU_META_PORT = 6008           # DT -> VU metadata (we listen here)

# camera settings
IMAGE_W = 640
IMAGE_H = 480
FOV = 90

# ------------------------------------------------
# minimal CARLA camera + vehicle spawn (adapt to your real recorder)
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()

bp_lib = world.get_blueprint_library()

# ---------- spawn VU ----------
# choose VU blueprint
vu_bp_candidates = bp_lib.filter("vehicle.tesla.model3") or bp_lib.filter("vehicle.*")
vehicle_bp = vu_bp_candidates[0]
spawn_points = world.get_map().get_spawn_points()
if not spawn_points:
    raise RuntimeError("No spawn points available in map")

# pick a spawn point for VU
spawn_point = random.choice(spawn_points)
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
if vehicle is None:
    raise RuntimeError("Could not spawn VU vehicle")
vehicle.set_autopilot(True)
time.sleep(0.5)

# ---------- spawn CoVs ----------
spawned_covs = []  # list of dicts: {"role": name, "actor": actor}
available_spawn_points = [p for p in spawn_points if p != spawn_point]
random.shuffle(available_spawn_points)

cov_blueprints = bp_lib.filter("vehicle.*")
covs_to_spawn = min(NUM_COVS, len(available_spawn_points))
for i in range(covs_to_spawn):
    try:
        bp = random.choice(cov_blueprints)
        # set a role_name so publisher/DT can identify them by name if needed
        role_name = f"cov_{i+1}"
        # many vehicle blueprints allow role_name attribute
        try:
            bp.set_attribute("role_name", role_name)
        except Exception:
            # ignore if blueprint doesn't support it
            pass
        spawn_pt = available_spawn_points[i]
        actor = world.try_spawn_actor(bp, spawn_pt)
        if actor is None:
            # try other spawn points
            for sp in available_spawn_points:
                actor = world.try_spawn_actor(bp, sp)
                if actor is not None:
                    break
        if actor is not None:
            actor.set_autopilot(True)
            spawned_covs.append({"role": role_name, "actor": actor})
            time.sleep(0.1)
    except Exception as e:
        print("Warning: failed to spawn a CoV:", e)

# report VU and CoV mapping
print("=== Spawn summary ===")
print(f"VU vehicle role: 'vu'    id: {vehicle.id}    transform: {vehicle.get_transform().location}")
if spawned_covs:
    for c in spawned_covs:
        a = c["actor"]
        print(f"CoV role: {c['role']}    id: {a.id}    transform: {a.get_transform().location}")
else:
    print("No CoV vehicles spawned (NUM_COVS set to 0 or spawn failed).")
print("======================")

# Make a list of actors we will record in CSV (VU + CoVs)
actors = [vehicle] + [c["actor"] for c in spawned_covs]

# ---------- attach camera to VU ----------
cam_bp = bp_lib.find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', str(IMAGE_W))
cam_bp.set_attribute('image_size_y', str(IMAGE_H))
cam_bp.set_attribute('fov', str(FOV))
cam_transform = carla.Transform(carla.Location(x=1.5, z=1.5), carla.Rotation(pitch=0))
camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
time.sleep(0.5)

# -------------- UDP sockets --------------
# Notify DT that CSV is ready (one-shot)
csv_ready_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
DT_CSV_READY_ADDR = (DT_HOST, DT_CSV_READY_PORT)

# Listen for selected cov metadata from DT
recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
recv_sock.bind((VU_HOST, VU_META_PORT))
recv_sock.settimeout(0.5)

latest_selected_cov_meta = None
meta_lock = threading.Lock()

def meta_listener():
    global latest_selected_cov_meta
    while True:
        try:
            data, addr = recv_sock.recvfrom(200000)
            msg = json.loads(data.decode('utf-8'))
            if msg.get('type') in ('selected_cov_metadata','selection'):
                with meta_lock:
                    latest_selected_cov_meta = msg
                    print("COV Data received from DT:", msg.get('type'), "cov_id:", msg.get('cov_id'), "objects:", len(msg.get('objects', [])) if msg.get('objects') else 0)
        except Exception:
            time.sleep(0.005)

threading.Thread(target=meta_listener, daemon=True).start()

# -------------- CSV writer setup --------------
csv_file = open(CSV_FILE, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["framew_idx", "sim_time", "vehicle_id","type", "x", "y", "z"])  # header

# -------------- helper: camera projection utilities (same as earlier) --------------
def get_camera_intrinsic(image_w: int, image_h: int, fov: float):
    f = image_w / (2.0 * math.tan(math.radians(fov) / 2.0))
    cx = image_w / 2.0
    cy = image_h / 2.0
    return np.array([[f,0,cx],[0,f,cy],[0,0,1]], dtype=np.float32)

def transform_to_matrix(transform):
    loc = transform.location
    rot = transform.rotation
    roll = math.radians(rot.roll); pitch = math.radians(rot.pitch); yaw = math.radians(rot.yaw)
    Rx = np.array([[1,0,0],[0,math.cos(roll),-math.sin(roll)],[0,math.sin(roll),math.cos(roll)]])
    Ry = np.array([[math.cos(pitch),0,math.sin(pitch)],[0,1,0],[-math.sin(pitch),0,math.cos(pitch)]])
    Rz = np.array([[math.cos(yaw),-math.sin(yaw),0],[math.sin(yaw),math.cos(yaw),0],[0,0,1]])
    R = Rz @ Ry @ Rx
    T = np.identity(4, dtype=np.float32)
    T[0:3,0:3] = R
    T[0:3,3] = np.array([loc.x, loc.y, loc.z], dtype=np.float32)
    return T

def world_to_camera_coords(world_point, cam_transform):
    Twc = transform_to_matrix(cam_transform)
    Twc_inv = np.linalg.inv(Twc)
    pw = np.array([world_point.x, world_point.y, world_point.z, 1.0], dtype=np.float32)
    p_cam = Twc_inv @ pw
    return p_cam[0:3]

def project_point(p_cam, K):
    x,y,z = p_cam
    if z <= 0.0001:
        return None
    p = K @ np.array([x,y,z], dtype=np.float32)
    return (float(p[0]/p[2]), float(p[1]/p[2]))

# draw helpers
def draw_actor_bbox_on_frame(frame, camera_sensor, actor, color=(0,255,0), thickness=2, label=None):
    try:
        image_w = int(camera_sensor.attributes.get('image_size_x', IMAGE_W))
        image_h = int(camera_sensor.attributes.get('image_size_y', IMAGE_H))
        fov = float(camera_sensor.attributes.get('fov', FOV))
    except Exception:
        image_w, image_h, fov = IMAGE_W, IMAGE_H, FOV
    K = get_camera_intrinsic(image_w, image_h, fov)
    cam_transform = camera_sensor.get_transform()
    bbox = actor.bounding_box
    ex, ey, ez = bbox.extent.x, bbox.extent.y, bbox.extent.z
    local_corners = [( ex, ey, ez),( ex,-ey, ez),(-ex,-ey, ez),(-ex, ey, ez),( ex, ey,-ez),( ex,-ey,-ez),(-ex,-ey,-ez),(-ex, ey,-ez)]
    actor_T = transform_to_matrix(actor.get_transform())
    bbox_offset_T = transform_to_matrix(carla.Transform(bbox.location))
    full = actor_T @ bbox_offset_T
    projected = []
    for (lx,ly,lz) in local_corners:
        local_p = np.array([lx,ly,lz,1.0], dtype=np.float32)
        world_p = full @ local_p
        p_cam = world_to_camera_coords(carla.Location(x=float(world_p[0]), y=float(world_p[1]), z=float(world_p[2])), cam_transform)
        pr = project_point(p_cam, K)
        if pr is not None:
            projected.append((int(round(pr[0])), int(round(pr[1]))))
    if not projected:
        return frame
    xs = [p[0] for p in projected]; ys = [p[1] for p in projected]
    img_h, img_w = frame.shape[:2]
    x_min = max(0, min(xs)); y_min = max(0, min(ys))
    x_max = min(img_w-1, max(xs)); y_max = min(img_h-1, max(ys))
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)
    if label is None:
        label = f"ID:{actor.id}"
    cv2.putText(frame, label, (x_min, y_min-6 if y_min-6>12 else y_min+12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    return frame

# -------------- frame callback (minimal) --------------
frame_buffer = []
is_recording = True
video_writer = None
frame_idx = 0

# print current actors for debug
print("Recording will include actor ids:", [a.id for a in actors])

def save_image(carla_image):
    global frame_idx, frame_buffer, video_writer, is_recording

    # convert raw CARLA image to BGR
    image_data = np.array(carla_image.raw_data)
    image_2d = image_data.reshape((carla_image.height, carla_image.width, 4))
    rgb_image = image_2d[:, :, :3]
    bgr_image = rgb_image[:, :, ::-1].copy()

    # initialize writer
    if video_writer is None:
        h,w = bgr_image.shape[:2]
        video_writer = cv2.VideoWriter(VIDEO_FILE, cv2.VideoWriter_fourcc(*'MJPG'), 15.0, (w,h))

    # write CSV rows for initial frames
    sim_time = float(carla_image.timestamp)
    if frame_idx < INIT_FRAMES:
        for veh in actors:
            loc = veh.get_location()
            vtype = veh.type_id
            # role_name if present
            role_name = veh.attributes.get('role_name') if hasattr(veh, 'attributes') else ""
            csv_writer.writerow([frame_idx, sim_time, veh.id, vtype, loc.x, loc.y, loc.z])
        # flush+fsync to ensure DT can read
        csv_file.flush()
        try:
            os.fsync(csv_file.fileno())
        except Exception:
            pass

    # overlay latest selected cov metadata if any
    with meta_lock:
        meta = None if latest_selected_cov_meta is None else dict(latest_selected_cov_meta)
    if meta and meta.get('type') == 'selected_cov_metadata':
        # draw objects reported by DT
        cam_tf = camera.get_transform()
        K = get_camera_intrinsic(IMAGE_W, IMAGE_H, FOV)
        objs = meta.get('objects', [])
        for o in objs:
            wp = o.get('world_pos')
            if not wp: continue
            p_cam = world_to_camera_coords(carla.Location(x=wp[0], y=wp[1], z=wp[2]), cam_tf)
            pr = project_point(p_cam, K)
            if pr is None: continue
            u,v = int(round(pr[0])), int(round(pr[1]))
            cv2.circle(bgr_image, (u,v), 6, (0,255,255), -1)
            cv2.putText(bgr_image, f"id:{o.get('object_id')}", (u+6, v+4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255),1)

        # HUD summary
        info = f"Selected CoV: {meta.get('cov_id')} objs:{len(objs)}"
        cv2.putText(bgr_image, info, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # show and buffer
    cv2.namedWindow("VU Dashcam", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("VU Dashcam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("VU Dashcam", bgr_image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        is_recording = False
    frame_buffer.append(bgr_image.copy())

    # after we've written INIT_FRAMES rows, send csv_ready once
    global csv_ready_sent
    if frame_idx == INIT_FRAMES - 1:
        try:
            msg = {"type":"csv_ready", "csv_path": CSV_FILE, "vu_id": int(vehicle.id), "vu_meta_port": VU_META_PORT}
            csv_ready_sock.sendto(json.dumps(msg).encode('utf-8'), DT_CSV_READY_ADDR)
            print("Sent csv_ready to DT:", DT_CSV_READY_ADDR)
        except Exception as e:
            print("Failed to send csv_ready:", e)

    frame_idx += 1

camera.listen(save_image)
print("Recording started (VU). Press 'q' in window to quit.")

# block for a fixed duration or until quit
try:
    while is_recording:
        time.sleep(0.1)
except KeyboardInterrupt:
    pass

# cleanup
is_recording = False
camera.stop()
try:
    csv_file.close()
    if video_writer is not None:
        video_writer.release()
    camera.destroy()
    # destroy spawned CoVs and VU
    for c in spawned_covs:
        try:
            c["actor"].destroy()
        except Exception:
            pass
    try:
        vehicle.destroy()
    except Exception:
        pass
except Exception:
    pass

print("Recorder stopped.")
