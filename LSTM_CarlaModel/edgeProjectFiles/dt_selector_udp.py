#!/usr/bin/env python3
"""
dt_selector_udp.py

Listens for 'csv_ready' from VU, invokes the selector script with --vu-id,
listens for cov_detections from publisher, forwards selected cov metadata to VU.

This variant will RETRY the selector repeatedly (indefinitely) until it returns
a cov_id. It prints stdout/stderr and uses exponential backoff between attempts.
"""
import socket, json, time, subprocess, threading, os, re
from collections import defaultdict, deque
import argparse

# optional psutil for robust process killing
try:
    import psutil
except Exception:
    psutil = None

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="0.0.0.0", help="DT bind host")
parser.add_argument("--csv-port", type=int, default=7010, help="VU -> DT csv_ready port")
parser.add_argument("--cov-port", type=int, default=7007, help="CoV -> DT detections port")
parser.add_argument("--vu-host", default="127.0.0.1", help="DT -> VU host")
parser.add_argument("--vu-port", type=int, default=6008, help="DT -> VU port")
parser.add_argument("--publisher-control-host", default="127.0.0.1", help="publisher control host")
parser.add_argument("--publisher-control-port", type=int, default=7012, help="publisher control port")
parser.add_argument("--selector-script", default="select_cov_using_trajectories.py", help="path to selector script")
parser.add_argument("--selector-output-csv", default="drl_cov_selection_result.csv", help="selector output file")
parser.add_argument("--selector-timeout", type=int, default=60, help="base selector timeout seconds (per attempt)")
parser.add_argument("--selector-backoff-base", type=float, default=2.0, help="exponential backoff base seconds")
parser.add_argument("--selector-max-backoff", type=float, default=60.0, help="max backoff seconds between attempts")
args = parser.parse_args()

DT_HOST = args.host
CSV_READY_PORT = args.csv_port
COV_PORT = args.cov_port
VU_HOST = args.vu_host
VU_META_PORT = args.vu_port
VU_ADDR = (VU_HOST, VU_META_PORT)
PUBLISHER_CONTROL_ADDR = (args.publisher_control_host, args.publisher_control_port)
SELECTOR_SCRIPT = args.selector_script
SELECTOR_OUTPUT_CSV = args.selector_output_csv
SELECTOR_TIMEOUT_BASE = args.selector_timeout
RETRY_BACKOFF_BASE = args.selector_backoff_base
MAX_BACKOFF = args.selector_max_backoff

SHORT_SLEEP = 0.01

# sockets
csv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
csv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
csv_sock.bind(("0.0.0.0", CSV_READY_PORT))
csv_sock.settimeout(0.5)

cov_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
cov_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
cov_sock.bind(("0.0.0.0", COV_PORT))
cov_sock.settimeout(0.5)

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"[DT] listening csv_ready on 0.0.0.0:{CSV_READY_PORT}, cov on 0.0.0.0:{COV_PORT}")
print(f"[DT] will forward selected metadata to VU at {VU_ADDR} and control publisher at {PUBLISHER_CONTROL_ADDR}")
print(f"[DT] selector script: {SELECTOR_SCRIPT}, selector output csv: {SELECTOR_OUTPUT_CSV}")

# state
cov_detection_buffers = defaultdict(lambda: deque(maxlen=16))
cov_lock = threading.Lock()
selected_cov_id = None
selected_lock = threading.Lock()

vu_id_runtime = None
vu_id_lock = threading.Lock()

# helper parse int from text
def parse_first_int(text):
    m = re.search(r"(-?\d+)", text)
    return int(m.group(1)) if m else None

def kill_proc_tree(proc):
    """
    Kill a subprocess and its children. Uses psutil if available, falls back to proc.kill().
    """
    try:
        if psutil is not None:
            p = psutil.Process(proc.pid)
            for child in p.children(recursive=True):
                try:
                    child.kill()
                except Exception:
                    pass
            p.kill()
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

def _read_selector_output_csv():
    """
    Try to read SELECTOR_OUTPUT_CSV and return an int cov_id or None.
    """
    try:
        if os.path.exists(SELECTOR_OUTPUT_CSV):
            import pandas as pd
            df = pd.read_csv(SELECTOR_OUTPUT_CSV)
            if not df.empty and 'cov_id' in df.columns:
                val = df.iloc[-1]['cov_id']
                if pd.notna(val):
                    return int(val) if val is not None else None
    except Exception as e:
        print("[DT] error reading selector output csv:", e)
    return None

def run_selector_on_csv(csv_path):
    """
    Robust invocation of SELECTOR_SCRIPT. Will retry indefinitely until a cov_id is obtained.
    Strategy:
      - Build cmd: python SELECTOR_SCRIPT --csv <csv_path> [--vu-id <vu_id_runtime>]
      - Attempt to run it, wait up to a per-attempt timeout (increasing slightly each attempt).
      - If the process times out, kill it (and children) and retry after exponential backoff.
      - If the process finishes, check SELECTOR_OUTPUT_CSV then stdout for an integer.
      - Continue until a cov_id is found (no hard retry cap).
    """
    global SELECTOR_SCRIPT, SELECTOR_OUTPUT_CSV, SELECTOR_TIMEOUT_BASE
    if not os.path.exists(csv_path):
        print("[DT] run_selector_on_csv: csv not found:", csv_path)
        return None

    # build base command, include runtime VU id if available
    with vu_id_lock:
        vu_arg = vu_id_runtime
    base_cmd = ["python", SELECTOR_SCRIPT, "--csv", csv_path]
    if vu_arg is not None:
        base_cmd += ["--vu-id", str(vu_arg)]

    attempt = 0
    while True:
        attempt += 1
        # slightly increase timeout as attempts go on
        timeout = SELECTOR_TIMEOUT_BASE * (1.0 + 0.25 * (attempt - 1))
        # cap the timeout to a reasonable upper bound (optional)
        if timeout > SELECTOR_TIMEOUT_BASE * 5:
            timeout = SELECTOR_TIMEOUT_BASE * 5

        print(f"[DT] selector attempt #{attempt} cmd={' '.join(base_cmd)} timeout={timeout:.1f}s")
        try:
            proc = subprocess.Popen(base_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                print(f"[DT] selector attempt #{attempt} timed out after {timeout:.1f}s -> killing selector")
                try:
                    kill_proc_tree(proc)
                except Exception as e:
                    print("[DT] error killing selector proc:", e)
                # small wait and then backoff/retry
                backoff = min(RETRY_BACKOFF_BASE ** attempt, MAX_BACKOFF)
                print(f"[DT] selector will retry after backoff {backoff:.1f}s")
                time.sleep(backoff)
                continue
            # proc finished within timeout
            if stderr:
                print(f"[DT] selector stderr (attempt {attempt}):\n{stderr.strip()[:2000]}")
            if stdout:
                print(f"[DT] selector stdout (attempt {attempt}):\n{stdout.strip()[:2000]}")

            # first check selector output CSV
            chosen = _read_selector_output_csv()
            if chosen is not None:
                print(f"[DT] selector wrote output csv and chose cov_id={chosen}")
                return int(chosen)

            # fallback: parse stdout for an integer
            if stdout:
                parsed = parse_first_int(stdout)
                if parsed is not None:
                    print(f"[DT] parsed cov_id={parsed} from selector stdout")
                    return int(parsed)

            # if we reach here, selector finished but returned no cov
            print(f"[DT] selector attempt {attempt} finished but produced no cov_id (returncode={proc.returncode})")
            # apply backoff then retry
            backoff = min(RETRY_BACKOFF_BASE ** attempt, MAX_BACKOFF)
            print(f"[DT] will retry selector after backoff {backoff:.1f}s")
            time.sleep(backoff)
            continue

        except FileNotFoundError as e:
            print("[DT] selector script or python not found:", e)
            # this is fatal — no point retrying until environment is fixed
            return None
        except Exception as e:
            print("[DT] selector invocation exception:", e)
            backoff = min(RETRY_BACKOFF_BASE ** attempt, MAX_BACKOFF)
            print(f"[DT] unexpected error, retrying after backoff {backoff:.1f}s")
            time.sleep(backoff)
            continue

def csv_ready_listener():
    global selected_cov_id, vu_id_runtime
    print("[DT] csv_ready listener started")
    while True:
        try:
            data, addr = csv_sock.recvfrom(8192)
        except Exception:
            time.sleep(SHORT_SLEEP)
            continue
        try:
            msg = json.loads(data.decode('utf-8'))
        except Exception:
            print("[DT] csv_ready_listener: bad JSON")
            continue
        if msg.get('type') != 'csv_ready':
            continue
        csv_path = msg.get('csv_path')
        runtime_vu = msg.get('vu_id')
        print("[DT] received csv_ready:", csv_path, "vu_id:", runtime_vu)
        # store vu_id_runtime
        with vu_id_lock:
            vu_id_runtime = int(runtime_vu) if runtime_vu is not None else None

        if csv_path is None or not os.path.exists(csv_path):
            print("[DT] csv path missing or not found:", csv_path)
            continue

        # run selector (this will RETRY until a cov_id is returned or fatal error)
        chosen = run_selector_on_csv(csv_path)
        if chosen is None:
            print("[DT] selector failed to choose a cov (fatal). Skipping this csv_ready.")
            continue

        with selected_lock:
            selected_cov_id = int(chosen)
        print("[DT] selector chose cov_id =", selected_cov_id)

        # send control to publisher
        try:
            # include expected pose if we have last detection in buffer
            expected_pose = None
            with cov_lock:
                buf = cov_detection_buffers.get(selected_cov_id, None)
                if buf and len(buf):
                    last = buf[-1]
                    dets = last.get('detections', [])
                    if dets:
                        wp = dets[0].get('world_pos', None)
                        if wp and len(wp) >= 3:
                            expected_pose = {"x": float(wp[0]), "y": float(wp[1]), "z": float(wp[2])}
            ctrl_msg = {"type":"set_target", "cov_id": selected_cov_id, "expected_pose": expected_pose}
            ctrl_sock.sendto(json.dumps(ctrl_msg).encode('utf-8'), PUBLISHER_CONTROL_ADDR)
            print("[DT] sent set_target to publisher:", PUBLISHER_CONTROL_ADDR, "expected_pose:", expected_pose)
        except Exception as e:
            print("[DT] failed to send set_target:", e)

        # notify VU of selection
        try:
            notif = {"type":"selection","cov_id": selected_cov_id, "chosen_source":"selector", "timestamp": time.time()}
            send_sock.sendto(json.dumps(notif).encode('utf-8'), VU_ADDR)
        except Exception as e:
            print("[DT] failed to notify VU:", e)

def cov_listener_and_forwarder():
    print("[DT] cov_listener started on port", COV_PORT)
    while True:
        try:
            data, addr = cov_sock.recvfrom(200000)
        except Exception:
            time.sleep(SHORT_SLEEP)
            continue
        print("[DT] recvfrom", addr, "bytes=", len(data))
        try:
            msg = json.loads(data.decode('utf-8'))
        except Exception:
            print("[DT] received invalid JSON from cov")
            continue
        if msg.get('type') != 'cov_detections':
            continue
        try:
            cid = int(msg.get('cov_id'))
        except Exception:
            continue
        with cov_lock:
            cov_detection_buffers[cid].append(msg)
        print(f"[DT] received cov_detections frame={msg.get('frame_idx')} cov_id={cid} detections={len(msg.get('detections',[]))}")
        with selected_lock:
            sel = selected_cov_id
        if sel is not None and cid == sel:
            objects = msg.get('detections', [])
            meta = {"type":"selected_cov_metadata", "cov_id": cid, "frame_idx": msg.get("frame_idx"), "timestamp": time.time(), "objects": objects}
            try:
                send_sock.sendto(json.dumps(meta).encode('utf-8'), VU_ADDR)
                print(f"[DT] forwarded selected_cov_metadata cov_id={cid} -> VU {VU_ADDR}")
            except Exception as e:
                print("[DT] forward to VU failed:", e)

if __name__ == "__main__":
    t1 = threading.Thread(target=csv_ready_listener, daemon=True)
    t2 = threading.Thread(target=cov_listener_and_forwarder, daemon=True)
    t1.start(); t2.start()
    print("[DT] running. Waiting for csv_ready and cov detections...")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[DT] stopped.")
