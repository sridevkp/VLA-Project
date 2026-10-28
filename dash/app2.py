import base64
import json
import threading
import time
from datetime import datetime

import requests
import streamlit as st

st.set_page_config(page_title="TIAGo AI Vision Console", page_icon="🤖", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:.6rem;padding-bottom:.6rem} [data-testid="stVerticalBlock"] {gap:.55rem}
[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.22);padding:10px;border-radius:12px}
</style>
""", unsafe_allow_html=True)


def make_state():
    return {
        "lock": threading.Lock(), "wake": threading.Event(),
        "stop": threading.Event(), "thread": None, "running": False,
        "config": {}, "camera": None, "telemetry": None, "snapshot": None,
        "output": "Ready. Press Start to send camera frames to the VLM.",
        "error": "", "last_request": None, "latency": None,
        "count": 0, "status": "Idle",
    }


def fetch_observation(base, timeout):
    base = base.rstrip("/")
    r = requests.get(f"{base}/api/camera.jpg", timeout=timeout)
    r.raise_for_status()
    image = r.content
    if not image:
        raise RuntimeError("Camera endpoint returned an empty image.")
    r = requests.get(f"{base}/api/telemetry", timeout=timeout)
    r.raise_for_status()
    telemetry = r.json()
    try:
        r = requests.get(f"{base}/api/snapshot", timeout=timeout)
        r.raise_for_status()
        snapshot = r.json()
    except Exception as e:
        snapshot = {"warning": f"Snapshot unavailable: {e}"}
    return image, telemetry, snapshot


def send_to_vlm(cfg, prompt, image, telemetry, snapshot):
    # Matches the supplied JS: {baseURL}/v1/chat/completions and image_url data URL.
    base = cfg["model_base_url"].strip().rstrip("/")
    endpoint = base if base.endswith("/v1/chat/completions") else base + "/v1/chat/completions"
    image_url = "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")
    context = (
        prompt.strip()
        + "\n\nTIAGo telemetry JSON:\n" + json.dumps(telemetry, ensure_ascii=False)
        + "\n\nSnapshot metadata JSON:\n" + json.dumps(snapshot, ensure_ascii=False)
        + "\n\nAnalyze the image and telemetry. State uncertainty clearly. "
          "Do not claim to have moved the robot."
    )
    headers = {"Content-Type": "application/json"}
    if cfg["api_key"].strip():
        headers["Authorization"] = "Bearer " + cfg["api_key"].strip()
    payload = {
        "model": cfg["model_name"].strip(),
        "max_tokens": int(cfg["max_tokens"]),
        "stream": False,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": context},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }],
    }
    r = requests.post(endpoint, headers=headers, json=payload, timeout=int(cfg["timeout"]))
    if not r.ok:
        raise RuntimeError(f"VLM HTTP {r.status_code}: {r.text[:1500]}")
    data = r.json()
    try:
        answer = data["choices"][0]["message"]["content"]
        if isinstance(answer, list):
            return "\n".join(
                str(x.get("text", x)) if isinstance(x, dict) else str(x)
                for x in answer
            )
        return str(answer)
    except (KeyError, IndexError, TypeError):
        return "Unexpected model response:\n" + json.dumps(data, ensure_ascii=False, indent=2)


def worker_loop(s):
    while not s["stop"].is_set():
        with s["lock"]:
            running, cfg = s["running"], dict(s["config"])
        if not running:
            s["wake"].wait(0.2)
            s["wake"].clear()
            continue

        prompt = cfg.get("prompt", "").strip()
        if not prompt:
            with s["lock"]:
                s["error"] = "Enter a prompt before starting."
                s["status"] = "Waiting for prompt"
                s["running"] = False
            continue

        started = time.monotonic()
        try:
            image, telemetry, snapshot = fetch_observation(
                cfg["robot_base_url"], min(10, int(cfg["timeout"]))
            )
            with s["lock"]:
                s["camera"], s["telemetry"], s["snapshot"] = image, telemetry, snapshot
                s["status"] = "Sending image + telemetry"
            answer = send_to_vlm(cfg, prompt, image, telemetry, snapshot)
            elapsed = time.monotonic() - started
            with s["lock"]:
                s["output"] = answer
                s["error"] = ""
                s["last_request"] = datetime.now().strftime("%H:%M:%S")
                s["latency"] = elapsed
                s["count"] += 1
                s["status"] = "Connected"
        except Exception as e:
            with s["lock"]:
                s["error"] = f"{type(e).__name__}: {e}"
                s["last_request"] = datetime.now().strftime("%H:%M:%S")
                s["latency"] = time.monotonic() - started
                s["status"] = "Request failed"

        with s["lock"]:
            running = s["running"]
            interval = int(s["config"].get("interval_ms", 1000)) / 1000
        if running:
            # Sequential calls only: no overlapping inference requests.
            s["stop"].wait(max(0.0, interval - (time.monotonic() - started)))


if "vlm_worker" not in st.session_state:
    st.session_state.vlm_worker = make_state()
worker = st.session_state.vlm_worker
if worker["thread"] is None or not worker["thread"].is_alive():
    worker["stop"].clear()
    worker["thread"] = threading.Thread(target=worker_loop, args=(worker,), daemon=True)
    worker["thread"].start()

# Sidebar
st.sidebar.title("🤖 TIAGo AI")
st.sidebar.caption("VISION • TELEMETRY • VLA WORKBENCH")
st.sidebar.divider()
robot_base = st.sidebar.text_input(
    "Robot API base URL", "http://127.0.0.1:8765"
).strip().rstrip("/")
model_base = st.sidebar.text_input(
    "VLM base URL", "http://127.0.0.1:8080",
    help="Enter the base URL; the app appends /v1/chat/completions."
).strip().rstrip("/")
model_name = st.sidebar.text_input("Model name", "local-model")
api_key = st.sidebar.text_input("API key (optional)", type="password")
timeout = st.sidebar.number_input("Request timeout (seconds)", 5, 600, 120)
max_tokens = st.sidebar.number_input("Maximum response tokens", 16, 2048, 150, step=16)
interval_ms = st.sidebar.select_slider(
    "Frame / inference interval",
    options=[100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000],
    value=1000,
    format_func=lambda n: f"{n} ms" + (" (10 Hz target)" if n == 100 else ""),
)

st.title("TIAGo / AI Vision Console")
st.caption("LIVE CAMERA  /  ROBOT TELEMETRY  /  VISION-LANGUAGE INTERACTION")
prompt = st.session_state.get("vlm_prompt", "What do you see?")

with worker["lock"]:
    worker["config"] = {
        "robot_base_url": robot_base, "model_base_url": model_base,
        "model_name": model_name, "api_key": api_key, "timeout": int(timeout),
        "max_tokens": int(max_tokens), "interval_ms": int(interval_ms), "prompt": prompt,
    }

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    start_clicked = st.button("▶ Start live VLM", type="primary",
                              use_container_width=True, disabled=worker["running"])
with c2:
    stop_clicked = st.button("■ Stop", use_container_width=True,
                             disabled=not worker["running"])
with c3:
    st.caption(f"Target interval: **{interval_ms} ms**. Slow inference lowers the actual rate.")

if start_clicked:
    with worker["lock"]:
        worker["running"] = True
        worker["error"] = ""
        worker["status"] = "Starting"
    worker["wake"].set()
    st.rerun()
if stop_clicked:
    with worker["lock"]:
        worker["running"] = False
        worker["status"] = "Stopped"
    worker["wake"].set()
    st.rerun()

# Check robot health separately from VLM connectivity.
h1, h2, h3 = st.columns(3)
try:
    r = requests.get(f"{robot_base}/health", timeout=2)
    r.raise_for_status()
    h1.success("Robot API: connected")
except Exception as e:
    h1.error(f"Robot API: {e}")
with worker["lock"]:
    running_now, status, count = worker["running"], worker["status"], worker["count"]
if status == "Connected":
    h2.success("VLM: responding")
elif running_now:
    h2.info(f"VLM: {status}")
else:
    h2.caption("VLM: not streaming")
h3.metric("Completed inferences", count)


@st.fragment(run_every=0.5)
def live_display():
    with worker["lock"]:
        camera, telemetry, snapshot = worker["camera"], worker["telemetry"], worker["snapshot"]
        output, error = worker["output"], worker["error"]
        request_time, latency, running, status = (
            worker["last_request"], worker["latency"], worker["running"], worker["status"]
        )

    # Compact 2x2 workspace: all four key panels remain in the same viewport.
    left, right = st.columns([1.15, 1], gap="medium")

    with left:
        st.subheader("Live camera")
        if camera:
            st.image(camera, caption="Latest frame sent to VLM", use_container_width=True)
        else:
            try:
                r = requests.get(f"{robot_base}/api/camera.jpg", timeout=2)
                r.raise_for_status()
                st.image(r.content, use_container_width=True)
            except Exception as e:
                st.warning(f"Camera preview unavailable: {e}")

        st.subheader("Prompt")
        prompt_value = st.text_area(
            "Instruction sent with every image",
            value=st.session_state.get("vlm_prompt", "What do you see?"),
            key="vlm_prompt", height=82,
            placeholder="Ask about the current Webots camera frame...",
            label_visibility="collapsed",
        )
        with worker["lock"]:
            worker["config"]["prompt"] = prompt_value

    with right:
        st.subheader("Robot telemetry")
        if telemetry is not None:
            st.json(telemetry, expanded=False)
        else:
            try:
                r = requests.get(f"{robot_base}/api/telemetry", timeout=2)
                r.raise_for_status()
                st.json(r.json(), expanded=False)
            except Exception as e:
                st.warning(f"Telemetry unavailable: {e}")

        if snapshot is not None:
            with st.expander("Camera / telemetry timestamps"):
                st.json(snapshot, expanded=False)

        st.subheader("Action preview")
        a, b, c = st.columns(3)
        a.metric("Stream", "RUNNING" if running else "STOPPED")
        b.metric("Inference", f"{latency:.2f}s" if latency is not None else "—")
        c.metric("Last request", request_time or "—")
        if error:
            st.error(error)
        st.markdown(output)


live_display()
st.caption(
    "The app sends the Webots JPEG, prompt, and telemetry to the VLM. "
    "It does not execute robot motor commands."
)
