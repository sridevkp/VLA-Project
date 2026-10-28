
import json
import time
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="TIAGo | AI Robotics",
    page_icon="🤖",
    layout="wide",
)

# ---------- Styling ----------
st.markdown("""
<style>
    .block-container {padding-top: 1.4rem; padding-bottom: 1rem;}
    .metric-card {
        padding: 14px;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 12px;
    }
    .small-muted {font-size: 0.8rem; opacity: 0.7;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.2);
        padding: 12px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Session state ----------
if "vlm_output" not in st.session_state:
    st.session_state.vlm_output = "Waiting for a prompt..."
if "history" not in st.session_state:
    st.session_state.history = []
if "prompt" not in st.session_state:
    st.session_state.prompt = ""

# ---------- Sidebar ----------
st.sidebar.title("🤖 TIAGo AI")
st.sidebar.caption("ROBOTICS DEVELOPMENT CONSOLE")
st.sidebar.divider()

robot_url = st.sidebar.text_input(
    "Robot controller URL",
    value="http://127.0.0.1:8765",
    help="Your Webots controller HTTP server",
).rstrip("/")

st.sidebar.subheader("Model connection")
model_url = st.sidebar.text_input(
    "VLA / Llama API URL",
    value="http://127.0.0.1:8000/v1/chat/completions",
    help="Example for an OpenAI-compatible server. Change to your model's actual endpoint.",
)
model_name = st.sidebar.text_input(
    "Model name",
    value="local-model",
)
api_key = st.sidebar.text_input(
    "API key (optional)",
    type="password",
)
model_timeout = st.sidebar.number_input(
    "Model timeout (seconds)",
    min_value=5,
    max_value=300,
    value=60,
)

st.sidebar.divider()
auto_refresh = st.sidebar.toggle("Live telemetry refresh", value=True)
refresh_seconds = st.sidebar.slider(
    "Refresh interval (seconds)", 1, 10, 2
)

# ---------- Header ----------
st.title("TIAGo / AI Robotics Dashboard")
st.caption(
    "SIMULATION MONITOR  /  VISION-LANGUAGE INTERFACE  /  ROBOT STATE"
)

# ---------- Endpoint helpers ----------
def get_json(endpoint):
    try:
        response = requests.get(
            f"{robot_url}{endpoint}",
            timeout=2,
        )
        response.raise_for_status()
        return response.json(), None
    except Exception as exc:
        return None, str(exc)


def extract_text(data):
    """Support common telemetry JSON structures."""
    if not isinstance(data, dict):
        return {"value": data}

    # Prefer a nested robot-state object if present.
    for key in ("telemetry", "robot_state", "state", "robot"):
        if isinstance(data.get(key), dict):
            return data[key]

    return data


# ---------- Connection status ----------
health, health_error = get_json("/health")
telemetry, telemetry_error = get_json("/api/telemetry")
snapshot, snapshot_error = get_json("/api/snapshot")

col_status, col_time, col_model = st.columns([1, 1, 2])

with col_status:
    if health_error:
        st.error("Robot API offline")
    else:
        st.success("Robot API connected")

with col_time:
    st.metric(
        "Dashboard time",
        datetime.now().strftime("%H:%M:%S"),
    )

with col_model:
    st.caption("MODEL ENDPOINT")
    st.code(model_url, language=None)

st.divider()

# ---------- Main display ----------
left, right = st.columns([1.15, 1])

with left:
    st.subheader("Live camera")
    camera_placeholder = st.empty()

    try:
        image_response = requests.get(
            f"{robot_url}/api/camera.jpg",
            timeout=3,
        )
        image_response.raise_for_status()
        camera_placeholder.image(
            image_response.content,
            caption="Latest frame from TIAGo camera",
            use_container_width=True,
        )
    except Exception as exc:
        camera_placeholder.warning(
            f"Camera unavailable: {exc}"
        )

    if isinstance(snapshot, dict):
        st.caption(
            "Snapshot timestamps: "
            + json.dumps(snapshot, ensure_ascii=False)
        )

with right:
    st.subheader("Robot telemetry")

    if telemetry_error:
        st.warning(f"Unable to read telemetry: {telemetry_error}")
    else:
        state = extract_text(telemetry)

        if isinstance(state, dict):
            # Show numeric values as metric cards.
            numeric_items = [
                (key, value)
                for key, value in state.items()
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
            ]

            if numeric_items:
                metric_cols = st.columns(2)
                for index, (key, value) in enumerate(numeric_items[:8]):
                    with metric_cols[index % 2]:
                        st.metric(
                            label=key.replace("_", " ").title(),
                            value=f"{value:.4f}"
                            if isinstance(value, float)
                            else value,
                        )

            with st.expander("Full robot-state JSON", expanded=True):
                st.json(telemetry)
        else:
            st.json(telemetry)

st.divider()

# ---------- Prompt and model interface ----------
st.subheader("Vision-language interaction")
st.caption(
    "Describe the scene or ask the model to reason about the robot's task. "
    "This sends a request only to the configured model endpoint."
)

with st.form("prompt_form", clear_on_submit=False):
    prompt = st.text_area(
        "Instruction",
        key="prompt",
        placeholder=(
            "Example: Describe what the camera sees and identify "
            "a safe grasping approach for the object in front of TIAGo."
        ),
        height=100,
    )

    submitted = st.form_submit_button(
        "Send to model",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not prompt.strip():
        st.warning("Enter a prompt first.")
    else:
        with st.spinner("Waiting for model response..."):
            try:
                headers = {"Content-Type": "application/json"}
                if api_key.strip():
                    headers["Authorization"] = f"Bearer {api_key.strip()}"

                # Attach current robot state as context.
                context = {
                    "instruction": prompt.strip(),
                    "robot_telemetry": telemetry,
                    "telemetry_snapshot": snapshot,
                    "timestamp": datetime.now().isoformat(),
                }

                # OpenAI-compatible chat-completions request.
                payload = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are an assistant for a simulated TIAGo "
                                "mobile manipulator. Analyze robot telemetry "
                                "and instructions. Explain your reasoning "
                                "concisely. Do not claim to have moved the "
                                "robot. Treat all robot actions as proposals "
                                "unless an explicit, safe control interface "
                                "is separately implemented."
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(context),
                        },
                    ],
                    "temperature": 0.2,
                }

                response = requests.post(
                    model_url,
                    headers=headers,
                    json=payload,
                    timeout=int(model_timeout),
                )
                response.raise_for_status()
                result = response.json()

                # Standard OpenAI-compatible response.
                output = result["choices"][0]["message"]["content"]

                st.session_state.vlm_output = output
                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "prompt": prompt.strip(),
                    "output": output,
                })

            except Exception as exc:
                st.session_state.vlm_output = (
                    f"Model request failed: {exc}\n\n"
                    "Check the URL, server availability, model name, "
                    "and API format."
                )

st.subheader("Model output")
st.markdown(st.session_state.vlm_output)

with st.expander("Interaction history"):
    if not st.session_state.history:
        st.caption("No interactions yet.")
    else:
        for item in reversed(st.session_state.history[-10:]):
            st.markdown(f"**[{item['time']}] You:** {item['prompt']}")
            st.markdown(f"**Model:** {item['output']}")
            st.divider()

st.caption(
    "Prototype dashboard · Model output is advisory; robot motion is not "
    "executed by this app."
)

# ---------- Refresh ----------
if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
