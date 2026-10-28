import io, os, time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import torch

ROBOT_API = os.getenv("ROBOT_API_URL", "http://127.0.0.1:8765").rstrip("/")
HTTP_TIMEOUT = float(os.getenv("ROBOT_HTTP_TIMEOUT", "3"))

st.set_page_config(page_title="SmolVLA Action Studio", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.block-container{max-width:1500px;padding-top:1.15rem}
[data-testid="stAppViewContainer"]{background:#0a0e14}
[data-testid="stSidebar"]{background:#0d131c}
.hero{padding:1.15rem 1.4rem;border:1px solid #263447;border-radius:18px;background:linear-gradient(120deg,#101d2b,#10151f 70%,#10241f);margin-bottom:1rem}
.eyebrow{color:#7dd3fc;font-size:.76rem;font-weight:700;letter-spacing:.16em}
.muted{color:#91a0b4;font-size:.9rem}
div[data-testid="stMetric"]{background:#0e141d;border:1px solid #263447;padding:.8rem;border-radius:12px}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_policy(model_id, device_name):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    device = torch.device(device_name)
    policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocess, postprocess, device

def get_camera():
    r = requests.get(f"{ROBOT_API}/api/camera.jpg", timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    im = Image.open(io.BytesIO(r.content)).convert("RGB")
    im.load()
    return im, r.headers.get("Last-Modified")

def get_json(path):
    r = requests.get(f"{ROBOT_API}{path}", timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()

def flatten(o, prefix=""):
    out=[]
    if isinstance(o,dict):
        for k,v in o.items(): out += flatten(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(o,list):
        for i,v in enumerate(o): out += flatten(v,f"{prefix}[{i}]")
    else: out.append((prefix,o))
    return out

def predict(policy, preprocess, postprocess, device, image, task, state_values):
    image_tensor = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.uint8).copy()).permute(2,0,1)
    # Demo-only: the SmolVLA base checkpoint declares three camera inputs.
    # Repeat the current controller frame in each slot; this is not true multi-camera input.
    frame = {
        "observation.state": torch.tensor(state_values, dtype=torch.float32),
        "observation.images.camera1": image_tensor,
        "observation.images.camera2": image_tensor.clone(),
        "observation.images.camera3": image_tensor.clone(),
        "task": task,
    }
    batch = preprocess(frame)
    with torch.inference_mode():
        action = postprocess(policy.predict_action_chunk(batch))
    if not isinstance(action, torch.Tensor):
        raise TypeError(f"Expected tensor action output, got {type(action).__name__}")
    action = action.detach().float().cpu()
    return action.squeeze(0).numpy(), list(action.shape)

with st.sidebar:
    st.markdown("## 🧠 SmolVLA Action Studio")
    st.caption("Live camera · Local model inference")
    robot_api = st.text_input("Robot read-only API", ROBOT_API, help="The app only GETs the listed endpoints. It never sends commands to Webots.")
    st.caption("Endpoints used")
    st.code("/health\n/api/telemetry\n/api/camera.jpg\n/api/snapshot")
    st.warning("READ-ONLY OBSERVATION. No robot actions are executed.")
    model_id = st.text_input("Model ID", "lerobot/smolvla_base")
    device_choice = st.selectbox("Inference device", ["cpu", "cuda"], index=0)
    refresh = st.slider("Camera refresh (seconds)", 1, 10, 2)

st.markdown('<div class="hero"><div class="eyebrow">EMBODIED AI · LIVE OBSERVATION</div><h1>SmolVLA Action Studio</h1><div class="muted">Robot camera + task prompt + demo state → real pretrained model action chunk</div></div>', unsafe_allow_html=True)

if "prediction" not in st.session_state: st.session_state.prediction=None
if "latest_frame" not in st.session_state: st.session_state.latest_frame=None
if "latest_telemetry" not in st.session_state: st.session_state.latest_telemetry=None
if "latest_snapshot" not in st.session_state: st.session_state.latest_snapshot=None
if "camera_error" not in st.session_state: st.session_state.camera_error=None

@st.fragment(run_every=refresh)
def live_observation():
    global robot_api
    a,b=st.columns([1.15,.85],gap="large")
    frame=None
    try:
        h=requests.get(f"{robot_api.rstrip('/')}/health",timeout=HTTP_TIMEOUT)
        online=h.ok
        health_detail=h.text[:300]
    except Exception as e:
        online=False
        health_detail=str(e)
    try:
        r=requests.get(f"{robot_api.rstrip('/')}/api/camera.jpg",timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        frame=Image.open(io.BytesIO(r.content)).convert("RGB")
        frame.load()
        st.session_state.latest_frame=frame
        st.session_state.camera_error=None
    except Exception as e:
        st.session_state.camera_error=f"{type(e).__name__}: {e}"
    try:
        r=requests.get(f"{robot_api.rstrip('/')}/api/telemetry",timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        st.session_state.latest_telemetry=r.json()
        st.session_state.telemetry_error=None
    except Exception as e:
        st.session_state.telemetry_error=f"{type(e).__name__}: {e}"
    try:
        r=requests.get(f"{robot_api.rstrip('/')}/api/snapshot",timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        st.session_state.latest_snapshot=r.json()
    except Exception as e:
        st.session_state.snapshot_error=f"{type(e).__name__}: {e}"

    with a:
        st.subheader("01 / Live robot camera")
        if frame is not None:
            st.image(frame,caption=f"Fetched from {robot_api}/api/camera.jpg · refreshed every {refresh}s",use_container_width=True)
            st.caption(f"Dashboard fetch time: {time.strftime('%H:%M:%S')}")
        else:
            st.error("Could not fetch the camera image.")
            st.caption(st.session_state.camera_error or "Check that the Webots controller server is running.")
    with b:
        st.subheader("02 / Controller telemetry")
        c1,c2=st.columns(2)
        c1.metric("Controller API","ONLINE" if online else "OFFLINE")
        c2.metric("Camera feed","LIVE" if frame is not None else "NO IMAGE")
        if st.session_state.latest_snapshot is not None:
            with st.expander("Camera / telemetry timestamps",expanded=True):
                st.json(st.session_state.latest_snapshot)
        if st.session_state.latest_telemetry is not None:
            values=[(k,v) for k,v in flatten(st.session_state.latest_telemetry) if isinstance(v,(int,float)) and not isinstance(v,bool)]
            if values:
                cols=st.columns(2)
                for i,(key,val) in enumerate(values[:16]):
                    cols[i%2].metric(key.split(".")[-1],f"{val:.4g}" if isinstance(val,float) else str(val))
            with st.expander("Full telemetry JSON"): st.json(st.session_state.latest_telemetry)
        else:
            st.warning("Telemetry unavailable.")
            st.caption(getattr(st.session_state,"telemetry_error",""))
live_observation()

st.divider()
st.subheader("03 / Instruction")
task=st.text_area("Task prompt", "Pick up the object and move it toward the target area.", height=85)
st.subheader("04 / Demo state vector")
st.caption("These six numbers remain synthetic placeholders for the SmolVLA base checkpoint. Live telemetry is displayed above but is not automatically mapped into the model state.")
cols=st.columns(6)
state_values=[]
for i in range(6):
    with cols[i]:
        state_values.append(st.number_input(f"state[{i}]",value=0.0,format="%.3f",key=f"demo_state_{i}"))

if st.button("Load model and run inference",type="primary",use_container_width=True):
    frame=st.session_state.latest_frame
    if frame is None:
        st.error("No camera frame available. Check the robot API address and /api/camera.jpg.")
    elif not task.strip():
        st.warning("Enter a task instruction.")
    elif device_choice=="cuda" and not torch.cuda.is_available():
        st.error("CUDA is selected but unavailable in this PyTorch installation. Select CPU or install compatible CUDA-enabled PyTorch.")
    else:
        try:
            with st.spinner("Loading pretrained SmolVLA weights; first run may take several minutes…"):
                policy,preprocess,postprocess,device=load_policy(model_id,device_choice)
            with st.spinner("Running inference on the latest controller image…"):
                start=time.perf_counter()
                actions,shape=predict(policy,preprocess,postprocess,device,frame,task,state_values)
                elapsed=time.perf_counter()-start
            st.session_state.prediction={
                "actions":actions.tolist(),"shape":shape,"seconds":elapsed,
                "device":str(device),"model_id":model_id,"task":task,
                "camera_size":[frame.width,frame.height],
            }
        except Exception as e:
            st.error(f"Inference failed: {type(e).__name__}: {e}")
            st.exception(e)

st.subheader("05 / Actual model action preview")
p=st.session_state.prediction
if p is None:
    st.info("Wait for a camera frame, enter a prompt, and run inference. The real model action output will appear here.")
else:
    m1,m2,m3=st.columns(3)
    m1.metric("Inference time",f"{p['seconds']:.2f} s")
    m2.metric("Device",p["device"])
    m3.metric("Action tensor shape",str(p["shape"]))
    st.markdown("**Prompt sent to model**"); st.write(p["task"])
    st.markdown("**Predicted action chunk**")
    st.caption("Real model output after the checkpoint postprocessor. It is not mapped to TIAGo joints and is never executed.")
    st.json(p["actions"])
    try:
        vals=np.asarray(p["actions"])
        if vals.ndim==2:
            st.markdown("**Action values per predicted step**")
            st.line_chart(pd.DataFrame(vals,columns=[f"action_{i}" for i in range(vals.shape[1])]))
    except Exception: pass
    with st.expander("Inference metadata"):
        st.json({k:v for k,v in p.items() if k!="actions"})

with st.expander("Important details"):
    st.markdown("""
- The app makes GET requests only to `/health`, `/api/telemetry`, `/api/camera.jpg`, and `/api/snapshot`.
- The live image is used as model input. The current frame is copied into all three image slots declared by this base checkpoint; that is a demo workaround, not real multi-camera input.
- Telemetry is displayed but deliberately not fed into the model because its schema has not been mapped to the checkpoint's expected state representation.
- Six state inputs are synthetic demo values, not live robot state.
- No robot action is executed or sent to Webots.
- The base checkpoint is not guaranteed to perform tasks successfully without robot-specific training/fine-tuning.
""")
