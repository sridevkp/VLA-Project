# TIAGo VLA Dashboard

## Run the dashboard (Windows PowerShell)

Start the Webots controller that exposes port 8765, then in this folder run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit URL, usually http://localhost:8501.

If PowerShell blocks activation, run:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app reads:
- `http://127.0.0.1:8765/health`
- `http://127.0.0.1:8765/api/telemetry`
- `http://127.0.0.1:8765/api/camera.jpg`
- `http://127.0.0.1:8765/api/snapshot`

## Connect an inference service

Set a base URL for a reachable VLA server:
```powershell
$env:VLA_API_URL = "http://YOUR-VLA-SERVER:8000"
streamlit run app.py
```
The dashboard POSTs to `/predict` with JSON keys `task`, `telemetry`, `image_base64`, `image_mime_type`. It expects an `action` or `actions` field in the JSON response.

## SmolVLA service scaffold

In a separate environment (preferably a remote GPU machine if local memory/GPU is insufficient):
```powershell
pip install "lerobot[smolvla]" fastapi "uvicorn[standard]" pillow
$env:VLA_MODEL_ID = "lerobot/smolvla_base"
$env:VLA_DEVICE = "cpu"
uvicorn vla_worker:app --host 0.0.0.0 --port 8000
```
For a GPU host set `VLA_DEVICE=cuda`. Check `/health` and `/model-info`.

**Important:** `vla_worker.py` intentionally returns HTTP 501 for `/predict` until a TIAGo-specific adapter is implemented. The base checkpoint's expected observation and action features may not match TIAGo. This prevents false confidence from arbitrary tensor mappings. The service currently demonstrates model loading and exposes the required feature metadata; it is not yet a working TIAGo action policy.

## Next step

Copy the JSON from `http://127.0.0.1:8765/api/telemetry` (remove secrets if any). Use it to map camera, torso, seven arm joint readings, gripper, and base state into a dataset/policy representation compatible with a TIAGo-trained or fine-tuned model. Keep all model output preview-only until units, dimensions, joint order, limits, and stopping behavior are validated.
