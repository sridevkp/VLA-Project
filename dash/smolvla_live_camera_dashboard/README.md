# SmolVLA Action Studio — live controller camera, no robot control

The app automatically reads the live image and telemetry from your existing controller HTTP server, but it never sends robot commands.

## Endpoints used
- `GET http://127.0.0.1:8765/health`
- `GET http://127.0.0.1:8765/api/telemetry`
- `GET http://127.0.0.1:8765/api/camera.jpg`
- `GET http://127.0.0.1:8765/api/snapshot`

## Install and run (Windows PowerShell)
Start the Webots controller's HTTP server first. Extract this ZIP and open PowerShell in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install "lerobot[smolvla]"
streamlit run app.py
```

If PowerShell activation is blocked:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install "lerobot[smolvla]"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If your controller server uses a different host or port:
```powershell
$env:ROBOT_API_URL = "http://127.0.0.1:8765"
streamlit run app.py
```
Use the actual reachable URL instead of the example if needed.

## Use
1. Wait for the live camera and telemetry to appear.
2. Enter a language instruction.
3. Keep the six demo state values at zero for the first run.
4. Click **Load model and run inference**.
5. Inspect the real model action chunk and chart.

The first run downloads model weights. CPU is the default because a 2 GB GPU may not have enough VRAM. 8 GB system RAM may be tight. The dashboard only reads robot observations; it does not send actions to the robot.

Telemetry is displayed but not passed to the model until it is deliberately mapped to the checkpoint's expected state features. The same live image is repeated across three camera slots because the base checkpoint config declares three images; this is only a pipeline demo.
