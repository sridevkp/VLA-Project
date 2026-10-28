# TIAGo AI Studio — Streamlit prototype

A lightweight, Python-only dashboard for a Webots TIAGo robotics project.

## Features
- Prompt/chat interface for a vision-language model
- Optional image upload to the model
- Demo mode that does not call any model
- Local model support:
  - Ollama `/api/chat` (vision-capable model required for images)
  - OpenAI-compatible `/chat/completions`
- Robot telemetry view with seven arm joints, base velocity, torso lift, gripper state
- Import/export telemetry JSON
- Event log
- No robot motion commands are sent

## Install (Windows PowerShell)

```powershell
cd tiago_ai_studio
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Local model setup

### Ollama
1. Start Ollama and make sure a vision-capable model is installed.
2. Open **Settings** in the dashboard.
3. Select **Ollama**.
4. Use `http://localhost:11434` as the base endpoint.
5. Enter the exact local model name (for example, the name shown by `ollama list`).
6. Save settings, test the connection, then switch **Model mode** in the sidebar to **Local model**.

### OpenAI-compatible local server
Choose **OpenAI-compatible**, enter the server base URL (for example `http://localhost:8000/v1`) and the model identifier expected by that server.

## Telemetry format

The Telemetry & State page can import a JSON object such as:

```json
{
  "connection": "Webots bridge",
  "simulation": "Running",
  "base_linear": 0.0,
  "base_angular": 0.0,
  "torso_lift": 0.12,
  "gripper": "Open",
  "joints": [0.07094, -0.01704, -0.00003, 0.00001, 0.00001, -0.000003, 0.000002],
  "battery": 96
}
```

The `joints` array must contain exactly seven values in radians.

## Planned Webots integration
This prototype is intentionally not connected to Webots yet. A next step is a small Python bridge that reads joint position sensors and camera frames from the Webots controller and publishes structured JSON to the dashboard (e.g. over a local WebSocket or HTTP endpoint). Keep robot actuation separate and guarded until simulation control is tested.
