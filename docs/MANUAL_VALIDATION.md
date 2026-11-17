# Manual Validation Procedures (self-contained)

Use these only when a step requires human action (Windows validation, visual checks, or after a failed automated attempt). Autonomous live campaign runs (run-episodes, bench-steps, determinism, servo, 20-episode gate) are executed by the agent without approval.

## 1) Linux/Webots manual run (GUI) — for visual checks/screenshots
**Prerequisites**
- Repo: `~/programs/VLA-Project` on branch/commit recorded in campaign manifest.
- Python env: `uv` installed; deps synced (`uv sync --extra dev`).
- Webots R2025a installed at `~/.local/opt/webots/webots` (or set path explicitly).
- Port free: run `uv run python -m vlarobot.cli doctor` and ensure no leftover Webots/controller/server processes.

**Steps (Terminal A - launch Webots world/controller)**
1. `cd ~/programs/VLA-Project`
2. (If telemetry needed for validation) edit `configs/robot.yaml`: set `export_sim_kinematics: true`.
3. Start world: `~/.local/opt/webots/webots ~/programs/VLA-Project/worlds/factory.wbt`
4. Wait until simulation loads and controller starts ZMQ server (check Terminal A logs; doctor should show port owner when ready).

**Steps (Terminal B - run command)**
1. `cd ~/programs/VLA-Project`
2. Preflight: `uv run python -m vlarobot.cli doctor`
3. Example validation sweep: `uv run python -m vlarobot.cli validate-kinematics --n 12 --seed 0 --bench-ms-per-step 447 --report runs/validate-kinematics-12-seed0.json`
4. Expected (MEASURED shape, values UNVERIFIED until run): per-pose residuals and summary; fails early with clear error if `kinematics_telemetry_enabled` is false/missing.
5. Time: depends on poses/Webots speed (bench-dependent; see Phase 0 ms/step when measured).

**Screenshots to capture (when requested)**
- Reset, PREGRASP exit, DESCEND mid, CLOSE, LIFT top, RELEASE, end (3–6 per episode for first 3/condition and every failed episode). Use Supervisor export/screenshot if available; otherwise capture Webots GUI.

**What to send back**
- Command output, `runs/...` report/trace paths, screenshots, and `uv run python -m vlarobot.cli doctor` output.

## 2) Linux headless/batch run (agent/campaign style)
1. `cd ~/programs/VLA-Project`
2. `uv run python -m vlarobot.cli doctor`
3. `uv run python -m vlarobot.cli run-episodes --n 1 --seed 0 --fresh-world --webots ~/.local/opt/webots/webots`
4. Expected: summary table with outcome/fail_state/dist/criteria/timings and trace/log paths.
5. Time: episode-dependent (Phase 0 baseline rerun MEASURED ~245.7s episode, 18.3s startup in that run; varies).

## 3) Windows manual validation (UNVERIFIED - cannot be executed in this Linux environment)
**Prerequisites**
- Windows machine with Webots R2025a installed, Python/uv, repo checkout at same commit.
- PowerShell terminal; working directory `C:\path\to\VLA-Project`.

**Steps**
1. `cd C:\path\to\VLA-Project`
2. Sync deps: `uv sync --extra dev`
3. Smoke/health: `uv run python -m vlarobot.cli doctor`
4. Webots smoke (if world/controller available): `$env:WEBOTS_HOME="C:\path\to\Webots"; uv run python -m vlarobot.cli smoke` (command availability/path may vary; **UNVERIFIED**)
5. Short run-episodes (if Webots installed): `uv run python -m vlarobot.cli run-episodes --n 1 --seed 0 --fresh-world --webots "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe"` (**UNVERIFIED** path/flags)
6. CI result: check GitHub Actions workflow `.github/workflows/ci.yml` (ubuntu + windows matrix, non-Webots tests) for the commit; record pass/fail link/status (**UNVERIFIED** here).

**Expected output**
- `doctor`: port/process status; no leftover servers. Smoke/run: successful startup and summary or clear errors. **UNVERIFIED**.

**What to send back**
- Full command outputs, `doctor` before/after, CI run URL/status, and any generated `runs/` artifacts.

## Notes
- Never use `--extern-urls` for the normal in-world controller/ZMQ server setup.
- After changing `configs/robot.yaml` (e.g. `export_sim_kinematics`), fully restart the Webots sim-server/controller process.
