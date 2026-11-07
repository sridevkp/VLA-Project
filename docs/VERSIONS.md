# Versions and compatibility

| Component | Pin/target | Verification |
|---|---|---|
| Python | 3.11.18 (`.python-version`; project supports Python 3.11) | Installed and tested on Fedora/Linux; Windows CI configured, not yet run |
| Webots | R2025a | `worlds/smoke.wbt` and external-controller smoke tested on Fedora/Linux; Windows unverified |
| PyTorch | Not pinned | Unverified; select a CUDA wheel per OS using the official selector |
| LeRobot | Not pinned | Unverified; model/training dependencies are not installed |
| MuJoCo | Not pinned | Unverified; surrogate backend is not implemented |
| ty | 0.0.99 (lockfile) | Installed and type-check passed on Fedora/Linux; Windows CI pending |
| uv | Lockfile managed by uv | `uv lock --check` passes in this environment |
| msgpack | 1.2.4 (lockfile) | Installed; wire round-trip tests pass |
| NumPy | 2.4.6 (lockfile) | Installed; array wire tests and live Webots RGB transfer pass |
| PyYAML | 6.0.3 (lockfile) | Installed; typed robot settings tests pass |
| pyzmq | 27.3.0 (lockfile) | Installed; FakeBackend transport and live Webots RPCs pass |
| Typer | 0.28.0 (lockfile) | Installed; `smoke`, `sim-server`, and `brain` CLI commands load |
| IKPy | 4.4.4 (lockfile, `sim` extra) | Installed; Webots backend initialized in the live factory-world check |
| SciPy | 1.17.1 (IKPy dependency) | Installed for the local backend check |
| SymPy | 1.14.0 (IKPy dependency) | Installed for the local backend check |

`pyproject.toml` constrains the supported interpreter to Python 3.11. CI and local development pin
the exact interpreter patch in `.python-version`. Webots is pinned to R2025a by the project world
files and was verified with the local external-controller smoke test. Cross-platform verification
is incomplete until Webots and CI pass on Windows.

## M1 dependency and backend verification

The versions above are the Fedora/Linux lockfile versions as of 2026-11-07; Windows CI has not yet
run. The Webots backend was exercised against the local R2025a factory world through the brain
process. The RGB observation shape was verified as HxWx3. `T_base_cam` is currently an identity
placeholder and must not be used for calibrated localization before M3. The built wheel contains
both `configs/robot.yaml` and the Webots TIAGo URDF.

## M0 compatibility and open-item audit

The versions below are project selections only; the Webots or learning-library compatibility
questions remain open wherever noted.

| Plan §9 item | M0 evidence | Status / remaining verification |
|---|---|---|
| Webots release, TIAGo PROTO, Python compatibility on Windows/Linux | R2025a smoke test passed on Fedora/Linux with Python 3.11.18 | Windows compatibility and smoke test remain unverified |
| Compatible torch/LeRobot/Transformers set and Windows training | None of these packages are pinned or installed | UNVERIFIED; choose versions and verify Windows/WSL2 training later |
| `smolvla_base` inputs, outputs, state/action dims, camera keys | Existing `model/main.py` references `lerobot/smolvla_base` but its config was not loaded | UNVERIFIED; inspect with the compatible LeRobot/model install and record in `VLA_NOTES.md` |
| TIAGo URDF import into MuJoCo | MuJoCo is not installed and surrogate import is not implemented | UNVERIFIED; test at M5 |
| Licences of selected model weights/software | No detector, segmenter, LLM, VLA runtime, or checkpoint has been selected for the new pipeline | Model licences remain UNVERIFIED until selection; the current locked Python dependencies have permissive licenses (see `DECISIONS.md`) |
| Webots collision/contact-event access | M0 smoke world only checks camera and stepping | UNVERIFIED; validate the contact metric when implementing M2 evaluation |
