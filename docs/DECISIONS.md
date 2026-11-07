# Architecture Decision Records

## ADR-0001: Establish M0 Python project foundation

- **Date:** 2026-11-07
- **Status:** Accepted
- **Context:** The repository had no Python package layout, lockfile, test configuration, or
  cross-platform CI. The architecture calls for Python 3.11, uv, Ruff, pytest, pre-commit, and a
  Windows/Linux CI matrix. M0 must not change controller behavior.
- **Decision:** Create a `src/vlarobot/` package, keep the existing controller files in place until
  the M1 migration, add empty configuration/script landing documentation, and use Python 3.11.18
  for local/CI tooling. Keep optional runtime extras declared but empty until runtime packages are
  selected. Configure pytest's `webots` and `gpu` markers.
- **Decision:** Use Typer for the CLI and import Webots' controller API only within
  `vlarobot.backends.webots`. Add `vlarobot.env.configure_webots_environment` to set `WEBOTS_HOME`
  and prepend the API path to `PYTHONPATH` and `sys.path` using the host OS path separator.
- **Decision:** Use `ty` for CI type checking rather than mypy. `ty` is MIT-licensed and has no
  runtime dependencies; it is currently beta, so the lockfile pins the tested version. This avoids
  introducing mypy's MPL-2.0 `pathspec` dependency.
- **Consequences:** CI installs and checks the package on Ubuntu and Windows; local Fedora checks
  pass. Actual Windows CI and Webots verification remain outstanding. Model, training, GPU, and
  simulator-surrogate dependencies remain unselected.
- **License audit:** Locked Python runtime and development packages are MIT, BSD, ISC, PSF, or
  Apache-2.0/BSD dual-licensed. The pre-commit hook repositories are MIT-licensed. No AGPL, GPL,
  or non-commercial dependency is in the lockfile. Package-license metadata was inspected from
  the resolved distributions on 2026-11-07. Model/checkpoint licences are not audited because
  none has been selected.
- **Source of truth:** `docs/ARCHITECTURE.md`; the older references to
  `docs/ARCHITECTURE_AND_PLAN.md` do not match the plan file present in this repository.

## ADR-0002: M1 core contracts and lockstep backend

- **Date:** 2026-11-07
- **Status:** Accepted, with M1-06 regression pending
- **Context:** M1 requires simulator-independent core contracts, a versioned simulator/brain
  protocol, a deterministic fake backend, and a Webots backend without changing the existing
  scripted controller logic.
- **Decision:** Implement the planned core types and protocols as dataclasses and `Protocol`
  interfaces. Use msgpack protocol version 1 with NumPy array shape/dtype metadata, strict
  envelope validation, and per-request sequence correlation. Implement a synchronous ZMQ REQ/REP
  transport with a default endpoint of `tcp://127.0.0.1:5555`.
- **Decision:** Keep the existing scripted controller available in
  `vlarobot.backends.webots.controller`. The Webots shim starts the lockstep server by default,
  selecting the repository `.venv` Python or `WEBOTS_PYTHON`; `VLAROBOT_WEBOTS_MODE=legacy`
  selects the retained controller for comparison.
- **Decision:** Load Webots/controller settings from `configs/robot.yaml` and workspace profiles
  from `configs/safety.yaml`; retain compatibility aliases for the old controller constants.
  Package both YAML files as install data, include the TIAGo URDF as package data, and resolve
  settings from the checkout or active Python prefix.
- **Dependencies and licences:** Added msgpack 1.2.4 (Apache-2.0), NumPy 2.4.6 (BSD-3-Clause and
  permissive notices), PyYAML 6.0.3 (MIT), pyzmq 27.3.0 (BSD-3-Clause), and Typer 0.28.0 (MIT);
  added IKPy 4.4.4 (Apache-2.0) to the `sim` extra. Its locked SciPy 1.17.1 and SymPy 1.14.0
  dependencies use BSD-style permissive licences. Installed distribution licence files were
  inspected on 2026-11-07; no GPL, AGPL, or non-commercial licence was selected.
- **Consequences:** Non-Webots tests and a live Fedora/Webots R2025a lockstep run pass. Camera
  extrinsics remain identity placeholders pending M3 calibration. Webots reset supports the
  loaded default scene only; seeded scenario spawning is deferred to M2.
- **Decision:** Derive workspace profiles from the original PickPlace flow's measured base-frame
  waypoints instead of guessing bounds. The `original_envelope` margin is 0.10 m and the tighter
  `strict` profile margin is 0.02 m. The legacy `PLACE_BASE = [-1.07, -3.9, 0.8]` is confirmed
  at `controllers/tiago_controller/config.py:6` in pre-M1 commit `1b2e4da` and remains unchanged
  in `configs/robot.yaml`. The original controller had no workspace limit. Its observed
  `original_envelope` is the measured axis-aligned waypoint bounds expanded by 0.10 m; `strict`
  uses the same observed bounds expanded by 0.02 m. The observed transit/placement points varied
  between two original-controller runs; the profiles include both runs' extrema. Neither profile
  is a validated safety limit: the envelope is a behavioral baseline only. Joint, velocity,
  IK-tolerance, and invalid-command checks remain enabled, and `SafetyMonitor.validate()` logs
  each rejection reason.
- **Measured original waypoints (base-frame meters):** home
  `(0.203672298, -0.985455309, 0.963048983)`, descend/grasp
  `(0.672736661, -0.213295065, 0.764821419)`, pregrasp/lift
  `(0.672736661, -0.213295065, 0.884821419)`, transit/retreat
  `(-1.077327982, -3.842045869, 0.936284755)` / `(-1.086885823, -3.858086446, 0.935119120)`,
  and place-descend/release `(-1.077327982, -3.842045869, 0.816284755)` /
  `(-1.086885823, -3.858086446, 0.815119120)`. Collected from two instrumented original-controller
  runs against `worlds/factory.wbt`; each run logged repeated names at initialization and
  terminal-state flush. The latest run terminated `FAILED`, so this establishes commanded
  waypoint evidence, not successful placement.
- **Open acceptance:** M1-06 remains blocked by the live regression. The current Webots-backed
  brain comparison was executed with `uv run python -m vlarobot.cli can-regression --seeds 10`; it
  produced `success=0/10` and `rejections=9`, with the first and only near-success still 0.560m from
  the target and repeated safety rejections in the `INIT`, `pregrasp`, and `place_descend` phases.
  Webots resets currently replay the default scene, so seed variation does not yet imply scenario
  randomization (scenario generation is M2). As of this date, the M1-06 gate is not met and no
  success claim is made.
