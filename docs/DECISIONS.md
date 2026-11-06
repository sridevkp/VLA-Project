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
