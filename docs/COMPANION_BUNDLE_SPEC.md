# COMPANION_BUNDLE_SPEC.md — gitignored reproducibility artifacts (DRAFT SPEC)

The repository deliberately excludes experiment outputs, datasets and model weights
(`runs/**` is gitignored; see `.gitignore`). A fresh checkout of the staging branch can
therefore run the code but cannot verify or reproduce any claimed result. This document
specifies the **companion bundle**: the exact set of gitignored artifacts that must
accompany a fresh checkout, with their authoritative hashes and a verification procedure.

**Publication constraint (owner decision, unresolved):** nothing in this bundle is
published anywhere by preparing this spec. Distributing model weights, the training
dataset, or large evidence archives requires separate, explicit owner authorization.
This document only defines *what* would need to move and *how to verify it*.

All hashes below are SHA-256, computed 2026-11-17 from the live worktrees:

- **Primary** = `/home/tj330/programs/VLA-Project` (branch `m1-checkpoint-ik-transit`)
- **Analytic** = `/home/tj330/programs/VLA-Project-analytic` (branch `work/analytic-risk-followup`)

## Bundle items

### Required

| # | Path | Located in | Size | What it is | Hash anchors |
|---|---|---|---|---|---|
| 1 | `runs/eval/baseline-train-s0-fresh-v2-registered/` | primary | 68 MB | Frozen scripted-baseline TRAIN campaign (23/24) incl. `frames/` (620 jpgs), per-episode recordings, `provenance.json` | `provenance.json` = `690f1597be07251200d74afe66ccec24fa06567ce29ba20f7c80175841d3aca9`; internal 12-file `files_sha256` lock re-verified fail-closed by `run_vla_campaign` |
| 2 | `runs/eval/vla-train-s0-step20000-recovered-v1/` | primary | 9.5 MB | Frozen VLA-only TRAIN campaign (0/24, gate FAILED): 225 files incl. per-episode `steps.npz`/`predicted.jsonl`/`actions.jsonl`/`latency.jsonl`/`meta.json`, `frames/` (24 jpgs), `summary.json`, `timings.json`, `provenance.json` | `provenance.json` = `2375d2ab188406f1bcd600eb7cede307af226222a248236a79fdad27545f8b02`; `summary.json` = `9695b1da66a2ec38dceba4b012e049cdd1c8412cc3c8aca7f0d4b076e7a3a416`; full per-file manifest committed at `docs/evidence/vla-train-step20000-files.json` (225 entries, analytic worktree) |
| 3 | `runs/validation-evidence-v1/` | analytic | 64 MB | Scripted VALIDATION campaign (15/18): `campaign/frames/` (425 jpgs), parallel-episode outputs, `completed-validation.json` | verify via the campaign's own recorded outputs; hash-anchor manifest to be generated at packaging time (see procedure) |
| 4 | `runs/colab-prep-v3/` | primary | 3.1 GB | Approved Colab training handoff: `source.zip`, `dataset-train-v1.zip`, `bundle-status.json` (164-entry source snapshot), packaging logs | `source.zip` = `2d3cb4bd06ac8313b1bb3cdb5aae7f916c2b614dd3884172c4570be9ff36b2f5`; `dataset-train-v1.zip` = `a2bc0842380ebc21440b41494b04b2ccba44961cd329fade9dd86f8976e7577c`; `bundle-status.json` = `b335fa8a4825688fcfeb539dd7c9539c73dccc35bcf98892c0a94482d79939ff` |
| 5 | `runs/datasets/tiago-train-v1` | primary | 3.1 GB | Native training dataset (80-success episode set) with `dataset-integrity.json` admission record | `dataset-integrity.json` = `b8f529a9371ab10d5afb663e3e11dd08f8c4ddb57c03ee4d081ac02b0dfc0770` |
| 6 | `runs/local-model-validation-v1/export-step-020000/` | analytic | 1.2 GB | Trained export at step 20,000 (the export the frozen VLA campaign ran against): `model.safetensors`, processor configs, optimizer evidence | `model.safetensors` = `7531991e27719e6fee402344aadbe43b2fc4ac858a87ae5ec77809c8efe385d8`; `config.json` = `4b03cc3cb3257ec32491bb56d44566590bb9cd6aceb4b02ab8e3cef25cb119f4`. **Weights: publication requires separate owner authorization** |
| 7 | `runs/ad1-train-clear-i00-s100-frame0-v1/` | analytic | 100 KB | AD1 numerical postmortem analysis (`docs/VLA_TRAIN_POSTMORTEM.md` pairs, episodes 08/11/14) | `provenance.json` = `b8c61fe030573ae12561f8cc9e8ca05b28f98c505b9c8bd19a47d3470da3d09c`; committed evidence mirror `docs/evidence/vla-train-postmortem-v1.json` = `4ef7434b3bef11de03a653edab47cfb420c626811f41cfa3be109aaa9d077d59` |

Notes on item 2: the per-episode metadata files referenced by
`docs/VLA_TRAIN_POSTMORTEM.md` (`episode-08.meta.json` = `46d6d2e7…`,
`episode-11.meta.json` = `e9f0903a…`, `episode-14.meta.json` = `90d324d0…`) are hash-pinned
in the committed 225-file manifest, but their **bytes exist only in the primary worktree**.
The bundle is what closes that gap for a fresh checkout.

### Optional (diagnostics; bundle only if authorized)

| Path | Located in | Size | What it is |
|---|---|---|---|
| `runs/eval/grasp-train-clear-i03-s0-registered-v1/` | primary | (part of `runs/eval` 132 MB total) | Grasp-repeat diagnostics frames (repeats 1–4) |
| `runs/independent-validation-candidates-v3/` | analytic | 4.4 MB | Independent-validation diagnostic candidates (`free_joint`/`closure`/`carried_joint`/`release` frames, 43 jpgs) — **not gate evidence** |
| `runs/colab-prep-v2/` | primary | — | Superseded earlier bundle generation (notebook `d3da4448…`); keep only for provenance archaeology |

### Excluded from any bundle

- Virtual environments (`.venv-vla/`), build artifacts (`model/compat/build/`,
  `*.egg-info`), `__pycache__`.
- Anything not listed above under `runs/` in either worktree unless the owner adds it.

## Verification procedure (fresh checkout)

Bundle layout: a single archive (or set of archives) that, when unpacked at the checkout
root, restores the paths exactly as listed above. No path rewriting, no symlinks
(Windows compatibility).

1. Check out the staging branch at the recorded HEAD and run `uv sync --locked`.
2. Unpack the bundle so that `runs/…` paths match this spec byte-for-byte.
3. Verify anchors, in order of authority:
   - **Item 2**: for every entry of the committed manifest
     `docs/evidence/vla-train-step20000-files.json`, recompute SHA-256 of the restored
     file and compare. All 225 must match.
   - **Item 1**: recompute the 12 `files_sha256` entries of
     `runs/eval/baseline-train-s0-fresh-v2-registered/provenance.json` **against the
     source tree** (these lock live files like `configs/robot.yaml`, not bundle files) and
     compare `provenance.json` itself against the anchor hash above.
   - **Items 4–7**: compare each named anchor file against the hashes in this document.
   - **Item 3**: `completed-validation.json` and the campaign outputs; a packaging-time
     manifest (generated with the procedure below) pins the remaining bytes.
4. Packaging-time manifest generation (deterministic, cross-platform, no symlinks):

   ```text
   uv run python - <<'PY'
   import hashlib, json, pathlib
   ROOTS = ["runs/eval/baseline-train-s0-fresh-v2-registered",
            "runs/validation-evidence-v1"]
   manifest = {}
   for root in ROOTS:
       for p in sorted(pathlib.Path(root).rglob("*")):
           if p.is_file():
               manifest[p.as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
   pathlib.Path("companion-bundle-manifest.json").write_text(
       json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
   PY
   ```

   The resulting `companion-bundle-manifest.json` is itself committed (or hash-anchored
   here) so the bundle is self-verifying.

5. A fresh checkout is **verified** only when every required item passes its anchor
   comparison. Any mismatch is a fail-closed condition: the artifact must not be used as
   evidence for any result.

## Status

- This is a **specification only**. No bundle has been built, moved or published.
- Building the bundle (and especially publishing items 4–6: training dataset, weights,
  Colab archives) requires explicit owner authorization and a decision on transport
  (e.g. institutional storage vs. Drive) recorded in `docs/DECISIONS.md`.
