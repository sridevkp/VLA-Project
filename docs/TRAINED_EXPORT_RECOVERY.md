# Production model transfer recovery

The downloaded ZIP `/home/tj330/Downloads/exports-20261115T123103Z-1-001.zip`
is 1,234,573 bytes, contains 20 members, passes ZIP CRC validation and contains no
`model.safetensors`. Every archived file matches the local portable export,
including `export-manifest.json`. Extraction did not omit the model.

`export_complete_checkpoint` first copies the whole pretrained-model directory
with `shutil.copytree`, then hashes actual files beneath the export directory and
writes its manifest. Copy/read errors propagate; there is no model exclusion.
The transferred manifest contains the model hash, so under this implementation
the model existed and was read at export manifest creation. Returning success
does not verify a later browser download or ZIP. No notebook step creates or
validates a downloaded archive. Precise upstream cause (Drive packaging omission,
incomplete multipart download, or later Drive-side removal) is not established;
no additional ZIP parts were found locally. Do not label this a training failure.

Minimal recovery, with no optimizer download and no production checkpoint edits:

1. In Google Drive, open `MyDrive/vlarobot_smolvla/training-tiago-v1/checkpoints/020000/pretrained_model/`.
2. Download **only `model.safetensors`** using the individual file's Download
   action, rather than downloading the parent folder as a ZIP. Expected size:
   1,197,789,224 bytes. No optimizer or other training-state download is needed.
3. Provide the downloaded file's local path to the agent. It will hash the file
   and require SHA-256
   `107698cfc0274cb2e0fbe2976be0e1e0a198f41c140807ec19bca4c48f04ed58`,
   matching both portable and owner-reported complete checkpoint manifests.
4. Preserve the original transfer. Build a new versioned portable export from its
   existing small files plus the verified model. Do not regenerate its manifest,
   replace weights with initialization, or alter its processors/normalization.
5. Run `require_trained_export` and the existing local import validator under
   `.venv-vla`, targeting a fresh report directory. Confirm exact final step,
   dataset identity and all file hashes, strict CPU loading, real TRAIN image/state
   processing, finite action7×50 chunk and measured latency. Compare final weights
   with pinned initialization before making a parameter-change conclusion.
6. Begin the frozen 24-episode TRAIN-only campaign only after those checks pass.

Local commands for step 5, from `/home/tj330/programs/VLA-Project`, after the agent
has assembled `runs/trained-export-recovered-v1`:

```text
.venv-vla/bin/python runs/week2-progress-v1/validate-import.py --export runs/trained-export-recovered-v1 --out runs/import-trained-recovered-v1
```

No inference or evaluation has started because the trained model file is absent.
TEST, gate, original checkpoint, approved training configuration and original
transferred export remain unchanged. Evidence:
`runs/import-trained-v1/transfer-forensics.json` in the primary project worktree.
