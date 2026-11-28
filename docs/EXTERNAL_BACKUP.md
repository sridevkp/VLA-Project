# Verified external evidence backup

Run these commands in Bash on the source machine from `~/programs/VLA-Project-v4`.
Mount a private external drive first. Use a fresh backup directory. Pause writers
before the final copy and verification so run logs and checkpoints are fixed.
The seven persistent worktrees contain the raw runs, datasets and checkpoint
directories. The Git bundle preserves the shared refs. The Downloads copy
preserves the original report and the known portable model archives and receipt.

```bash
set -euo pipefail
cd "$HOME/programs/VLA-Project-v4"
VLA_MOUNT=/run/media/tj330/REPLACE_WITH_DRIVE
mountpoint -q "$VLA_MOUNT"
test "$(findmnt -n -o MAJ:MIN -T "$PWD")" != "$(findmnt -n -o MAJ:MIN -T "$VLA_MOUNT")"
findmnt -T "$VLA_MOUNT"
df -h "$VLA_MOUNT"

vla_trees=(
  VLA-Project VLA-Project-analytic VLA-Project-m1-original
  VLA-Project-remediation VLA-Project-staging VLA-Project-week3 VLA-Project-v4
)
vla_originals=(
  Project_Report.pdf
  acct2-durable-v1-drive-v1-portable-020000-attempt-1.zip
  acct2-durable-v1-drive-v1-portable-020000-attempt-1.receipt.json
  tiago-checkpoint-017000-backup.zip
)
vla_sources=()
for tree in "${vla_trees[@]}"; do vla_sources+=("$HOME/programs/$tree"); done
for original in "${vla_originals[@]}"; do vla_sources+=("$HOME/Downloads/$original"); done
vla_git=$(realpath "$(git rev-parse --git-common-dir)")
vla_need=$(du -sc --apparent-size -B1 --exclude='.venv*' --exclude='.git' "${vla_sources[@]}" | tail -1 | cut -f1)
vla_git_size=$(du -s --apparent-size -B1 "$vla_git" | cut -f1)
vla_free=$(df -B1 --output=avail "$VLA_MOUNT" | tail -1)
test "$vla_free" -gt "$((vla_need + vla_git_size + 10 * 1024**3))"

VLA_BACKUP="$VLA_MOUNT/worldmind-final"
mkdir "$VLA_BACKUP"
mkdir "$VLA_BACKUP"/{worktrees,originals}
git bundle create "$VLA_BACKUP/worldmind-history.bundle" --all
git bundle verify "$VLA_BACKUP/worldmind-history.bundle"
git for-each-ref --format='%(objectname) %(refname)' > "$VLA_BACKUP/source-git-refs.txt"
git bundle list-heads "$VLA_BACKUP/worldmind-history.bundle" > "$VLA_BACKUP/bundled-git-refs.txt"
git worktree list --porcelain > "$VLA_BACKUP/source-worktree-heads.txt"

for tree in "${vla_trees[@]}"; do
  rsync -aH --exclude='/.git' --exclude='/.venv*/' \
    "$HOME/programs/$tree/" "$VLA_BACKUP/worktrees/$tree/"
done
for original in "${vla_originals[@]}"; do
  rsync -a "$HOME/Downloads/$original" "$VLA_BACKUP/originals/"
done

{
  for tree in "${vla_trees[@]}"; do
    rsync -aHnci --exclude='/.git' --exclude='/.venv*/' \
      "$HOME/programs/$tree/" "$VLA_BACKUP/worktrees/$tree/"
  done
  for original in "${vla_originals[@]}"; do
    rsync -anci "$HOME/Downloads/$original" "$VLA_BACKUP/originals/"
  done
} | tee "$VLA_BACKUP/verification-differences.txt"
test ! -s "$VLA_BACKUP/verification-differences.txt"
cd "$VLA_BACKUP/worktrees/VLA-Project-week3"
sha256sum -c docs/evidence/worldmind-submission.sha256
sha256sum -c docs/figures/SHA256SUMS
(cd "$VLA_BACKUP/originals" && sha256sum "${vla_originals[@]}") | tee "$VLA_BACKUP/originals.sha256"
```

Compare the bundle refs with the source ref list and retain the verification
outputs. If protected evidence is linked to an archive outside the listed
worktrees, copy and verify the archive itself as well. The byte verification
covers the stored representation. Keep any existing logical-payload manifests
alongside it. Verification is complete only after all checks succeed on the
external destination.
