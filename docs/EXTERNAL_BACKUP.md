# Verified external evidence backup

Run these commands in Bash on the source machine from `~/programs/VLA-Project-v4`.
Mount a private external drive first. Use a fresh backup directory. Pause writers
before the final copy and verification so run logs and checkpoints are fixed.
The seven persistent worktrees contain the raw runs, datasets and checkpoint
directories. The Git bundle preserves the shared refs. The Downloads copy
preserves the report, the two portable checkpoints, the V2 recovery source
archive, and the full V2 recovery directory with optimizer records and receipts.

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
  vla-recovery-55c8938.zip
  step-021000-20261126T022556Z-1-001.zip
  v2-transition-warmstart-01
)
vla_sources=()
for tree in "${vla_trees[@]}"; do vla_sources+=("$HOME/programs/$tree"); done
for original in "${vla_originals[@]}"; do vla_sources+=("$HOME/Downloads/$original"); done
# Add every referenced archive outside these roots by its absolute source path.
vla_external=()
vla_sources+=("${vla_external[@]}")
for source in "${vla_sources[@]}"; do test -r "$source"; done
vla_v2="$HOME/Downloads/v2-transition-warmstart-01"
for file in optimizer-steps.jsonl step-021000/{model.safetensors,recovery-optimizer.pt,optimizer-steps.jsonl,export-manifest.json,recovery-provenance.json,training-approval.json}; do
  test -f "$vla_v2/$file"
done
printf '%s  %s\n' \
  dd27b10140fadfba257b7a9696186420b05f02f30894b05938729d6db9e8537e "$HOME/Downloads/vla-recovery-55c8938.zip" \
  0d5059dc5051b12a0d19614cc93f4b6a8ec77887f7e1e97b6e830858399504d6 "$vla_v2/step-021000/export-manifest.json" \
  | sha256sum -c -
vla_git=$(realpath "$(git rev-parse --git-common-dir)")
vla_need=$(du -sc --apparent-size -B1 --exclude='.venv*' --exclude='.git' "${vla_sources[@]}" | tail -1 | cut -f1)
vla_git_size=$(du -s --apparent-size -B1 "$vla_git" | cut -f1)
vla_free=$(df -B1 --output=avail "$VLA_MOUNT" | tail -1)
test "$vla_free" -gt "$((vla_need + 2 * vla_git_size + 10 * 1024**3))"

VLA_BACKUP="$VLA_MOUNT/worldmind-final"
mkdir "$VLA_BACKUP"
mkdir "$VLA_BACKUP"/{worktrees,originals,external}
vla_probe=$(mktemp -d "$VLA_BACKUP/.filesystem-check.XXXXXX")
mkdir "$vla_probe/source"
printf 'preservation check\n' > "$vla_probe/source/file"
chmod 640 "$vla_probe/source/file"
ln "$vla_probe/source/file" "$vla_probe/source/hardlink"
ln -s file "$vla_probe/source/symlink"
rsync -aH "$vla_probe/source/" "$vla_probe/copy/"
test "$(stat -c %a "$vla_probe/copy/file")" = 640
test "$(stat -c %i "$vla_probe/copy/file")" = "$(stat -c %i "$vla_probe/copy/hardlink")"
test "$(readlink "$vla_probe/copy/symlink")" = file
test "$(stat -c %Y "$vla_probe/source/file")" = "$(stat -c %Y "$vla_probe/copy/file")"
printf 'mode, hard links, symlinks, and timestamps verified\n' > "$VLA_BACKUP/filesystem-check.txt"

git for-each-ref --format='%(objectname) %(refname)' > "$VLA_BACKUP/source-git-refs.txt"
git worktree list --porcelain > "$VLA_BACKUP/source-worktree-heads.txt"
vla_stage="$VLA_BACKUP/repository.git"
git init --bare "$vla_stage"
git --git-dir="$vla_stage" fetch --no-tags "$vla_git" '+refs/*:refs/*'
awk '/^HEAD / {head=$2} /^detached$/ {print head}' "$VLA_BACKUP/source-worktree-heads.txt" |
  sort -u > "$VLA_BACKUP/detached-heads.txt"
while IFS= read -r revision; do
  git cat-file -e "$revision^{commit}"
  git --git-dir="$vla_stage" fetch --no-tags "$vla_git" \
    "$revision:refs/heads/backup-detached/$revision"
done < "$VLA_BACKUP/detached-heads.txt"
git --git-dir="$vla_stage" bundle create "$VLA_BACKUP/worldmind-history.bundle" --all
git --git-dir="$vla_stage" bundle verify "$VLA_BACKUP/worldmind-history.bundle" \
  > "$VLA_BACKUP/bundle-verification.txt" 2>&1
git bundle list-heads "$VLA_BACKUP/worldmind-history.bundle" > "$VLA_BACKUP/bundled-git-refs.txt"
while IFS= read -r ref; do
  grep -Fqx "$ref" "$VLA_BACKUP/bundled-git-refs.txt"
done < "$VLA_BACKUP/source-git-refs.txt"
while IFS= read -r revision; do
  grep -Fqx "$revision refs/heads/backup-detached/$revision" "$VLA_BACKUP/bundled-git-refs.txt"
done < "$VLA_BACKUP/detached-heads.txt"

for tree in "${vla_trees[@]}"; do
  rsync -aH --exclude='/.git' --exclude='/.venv*/' \
    "$HOME/programs/$tree/" "$VLA_BACKUP/worktrees/$tree/"
done
for original in "${vla_originals[@]}"; do
  rsync -aH "$HOME/Downloads/$original" "$VLA_BACKUP/originals/"
done
for archive in "${vla_external[@]}"; do
  rsync -aHR "$archive" "$VLA_BACKUP/external/"
done

{
  for tree in "${vla_trees[@]}"; do
    rsync -aHnci --exclude='/.git' --exclude='/.venv*/' \
      "$HOME/programs/$tree/" "$VLA_BACKUP/worktrees/$tree/"
  done
  for original in "${vla_originals[@]}"; do
    rsync -aHnci "$HOME/Downloads/$original" "$VLA_BACKUP/originals/"
  done
  for archive in "${vla_external[@]}"; do
    rsync -aHRnci "$archive" "$VLA_BACKUP/external/"
  done
} | tee "$VLA_BACKUP/verification-differences.txt"
test ! -s "$VLA_BACKUP/verification-differences.txt"
cd "$VLA_BACKUP/worktrees/VLA-Project-week3"
sha256sum -c docs/evidence/worldmind-submission.sha256
sha256sum -c docs/figures/SHA256SUMS
(cd "$VLA_BACKUP/originals" && find . -type f -print0 | sort -z | xargs -0 sha256sum) | tee "$VLA_BACKUP/originals.sha256"
```

Compare the bundle refs with the source ref list and retain the verification
outputs. If protected evidence is linked to an archive outside the listed
worktrees, copy and verify the archive itself as well. The byte verification
covers the stored representation. Keep any existing logical-payload manifests
alongside it. Verification is complete only after all checks succeed on the
external destination.
