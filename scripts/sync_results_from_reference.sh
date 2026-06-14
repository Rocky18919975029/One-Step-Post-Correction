#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Mirror a results directory from a reference server to this machine.

Run this script on the target server:

  scripts/sync_results_from_reference.sh \
    --reference rocky@reference-host:/path/to/results/run \
    --target /path/to/results/run

The script first compares file contents with rsync checksums. If differences
exist, the target is made identical to the reference, including deletion of
files that only exist on the target.

Options:
  --reference SRC   Reference directory. May be local or user@host:/path.
  --target DIR      Local target directory.
  --ssh-option OPT  Additional SSH option; may be specified multiple times.
  --dry-run         Report differences without synchronizing.
  --no-delete       Do not remove files that only exist in the target.
  -h, --help        Show this help.
EOF
}

reference=""
target=""
dry_run=0
delete_extra=1
ssh_options=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reference)
      reference="${2:?Missing value for --reference}"
      shift 2
      ;;
    --target)
      target="${2:?Missing value for --target}"
      shift 2
      ;;
    --ssh-option)
      ssh_options+=("${2:?Missing value for --ssh-option}")
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --no-delete)
      delete_extra=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$reference" || -z "$target" ]]; then
  echo "Both --reference and --target are required." >&2
  usage >&2
  exit 2
fi

if [[ "$target" == *:* ]]; then
  echo "--target must be a local path. Run this script on the target server." >&2
  exit 2
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required but was not found." >&2
  exit 1
fi

# A trailing slash means mirror the directory contents, not its parent name.
reference="${reference%/}/"
target="${target%/}/"
mkdir -p "$target"

rsync_args=(
  --archive
  --hard-links
  --checksum
  --itemize-changes
  --human-readable
  --partial
  --partial-dir=.rsync-partial
)

if [[ $delete_extra -eq 1 ]]; then
  rsync_args+=(--delete --delete-after)
fi

if [[ ${#ssh_options[@]} -gt 0 ]]; then
  ssh_command=(ssh)
  for option in "${ssh_options[@]}"; do
    ssh_command+=("$option")
  done
  printf -v ssh_command_string '%q ' "${ssh_command[@]}"
  rsync_args+=(--rsh "${ssh_command_string% }")
fi

compare_output="$(mktemp)"
trap 'rm -f "$compare_output"' EXIT

echo "Reference: $reference"
echo "Target:    $target"
echo "Checking file contents..."

rsync "${rsync_args[@]}" --dry-run "$reference" "$target" >"$compare_output"

if [[ ! -s "$compare_output" ]]; then
  echo "Directories are identical. Nothing to synchronize."
  exit 0
fi

echo "Differences found:"
cat "$compare_output"

if [[ $dry_run -eq 1 ]]; then
  echo "Dry run requested; no files were changed."
  exit 0
fi

echo "Synchronizing target from reference..."
rsync "${rsync_args[@]}" --progress "$reference" "$target"

echo "Verifying synchronized contents..."
: >"$compare_output"
rsync "${rsync_args[@]}" --dry-run "$reference" "$target" >"$compare_output"
if [[ -s "$compare_output" ]]; then
  echo "Verification failed; differences remain:" >&2
  cat "$compare_output" >&2
  exit 1
fi

echo "Synchronization complete. Target matches reference."
