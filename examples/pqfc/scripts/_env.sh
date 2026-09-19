#!/usr/bin/env bash
# Shared env for PQFC scripts: SOURCE_ROOT = this worktree, PYTHONPATH first.
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ -z "${PROJECT_ROOT:-}" && -f "$SOURCE_ROOT/.env" ]]; then
  PROJECT_ROOT="$(awk -F= '/^[[:space:]]*PROJECT_ROOT[[:space:]]*=/ {
    sub(/^[^=]*=[[:space:]]*/, ""); print; exit
  }' "$SOURCE_ROOT/.env")"
fi
export SOURCE_ROOT
export PROJECT_ROOT="${PROJECT_ROOT:-$SOURCE_ROOT}"
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-poetry -C $PROJECT_ROOT run python}"
