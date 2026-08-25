#!/usr/bin/env bash
set -euo pipefail

repo=${1:-/workspace}
dist_dir=${SEQATTN_CHECK_DIST_DIR:-$(mktemp -d /tmp/seqattn-dist.XXXXXX)}
build_dir=$(mktemp -d /tmp/seqattn-build.XXXXXX)
source_git_dir=$(mktemp -d /tmp/seqattn-git.XXXXXX)
export PYTHONPATH="/opt/ComfyUI${PYTHONPATH:+:${PYTHONPATH}}"

cd "$repo"

ruff check --no-cache .
python -m pytest -p no:cacheprovider
git -C "$source_git_dir" init --quiet
git --git-dir="$source_git_dir/.git" --work-tree="$repo" add --all
git --git-dir="$source_git_dir/.git" --work-tree="$repo" diff --cached --check
mkdir -p "$dist_dir"
git --git-dir="$source_git_dir/.git" --work-tree="$repo" \
    checkout-index --all --prefix="$build_dir/"
python -m build --outdir "$dist_dir" "$build_dir"

printf 'built artifacts:\n'
find "$dist_dir" -maxdepth 1 -type f -printf '%f\n' | sort
