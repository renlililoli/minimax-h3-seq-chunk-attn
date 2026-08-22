#!/usr/bin/env bash
set -euo pipefail

example_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

for scenario in t2va fl2va ref2va_images ref2va_video; do
  "$example_dir/_run_example.sh" "$scenario" "$@"
done
