from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import tomllib


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install example dependencies from pyproject.toml."
    )
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()

    pyproject_path = args.package_root / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    dependencies = project.get("dependencies", [])
    if not dependencies:
        return 0

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--no-deps",
            "--target",
            str(args.target),
            *dependencies,
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
