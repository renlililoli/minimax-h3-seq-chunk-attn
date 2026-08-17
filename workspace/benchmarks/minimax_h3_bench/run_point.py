from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def git_output(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def inject_docker_exec_environment(command: list[str], values: dict[str, str]) -> list[str]:
    if len(command) < 3 or command[:3] != ["docker", "compose", "exec"]:
        return command
    insert_at = 3
    while insert_at < len(command) and command[insert_at] in ("-T", "--no-TTY"):
        insert_at += 1
    injected = []
    for key, value in values.items():
        injected.extend(["-e", f"{key}={value}"])
    return command[:insert_at] + injected + command[insert_at:]


def main():
    parser = argparse.ArgumentParser(description="Run one benchmark point in an isolated process")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--container-output",
        help="Path to the same result file as seen by the worker container.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a worker command is required after --")
    metadata = {
        "MINIMAX_H3_RESULT_JSON": args.container_output or str(args.output),
        "MINIMAX_H3_MODEL_COMMIT": git_output("rev-parse", "HEAD"),
        "MINIMAX_H3_DIFFSYNTH_COMMIT": git_output(
            "-C", "extern/DiffSynth-Studio", "rev-parse", "HEAD"
        ),
        "MINIMAX_H3_MODEL_DIRTY": str(bool(git_output("status", "--porcelain"))).lower(),
        "MINIMAX_H3_DIFFSYNTH_DIRTY": str(bool(git_output(
            "-C", "extern/DiffSynth-Studio", "status", "--porcelain"
        ))).lower(),
        "MINIMAX_H3_CONTAINER_IMAGE": command_output(
            ["docker", "image", "inspect", "diffsynth:cu128", "--format", "{{.Id}}"]
        ),
    }
    environment = os.environ.copy()
    environment.update(metadata)
    command = inject_docker_exec_environment(command, metadata)
    process = subprocess.Popen(command, env=environment, start_new_session=True)
    try:
        returncode = process.wait(timeout=args.timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        atomic_write_json(args.output, {
            "status": "timeout",
            "failure_message": f"exceeded {args.timeout_seconds} seconds",
            "command": command,
        })
        return 124
    if not args.output.is_file():
        atomic_write_json(args.output, {
            "status": "runtime_error",
            "failure_message": f"worker exited {returncode} without writing JSON",
            "command": command,
        })
    return returncode


if __name__ == "__main__":
    sys.exit(main())
