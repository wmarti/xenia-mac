#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_XENIA_CANDIDATES = [
    "./build/bin/Release/xenia",
    "./build/bin/Debug/xenia",
    "./build/bin/xenia",
    "./xenia",
    "./build/bin/Release/xenia.app/Contents/MacOS/xenia",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2))


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def find_xenia_binary(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    env = os.environ.get("XENIA_BIN")
    if env:
        return Path(env)
    for candidate in DEFAULT_XENIA_CANDIDATES:
        path = Path(candidate)
        if path.exists() and path.is_file():
            return path
    raise SystemExit("Could not find xenia binary. Pass --xenia or set XENIA_BIN.")


def run_process(cmd: list[str], log_path: Path, timeout: int) -> dict:
    start = time.time()
    with log_path.open("w") as log_file:
        log_file.write("COMMAND: {}\n".format(" ".join(cmd)))
        log_file.flush()
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
        try:
            exit_code = proc.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = None
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        duration = time.time() - start
    return {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_sec": round(duration, 3),
    }


def classify(log_text: str, timed_out: bool, exit_code: int | None, milestones: list[dict]) -> tuple[str, int, str]:
    if timed_out:
        status = "timeout"
    elif "CRASH DUMP" in log_text:
        status = "crash"
    elif re.search(r"Fatal error|assert", log_text, re.IGNORECASE):
        status = "crash"
    elif exit_code == 0:
        status = "exited"
    else:
        status = "error"

    best_rank = -1
    best_name = ""
    for m in milestones:
        pattern = m.get("pattern", "")
        try:
            if re.search(pattern, log_text):
                rank = int(m.get("rank", 0))
                if rank > best_rank:
                    best_rank = rank
                    best_name = m.get("name", "")
        except re.error:
            continue
    return status, best_rank, best_name


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Xenia boot tests for all games in a manifest.")
    parser.add_argument("--manifest", default=str(Path(__file__).with_name("manifest.json")))
    parser.add_argument("--milestones", default=str(Path(__file__).with_name("milestones.json")))
    parser.add_argument("--xenia", default=None)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--runs-dir", default=str(Path(__file__).with_name("runs")))
    parser.add_argument("--filter", default="")
    parser.add_argument("--args", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    milestones_path = Path(args.milestones)
    if not milestones_path.exists():
        raise SystemExit(f"Milestones not found: {milestones_path}")

    manifest = load_json(manifest_path)
    milestones = load_json(milestones_path)

    xenia_bin = find_xenia_binary(args.xenia)
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = Path(args.runs_dir)
    run_dir = runs_dir / f"{timestamp}_{commit}"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Copy manifest and milestones for reproducibility
    shutil.copy2(manifest_path, run_dir / "manifest.json")
    shutil.copy2(milestones_path, run_dir / "milestones.json")

    filter_re = re.compile(args.filter, re.IGNORECASE) if args.filter else None

    results = []
    for entry in manifest.get("entries", []):
        title = entry.get("title", "")
        launch_path = entry.get("launch_path", "")
        if filter_re and not filter_re.search(title):
            continue
        if not launch_path:
            results.append({
                "title": title,
                "root": entry.get("root", ""),
                "launch_path": "",
                "signature": entry.get("signature", ""),
                "status": "no_launch_path",
                "milestone": "",
                "milestone_rank": -1,
                "exit_code": None,
                "duration_sec": 0,
                "log_path": "",
            })
            continue

        log_name = slugify(title) or "untitled"
        log_path = logs_dir / f"{log_name}.log"

        cmd = [str(xenia_bin), f"--log_file={log_path}"]
        if args.args:
            cmd.extend(args.args.split())
        cmd.append(launch_path)

        if args.dry_run:
            result = {"exit_code": None, "timed_out": False, "duration_sec": 0}
            status, best_rank, best_name = "dry_run", -1, ""
        else:
            result = run_process(cmd, log_path, args.timeout)
            log_text = log_path.read_text(errors="ignore") if log_path.exists() else ""
            status, best_rank, best_name = classify(
                log_text, result["timed_out"], result["exit_code"], milestones
            )

        results.append({
            "title": title,
            "root": entry.get("root", ""),
            "launch_path": launch_path,
            "signature": entry.get("signature", ""),
            "status": status,
            "milestone": best_name,
            "milestone_rank": best_rank,
            "exit_code": result["exit_code"],
            "duration_sec": result["duration_sec"],
            "log_path": str(log_path.relative_to(run_dir)) if log_path.exists() else "",
        })

    summary = {
        "run": {
            "commit": commit,
            "timestamp": timestamp,
            "xenia": str(xenia_bin),
            "args": args.args,
            "timeout": args.timeout,
            "games_root": manifest.get("games_root", ""),
        },
        "results": results,
    }

    save_json(run_dir / "summary.json", summary)

    # Write CSV
    csv_path = run_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else [])
        if results:
            writer.writeheader()
            writer.writerows(results)

    print(f"Run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
