#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

SIGNATURES = {
    b"XEX2": "XEX2",
    b"XEX1": "XEX1",
    b"CON ": "CON",
    b"LIVE": "LIVE",
    b"PIRS": "PIRS",
    b"XSF ": "XISO",
}

PRIORITY = {
    "XEX2": 100,
    "XEX1": 90,
    "XISO": 80,
    "CON": 70,
    "LIVE": 70,
    "PIRS": 70,
}


def detect_signature(path: Path) -> Optional[str]:
    try:
        with path.open("rb") as f:
            magic = f.read(4)
    except OSError:
        return None
    return SIGNATURES.get(magic)


def find_candidates(root: Path, max_depth: int) -> list[Tuple[int, Path, str]]:
    candidates = []
    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).parts) - root_depth
        if depth > max_depth:
            dirnames[:] = []
            continue
        for name in filenames:
            path = Path(dirpath) / name
            sig = detect_signature(path)
            if sig:
                rank = PRIORITY.get(sig, 0)
                if name.lower() == "default.xex":
                    rank += 1000
                candidates.append((rank, path, sig))
    return candidates


def choose_launch_path(root: Path, max_depth: int) -> Tuple[Optional[Path], Optional[str]]:
    candidates = find_candidates(root, max_depth)
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: (-x[0], str(x[1])))
    _, path, sig = candidates[0]
    return path, sig


def build_manifest(games_root: Path, max_depth: int) -> dict:
    entries = []
    for entry in sorted(games_root.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            launch_path, sig = choose_launch_path(entry, max_depth)
            entries.append({
                "title": entry.name,
                "root": str(entry),
                "launch_path": str(launch_path) if launch_path else "",
                "signature": sig or "",
                "notes": "",
            })
        elif entry.is_file():
            sig = detect_signature(entry)
            if sig:
                entries.append({
                    "title": entry.stem,
                    "root": str(entry),
                    "launch_path": str(entry),
                    "signature": sig,
                    "notes": "",
                })
    return {
        "games_root": str(games_root),
        "max_depth": max_depth,
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Xenia boot test manifest.")
    parser.add_argument("--games-root", default="/Users/admin/Documents/X360-Games")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--out", default=str(Path(__file__).with_name("manifest.json")))
    args = parser.parse_args()

    games_root = Path(args.games_root)
    if not games_root.exists():
        raise SystemExit(f"Games root not found: {games_root}")

    manifest = build_manifest(games_root, args.max_depth)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
