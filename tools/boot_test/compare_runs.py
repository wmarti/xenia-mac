#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def index_results(results: list[dict]) -> dict:
    return {r["title"]: r for r in results}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two boot test runs.")
    parser.add_argument("baseline", help="Baseline run directory")
    parser.add_argument("current", help="Current run directory")
    parser.add_argument("--out", default="diff.csv")
    args = parser.parse_args()

    base_dir = Path(args.baseline)
    curr_dir = Path(args.current)

    base = load_summary(base_dir / "summary.json")
    curr = load_summary(curr_dir / "summary.json")

    base_idx = index_results(base.get("results", []))
    curr_idx = index_results(curr.get("results", []))

    titles = sorted(set(base_idx.keys()) | set(curr_idx.keys()))
    rows = []
    for title in titles:
        b = base_idx.get(title)
        c = curr_idx.get(title)
        if not b or not c:
            continue
        delta = (c.get("milestone_rank", -1) - b.get("milestone_rank", -1))
        rows.append({
            "title": title,
            "baseline_status": b.get("status", ""),
            "current_status": c.get("status", ""),
            "baseline_milestone": b.get("milestone", ""),
            "current_milestone": c.get("milestone", ""),
            "delta_rank": delta,
        })

    out_path = curr_dir / args.out
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote diff to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
