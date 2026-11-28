"""Discover dated raw CSV snapshots; never calculate or rewrite their contents."""

import argparse
from datetime import date
import json
from pathlib import Path


def discover(root):
    runs = []
    for folder in root.iterdir():
        if not folder.is_dir() or not (folder / "results.csv").is_file():
            continue
        try:
            parsed = date.fromisoformat(folder.name)
        except ValueError:
            continue
        if parsed.isoformat() != folder.name:
            continue
        runs.append({"id": folder.name, "games": (folder / "games.csv").is_file()})
    return sorted(runs, key=lambda run: run["id"], reverse=True)


def generate(root):
    manifest = {"runs": discover(root)}
    output = root / "runs.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    result = generate(args.root)
    print(f"Indexed {len(result['runs'])} tournament snapshots.")
