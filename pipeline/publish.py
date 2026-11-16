#!/usr/bin/env python3
"""Merge per-worker clip results into the gallery index (clips.json)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "clips.json"
PART = ROOT / "clips.partial"


def main():
    idx = {"jobs": []}
    if INDEX.exists():
        try:
            idx = json.loads(INDEX.read_text())
        except Exception:
            pass
    entries = {j.get("id"): j for j in idx.get("jobs", []) if j.get("id")}
    merged = 0
    if PART.exists():
        for f in sorted(PART.glob("*.json")):
            try:
                e = json.loads(f.read_text())
            except Exception:
                continue
            if e.get("id"):
                entries[e["id"]] = e
                merged += 1
    jobs = sorted(entries.values(), key=lambda j: (j.get("when") or "", j.get("id") or ""), reverse=True)[:25]
    INDEX.write_text(json.dumps({"jobs": jobs}, indent=1))
    print(f"merged {merged} worker results, {len(jobs)} jobs in index")
    for f in sorted(PART.glob("*.json")):
        f.unlink()


if __name__ == "__main__":
    main()
