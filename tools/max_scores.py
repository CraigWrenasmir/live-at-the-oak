#!/usr/bin/env python3
"""Compute the maximum possible score per chart (all perfects, full holds).
Mirrors the engine: combo increments per hit, mult = 1+min(3, combo//10),
tap = 100*mult, hold completion bonus = round(len*100), unmultiplied."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = {}
for f in sorted((root / "charts").glob("*.json")):
    if f.name == "setlist.json":
        continue
    c = json.loads(f.read_text())
    combo = score = 0
    for n in sorted(c["notes"], key=lambda n: n["t"]):
        combo += 1
        score += 100 * (1 + min(3, combo // 10))
        if n.get("len"):
            score += round(n["len"] * 100)
    slug = f.stem.replace("-hard", "")
    diff = "hard" if f.stem.endswith("-hard") else "standard"
    out.setdefault(slug, {})[diff] = score
print(json.dumps(out, indent=1))
(root / "charts" / "max_scores.json").write_text(json.dumps(out, indent=1))
