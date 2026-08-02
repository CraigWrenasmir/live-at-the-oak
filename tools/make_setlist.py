#!/usr/bin/env python3
"""Build charts/setlist.json from the numbered set order + generated charts."""
import json
from pathlib import Path

SET = [(1, "rdst-gm", "Rdst Gm"), (2, "lucempight", "Lucempight"),
       (3, "spinet-destroiet", "Spinet Destroiet"), (4, "what-is-better", "What Is Better"),
       (5, "the-way-that-i-go", "The Way That I Go"), (6, "mabon", "Mabon"),
       (7, "crafluropi", "Crafluropi")]

root = Path(__file__).resolve().parent.parent
songs = []
for n, slug, title in SET:
    c = json.loads((root / "charts" / f"{slug}.json").read_text())
    songs.append({
        "n": n, "slug": slug, "title": title,
        "bpm": c["bpm"], "duration": c["duration"], "audio": c["audio"],
        "charts": {"standard": f"charts/{slug}.json", "hard": f"charts/{slug}-hard.json"},
    })
(root / "charts" / "setlist.json").write_text(json.dumps(songs, indent=1))
print("setlist:", ", ".join(f"{s['n']}. {s['title']} ({s['bpm']:.0f}bpm)" for s in songs))
