#!/usr/bin/env python3
"""Renders an audible chart audit: the song with its chart played as
clicks on top — lane-pitched so you can hear the lanes move.
  lane 0 (kick/green) = low thump   lane 1/2 (red/yellow) = mid ticks
  lane 3 (blue) = high tick         holds = soft sustained tone

Usage: render_audit.py <slug> [start_s] [dur_s] [--hard] [--base]
Writes promo/audits/<slug>[-hard][-base]-audit.wav
"""
import json
import sys
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

slug = sys.argv[1]
start = float(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else 60.0
dur = float(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("-") else 25.0
hard = "--hard" in sys.argv
base = "--base" in sys.argv

root = Path(__file__).resolve().parent.parent
name = slug + ("-hard" if hard else "") + ".json"
chart_path = (root / "charts" / name) if base else (root / "charts" / "drafts" / name)
chart = json.loads(chart_path.read_text())

y, sr = librosa.load(root / "audio" / f"{slug}.mp3", sr=44100, mono=True,
                     offset=start, duration=dur)
y *= 0.55

LANE_F = [180, 850, 1100, 1500]


def click(f, length, amp):
    t = np.arange(int(length * sr)) / sr
    env = np.exp(-t * (30 if length < 0.1 else 6))
    return amp * env * np.sin(2 * np.pi * f * t)


for n in chart["notes"]:
    t0 = n["t"] - start
    if t0 < 0 or t0 > dur - 0.05:
        continue
    i = int(t0 * sr)
    if "len" in n:
        c = click(LANE_F[n["lane"]] * 0.5, min(n["len"], dur - t0), 0.12)
    else:
        c = click(LANE_F[n["lane"]], 0.09, 0.5 if n["lane"] == 0 else 0.35)
    end = min(i + len(c), len(y))
    y[i:end] += c[:end - i]

y = np.clip(y, -1, 1)
out = root / "promo" / "audits"
out.mkdir(exist_ok=True)
fn = out / (slug + ("-hard" if hard else "") + ("-base" if base else "") + "-audit.wav")
sf.write(fn, y, sr)
print("wrote", fn, f"({start:.0f}s–{start+dur:.0f}s)")
