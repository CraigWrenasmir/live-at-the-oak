#!/usr/bin/env python3
"""First-pass chart generator for Live at The Oak.

Beat-tracks the song, detects onsets, snaps them to a 16th-note grid,
and assigns each note to one of 4 lanes by spectral centroid at the
onset (low frequency content -> left lanes, bright -> right lanes).

Output is a chart JSON meant to be hand-tuned in the chart editor later.

Usage: make_chart.py <audio file> <out.json> [min_gap] [strength_floor_pct]
"""
import json
import sys
from pathlib import Path

import numpy as np
import librosa

LANES = 4
MIN_GAP = 0.11          # seconds; drop onsets closer together than this
CHORD_STRENGTH = 0.92   # percentile of onset strength that earns a 2-note chord


def main():
    audio_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    min_gap = float(sys.argv[3]) if len(sys.argv) > 3 else MIN_GAP
    floor_pct = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = float(len(y) / sr)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    tempo = float(np.atleast_1d(tempo)[0])

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, backtrack=False)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    onset_strengths = onset_env[onset_frames]

    # 16th-note grid extended across the whole track from the beat grid
    if len(beats) >= 2:
        beat_len = float(np.median(np.diff(beats)))
    else:
        beat_len = 60.0 / tempo
    grid_start = float(beats[0]) if len(beats) else 0.0
    step = beat_len / 4.0
    n_steps = int((duration - grid_start) / step) + 1
    grid = grid_start + np.arange(n_steps) * step

    # spectral centroid per frame for lane assignment
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    strong = np.percentile(onset_strengths, CHORD_STRENGTH * 100)
    floor = np.percentile(onset_strengths, floor_pct) if floor_pct else -np.inf

    # quantile boundaries so lanes are used evenly, ordered low->bright
    onset_centroids = centroid[np.minimum(onset_frames, len(centroid) - 1)]
    lane_bounds = np.percentile(onset_centroids, [25, 50, 75])

    notes = []
    last_t = -1.0
    for t, s, f in zip(onset_times, onset_strengths, onset_frames):
        if s < floor:
            continue
        # snap to grid
        t = float(grid[np.argmin(np.abs(grid - t))]) if len(grid) else float(t)
        if t - last_t < min_gap:
            continue
        last_t = t

        c = centroid[min(f, len(centroid) - 1)]
        lane = int(np.searchsorted(lane_bounds, c))

        notes.append({"t": round(t, 3), "lane": lane})
        if s >= strong:
            # strong hit: add a chord partner one lane over
            partner = lane + 1 if lane < LANES - 1 else lane - 1
            notes.append({"t": round(t, 3), "lane": partner})

    chart = {
        "title": audio_path.stem,
        "artist": "Wrenasmir",
        "album": "Live at The Oak",
        "audio": f"audio/{audio_path.name}",
        "bpm": round(tempo, 2),
        "beatOffset": round(grid_start, 3),
        "duration": round(duration, 2),
        "lanes": LANES,
        "notes": notes,
    }
    out_path.write_text(json.dumps(chart, indent=1))
    print(f"{audio_path.name}: {tempo:.1f} BPM, {len(notes)} notes over {duration:.0f}s "
          f"({len(notes)/duration:.1f} notes/sec)")


if __name__ == "__main__":
    main()
