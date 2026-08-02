#!/usr/bin/env python3
"""Chart generator v2 for Live at The Oak.

Craig's charting direction (Aug 2026):
- notes follow the KICK and SNARE, that's the fun part to keep time with
- quiet/pad sections get HOLD notes instead of nothing, through to the end
- hats only garnish the hard chart

Band-split onset detection: kick from the low band -> lane 0 (green),
snare from the mid band -> lanes 1/2 alternating, hats from the high
band -> lane 3 (hard only). Everything snaps to a 16th-note grid.
Gaps longer than GAP_MIN with real energy become chains of hold notes,
lane picked from the chroma register of each chunk.

Usage: make_chart.py <audio> <out.json> [min_gap] [strength_floor_pct] [--no-hats]
  standard: min_gap 0.22, floor 45, --no-hats     hard: min_gap 0.11, floor 10
"""
import json
import sys
from pathlib import Path

import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt

LANES = 4
GAP_MIN = 2.0          # seconds without percussion -> hold territory
HOLD_MAX_BEATS = 4.0   # longest single hold
RMS_FLOOR_PCT = 8      # section must have energy above this percentile
                       # (low so the outro fade still earns holds to the end)


def band(y, sr, lo=None, hi=None):
    if lo and hi:
        sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    elif lo:
        sos = butter(4, lo, btype="high", fs=sr, output="sos")
    else:
        sos = butter(4, hi, btype="low", fs=sr, output="sos")
    return sosfiltfilt(sos, y)


def onsets_of(y, sr, floor_pct):
    env = librosa.onset.onset_strength(y=y, sr=sr)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr)
    if len(frames) == 0:
        return np.array([]), np.array([])
    times = librosa.frames_to_time(frames, sr=sr)
    strengths = env[frames]
    if floor_pct > 0:
        keep = strengths >= np.percentile(strengths, floor_pct)
        times, strengths = times[keep], strengths[keep]
    return times, strengths


def main():
    audio_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    min_gap = float(sys.argv[3]) if len(sys.argv) > 3 else 0.22
    floor_pct = float(sys.argv[4]) if len(sys.argv) > 4 else 45
    hats = "--no-hats" not in sys.argv

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = float(len(y) / sr)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    beat_len = float(np.median(np.diff(beats))) if len(beats) >= 2 else 60.0 / tempo
    grid_start = float(beats[0]) if len(beats) else 0.0
    step = beat_len / 4.0
    grid = grid_start + np.arange(int((duration - grid_start) / step) + 1) * step

    def snap(t):
        return float(grid[np.argmin(np.abs(grid - t))]) if len(grid) else float(t)

    def snap_beat(t):
        k = round((t - grid_start) / beat_len)
        return grid_start + k * beat_len

    # ---- band-split percussion ----
    kick_t, kick_s = onsets_of(band(y, sr, hi=140), sr, floor_pct)
    snare_t, snare_s = onsets_of(band(y, sr, lo=140, hi=2500), sr, floor_pct)
    hat_t, hat_s = (onsets_of(band(y, sr, lo=4000), sr, max(floor_pct, 30))
                    if hats else (np.array([]), np.array([])))

    events = []   # (t, lane, priority)
    for t in kick_t:
        events.append((snap(t), 0, 3))
    # accented snares (top quartile) take the blue lane; the rest alternate 1/2
    snare_accent = np.percentile(snare_s, 75) if len(snare_s) else 0
    alt = 0
    for t, s in zip(snare_t, snare_s):
        if s >= snare_accent:
            events.append((snap(t), 3, 2))
        else:
            events.append((snap(t), 1 + alt % 2, 2))
            alt += 1
    for t in hat_t:
        events.append((snap(t), 3, 1))
    events.sort(key=lambda e: (e[0], -e[2]))

    notes = []
    last_by_lane = {l: -9.9 for l in range(LANES)}
    last_any = -9.9
    for t, lane, pri in events:
        if t - last_by_lane[lane] < min_gap * 1.6:
            continue
        # allow same-instant chords (kick+snare), otherwise enforce global gap
        if t != last_any and t - last_any < min_gap:
            continue
        notes.append({"t": round(t, 3), "lane": lane})
        last_by_lane[lane] = t
        last_any = t

    # ---- holds over the quiet stretches, through to the end ----
    rms = librosa.feature.rms(y=y)[0]
    rms_t = librosa.times_like(rms, sr=sr)
    rms_floor = np.percentile(rms, RMS_FLOOR_PCT)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_t = librosa.times_like(chroma[0], sr=sr)

    perc_times = sorted(n["t"] for n in notes)
    edges = [0.0] + perc_times + [duration]
    holds = []
    hold_max = HOLD_MAX_BEATS * beat_len
    for a, b in zip(edges[:-1], edges[1:]):
        a, b = a + 0.45, b - 0.45          # breathe around the percussion
        if b - a < GAP_MIN:
            continue
        seg = rms[(rms_t >= a) & (rms_t <= b)]
        if len(seg) == 0 or seg.mean() < rms_floor:
            continue                        # true silence stays silent
        t0 = max(a, snap_beat(a))
        while t0 + beat_len * 0.9 < b:
            ln = min(hold_max, b - t0)
            ln = max(beat_len, round(ln / beat_len) * beat_len)
            ln = min(ln, b - t0)
            m = (chroma_t >= t0) & (chroma_t <= t0 + ln)
            if m.any():
                reg = float((chroma[:, m].mean(axis=1) * np.arange(12)).sum()
                            / max(chroma[:, m].mean(axis=1).sum(), 1e-6))
            else:
                reg = 5.5
            holds.append({"t": round(t0, 3), "lane": reg, "len": round(ln, 3)})
            t0 += ln + beat_len            # a beat of air between holds
    # spread hold lanes across all four by quantile rank of register
    if holds:
        regs = np.array([h["lane"] for h in holds], dtype=float)
        order = regs.argsort().argsort()          # rank of each hold's register
        for h, r in zip(holds, order):
            h["lane"] = int(r * LANES / len(holds))
    notes += holds
    notes.sort(key=lambda n: n["t"])

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
    taps = len(notes) - len(holds)
    print(f"{audio_path.name}: {tempo:.1f} BPM, {taps} taps + {len(holds)} holds "
          f"over {duration:.0f}s (kick {len(kick_t)}, snare {len(snare_t)}, hat {len(hat_t)})")


if __name__ == "__main__":
    main()
