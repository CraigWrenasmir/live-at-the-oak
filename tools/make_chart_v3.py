#!/usr/bin/env python3
"""Chart generator v3 — plays the song, not the grid.

What's new over v2:
- HPSS: drums are detected on the percussive layer (no bass-note bleed),
  melody on the harmonic layer.
- DYNAMIC beat grid: every beat's true position from the live performance,
  so charts breathe with the band instead of a fixed metronome.
- Tiles land on TRUE transient times — the grid is only used for
  structural decisions, never to move a tile off its sound.
- Melody contour layer (the Guitar Hero lesson): pitch-tracked lines
  become tiles whose lanes walk up as the tune rises and down as it
  falls; sustained notes become holds.
- Intro ease-in: the first bars keep only their strongest hits.
- v2's monotone-run breaker and sparse-section filler retained.

Writes both difficulties to charts/drafts/ for ?draft A/B testing.

Usage: make_chart_v3.py <audio> <slug>
"""
import json
import sys
from pathlib import Path

import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt

LANES = 4
HOP = 512


def band(y, sr, lo=None, hi=None):
    if lo and hi:
        sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    elif lo:
        sos = butter(4, lo, btype="high", fs=sr, output="sos")
    else:
        sos = butter(4, hi, btype="low", fs=sr, output="sos")
    return sosfiltfilt(sos, y)


def onsets_of(y, sr):
    env = librosa.onset.onset_strength(y=y, sr=sr)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr)
    if len(frames) == 0:
        return np.array([]), np.array([])
    return librosa.frames_to_time(frames, sr=sr, hop_length=HOP), env[frames]


def analyze(audio_path):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = float(len(y) / sr)
    y_harm, y_perc = librosa.effects.hpss(y)

    # dynamic beat grid — the live performance's actual beats
    tempo, beats = librosa.beat.beat_track(y=y_perc, sr=sr, units="time", trim=False)
    tempo = float(np.atleast_1d(tempo)[0])
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 8:                      # degenerate tracking: even grid
        bl = 60.0 / tempo
        beats = np.arange(0, duration, bl)
    # extend the grid to song end with the last interval
    tail = np.median(np.diff(beats[-9:])) if len(beats) > 9 else 60.0 / tempo
    while beats[-1] + tail < duration:
        beats = np.append(beats, beats[-1] + tail)

    # 16th slots that stretch with the tempo
    slots = []
    for a, b in zip(beats[:-1], beats[1:]):
        slots.extend(np.linspace(a, b, 4, endpoint=False))
    slots = np.array(slots)

    # percussion on the percussive layer, per band
    kick_t, kick_s = onsets_of(band(y_perc, sr, hi=140), sr)
    snare_t, snare_s = onsets_of(band(y_perc, sr, lo=140, hi=2500), sr)
    hat_t, hat_s = onsets_of(band(y_perc, sr, lo=4000), sr)

    # melody on the harmonic layer
    f0, voiced, vprob = librosa.pyin(y_harm, fmin=65, fmax=900, sr=sr,
                                     frame_length=2048, hop_length=HOP)
    t_frames = librosa.times_like(f0, sr=sr, hop_length=HOP)
    harm_rms = librosa.feature.rms(y=y_harm, hop_length=HOP)[0]
    rms_gate = np.percentile(harm_rms[harm_rms > 0], 35)

    melody = []   # (t, dur, midi)
    i = 0
    n = len(f0)
    rms_gate = np.percentile(harm_rms[harm_rms > 0], 25)
    while i < n:
        if not voiced[i] or vprob[i] < 0.35 or harm_rms[min(i, len(harm_rms)-1)] < rms_gate:
            i += 1
            continue
        j = i
        base = librosa.hz_to_midi(f0[i])
        while (j + 1 < n and voiced[j + 1] and vprob[j + 1] >= 0.3 and
               abs(librosa.hz_to_midi(f0[j + 1]) - base) < 1.2):
            j += 1
        dur = t_frames[j] - t_frames[i]
        if dur >= 0.10:
            seg = librosa.hz_to_midi(f0[i:j + 1])
            melody.append((float(t_frames[i]), float(dur), float(np.nanmedian(seg))))
        i = j + 1

    rms = librosa.feature.rms(y=y)[0]
    rms_t = librosa.times_like(rms)
    chroma = librosa.feature.chroma_stft(y=y_harm, sr=sr)
    chroma_t = librosa.times_like(chroma[0], sr=sr)

    return dict(duration=duration, tempo=tempo, beats=beats, slots=slots,
                kick=(kick_t, kick_s), snare=(snare_t, snare_s), hat=(hat_t, hat_s),
                melody=melody, rms=rms, rms_t=rms_t, chroma=chroma, chroma_t=chroma_t)


def emit(A, min_gap, floor_pct, hats, melody_density):
    beats, slots = A["beats"], A["slots"]
    duration = A["duration"]

    def floor(v, pct):
        return np.percentile(v, pct) if len(v) and pct > 0 else -np.inf

    kt, ks = A["kick"]; st, ss = A["snare"]; ht, hs = A["hat"]
    kf, sf = floor(ks, floor_pct), floor(ss, floor_pct)
    hf = floor(hs, max(floor_pct, 30))

    events = []   # (t, lane, priority)
    for t, s in zip(kt, ks):
        if s >= kf:
            events.append((float(t), 0, 4))
    accent = np.percentile(ss, 75) if len(ss) else 0
    alt = 0
    for t, s in zip(st, ss):
        if s < sf:
            continue
        if s >= accent:
            events.append((float(t), 3, 3))
        else:
            events.append((float(t), 1 + alt % 2, 3))
            alt += 1
    if hats:
        for t, s in zip(ht, hs):
            if s >= hf:
                events.append((float(t), 3, 1))

    # percussion density per 2s window, for melody placement
    perc_times = np.array(sorted(e[0] for e in events)) if events else np.array([0.0])

    def perc_density(t):
        return np.sum(np.abs(perc_times - t) < 1.0) / 2.0

    # melody contour layer: lanes walk with the tune
    mel_notes = []
    midis = np.array([m for _, _, m in A["melody"]]) if A["melody"] else np.array([60.0])
    qs = np.percentile(midis, [25, 50, 75])
    prev_lane, prev_midi, prev_t = None, None, -9
    for t, dur, midi in A["melody"]:
        if perc_density(t) > melody_density:
            prev_lane = None
            continue
        if prev_lane is None or t - prev_t > 0.8:
            lane = int(np.searchsorted(qs, midi))          # phrase starts at register
        else:
            step = 0 if abs(midi - prev_midi) < 0.5 else (1 if midi > prev_midi else -1)
            lane = int(np.clip(prev_lane + step, 0, 3))
        note = {"t": round(float(t), 3), "lane": lane}
        if dur >= 1.1:
            bl = float(np.median(np.diff(beats)))
            note["len"] = round(float(min(dur, 4 * bl)), 3)
        mel_notes.append((note, 2))
        prev_lane, prev_midi, prev_t = lane, midi, t

    # merge with priorities and gaps; timestamps stay TRUE
    merged = sorted([(t, {"t": round(t, 3), "lane": l}, p) for t, l, p in events] +
                    [(n["t"], n, p) for n, p in mel_notes],
                    key=lambda e: (e[0], -e[2]))
    notes = []
    last_any, last_by_lane = -9.9, {l: -9.9 for l in range(LANES)}
    for t, note, pri in merged:
        if t - last_by_lane[note["lane"]] < min_gap * 1.6:
            continue
        if abs(t - last_any) > 1e-9 and t - last_any < min_gap:
            continue
        notes.append(dict(note))
        last_by_lane[note["lane"]] = t
        last_any = t

    # intro ease-in: first 8 beats keep only their strongest half
    if len(beats) > 8:
        ease_end = beats[8]
        intro = [n for n in notes if n["t"] < ease_end]
        if len(intro) > 4:
            keep = set(id(n) for n in intro[::2])
            notes = [n for n in notes if n["t"] >= ease_end or id(n) in keep]

    # v2 passes: fill sparse-loud bins, break monotone runs
    rms, rms_t = A["rms"], A["rms_t"]
    bin_w = 2.0
    n_bins = int(duration / bin_w) + 1
    dens = np.zeros(n_bins)
    for n in notes:
        dens[int(n["t"] / bin_w)] += 1
    rms_bin = np.array([rms[(rms_t >= i*bin_w) & (rms_t < (i+1)*bin_w)].mean()
                        if ((rms_t >= i*bin_w) & (rms_t < (i+1)*bin_w)).any() else 0
                        for i in range(n_bins)])
    live = dens[dens > 0]
    med_d = np.median(live) if len(live) else 0
    med_r = np.median(rms_bin[rms_bin > 0]) if (rms_bin > 0).any() else 0
    pool = sorted([(s, t) for t, s in zip(*A["kick"])] +
                  [(s, t) for t, s in zip(*A["snare"])] +
                  [(s, t) for t, s in zip(*A["hat"])], reverse=True)
    if med_d > 0:
        taps_t = np.array(sorted(n["t"] for n in notes))
        for i in range(n_bins):
            if dens[i] >= 0.45 * med_d or rms_bin[i] < 0.75 * med_r:
                continue
            want = int(0.8 * med_d - dens[i])
            for s, t in pool:
                if want <= 0:
                    break
                if not (i * bin_w <= t < (i+1) * bin_w):
                    continue
                if len(taps_t) and np.abs(taps_t - t).min() < min_gap * 0.8:
                    continue
                lane = int(np.searchsorted(qs, 60))   # neutral middle register
                notes.append({"t": round(float(t), 3), "lane": 1 + (want % 2)})
                taps_t = np.append(taps_t, t)
                want -= 1
    notes.sort(key=lambda n: n["t"])

    run = []
    for n in notes + [{"t": -1, "lane": -1}]:
        if run and n["lane"] == run[-1]["lane"] and n["t"] != run[-1]["t"]:
            run.append(n)
            continue
        if len(run) >= 5:
            base = run[0]["lane"]
            for j, m in enumerate(run):
                if j % 2:
                    m["lane"] = (base + (1, 3, 2)[(j // 2) % 3]) % LANES
        run = [n] if n["lane"] >= 0 else []

    # v2's pad holds: percussion-free but energetic stretches get sustains
    # through to the end of the song (only where the melody layer left room)
    bl = float(np.median(np.diff(beats)))
    hold_max = 4.0 * bl
    edges = [0.0] + sorted(n["t"] + n.get("len", 0) for n in notes) + [duration]
    rms_floor = np.percentile(rms[rms > 0], 8) if (rms > 0).any() else 0
    chroma, chroma_t = A["chroma"], A["chroma_t"]
    gap_holds = []
    for a, b in zip(edges[:-1], edges[1:]):
        a, b = a + 0.45, b - 0.45
        if b - a < 2.0:
            continue
        seg = rms[(rms_t >= a) & (rms_t <= b)]
        if len(seg) == 0 or seg.mean() < rms_floor:
            continue
        t0 = a
        while t0 + bl * 0.9 < b:
            ln = min(hold_max, b - t0)
            ln = max(bl, round(ln / bl) * bl)
            ln = min(ln, b - t0)
            m = (chroma_t >= t0) & (chroma_t <= t0 + ln)
            reg = (float((chroma[:, m].mean(axis=1) * np.arange(12)).sum() /
                         max(chroma[:, m].mean(axis=1).sum(), 1e-6)) if m.any() else 5.5)
            gap_holds.append({"t": round(t0, 3), "lane": reg, "len": round(float(ln), 3)})
            t0 += ln + bl
    if gap_holds:
        regs = np.array([h["lane"] for h in gap_holds], dtype=float)
        order = regs.argsort().argsort()
        for h, r in zip(gap_holds, order):
            h["lane"] = int(r * LANES / len(gap_holds))
    notes += gap_holds
    notes.sort(key=lambda n: n["t"])
    return notes


def main():
    audio_path = Path(sys.argv[1])
    slug = sys.argv[2]
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "charts" / "drafts"
    out_dir.mkdir(exist_ok=True)

    A = analyze(audio_path)
    base = json.loads((root / "charts" / f"{slug}.json").read_text())

    for diff, (min_gap, floor_pct, hats, mel_d) in {
        "standard": (0.26, 55, False, 2.2),
        "hard": (0.14, 30, True, 3.5),
    }.items():
        notes = emit(A, min_gap, floor_pct, hats, mel_d)
        chart = dict(base)
        chart["notes"] = notes
        chart["beats"] = [round(float(b), 3) for b in A["beats"]]
        chart["generator"] = "v3"
        name = f"{slug}.json" if diff == "standard" else f"{slug}-hard.json"
        (out_dir / name).write_text(json.dumps(chart, indent=1))
        holds = sum(1 for n in notes if "len" in n)
        mel = sum(1 for n in notes if n.get("_m"))
        print(f"{slug} [{diff}]: {len(notes)-holds} taps + {holds} holds "
              f"({len(A['melody'])} melody notes found, {len(A['beats'])} live beats)")


if __name__ == "__main__":
    main()
