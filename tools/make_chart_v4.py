#!/usr/bin/env python3
"""Chart generator v4 — for grid-authored music (Ableton, strict BPM).

The songs were sequenced on a rigid grid, so the grid IS the truth:
- One exact (BPM, phase) fitted for the whole song by maximising the
  alignment of strong percussive onsets to a 16th grid (circular fit).
  Pass the known Ableton BPM to pin it exactly: --bpm 120
- Detections snap TO the grid (the sequencer put them there).
- Bar-fold pattern voting: each (slot-in-bar, band) must recur across
  neighbouring bars to count — one-off detector noise dies, programmed
  patterns survive. Strong one-offs live on as accents (real fills).
- Melody tiles require a real attack (no tiles on pad swells); sustained
  confident lines become holds. Pad holds and intro ease-in retained.
- No sparse-filler: density now comes from the actual patterns.

Writes both difficulties to charts/drafts/.

Usage: make_chart_v4.py <audio> <slug> [--bpm N]
"""
import json
import sys
from pathlib import Path

import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt

LANES = 4
HOP = 512
SLOTS_PER_BAR = 16          # 4/4, 16th resolution


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


def grid_concentration(times, weights, slot):
    ph = (times % slot) / slot * 2 * np.pi
    z = np.sum(weights * np.exp(1j * ph))
    return np.abs(z) / max(np.sum(weights), 1e-9), np.angle(z)


def fit_grid(times, weights, bpm0, fixed_bpm=None):
    """Exact (bpm, offset) by maximising onset concentration on the 16th grid."""
    if fixed_bpm:
        cands = [float(fixed_bpm)]
    else:
        cands = list(np.arange(bpm0 - 2.0, bpm0 + 2.0, 0.01))
        # sequencers love round numbers — try them at full precision too
        cands += [round(bpm0), round(bpm0 * 2) / 2]
    best = (-1, bpm0, 0.0)
    for bpm in cands:
        slot = 60.0 / bpm / 4
        R, ang = grid_concentration(times, weights, slot)
        if R > best[0]:
            offset = (ang / (2 * np.pi)) * slot % slot
            best = (R, float(bpm), float(offset))
    return best  # (R, bpm, offset)


def main():
    audio_path = Path(sys.argv[1])
    slug = sys.argv[2]
    fixed_bpm = None
    if "--bpm" in sys.argv:
        fixed_bpm = float(sys.argv[sys.argv.index("--bpm") + 1])

    root = Path(__file__).resolve().parent.parent
    out_dir = root / "charts" / "drafts"
    out_dir.mkdir(exist_ok=True)

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = float(len(y) / sr)
    y_harm, y_perc = librosa.effects.hpss(y)

    tempo0, _ = librosa.beat.beat_track(y=y_perc, sr=sr, units="time")
    tempo0 = float(np.atleast_1d(tempo0)[0])

    kick_t, kick_s = onsets_of(band(y_perc, sr, hi=140), sr)
    snare_t, snare_s = onsets_of(band(y_perc, sr, lo=140, hi=2500), sr)
    hat_t, hat_s = onsets_of(band(y_perc, sr, lo=4000), sr)

    # fit the sequencer grid on strong kick+snare onsets
    ft = np.concatenate([kick_t, snare_t])
    fw = np.concatenate([kick_s, snare_s])
    strong = fw >= np.percentile(fw, 40)
    R, bpm, offset = fit_grid(ft[strong], fw[strong], tempo0, fixed_bpm)
    slot = 60.0 / bpm / 4
    print(f"{slug}: grid fit bpm={bpm:.3f} offset={offset*1000:.0f}ms "
          f"concentration R={R:.2f} {'(LOW - grid uncertain!)' if R < 0.5 else '(strict grid confirmed)'}")

    def to_slot(t):
        return int(round((t - offset) / slot))

    def slot_time(k):
        return offset + k * slot

    # bar-fold pattern voting per band
    def voted(times, strengths, min_support, accent_pct, floor_pct):
        if len(times) == 0:
            return []
        floor = np.percentile(strengths, floor_pct) if floor_pct else -np.inf
        m = strengths >= floor
        times, strengths = times[m], strengths[m]
        if len(times) == 0:
            return []
        ks = np.array([to_slot(t) for t in times])
        bars = ks // SLOTS_PER_BAR
        pos = ks % SLOTS_PER_BAR
        accent = np.percentile(strengths, accent_pct)
        keep = []
        for i in range(len(times)):
            window = (np.abs(bars - bars[i]) <= 4) & (pos == pos[i])
            support = np.sum(window)
            if support >= min_support or strengths[i] >= accent:
                keep.append((slot_time(ks[i]), strengths[i]))
        return keep

    # melody on the harmonic layer, attack-gated
    f0, voiced, vprob = librosa.pyin(y_harm, fmin=65, fmax=900, sr=sr,
                                     frame_length=2048, hop_length=HOP)
    t_frames = librosa.times_like(f0, sr=sr, hop_length=HOP)
    harm_on_t, harm_on_s = onsets_of(y_harm, sr)
    harm_rms = librosa.feature.rms(y=y_harm, hop_length=HOP)[0]
    rms_gate = np.percentile(harm_rms[harm_rms > 0], 30)
    melody = []
    i, n = 0, len(f0)
    while i < n:
        if not voiced[i] or vprob[i] < 0.4 or harm_rms[min(i, len(harm_rms)-1)] < rms_gate:
            i += 1
            continue
        j = i
        base = librosa.hz_to_midi(f0[i])
        while (j + 1 < n and voiced[j + 1] and vprob[j + 1] >= 0.35 and
               abs(librosa.hz_to_midi(f0[j + 1]) - base) < 1.2):
            j += 1
        t0, dur = t_frames[i], t_frames[j] - t_frames[i]
        has_attack = len(harm_on_t) and np.abs(harm_on_t - t0).min() < 0.08
        if dur >= 0.10 and (has_attack or dur >= 1.1):
            seg = librosa.hz_to_midi(f0[i:j + 1])
            melody.append((float(t0), float(dur), float(np.nanmedian(seg))))
        i = j + 1

    rms = librosa.feature.rms(y=y)[0]
    rms_t = librosa.times_like(rms)
    chroma = librosa.feature.chroma_stft(y=y_harm, sr=sr)
    chroma_t = librosa.times_like(chroma[0], sr=sr)

    base_chart = json.loads((root / "charts" / f"{slug}.json").read_text())

    for diff, (min_gap, support, accent_pct, hats, mel_dens, floor_pct) in {
        "standard": (0.26, 4, 92, False, 2.2, 55),
        "hard": (0.14, 3, 85, True, 3.5, 30),
    }.items():
        kick = voted(kick_t, kick_s, support, accent_pct, floor_pct)
        snare = voted(snare_t, snare_s, support, accent_pct, floor_pct)
        hat = voted(hat_t, hat_s, support + 1, 97, max(floor_pct, 40)) if hats else []

        events = [(t, 0, 4, s) for t, s in kick]
        acc = np.percentile([s for _, s in snare], 75) if snare else 0
        alt = 0
        for t, s in snare:
            if s >= acc:
                events.append((t, 3, 3, s))
            else:
                events.append((t, 1 + alt % 2, 3, s))
                alt += 1
        events += [(t, 3, 1, s) for t, s in hat]

        perc_times = np.array(sorted(e[0] for e in events)) if events else np.array([0.0])

        def perc_density(t):
            return np.sum(np.abs(perc_times - t) < 1.0) / 2.0

        mel_notes = []
        midis = np.array([m for _, _, m in melody]) if melody else np.array([60.0])
        qs = np.percentile(midis, [25, 50, 75])
        prev_lane, prev_midi, prev_t = None, None, -9
        for t0, dur, midi in melody:
            if perc_density(t0) > mel_dens:
                prev_lane = None
                continue
            ts = slot_time(to_slot(t0))          # melody snaps to the grid too
            if prev_lane is None or t0 - prev_t > 0.8:
                lane = int(np.searchsorted(qs, midi))
            else:
                step = 0 if abs(midi - prev_midi) < 0.5 else (1 if midi > prev_midi else -1)
                lane = int(np.clip(prev_lane + step, 0, 3))
            note = {"t": round(float(ts), 3), "lane": lane}
            if dur >= 1.1:
                note["len"] = round(float(min(dur, 16 * slot)), 3)
            mel_notes.append((ts, note, 2, 1.0))
            prev_lane, prev_midi, prev_t = lane, midi, t0

        merged = sorted([(t, {"t": round(float(t), 3), "lane": l}, p, s) for t, l, p, s in events] +
                        mel_notes, key=lambda e: (e[0], -e[2]))
        notes = []
        last_any, last_by_lane = -9.9, {l: -9.9 for l in range(LANES)}
        for t, note, pri, s in merged:
            if t - last_by_lane[note["lane"]] < min_gap * 1.6:
                continue
            if abs(t - last_any) > 1e-9 and t - last_any < min_gap:
                continue
            notes.append(dict(note))
            last_by_lane[note["lane"]] = t
            last_any = t

        # intro ease-in
        bar = slot * SLOTS_PER_BAR
        ease_end = offset + 2 * bar
        intro = [n_ for n_ in notes if n_["t"] < ease_end]
        if len(intro) > 4:
            keep = set(id(n_) for n_ in intro[::2])
            notes = [n_ for n_ in notes if n_["t"] >= ease_end or id(n_) in keep]

        # monotone-run breaker
        run = []
        for n_ in notes + [{"t": -1, "lane": -1}]:
            if run and n_["lane"] == run[-1]["lane"] and n_["t"] != run[-1]["t"]:
                run.append(n_)
                continue
            if len(run) >= 5:
                b0 = run[0]["lane"]
                for j, m in enumerate(run):
                    if j % 2:
                        m["lane"] = (b0 + (1, 3, 2)[(j // 2) % 3]) % LANES
            run = [n_] if n_["lane"] >= 0 else []

        # pad holds through percussion-free energetic stretches
        bl = slot * 4
        hold_max = 4 * bl
        edges = [0.0] + sorted(n_["t"] + n_.get("len", 0) for n_ in notes) + [duration]
        rms_floor = np.percentile(rms[rms > 0], 8) if (rms > 0).any() else 0
        gap_holds = []
        for a, b in zip(edges[:-1], edges[1:]):
            a, b = a + 0.45, b - 0.45
            if b - a < 2.0:
                continue
            seg = rms[(rms_t >= a) & (rms_t <= b)]
            if len(seg) == 0 or seg.mean() < rms_floor:
                continue
            t0 = slot_time(to_slot(a) + (4 - to_slot(a) % 4) % 4)   # next beat
            while t0 + bl * 0.9 < b:
                ln = min(hold_max, b - t0)
                ln = max(bl, round(ln / bl) * bl)
                ln = min(ln, b - t0)
                m = (chroma_t >= t0) & (chroma_t <= t0 + ln)
                reg = (float((chroma[:, m].mean(axis=1) * np.arange(12)).sum() /
                             max(chroma[:, m].mean(axis=1).sum(), 1e-6)) if m.any() else 5.5)
                gap_holds.append({"t": round(float(t0), 3), "lane": reg,
                                  "len": round(float(ln), 3)})
                t0 += ln + bl
        if gap_holds:
            regs = np.array([h["lane"] for h in gap_holds], dtype=float)
            order = regs.argsort().argsort()
            for h, r_ in zip(gap_holds, order):
                h["lane"] = int(r_ * LANES / len(gap_holds))
        notes += gap_holds
        notes.sort(key=lambda n_: n_["t"])

        chart = dict(base_chart)
        chart["notes"] = notes
        chart["bpm"] = round(bpm, 3)
        chart["beatOffset"] = round(offset, 4)
        chart.pop("beats", None)
        chart["generator"] = "v4"
        name = f"{slug}.json" if diff == "standard" else f"{slug}-hard.json"
        (out_dir / name).write_text(json.dumps(chart, indent=1))
        holds = sum(1 for n_ in notes if "len" in n_)
        print(f"  [{diff}] {len(notes)-holds} taps + {holds} holds")


if __name__ == "__main__":
    main()
