#!/usr/bin/env python3
"""Chart generator v5 — charts from the instruments, not the mix.

Requires Demucs stems in stems/htdemucs/<slug>/{drums,bass,other,vocals}.mp3.

- Grid fitted on the ISOLATED DRUM STEM (exact Ableton bpm + phase).
- Every drum tile is a real drum hit, classified kick/snare/hat by the
  band-energy profile of the isolated hit. Phantom tiles are impossible.
- Melody from the other/vocals stems (clean pitch tracking), contour
  lanes, attack-gated; bassline tiles on hard where drums leave room.
- Section energy tiers from the drum stem shape density.
- Self-audit: phantom rate, miss rate of the loudest hits, grid deviation.

Usage: make_chart_v5.py <slug> [--bpm N]
"""
import json
import sys
from pathlib import Path

import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt

LANES = 4
HOP = 512
SLOTS_PER_BAR = 16


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
    if fixed_bpm:
        cands = [float(fixed_bpm)]
    else:
        # the tempo estimator often lands on a harmonic — search around the
        # estimate AND its musical ratios, let concentration pick the winner
        cands = []
        for ratio in (1, 0.5, 2, 2/3, 3/4, 4/3, 3/2):
            c = bpm0 * ratio
            if 50 <= c <= 220:
                cands += list(np.arange(c * 0.93, c * 1.07, 0.01))   # ±7%: windows tile
    best = (-1, bpm0, 0.0)
    for bpm in cands:
        slot = 60.0 / bpm / 4
        R, ang = grid_concentration(times, weights, slot)
        if R > best[0]:
            best = (R, float(bpm), float((ang / (2 * np.pi)) * slot % slot))
    return best


def classify_hits(y, sr, times):
    """kick / snare / hat from the isolated drum stem — per-bin band energy
    (mean, not sum: wide bands must not win by bin count alone)."""
    S = np.abs(librosa.stft(y, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=sr)
    lo = S[freqs < 150].mean(axis=0)
    mid = S[(freqs >= 200) & (freqs < 2500)].mean(axis=0)
    hi = S[freqs >= 6000].mean(axis=0)
    out = []
    for t in times:
        f = int(t * sr / HOP)
        a, b = max(0, f - 1), min(len(lo), f + 4)
        e_lo, e_mid, e_hi = lo[a:b].max(), mid[a:b].max(), hi[a:b].max()
        if e_lo > 1.5 * e_mid and e_lo > 3 * e_hi:
            out.append("kick")
        elif e_hi > 2.0 * e_mid and e_lo < 0.5 * e_mid:
            out.append("hat")
        elif e_lo > e_mid:
            out.append("kick")
        else:
            out.append("snare")
    return out


def melody_notes(y, sr, fmin, fmax, vp_gate=0.4):
    if y is None or not np.any(np.abs(y) > 1e-4):
        return []
    f0, voiced, vprob = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr,
                                     frame_length=2048, hop_length=HOP)
    t_frames = librosa.times_like(f0, sr=sr, hop_length=HOP)
    on_t, _ = onsets_of(y, sr)
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    gate = np.percentile(rms[rms > 0], 30) if (rms > 0).any() else 0
    out = []
    i, n = 0, len(f0)
    while i < n:
        if not voiced[i] or vprob[i] < vp_gate or rms[min(i, len(rms)-1)] < gate:
            i += 1
            continue
        j = i
        base = librosa.hz_to_midi(f0[i])
        while (j + 1 < n and voiced[j + 1] and vprob[j + 1] >= vp_gate - 0.05 and
               abs(librosa.hz_to_midi(f0[j + 1]) - base) < 1.2):
            j += 1
        t0, dur = t_frames[i], t_frames[j] - t_frames[i]
        has_attack = len(on_t) and np.abs(on_t - t0).min() < 0.08
        if dur >= 0.10 and (has_attack or dur >= 1.1):
            seg = librosa.hz_to_midi(f0[i:j + 1])
            out.append((float(t0), float(dur), float(np.nanmedian(seg))))
        i = j + 1
    return out


def main():
    slug = sys.argv[1]
    fixed_bpm = float(sys.argv[sys.argv.index("--bpm") + 1]) if "--bpm" in sys.argv else None
    # True hit times are ALWAYS truth now (drift-proof by construction);
    # --snap restores grid quantisation if ever wanted for comparison.
    no_snap = "--snap" not in sys.argv

    root = Path(__file__).resolve().parent.parent
    stem_dir = root / "stems" / "htdemucs" / slug
    out_dir = root / "charts" / "drafts"
    out_dir.mkdir(exist_ok=True)

    drums, sr = librosa.load(stem_dir / "drums.mp3", sr=None, mono=True)
    bass, _ = librosa.load(stem_dir / "bass.mp3", sr=sr, mono=True)
    other, _ = librosa.load(stem_dir / "other.mp3", sr=sr, mono=True)
    vocals, _ = librosa.load(stem_dir / "vocals.mp3", sr=sr, mono=True)
    duration = float(len(drums) / sr)

    d_t, d_s = onsets_of(drums, sr)
    tempo0, _b = librosa.beat.beat_track(y=drums, sr=sr, units="time")
    tempo0 = float(np.atleast_1d(tempo0)[0])
    strong = d_s >= np.percentile(d_s, 40)
    R, bpm, offset = fit_grid(d_t[strong], d_s[strong], tempo0, fixed_bpm)
    slot = 60.0 / bpm / 4
    kinds = classify_hits(drums, sr, d_t)
    print(f"{slug}: drum-stem grid bpm={bpm:.3f} offset={offset*1000:.0f}ms R={R:.2f} "
          f"| {len(d_t)} drum hits: "
          f"{sum(1 for k in kinds if k=='kick')}k/"
          f"{sum(1 for k in kinds if k=='snare')}s/"
          f"{sum(1 for k in kinds if k=='hat')}h")

    def to_slot(t): return int(round((t - offset) / slot))
    def slot_time(k): return offset + k * slot
    def place(t):
        """Tile timestamp: grid slot normally; the TRUE hit time in no-snap
        mode (chopped/swung breaks keep their micro-timing)."""
        return float(t) if no_snap else slot_time(to_slot(t))

    # per-slot best hit per kind (merges flams); no-snap keeps true times
    hits = {}
    for t, s, kind in zip(d_t, d_s, kinds):
        key = (to_slot(t), kind)
        if key not in hits or s > hits[key][1]:
            hits[key] = (place(t), s)
    kick = sorted(v for (k, kd), v in hits.items() if kd == "kick")
    snare = sorted(v for (k, kd), v in hits.items() if kd == "snare")
    hat = sorted(v for (k, kd), v in hits.items() if kd == "hat")

    # section energy tiers from drum-stem rms (8-bar windows)
    bar = slot * SLOTS_PER_BAR
    d_rms = librosa.feature.rms(y=drums, hop_length=HOP)[0]
    d_rms_t = librosa.times_like(d_rms, sr=sr, hop_length=HOP)

    def tier(t):
        m = (d_rms_t >= t - 2*bar) & (d_rms_t <= t + 2*bar)
        if not m.any():
            return 1.0
        lvl = d_rms[m].mean() / (np.percentile(d_rms[d_rms > 0], 75) + 1e-9)
        return float(np.clip(lvl, 0.35, 1.0))

    mel = melody_notes(other, sr, 65, 900) + melody_notes(vocals, sr, 80, 800, 0.45)
    mel.sort()
    bassline = melody_notes(bass, sr, 30, 300, 0.35)

    mix_rms_y, _ = librosa.load(root / "audio" / f"{slug}.mp3", sr=22050, mono=True, duration=None)
    rms = librosa.feature.rms(y=mix_rms_y)[0]
    rms_t = librosa.times_like(rms, sr=22050)
    chroma = librosa.feature.chroma_stft(y=other, sr=sr)
    chroma_t = librosa.times_like(chroma[0], sr=sr)

    base_chart = json.loads((root / "charts" / f"{slug}.json").read_text())
    audits = {}

    beat = 60.0 / bpm
    for diff, (min_gap, k_floor, s_floor, use_hats, use_bass, mel_dens) in {
        # gaps in musical time: standard admits 8ths, hard admits 16ths
        "standard": (beat / 2 * 0.9, 25, 35, False, False, 2.2),
        "hard": (beat / 4 * 0.9, 5, 10, True, True, 3.5),
    }.items():
        kf = np.percentile([s for _, s in kick], k_floor) if kick else 0
        sf = np.percentile([s for _, s in snare], s_floor) if snare else 0
        events = []
        for t, s in kick:
            if s >= kf * tier(t):
                events.append((t, 0, 4, s))
        acc = np.percentile([s for _, s in snare], 75) if snare else 0
        alt = 0
        for t, s in snare:
            if s < sf * tier(t):
                continue
            if s >= acc:
                events.append((t, 3, 3, s))
            else:
                events.append((t, 1 + alt % 2, 3, s))
                alt += 1
        if use_hats and hat:
            hf = np.percentile([s for _, s in hat], 45)
            for t, s in hat:
                if s >= hf:
                    events.append((t, 3, 1, s))

        perc_times = np.array(sorted(e[0] for e in events)) if events else np.array([0.0])
        def perc_density(t):
            return np.sum(np.abs(perc_times - t) < 1.0) / 2.0

        # melody contour (clean stems)
        mel_events = []
        midis = np.array([m for _, _, m in mel]) if mel else np.array([60.0])
        qs = np.percentile(midis, [25, 50, 75])
        prev_lane, prev_midi, prev_t = None, None, -9
        for t0, dur, midi in mel:
            if perc_density(t0) > mel_dens:
                prev_lane = None
                continue
            ts = place(t0)
            if prev_lane is None or t0 - prev_t > 0.8:
                lane = int(np.searchsorted(qs, midi))
            else:
                step = 0 if abs(midi - prev_midi) < 0.5 else (1 if midi > prev_midi else -1)
                lane = int(np.clip(prev_lane + step, 0, 3))
            note = {"t": round(float(ts), 3), "lane": lane}
            if dur >= 1.1:
                note["len"] = round(float(min(dur, 16 * slot)), 3)
            mel_events.append((ts, note, 2, 1.0))
            prev_lane, prev_midi, prev_t = lane, midi, t0

        # bassline tiles (hard): walking low lanes where drums leave room
        if use_bass:
            for t0, dur, midi in bassline:
                if perc_density(t0) > 4.5:
                    continue
                ts = place(t0)
                mel_events.append((ts, {"t": round(float(ts), 3),
                                        "lane": 0 if midi < np.median([m for _,_,m in bassline] or [40]) else 1},
                                   1, 0.5))

        merged = sorted([(t, {"t": round(float(t), 3), "lane": l}, p, s) for t, l, p, s in events] +
                        mel_events, key=lambda e: (e[0], -e[2]))
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

        # ease-in
        ease_end = offset + 2 * bar
        intro = [n_ for n_ in notes if n_["t"] < ease_end]
        if len(intro) > 4:
            keep = set(id(n_) for n_ in intro[::2])
            notes = [n_ for n_ in notes if n_["t"] >= ease_end or id(n_) in keep]

        # run breaker
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

        # pad holds
        bl = slot * 4
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
            t0 = slot_time(to_slot(a) + (4 - to_slot(a) % 4) % 4)
            while t0 + bl * 0.9 < b:
                ln = min(4 * bl, b - t0)
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

        # ---- audit against the stems ----
        all_stem_onsets = np.sort(np.concatenate([
            d_t,
            onsets_of(bass, sr)[0] if np.any(np.abs(bass) > 1e-4) else np.array([]),
            onsets_of(other, sr)[0] if np.any(np.abs(other) > 1e-4) else np.array([]),
            onsets_of(vocals, sr)[0] if np.any(np.abs(vocals) > 1e-4) else np.array([]),
        ]))
        taps = [n_ for n_ in notes if "len" not in n_]
        phantom = sum(1 for n_ in taps
                      if len(all_stem_onsets) and np.abs(all_stem_onsets - n_["t"]).min() > 0.06)
        top = d_t[np.argsort(d_s)[-100:]]
        note_ts = np.array([n_["t"] for n_ in taps]) if taps else np.array([0.0])
        missed = sum(1 for t in top if np.abs(note_ts - t).min() > 0.06)
        audits[diff] = (len(taps), len(notes) - len(taps),
                        100 * phantom / max(len(taps), 1), missed)

        chart = dict(base_chart)
        chart["notes"] = notes
        chart["bpm"] = round(bpm, 3)
        chart["beatOffset"] = round(offset, 4)
        chart.pop("beats", None)
        chart["generator"] = "v5"
        name = f"{slug}.json" if diff == "standard" else f"{slug}-hard.json"
        (out_dir / name).write_text(json.dumps(chart, indent=1))

    for diff, (t, h, ph, miss) in audits.items():
        print(f"  [{diff}] {t} taps + {h} holds | phantom {ph:.1f}% | "
              f"top-100 drum hits missed: {miss}")

    # drift detector: median |tap - nearest drum hit| in the first vs last minute.
    # If these diverge, the chart is walking away from the audio.
    std = json.loads((out_dir / f"{slug}.json").read_text())
    taps_all = np.array([n_["t"] for n_ in std["notes"] if "len" not in n_])
    ref = np.sort(np.concatenate([
        d_t,
        onsets_of(bass, sr)[0] if np.any(np.abs(bass) > 1e-4) else np.array([]),
        onsets_of(other, sr)[0] if np.any(np.abs(other) > 1e-4) else np.array([]),
        onsets_of(vocals, sr)[0] if np.any(np.abs(vocals) > 1e-4) else np.array([]),
    ]))
    def med_dev(a, b):
        w = taps_all[(taps_all >= a) & (taps_all <= b)]
        if not len(w) or not len(ref):
            return 0.0
        return float(np.median([np.abs(ref - t).min() for t in w]) * 1000)
    print(f"  drift check: first-min dev {med_dev(0, 60):.0f}ms | "
          f"last-min dev {med_dev(duration - 60, duration):.0f}ms")


if __name__ == "__main__":
    main()
