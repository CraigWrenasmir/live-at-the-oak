# Wrenasmir — Live at The Oak

A retro (PSX / Guitar Hero era) rhythm game built around the *Live at The Oak* set.
Web-first; iOS wrap planned via Capacitor/Expo once the core game is solid.

## Play locally

```sh
cd ~/Live-at-The-Oak
python3 -m http.server 8041
# open http://localhost:8041
```

Keys **D F J K** (or tap the four lanes on touch). Run **Calibrate timing**
once per device — the offset is stored in localStorage.

## Architecture

- `index.html` — the whole game. All gameplay time derives from
  `AudioContext.currentTime`; rendering is a pure function of song position,
  so dropped frames can never desync judgement. Internal render is 640×360
  upscaled with `image-rendering: pixelated` for the PSX look.
- `charts/*.json` — note charts: `{bpm, beatOffset, duration, lanes, notes:[{t, lane}]}`.
- `tools/make_chart.py` — first-pass chart generator (librosa): beat-tracks the
  song, detects onsets, snaps to a 16th-note grid, assigns lanes by spectral
  centroid quantiles (bassy → left, bright → right). Charts are meant to be
  hand-tuned in the chart editor (next up).
  ```sh
  .venv/bin/python tools/make_chart.py audio/Song.mp3 charts/song.json [min_gap] [strength_floor_pct]
  # standard: min_gap 0.22, floor 45 — hard: min_gap 0.11, floor 0
  ```

## Roadmap

1. ✅ Core timing engine + playable highway (Crafluropi, Standard/Hard)
2. Chart editor (waveform, scrub, place/move notes, instant playtest)
3. Low-poly Wrenasmir + Oak stage in Blender (via MCP) → glTF → Three.js scene
4. Retro post shader (dither, chromatic aberration, VHS glitch banner)
5. Menus / full set list, star power, hold notes
6. Capacitor/Expo wrap → TestFlight → App Store
