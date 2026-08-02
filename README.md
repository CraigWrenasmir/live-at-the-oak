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
- `tools/make_chart.py` — chart generator v2 (librosa + scipy): band-split
  onsets — kick (<140Hz) → lane 0, snare (140–2500Hz) → lanes 1/2, hats
  (>4kHz) → lane 3 — snapped to a 16th grid; quiet stretches with energy
  become chains of hold notes (`len` in seconds) through to the end of song.
  ```sh
  .venv/bin/python tools/make_chart.py audio/Song.mp3 charts/song.json [min_gap] [floor_pct] [--no-hats]
  # standard: 0.26 55 --no-hats — hard: 0.14 30
  ```

## Roadmap

1. ✅ Core timing engine + playable highway (Crafluropi, Standard/Hard)
2. Chart editor (waveform, scrub, place/move notes, instant playtest)
3. Low-poly Wrenasmir + Oak stage in Blender (via MCP) → glTF → Three.js scene
4. Retro post shader (dither, chromatic aberration, VHS glitch banner)
5. Menus / full set list, star power, hold notes
6. Capacitor/Expo wrap → TestFlight → App Store
