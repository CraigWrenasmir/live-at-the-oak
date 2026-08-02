#!/bin/bash
# encode wavs -> mp3 and chart both difficulties for every track
set -e
cd "$(dirname "$0")/.."
declare -a NUMS=(1 2 3 4 5 6 7)
declare -a WAVS=("1 Rdst Gm" "2 Lucempight" "3 Spinet Destroiet" "4 What Is Better" "5 The Way That I Go" "6 Mabon" "7 Crafluropi")
declare -a SLUGS=(rdst-gm lucempight spinet-destroiet what-is-better the-way-that-i-go mabon crafluropi)
for i in "${!WAVS[@]}"; do
  slug="${SLUGS[$i]}"
  echo "=== ${WAVS[$i]} -> $slug"
  ffmpeg -y -loglevel error -i "audio/${WAVS[$i]}.wav" -codec:a libmp3lame -b:a 192k "audio/$slug.mp3"
  .venv/bin/python tools/make_chart.py "audio/$slug.mp3" "charts/$slug.json" 0.26 55 --no-hats 2>/dev/null
  .venv/bin/python tools/make_chart.py "audio/$slug.mp3" "charts/$slug-hard.json" 0.14 30 2>/dev/null
done
echo "ALL DONE"
