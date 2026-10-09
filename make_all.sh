#!/usr/bin/env bash
# Render every animation in the repo into videos/ (needs ffmpeg on PATH). ~1 min each.
set -e
cd "$(dirname "$0")"

# clustering, main video: lightbulb (left) vs bursty short-tau DRW (right), dark and white
python clustering_anim.py drw   --bulb --info --crop
python clustering_anim.py drw   --bulb --full
python clustering_anim.py drw   --bulb --info --crop --light
python clustering_anim.py drw   --bulb --full --light
# clustering, other variants
python clustering_anim.py onoff --extreme --info --crop
python clustering_anim.py drw   --extreme --info --crop
python clustering_anim.py onoff --extreme --full
python clustering_anim.py drw   --extreme --full
python clustering_anim.py drw   --edd --info --crop
python clustering_anim.py drw   --edd --full
# proximity zones
python proximity_tq.py
python proximity_drw.py

# keep the deliverables, drop the uncropped --info intermediates
mkdir -p videos
mv -f clustering_panels_*.mp4 clustering_dutycycle_*_full_*.mp4 proximity_zone_*.mp4 videos/
rm -f clustering_dutycycle_*_info_*.mp4
