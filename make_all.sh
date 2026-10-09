#!/usr/bin/env bash
# Render every animation in the repo (needs ffmpeg on PATH). ~1 min each.
set -e
# clustering: cropped panels (boxes + lightcurves + counters) and full-slide versions
python clustering_anim.py onoff --extreme --info --crop
python clustering_anim.py drw   --extreme --info --crop
python clustering_anim.py onoff --extreme --full
python clustering_anim.py drw   --extreme --full
# clustering: long Eddington-limited episodes vs short rapid bursts (DRW coherence time)
python clustering_anim.py drw   --edd --info --crop
python clustering_anim.py drw   --edd --full
# clustering: toy lightbulb (left) vs bursty short-tau DRW (right), dark and white slides
python clustering_anim.py drw   --bulb --info --crop
python clustering_anim.py drw   --bulb --full
python clustering_anim.py drw   --bulb --info --crop --light
python clustering_anim.py drw   --bulb --full --light
# proximity zones
python proximity_tq.py
python proximity_drw.py
