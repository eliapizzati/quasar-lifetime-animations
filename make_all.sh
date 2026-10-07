#!/usr/bin/env bash
# Render every animation in the repo (needs ffmpeg on PATH). ~1 min each.
set -e
python clustering_anim.py onoff --extreme --info --crop
python clustering_anim.py drw   --extreme --info --crop
python proximity_tq.py
python proximity_drw.py
