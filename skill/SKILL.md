---
name: climbing-cam
description: Split a rock-climbing video into one clip per attempt, with on-wall rests sped up. Clones/updates the climbing-cam tool from GitHub and runs it on a given video. Use when the user wants to process, analyze, split, or clip a climbing video (.mov/.mp4) from anywhere on their machine.
---

# Climbing Cam — clip a climbing video (clone & run)

Splits a fixed-camera rock-climbing video into one clip per attempt ("burn"),
trims dead time, and speeds up on-wall hangs. The tool lives at
https://github.com/lalexgap/climbing-cam and runs locally (ffmpeg + YOLO; uses
an Apple/NVIDIA GPU if present, else CPU). Requires `git`, `uv`, and `ffmpeg`.

## 1. Get the tool (clone, or update if already cloned)

Keep a fixed checkout at `~/.local/share/climbing-cam`:

```bash
REPO="$HOME/.local/share/climbing-cam"
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" pull --ff-only
else
  git clone https://github.com/lalexgap/climbing-cam "$REPO"
fi
cd "$REPO" && uv sync
```

(First run also auto-downloads YOLO weights, ~40 MB.)

## 2. Run on a video

```bash
cd "$HOME/.local/share/climbing-cam"
uv run python -m app.cli "<path-to-video>"
```

Detection is the slow stage (minutes; much slower on CPU-only boxes) — **run it
in the background** and report when it finishes; don't poll. Clips land in
`~/.local/share/climbing-cam/data/outputs/<job>/attempt_NN.mp4`; the job id is
printed at the start, and the folder opens on macOS.

## 3. Re-cut after tuning (fast, no re-detection)

Detections are cached, so to change the result, edit thresholds in
`app/config.py` and re-cut in seconds:

```bash
cd "$HOME/.local/share/climbing-cam"
uv run python -m app.cli --recut <job_id>
```

## Notes

- Key knobs in `app/config.py`: `analysis_width`/`imgsz` (detection recall),
  `merge_gap_seconds` (attempt splitting), `rest_speedup` / `min_rest_seconds` /
  `rest_band_bh` (rest speed-ramp), `ramp_marker` (the "8×" review badge — set
  False for final exports).
- Assumes a fixed camera with the base of the route in frame (portrait handled).
  Rest detection uses flat height, so a flat crux you're *working* may get sped
  up — tell the user and offer to tune if rests look wrong.
- The repo's `README.md` and `CLAUDE.md` have full architecture and tuning notes.
