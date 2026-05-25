---
name: climbing-cam
description: Split a fixed-camera rock-climbing video into one clip per attempt, with on-wall rests sped up, using this repo's pipeline. Use when the user wants to process, analyze, split, or clip a climbing video (.mov/.mp4) into per-attempt clips, or points at climbing footage to turn into clips.
---

# Climbing Cam — process a climbing video

Turns a long climbing video into one clip per attempt ("burn"), trimming dead
time and speeding up on-wall rests. Runs locally (ffmpeg + YOLO on Apple MPS).
**Run from the climbing-cam repo root** (it uses the repo's `uv` environment).

## Process a video

```
uv run python -m app.cli <path-to-video>
```

Detection is the slow stage (~6–15 min for a 30-min 4K clip), so **run it in the
background** (`run_in_background: true`) and report when it finishes — don't poll.
It writes `data/outputs/<job>/attempt_NN.mp4`, caches detections to
`_detections.json`, and opens the output folder on macOS. The job id is the
`data/outputs/<job>/` folder name (printed at the start).

## Re-cut after tuning (fast — no re-detection)

Detections are cached, so to change the result, edit thresholds in
`app/config.py` and re-cut from cache (seconds, not minutes):

```
uv run python -m app.cli --recut <job_id>
```

This is the main tuning loop — prefer it over re-running detection.

## Key knobs (`app/config.py`)

- `analysis_width` / `imgsz` — detection recall of a small/distant climber
  (default 1920; lower → ~2× faster, worse on distant climbers).
- `merge_gap_seconds` — how long off the wall counts as a *separate* attempt
  (~5 min). Lower to split goes that are close together; raise to merge.
- Speed-ramp: `speed_ramp`, `rest_speedup` (e.g. 8×), `min_rest_seconds` and
  `rest_band_bh` (a flat-height stretch this long/tight = a hang),
  `rest_inset_seconds` (start the speed-up later / end earlier),
  `ramp_marker` (the green "8×" badge on sped sections — **set False for final
  clips**, it's a review aid).

## Assumes / known limits

- Fixed camera, base of route in frame (portrait videos are handled).
- Splitting into multiple attempts relies on a long off-wall gap between goes.
- Rest detection = flat height (no net climbing progress); it can over-speed a
  flat crux where you're working hard moves with little height gain. Tell the
  user this if rests look wrong, and offer to tune `rest_band_bh` / `min_rest_seconds`.

## Verify

After a run, open `data/outputs/<job>/` and check: the number of attempts is
right, each clip starts as the climber leaves the ground and ends just after the
high point, and the badge-marked sped sections are genuine rests. Tune
`config.py` and `--recut` until right; turn `ramp_marker` off for the finals.
