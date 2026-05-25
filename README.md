# 🧗 Climbing Cam

Drop a long, mostly-dead climbing video in, get one tight clip per attempt out.

Climbing Cam analyzes a route video, detects when you're actually on the wall,
and exports a separate clip for each **burn** — trimmed to start as you leave the
ground and end just after your high point. All processing runs **locally** on
your Mac with GPU (Metal/MPS) acceleration; nothing is uploaded anywhere.

## What it's tuned for

- **Outdoor sport climbing**, **fixed camera**, base of the route in frame.
- One video = **one route**, several burns, ~30 min, 1080p or 4K.
- Follows whoever rises out of the ground band, so the belayer / people at the
  base (elevation ≈ 0) are ignored automatically. This is track-agnostic — it
  survives the detector re-ID'ing you into fragments as you shrink high on the
  wall. (Two people climbing in frame at once is a known limitation; it follows
  the higher one.)
- A *framed-up-on-the-wall* shot (no ground visible) is handled by a fallback
  that treats "in frame = climbing."

## Requirements

Already present on this machine: **ffmpeg**, **uv**, an Apple-Silicon GPU. The
first run auto-downloads the YOLO weights (~40 MB).

## Run it

```bash
uv run uvicorn app.main:app --port 8000
# open http://localhost:8000
```

Drop a video → watch progress → preview & download clips. "Reveal in Finder"
opens the output folder.

Or from the command line:

```bash
uv run python -m app.cli <video>           # full run: detect → split → cut
uv run python -m app.cli --recut <job_id>  # re-cut from cached detections (fast)
```

Clips are written to `data/outputs/<job>/attempt_NN.mp4` at source resolution
(H.264 via `h264_videotoolbox`, original audio + orientation preserved).
Detections are cached to `_detections.json`, so `--recut` re-segments and
re-cuts in seconds after you tune thresholds in `config.py` — no re-detection.

## How it works

1. **Sample** frames at ~3 fps, downscaled (ffmpeg, hardware decode).
2. **Detect + track** people with YOLO (`yolo11m`) + ByteTrack on the GPU —
   plain bounding boxes, which stay reliable when you're small and far up.
3. **Elevation signal**: per frame, take the height of the *highest* person out
   of the estimated ground band (in self-calibrating *body-heights*). Being
   track-agnostic, it's immune to ByteTrack re-IDing the climber mid-ascent.
4. **Segment burns**: one burn spans from leaving the ground to your last
   moment on the wall. Short gaps — detection dropouts and on-wall hangs — are
   **merged through**; a burn splits into a new attempt only where on-wall
   activity stops for longer than `merge_gap_seconds` (~5 min: you came down and
   rested between goes). So a single long attempt with multi-minute hangs stays
   one clip, while genuinely separate goes split. Sub-15 s blips are discarded.
5. **Cut** each burn from the original file: starts ~3 s before you leave the
   ground, ends ~3 s after your last on-wall moment (trims the lower-off).

## Tuning

Every threshold lives in [`app/config.py`](app/config.py) — sampling fps,
analysis resolution, the elevation enter/exit thresholds, the ground/min-burn/
padding/apex timings, and encode bitrate. Expect to calibrate `analysis_width`
and the `*_bh` thresholds on a couple of your real videos (a climber tiny near
the top of a tall route is the main accuracy risk; raise `analysis_width` /
`imgsz` if detection drops out up high).

## Tests

```bash
uv run pytest
```

The burn-segmentation logic is pure and unit-tested on synthetic signals
(`tests/test_classify.py`) — no GPU/video needed.

## Speed-ramp the rests

Within each clip, **hangs** are sped up while climbing plays at 1x. A hang is a
sustained stretch where your **height stays flat** (no net climbing progress) —
measured on the dropout-interpolated elevation, so it catches both visible and
undetected hangs. (We tried optical-flow body motion, but on real footage
resting/shaking and small climbing moves overlap too much to separate; flat
height is the reliable signal — though it can over-speed a flat crux you're
working with little height gain.) The speed-up is inset a few seconds on each
end (`rest_inset_seconds`) and marked with an "8×" badge (`ramp_marker`, a review
aid — turn off for finals). Tune `rest_speedup`, `min_rest_seconds`,
`rest_band_bh` in `config.py`.

## Known assumptions (v1)

- Camera is roughly **static** (propped/tripod). Panning to follow breaks the
  vertical-travel signal.
- Base of the route is usually in frame (framed-up has a coarser fallback).
- No repositioning mid-video (one route per file).
- Splitting one video into multiple attempts is based on a **gap in on-wall
  activity longer than `merge_gap_seconds`** (~5 min). On-wall hangs are
  typically shorter than this and stay one attempt; a real lower-off-and-rest is
  longer and splits. Two goes less than ~5 min apart will merge; tune
  `merge_gap_seconds` in `config.py`. (A sustained 2+ people-at-base stretch also
  forces a split when the belayer + you are both visible at the base, but on real
  footage the base is often hidden, so the gap rule does the heavy lifting.)
- Portrait phone videos are handled (rotation is read from the Display Matrix);
  frames are analyzed upright, not squished.
