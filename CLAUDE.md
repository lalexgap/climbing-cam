# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local macOS web app that ingests a fixed-camera rock-climbing video and exports one clip per climbing attempt. All processing runs locally on Apple Silicon (MPS); nothing is uploaded. Tuned for **outdoor sport climbing, fixed camera, one route per video** (often portrait, 1080p or 4K). See `README.md` for the user-facing overview.

## Commands

```bash
uv sync --extra dev                      # install deps (Python 3.12 via uv)
uv run uvicorn app.main:app --reload --port 8000   # run the app -> http://localhost:8000
uv run pytest                            # run tests
uv run pytest tests/test_classify.py::test_single_clean_burn   # single test
```

YOLO weights (`yolo11m.pt`) auto-download on first detection run. Detection requires ffmpeg on PATH and an Apple GPU (falls back to CPU).

## Architecture

Single FastAPI process (`app/main.py`) that both serves the vanilla-JS frontend in `web/` and runs the analysis pipeline, streaming progress to the browser over **SSE**. The pipeline is exposed as event *generators* (`app/pipeline.py`); each runs its blocking work (ffmpeg/YOLO) in a worker thread via `_stream()` and yields progress dicts.

Pipeline stages (`pipeline.analyze_stream`):
1. `clip.probe` — ffmpeg metadata. **Reads display rotation** (`stream_side_data=rotation`) and reports rotated dims; portrait phone videos must be treated as portrait or frames get squished and detection collapses.
2. `clip.sample_frames` — ffmpeg decodes at ~`analysis_fps`, downscaled to `analysis_width`, hardware-decoded.
3. `detect.run_detection` — YOLO **person bounding boxes** (not pose) + ByteTrack on MPS, per sampled frame.
4. `classify` — estimate the ground band, then segment burns (see below).
5. `clip.cut_clip` — cut each burn from the **original** file with `h264_videotoolbox`, audio + orientation preserved.

### The core signal (the non-obvious part)

`classify.py` is pure (numpy in, intervals out) and fully unit-tested without YOLO/ffmpeg — it holds all the segmentation logic and is where most tuning happens.

- Elevation is measured as **body-heights above an estimated ground band** (`estimate_ground`): self-calibrating across camera distance / resolution.
- The signal is **track-agnostic** (`frame_max_elevation`: the highest person per frame), *deliberately not* per-track — ByteTrack re-IDs a small/distant climber into many fragments, so following any single track is unreliable. The belayer / people at the base sit near elevation 0 and are ignored automatically.
- `detect_burns` confirms a burst where elevation reaches `enter_bh`, **merges through** short gaps (detection dropouts, on-wall hangs), and **splits** into a new attempt only when on-wall activity stops for longer than `merge_gap_seconds` (you came down and rested — far more reliable than counting people at the base, which is often hidden). A burn's **start** is extended back over the first low moves (`leave_bh`); its **end** is the last `enter_bh` moment, trimming the lower-off.

### Tuning surface

`app/config.py` is a frozen `Config` dataclass holding every threshold (sampling fps, `analysis_width`/`imgsz`, the `*_bh` elevation thresholds, `merge_gap_seconds`, padding). Calibrating against real footage means editing these. The two knobs that matter most: `analysis_width`/`imgsz` (recall of a small distant climber — default 1920; drop to 1280 for ~2x speed) and `merge_gap_seconds` (how far apart two goes must be to count as separate attempts).

### Fast iteration: re-cut from cache

Detection (the slow stage) is cached to `data/outputs/<job>/_detections.json`. `pipeline.recut_stream` (and `GET /api/recut/{job_id}`) re-segments and re-cuts from that cache **without re-running detection** — use this to test `config.py` changes in seconds instead of re-detecting. When debugging segmentation, load a cached `_detections.json` and call the `classify` functions directly rather than re-processing a video.

`data/` (uploads + generated clips) is gitignored and can be large; delete `data/outputs/<job>` / `data/uploads/<job>` to reclaim space.

## Roadmap

Phase 2 (not yet built): **speed-ramp the rests** — within a burn, detect on-wall hangs (elevated but low motion) and speed them up with ffmpeg `setpts`, returning to normal speed while moving. Reuses the elevation/motion signal; pose keypoints become useful for the lower, larger part of the wall.
