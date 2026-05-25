# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local macOS web app that ingests a fixed-camera rock-climbing video and exports one clip per climbing attempt. All processing runs locally on Apple Silicon (MPS); nothing is uploaded. Tuned for **outdoor sport climbing, fixed camera, one route per video** (often portrait, 1080p or 4K). See `README.md` for the user-facing overview.

## Commands

```bash
uv sync --extra dev                      # install deps (Python 3.12 via uv)
uv run uvicorn app.main:app --reload --port 8000   # run the web app
uv run python -m app.cli <video>         # CLI: detect -> split -> cut clips
uv run python -m app.cli --recut <job>   # re-cut from cached detections (fast)
uv run pytest                            # run tests
uv run pytest tests/test_classify.py::test_flat_hang_is_a_rest   # single test
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

## Speed-ramp the rests (Phase 2, built)

Each clip speeds up hangs (`clip.cut_clip_ramped`, a trim/setpts/concat filtergraph; audio sped with `atempo`). A hang is detected in `classify.rest_intervals` as a sustained **flat-height** stretch (no net climbing progress) on the dropout-interpolated elevation — this catches visible and undetected hangs. Note the limit: a flat crux you're *working* (little height gain) can be mistaken for a rest. We deliberately rejected optical-flow body motion (resting/shaking vs small climbing moves overlap too much) — see git history. Sped sections get an "8×" badge overlaid (`ramp_marker`, a review aid; this ffmpeg build has no `drawtext`, so the badge is a generated PNG overlaid via `overlay`). Tune `rest_speedup`, `min_rest_seconds`, `rest_band_bh`, `rest_inset_seconds` in `config.py`.

## Plugin packaging

This repo is also a Claude Code plugin (`.claude-plugin/plugin.json` + `.claude-plugin/marketplace.json`) bundling two skills under `skills/`: `split` (= `app.cli <video>`) and `check` (= `app.cli --check <video>`), invoked as `/climbing-cam:split` and `/climbing-cam:check`. The plugin carries the code, so the skills run from `$CLAUDE_PLUGIN_ROOT` (falling back to a clone). The marketplace plugin `source` is a `github` object (the plugin is the whole repo, not a subdir). Validate with `claude plugin validate .`. OpenClaw ingests the same Claude-layout bundle via `openclaw plugins install git:github.com/lalexgap/climbing-cam@main`.
