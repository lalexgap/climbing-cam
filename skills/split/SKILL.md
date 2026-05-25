---
name: split
description: Split a fixed-camera rock-climbing video into one clip per attempt, with on-wall rests sped up. Use when the user wants to process, split, or clip a climbing video (.mov/.mp4) into per-attempt clips.
---

# Climbing Cam — split a video into per-attempt clips

Turns a long climbing video into one clip per attempt ("burn"), trims dead time,
and speeds up on-wall hangs. Runs locally (ffmpeg + YOLO; uses an Apple/NVIDIA
GPU if present, else CPU). Needs `git`, `uv`, and `ffmpeg` on the host.

## Locate the tool

This plugin ships the tool. Use the plugin's own checkout when available
(`$CLAUDE_PLUGIN_ROOT`), otherwise clone it:

```bash
ROOT="${CLAUDE_PLUGIN_ROOT:-$HOME/.local/share/climbing-cam}"
[ -f "$ROOT/app/cli.py" ] || git clone https://github.com/lalexgap/climbing-cam "$ROOT"
cd "$ROOT" && uv sync
```

## Run

```bash
cd "$ROOT" && uv run python -m app.cli "<path-to-video>"
```

Detection is the slow stage (minutes; slower on CPU) — **run it in the
background** and report when done. Clips land in
`$ROOT/data/outputs/<job>/attempt_NN.mp4`; the job id prints at the start and the
folder opens on macOS.

## Notes

- Re-cut after tuning (fast, no re-detection): `uv run python -m app.cli --recut <job_id>`.
- Knobs in `app/config.py`: `merge_gap_seconds` (attempt splitting), `rest_speedup`
  / `min_rest_seconds` / `rest_band_bh` (rest speed-ramp), `ramp_marker` (the "8×"
  review badge — set False for final exports), `analysis_width`/`imgsz` (recall).
- To just check whether a video has climbing, use the **check** skill.
- `README.md` / `CLAUDE.md` in `$ROOT` have the architecture and known limits.
