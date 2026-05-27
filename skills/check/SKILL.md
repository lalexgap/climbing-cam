---
name: check
description: Quickly check whether a video contains rock climbing — a fast yes/no screen, not full clipping. Use when the user wants to know if a video has climbing in it, or to triage/filter a set of videos.
---

# Climbing Cam — is there climbing in this video?

A fast yes/no screen: detects people and checks whether anyone is clearly off the
ground for a sustained stretch. Runs locally (ffmpeg + YOLO). Needs `git`, `uv`,
and `ffmpeg` on the host.

## Locate the tool

```bash
ROOT="${CLAUDE_PLUGIN_ROOT:-$HOME/.local/share/climbing-cam}"
[ -f "$ROOT/app/cli.py" ] || git clone https://github.com/lalexgap/climbing-cam "$ROOT"
cd "$ROOT" && uv sync
```

## Check a video

```bash
cd "$ROOT" && uv run python -m app.cli --check "<path-to-video>"
```

Prints **"YES — climbing detected"** or **"no climbing detected"** plus evidence
(peak height in body-heights, longest elevated stretch, total on-wall time).
Faster than a full run (1 fps, small model) but still a detection pass — **run it
in the background** for long videos. Exit code `0` if climbing is found, `3` if
not (handy for scripting / batch-filtering a folder).

## Triage a folder

```bash
cd "$ROOT"
for f in /path/to/videos/*.mov /path/to/videos/*.mp4; do
  uv run python -m app.cli --check "$f" >/dev/null 2>&1 && echo "CLIMBING: $f" || echo "skip:     $f"
done
```

## Notes

- To actually split a climbing video into clips, use the **split** skill.
- Tune sensitivity via `min_climb_seconds` / `ascend_bh` in `app/config.py`.
- Experimental: add `--clip` (`--check --clip <video>`) to screen with a zero-shot
  CLIP classifier instead of person-detection + elevation. Needs the optional extra
  (`uv sync --extra clip`). Useful when the base isn't in frame, where the geometric
  signal is weakest. Tune `clip_threshold` in `app/config.py`.
