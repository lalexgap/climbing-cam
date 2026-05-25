---
name: climbing-check
description: Quickly check whether a video contains rock climbing — a fast yes/no screen, not full clipping. Clones/updates the climbing-cam tool from GitHub and runs the check. Use when the user wants to know if a video has climbing in it, or to triage/filter a set of videos.
---

# Climbing Check — is there climbing in this video?

A fast yes/no screen: detects people and checks whether anyone is clearly off
the ground for a sustained stretch. Uses the climbing-cam tool at
https://github.com/lalexgap/climbing-cam (ffmpeg + YOLO; Apple/NVIDIA GPU if
present, else CPU). Requires `git`, `uv`, and `ffmpeg`.

## 1. Get the tool (clone, or update if already cloned)

```bash
REPO="$HOME/.local/share/climbing-cam"
if [ -d "$REPO/.git" ]; then git -C "$REPO" pull --ff-only; else git clone https://github.com/lalexgap/climbing-cam "$REPO"; fi
cd "$REPO" && uv sync
```

## 2. Check a video

```bash
cd "$HOME/.local/share/climbing-cam"
uv run python -m app.cli --check "<path-to-video>"
```

Prints **"YES — climbing detected"** or **"no climbing detected"** plus evidence
(peak height in body-heights, longest elevated stretch, total on-wall time).
It's faster than a full run (1 fps, small model) but still a detection pass — for
a long video, **run it in the background**. Exit code is `0` if climbing is found,
`3` if not (handy for scripting).

## Triage a folder

To filter many videos, run `--check` on each and group by verdict, e.g.:

```bash
for f in *.mov *.mp4; do
  uv run python -m app.cli --check "$f" >/dev/null 2>&1 && echo "CLIMBING: $f" || echo "skip:     $f"
done
```

## Notes

- To actually split a climbing video into per-attempt clips, use the
  **climbing-cam** skill (or `app.cli <video>`).
- Tune sensitivity via `min_climb_seconds` / `ascend_bh` in `app/config.py`.
