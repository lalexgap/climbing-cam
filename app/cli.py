"""Command-line entry point for the climbing-cam pipeline.

    uv run python -m app.cli <video>            # full run: detect -> split -> cut
    uv run python -m app.cli --check <video>    # quick: is there climbing in it?
    uv run python -m app.cli --recut <job_id>   # re-cut from cached detections

Flags (anywhere on the line):
    --pose    full/check runs: use the keypoint-anchored pose detector (experimental)
    --check --clip <video>   screen with the zero-shot CLIP classifier instead of YOLO

Full runs point the job's source directly at <video> (no copy). Clips land in
data/outputs/<job>/ and the folder is opened on macOS.
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from pathlib import Path

from . import pipeline
from .config import Config


def _print_event(ev: dict) -> None:
    if ev.get("stage"):
        pct = ev.get("pct")
        tail = f" {int(pct * 100)}%" if pct is not None else ""
        print(f"  [{ev['stage']}] {ev.get('message', '')}{tail}", flush=True)


def _report(clips: list[dict], out_dir: Path) -> None:
    print(f"\n{len(clips)} clip(s) in {out_dir}:")
    for c in clips:
        dur = c.get("duration")
        print(f"  {c['name']}" + (f"  ({dur}s)" if dur else ""))
    if sys.platform == "darwin":
        subprocess.run(["open", str(out_dir)], check=False)


def main(argv: list[str] | None = None) -> int:
    import dataclasses

    argv = list(sys.argv[1:] if argv is None else argv)
    # Pull out boolean flags so they can appear anywhere; keep mode + positionals.
    use_clip = "--clip" in argv
    use_pose = "--pose" in argv
    argv = [a for a in argv if a not in ("--clip", "--pose")]
    cfg = Config()
    if use_pose:
        cfg = dataclasses.replace(cfg, pose=True)

    if argv and argv[0] == "--check":
        if len(argv) < 2:
            print("usage: python -m app.cli --check [--clip] <video>")
            return 2
        video = Path(argv[1]).expanduser().resolve()
        if not video.exists():
            print(f"video not found: {video}")
            return 2
        print(f"Screening {video.name} for climbing…")
        last = [0]

        def tick(f: float) -> None:
            if int(f * 100) >= last[0] + 25:
                last[0] = int(f * 100)
                print(f"  …{last[0]}%", flush=True)

        if use_clip:
            from . import screen_clip
            ev = screen_clip.screen_video_clip(video, cfg, progress=tick)
            verdict = "YES — climbing detected" if ev["present"] else "no climbing detected"
            print(f"\n{verdict}")
            print(f"  peak score: {ev['peak_score']} | "
                  f"longest stretch: {ev['longest_stretch_seconds']}s | "
                  f"total climbing: {ev['climbing_seconds']}s of {ev['duration']}s")
            return 0 if ev["present"] else 3

        from . import screen
        ev = screen.screen_video(video, cfg, progress=tick)
        verdict = "YES — climbing detected" if ev["present"] else "no climbing detected"
        print(f"\n{verdict}")
        print(f"  peak height: {ev['peak_bh']} body-heights | "
              f"longest stretch: {ev['longest_stretch_seconds']}s | "
              f"total on-wall: {ev['climbing_seconds']}s of {ev['duration']}s")
        return 0 if ev["present"] else 3

    if argv and argv[0] == "--recut":
        if len(argv) < 2:
            print("usage: python -m app.cli --recut <job_id>")
            return 2
        job = pipeline.Job.load(argv[1])
        events = pipeline.recut_stream(job, cfg)
    elif argv and not argv[0].startswith("-"):
        video = Path(argv[0]).expanduser().resolve()
        if not video.exists():
            print(f"video not found: {video}")
            return 2
        job = pipeline.Job(id=uuid.uuid4().hex[:12], source=str(video), filename=video.name)
        job.out_dir.mkdir(parents=True, exist_ok=True)
        job.save_meta()
        print(f"Processing {video.name}  (job {job.id})")
        events = pipeline.analyze_stream(job, cfg)
    else:
        print(__doc__)
        return 2

    clips: list[dict] = []
    for ev in events:
        if ev.get("type") == "error":
            print(f"ERROR: {ev.get('message')}")
            return 1
        if ev.get("type") == "done":
            clips = ev.get("clips", [])
        else:
            _print_event(ev)
    _report(clips, job.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
