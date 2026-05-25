"""Command-line entry point for the climbing-cam pipeline.

    uv run python -m app.cli <video>            # full run: detect -> split -> cut
    uv run python -m app.cli --recut <job_id>   # re-cut from cached detections

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
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = Config()

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
