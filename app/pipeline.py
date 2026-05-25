"""Pipeline orchestration: upload -> detect -> classify -> (confirm) -> cut.

Each long-running step is exposed as an event *generator* so the FastAPI layer
can stream progress over SSE. Detection runs in a worker thread that pushes
progress onto a queue; the generator drains it.
"""

from __future__ import annotations

import json
import queue
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from . import classify, clip, detect
from .config import OUTPUT_DIR, UPLOAD_DIR, Config


@dataclass
class Job:
    id: str
    source: str          # path to uploaded video
    filename: str        # original filename

    @property
    def out_dir(self) -> Path:
        return OUTPUT_DIR / self.id

    @property
    def cache_path(self) -> Path:
        return self.out_dir / "_detections.json"

    @property
    def meta_path(self) -> Path:
        return self.out_dir / "job.json"

    @property
    def frame_path(self) -> Path:
        return self.out_dir / "_confirm.jpg"

    def save_meta(self, extra: dict | None = None) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        data = {"id": self.id, "source": self.source, "filename": self.filename}
        if extra:
            data.update(extra)
        self.meta_path.write_text(json.dumps(data))

    @classmethod
    def load(cls, job_id: str) -> "Job":
        data = json.loads((OUTPUT_DIR / job_id / "job.json").read_text())
        return cls(id=data["id"], source=data["source"], filename=data["filename"])


def create_job(filename: str) -> Job:
    job_id = uuid.uuid4().hex[:12]
    up_dir = UPLOAD_DIR / job_id
    up_dir.mkdir(parents=True, exist_ok=True)
    safe = Path(filename).name or "video.mp4"
    return Job(id=job_id, source=str(up_dir / safe), filename=safe)


# --- Shared helpers ----------------------------------------------------------

def _times(det: detect.DetectionResult) -> np.ndarray:
    return np.array([fd.t for fd in det.frames], dtype=float)


def _compute_burns(det, est, cfg, duration) -> list[classify.Burn]:
    """Track-agnostic burn detection: follow the highest person per frame.

    Robust to ByteTrack re-IDing the climber into fragments high on the wall.
    Falls back to presence when there's no reliable ground band (framed up)."""
    times = _times(det)
    if est.reliable:
        elev = classify.frame_max_elevation(det, est)
        if np.isfinite(elev).any() and np.nanmax(elev) >= cfg.ascend_bh:
            ground_count = classify.frame_ground_count(det, est, cfg)
            return classify.detect_burns(times, elev, ground_count, cfg, duration)
    present = classify.frame_presence(det)
    return classify.detect_burns_presence(times, present, cfg, duration)


def _cut_burns(job, info, burns, cfg, emit) -> list[dict]:
    src = Path(job.source)
    clips: list[dict] = []
    for i, burn in enumerate(burns, 1):
        name = f"attempt_{i:02d}.mp4"
        clip.cut_clip(src, job.out_dir / name, burn.start, burn.end, info, cfg)
        clips.append({
            "name": name,
            "url": f"/api/clip/{job.id}/{name}",
            "start": round(burn.start, 2),
            "end": round(burn.end, 2),
            "duration": round(burn.end - burn.start, 2),
        })
        emit({"stage": "cut", "pct": round(i / len(burns), 3),
              "message": f"Exported {i}/{len(burns)} clips"})
    return clips


def _stream(worker: Callable[[Callable[[dict], None]], None]) -> Iterator[dict]:
    """Run `worker(emit)` in a thread, yielding emitted events until it ends."""
    q: queue.Queue = queue.Queue()

    def run():
        try:
            worker(q.put)
        except Exception as exc:  # surface failures to the client
            q.put({"type": "error", "message": str(exc)})
        finally:
            q.put(None)

    threading.Thread(target=run, daemon=True).start()
    while True:
        ev = q.get()
        if ev is None:
            break
        yield ev


# --- Public streaming entrypoints -------------------------------------------

def analyze_stream(job: Job, cfg: Config = Config()) -> Iterator[dict]:
    def worker(emit):
        emit({"stage": "probe", "message": "Reading video…"})
        info = clip.probe(Path(job.source))
        emit({"stage": "probe", "duration": round(info.duration, 1),
              "message": f"{info.duration/60:.1f} min, {info.width}x{info.height}"})

        emit({"stage": "detect", "pct": 0.0, "message": "Detecting climber…"})
        det = detect.run_detection(
            Path(job.source), info, cfg,
            progress=lambda f: emit({"stage": "detect", "pct": round(f, 3),
                                     "message": "Detecting climber…"}),
        )
        job.out_dir.mkdir(parents=True, exist_ok=True)
        job.cache_path.write_text(json.dumps(detect.result_to_dict(det)))
        job.save_meta({"duration": info.duration, "width": info.width, "height": info.height})

        est = classify.estimate_ground(det, cfg)
        if sum(len(fd.detections) for fd in det.frames) == 0:
            emit({"type": "done", "clips": [],
                  "message": "No people detected in this video."})
            return

        burns = _compute_burns(det, est, cfg, info.duration)
        emit({"stage": "segment", "message": f"Found {len(burns)} attempt(s)."})
        clips = _cut_burns(job, info, burns, cfg, emit)
        emit({"type": "done", "clips": clips})

    return _stream(worker)


def recut_stream(job: Job, cfg: Config = Config()) -> Iterator[dict]:
    """Re-segment and re-cut from cached detections — no re-detection needed.
    Handy for re-tuning config.py thresholds against an already-analyzed video."""
    def worker(emit):
        det = detect.result_from_dict(json.loads(job.cache_path.read_text()))
        meta = json.loads(job.meta_path.read_text())
        info = clip.VideoInfo(meta["width"], meta["height"], meta["duration"], 0.0)
        est = classify.estimate_ground(det, cfg)
        burns = _compute_burns(det, est, cfg, info.duration)
        emit({"stage": "segment", "message": f"Found {len(burns)} attempt(s)."})
        clips = _cut_burns(job, info, burns, cfg, emit)
        emit({"type": "done", "clips": clips})

    return _stream(worker)


def list_clips(job_id: str) -> list[dict]:
    out_dir = OUTPUT_DIR / job_id
    clips = []
    for p in sorted(out_dir.glob("attempt_*.mp4")):
        clips.append({"name": p.name, "url": f"/api/clip/{job_id}/{p.name}"})
    return clips
