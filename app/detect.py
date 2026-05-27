"""Person detection + tracking over sampled frames using Ultralytics YOLO.

Default models are plain box detectors: they stay reliable when the climber is
small and distant high on the wall, and the box is all the elevation signal
needs. An opt-in pose variant (`cfg.pose`) instead anchors each detection's
bottom and height to keypoints (ankles -> nose) for a steadier signal, falling
back to the box wherever keypoints are too low-confidence. Either way the
emitted `Detection` is box-shaped, so `classify.py` is agnostic to the choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .clip import VideoInfo, sample_frames
from .config import Config


@dataclass
class Detection:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def bottom(self) -> float:
        return self.y2

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2


@dataclass
class FrameDetections:
    t: float
    detections: list[Detection] = field(default_factory=list)


@dataclass
class DetectionResult:
    frames: list[FrameDetections]
    frame_width: int
    frame_height: int


def _resolve_device(requested: str) -> str:
    """Pick the best available device. "mps"/"auto" prefer Apple GPU, then CUDA
    (cloud GPUs), then CPU — so the same config works on a Mac or a Linux box."""
    try:
        import torch

        if requested in ("mps", "auto") and torch.backends.mps.is_available():
            return "mps"
        if requested in ("cuda", "auto", "mps") and torch.cuda.is_available():
            return "cuda"
        if requested == "cuda" and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# COCO pose keypoint indices used to anchor the body span.
_NOSE, _L_ANKLE, _R_ANKLE = 0, 15, 16


def _pose_anchor(kxy: np.ndarray, kconf: np.ndarray, box: tuple[float, float, float, float],
                 cfg: Config) -> tuple[float, float]:
    """Return (top_y, bottom_y) for one person from its keypoints.

    Bottom = lowest confident ankle (where the climber's feet are on the wall);
    top = the nose. Falls back to the box edge for whichever end lacks a
    confident keypoint, so a distant climber with only a torso visible still
    yields a usable span rather than being dropped."""
    bx1, by1, bx2, by2 = box
    ankles = [kxy[k][1] for k in (_L_ANKLE, _R_ANKLE) if kconf[k] >= cfg.kpt_conf]
    bottom = max(ankles) if ankles else by2
    top = float(kxy[_NOSE][1]) if kconf[_NOSE] >= cfg.kpt_conf else by1
    # Guard against degenerate/inverted spans (e.g. nose below ankle on a fall).
    if bottom - top < 1.0:
        return by1, by2
    return top, bottom


def run_detection(
    video_path: Path,
    info: VideoInfo,
    cfg: Config,
    progress: Callable[[float], None] | None = None,
) -> DetectionResult:
    """Sample frames and run tracked person detection on each.

    `progress` is called with a 0..1 fraction as frames are processed.
    """
    from ultralytics import YOLO

    device = _resolve_device(cfg.device)
    model = YOLO(cfg.pose_model if cfg.pose else cfg.model)

    total = max(1, int(info.duration * cfg.analysis_fps))
    frames: list[FrameDetections] = []
    frame_w = frame_h = 0

    for i, (t, frame) in enumerate(sample_frames(video_path, info, cfg)):
        frame_h, frame_w = frame.shape[:2]
        results = model.track(
            frame,
            persist=True,
            classes=[cfg.person_class],
            conf=cfg.conf,
            imgsz=cfg.imgsz,
            device=device,
            tracker=cfg.tracker,
            verbose=False,
        )
        fd = FrameDetections(t=t)
        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            kxy = kconf = None
            if cfg.pose and results[0].keypoints is not None:
                kxy = results[0].keypoints.xy.cpu().numpy()
                kconf = results[0].keypoints.conf
                kconf = kconf.cpu().numpy() if kconf is not None else None
            for j, ((x1, y1, x2, y2), tid, c) in enumerate(zip(xyxy, ids, confs)):
                if kxy is not None and kconf is not None:
                    y1, y2 = _pose_anchor(kxy[j], kconf[j], (x1, y1, x2, y2), cfg)
                fd.detections.append(
                    Detection(int(tid), float(x1), float(y1), float(x2), float(y2), float(c))
                )
        frames.append(fd)
        if progress:
            progress(min(1.0, (i + 1) / total))

    return DetectionResult(frames=frames, frame_width=frame_w, frame_height=frame_h)


# --- Serialization (cache detections between analyze and confirm/finalize) ---

def result_to_dict(result: DetectionResult) -> dict:
    return {
        "frame_width": result.frame_width,
        "frame_height": result.frame_height,
        "frames": [
            {
                "t": fd.t,
                "detections": [
                    [d.track_id, d.x1, d.y1, d.x2, d.y2, d.conf] for d in fd.detections
                ],
            }
            for fd in result.frames
        ],
    }


def result_from_dict(data: dict) -> DetectionResult:
    frames = [
        FrameDetections(
            t=fd["t"],
            detections=[Detection(int(d[0]), *map(float, d[1:])) for d in fd["detections"]],
        )
        for fd in data["frames"]
    ]
    return DetectionResult(
        frames=frames,
        frame_width=data["frame_width"],
        frame_height=data["frame_height"],
    )
