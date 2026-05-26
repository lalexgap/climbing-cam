"""Quick check: does a video contain climbing?

Reuses the detection + elevation signal, but at a faster/cheaper setting (lower
fps + smaller model) since we only need a yes/no, not frame-accurate clips.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

import numpy as np

from . import classify, clip, detect
from .config import Config


def screen_video(
    video_path: Path, cfg: Config = Config(),
    progress: Callable[[float], None] | None = None,
) -> dict:
    """Return {present, peak_bh, climbing_seconds, longest_stretch_seconds, duration}."""
    video_path = Path(video_path)
    info = clip.probe(video_path)
    # Fast screening config: 1 fps, smaller model/resolution. Enough to spot a
    # climber elevated somewhere in the clip without a full-quality pass.
    scfg = dataclasses.replace(
        cfg, analysis_fps=1.0, analysis_width=960, imgsz=960,
        model=cfg.screen_model, conf=0.20,
    )
    det = detect.run_detection(video_path, info, scfg, progress)
    est = classify.estimate_ground(det, scfg)
    times = np.array([fd.t for fd in det.frames], dtype=float)
    elev = classify.frame_max_elevation(det, est)
    result = classify.climbing_evidence(times, elev, scfg)
    result["duration"] = round(info.duration, 1)
    return result
