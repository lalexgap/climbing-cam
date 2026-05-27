"""A/B harness: box vs pose detection on the same video.

Runs detection twice over one video — once with the plain person box, once with
the keypoint-anchored pose variant (`cfg.pose`) — then lays the two elevation
signals side by side so you can *see* whether pose actually sharpens the signal
before committing to it. Both detection results are cached so you can recut from
either, and per-frame elevations are dumped to CSV for closer inspection.

    uv run python -m app.ab <video>
    uv run python -m app.ab <video> --csv out.csv

Prints, for each variant: the ground/body-height estimate, peak elevation, the
burns it would produce, and an ASCII sparkline of the elevation curve. Detection
is the slow part and runs once per variant, so expect ~2x a normal detect pass.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

from . import classify, clip, detect
from .config import Config


def _sparkline(values: np.ndarray) -> str:
    """8-level unicode sparkline; NaN (no detection) renders as a space."""
    bars = "▁▂▃▄▅▆▇█"
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return " " * len(values)
    lo, hi = float(finite.min()), float(finite.max())
    span = hi - lo or 1.0
    out = []
    for v in values:
        if not np.isfinite(v):
            out.append(" ")
        else:
            out.append(bars[min(7, int((v - lo) / span * 7.999))])
    return "".join(out)


def _analyze(video_path: Path, info: clip.VideoInfo, cfg: Config) -> dict:
    """Detect, estimate ground, and segment burns for one config variant."""
    det = detect.run_detection(video_path, info, cfg)
    est = classify.estimate_ground(det, cfg)
    times = np.array([fd.t for fd in det.frames], dtype=float)
    elev = classify.frame_max_elevation(det, est)
    gcount = classify.frame_ground_count(det, est, cfg)
    burns = classify.detect_burns(times, elev, gcount, cfg, info.duration)
    return {"det": det, "est": est, "times": times, "elev": elev, "burns": burns}


def _report(label: str, a: dict) -> None:
    est, elev, burns = a["est"], a["elev"], a["burns"]
    peak = float(np.nanmax(elev)) if np.isfinite(elev).any() else 0.0
    print(f"\n=== {label} ===")
    print(f"  ground_y={est.ground_y:.0f}  body_h={est.body_h:.0f}  "
          f"reliable={est.reliable}")
    print(f"  peak elevation: {peak:.2f} body-heights")
    print(f"  burns: {len(burns)}")
    for i, b in enumerate(burns, 1):
        print(f"    {i}. {b.start:6.1f}s -> {b.end:6.1f}s  (apex {b.apex_t:.1f}s)")
    print(f"  signal: |{_sparkline(elev)}|")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    csv_path = None
    if "--csv" in argv:
        k = argv.index("--csv")
        csv_path = Path(argv[k + 1])
        del argv[k:k + 2]
    if not argv:
        print("usage: python -m app.ab <video> [--csv out.csv]")
        return 2
    video = Path(argv[0]).expanduser().resolve()
    if not video.exists():
        print(f"video not found: {video}")
        return 2

    info = clip.probe(video)
    base = Config()
    print(f"Detecting {video.name}  ({info.duration:.0f}s)  — box pass…")
    box = _analyze(video, info, base)
    print("  — pose pass…")
    pose = _analyze(video, info, dataclasses.replace(base, pose=True))

    _report("box  (yolo11m.pt)", box)
    _report(f"pose ({base.pose_model})", pose)

    # Correlation on frames where both signals exist — a quick read on whether
    # pose merely tracks the box or genuinely diverges.
    both = np.isfinite(box["elev"]) & np.isfinite(pose["elev"])
    if both.sum() >= 2 and np.std(box["elev"][both]) > 0 and np.std(pose["elev"][both]) > 0:
        r = float(np.corrcoef(box["elev"][both], pose["elev"][both])[0, 1])
        print(f"\noverlap: {int(both.sum())}/{both.size} frames  |  correlation r={r:.3f}")

    if csv_path:
        times = box["times"]
        rows = ["t,box_elev,pose_elev"]
        for i, t in enumerate(times):
            be = box["elev"][i]
            pe = pose["elev"][i]
            rows.append(f"{t:.3f},{'' if not np.isfinite(be) else f'{be:.4f}'},"
                        f"{'' if not np.isfinite(pe) else f'{pe:.4f}'}")
        csv_path.write_text("\n".join(rows) + "\n")
        print(f"\nwrote {csv_path}  ({len(times)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
