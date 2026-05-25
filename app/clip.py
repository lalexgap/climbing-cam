"""ffmpeg/ffprobe helpers: probe metadata, sample analysis frames, cut clips.

We decode at a low fps / downscaled for analysis (cheap), but cut the final
clips from the *original* file at full resolution with the hardware encoder.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import Config


@dataclass
class VideoInfo:
    width: int          # display width (after rotation)
    height: int         # display height (after rotation)
    duration: float     # seconds
    fps: float


def probe(path: Path) -> VideoInfo:
    """Read video metadata via ffprobe, accounting for display rotation.

    Phone videos are usually stored landscape with a rotation in the Display
    Matrix side-data; ffmpeg auto-rotates on decode, so we must report the
    *rotated* (display) dimensions or analysis frames get squished."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate:format=duration",
        "-of", "json", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    stream = json.loads(out)["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])

    if abs(_probe_rotation(path)) % 180 == 90:
        width, height = height, width

    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    duration = float(json.loads(out)["format"]["duration"])
    return VideoInfo(width=width, height=height, duration=duration, fps=fps)


def _probe_rotation(path: Path) -> int:
    """Display rotation in degrees (e.g. -90 for portrait), via the Display
    Matrix side-data. Returns 0 if absent."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream_side_data=rotation",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    ).stdout
    for line in out.splitlines():
        try:
            return int(line.strip())
        except ValueError:
            continue
    return 0


def _scaled_height(info: VideoInfo, target_w: int) -> int:
    h = round(info.height * target_w / info.width)
    return h - (h % 2)  # keep even for rawvideo


def sample_frames(
    path: Path, info: VideoInfo, cfg: Config
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp_seconds, BGR frame) sampled at cfg.analysis_fps.

    Uses ffmpeg with autorotation so frames match display orientation, and
    hardware decode where available. Frames are downscaled to cfg.analysis_width.
    """
    out_w = min(cfg.analysis_width, info.width)
    out_h = _scaled_height(info, out_w)
    frame_bytes = out_w * out_h * 3

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-hwaccel", "videotoolbox",
        "-i", str(path),
        "-vf", f"fps={cfg.analysis_fps},scale={out_w}:{out_h}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None

    idx = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(out_h, out_w, 3)
            yield idx / cfg.analysis_fps, frame
            idx += 1
    finally:
        proc.stdout.close()
        ret = proc.wait()
        if ret not in (0, None) and idx == 0:
            err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            raise RuntimeError(f"ffmpeg sampling failed: {err.strip()}")


def extract_frame(path: Path, t: float, info: VideoInfo, cfg: Config) -> np.ndarray:
    """Grab a single BGR frame at time `t`, downscaled to analysis width so the
    coordinates line up with detection boxes."""
    out_w = min(cfg.analysis_width, info.width)
    out_h = _scaled_height(info, out_w)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
        "-vf", f"scale={out_w}:{out_h}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    # .copy() -> writable array (frombuffer is read-only, breaks cv2 drawing).
    return np.frombuffer(out[: out_w * out_h * 3], np.uint8).reshape(out_h, out_w, 3).copy()


def cut_clip(
    src: Path, out_path: Path, start: float, end: float, info: VideoInfo, cfg: Config
) -> None:
    """Cut [start, end] from the source at full resolution, re-encoding with the
    hardware H.264 encoder for frame-accurate, fast output."""
    duration = max(0.1, end - start)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{duration:.3f}",
        "-c:v", "h264_videotoolbox", "-b:v", cfg.bitrate_for_height(info.height),
    ]
    if cfg.audio:
        cmd += ["-c:a", "aac", "-b:a", "160k"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)
