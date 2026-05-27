"""ffmpeg/ffprobe helpers: probe metadata, sample analysis frames, cut clips.

We decode at a low fps / downscaled for analysis (cheap), but cut the final
clips from the *original* file at full resolution with the hardware encoder.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import Config


IS_DARWIN = platform.system() == "Darwin"


def _command_text(cmd: list[str]) -> str:
    return " ".join(str(part) for part in cmd)


def _run_logged(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess and preserve stderr/stdout when it fails.

    subprocess.CalledProcessError only prints the command by default, which hid
    the useful ffmpeg error text from unattended backfills.
    """
    result = subprocess.run(cmd, capture_output=True, **kwargs)
    if result.returncode == 0:
        return result
    stderr_raw = result.stderr or b""
    stdout_raw = result.stdout or b""
    if isinstance(stderr_raw, bytes):
        stderr = stderr_raw.decode(errors="replace").strip()
    else:
        stderr = stderr_raw.strip()
    if isinstance(stdout_raw, bytes):
        stdout = stdout_raw.decode(errors="replace").strip()
    else:
        stdout = stdout_raw.strip()
    details = []
    if stderr:
        details.append(f"stderr:\n{stderr[-4000:]}")
    if stdout:
        details.append(f"stdout:\n{stdout[-2000:]}")
    suffix = "\n" + "\n".join(details) if details else ""
    raise RuntimeError(
        f"command failed ({result.returncode}): {_command_text(cmd)}{suffix}"
    )


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
    out = _run_logged(cmd, text=True).stdout
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


def _decode_args() -> list[str]:
    if IS_DARWIN:
        return ["-hwaccel", "videotoolbox"]
    return []


def _encode_args(info: VideoInfo, cfg: Config) -> list[str]:
    bitrate = cfg.bitrate_for_height(info.height)
    if IS_DARWIN:
        return ["-c:v", "h264_videotoolbox", "-b:v", bitrate]
    return ["-c:v", "libx264", "-preset", "veryfast", "-threads", "2", "-b:v", bitrate]


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
        *_decode_args(), "-i", str(path),
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
    out = _run_logged(cmd).stdout
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
        *_encode_args(info, cfg),
    ]
    if cfg.audio:
        cmd += ["-c:a", "aac", "-b:a", "160k"]
    else:
        cmd += ["-an"]
    # Carry the original's creation date + GPS location (com.apple.quicktime.*)
    # into the clip; use_metadata_tags preserves the custom QuickTime tags that
    # -map_metadata alone would drop on re-encode.
    cmd += ["-map_metadata", "0",
            "-movflags", "use_metadata_tags+faststart", str(out_path)]
    _run_logged(cmd)


def _atempo_chain(speed: float) -> str:
    """ffmpeg atempo accepts 0.5..2.0; chain filters for larger speed-ups."""
    factors, s = [], speed
    while s > 2.0:
        factors.append(2.0)
        s /= 2.0
    factors.append(s)
    return "".join(f",atempo={f:.4f}" for f in factors)


def _make_badge(path: Path, speed: float) -> None:
    """Render a small RGBA ">> Nx" badge PNG (this ffmpeg build lacks drawtext,
    so we overlay an image instead)."""
    import cv2

    W, H = 520, 180
    img = np.zeros((H, W, 4), np.uint8)
    cv2.rectangle(img, (0, 0), (W - 1, H - 1), (45, 105, 35, 205), -1)      # BGRA fill
    cv2.rectangle(img, (4, 4), (W - 5, H - 5), (130, 235, 130, 255), 6)     # border
    cv2.putText(img, f">> {int(speed)}x", (30, 122), cv2.FONT_HERSHEY_DUPLEX,
                3.2, (255, 255, 255, 255), 6, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def cut_clip_ramped(
    src: Path, out_path: Path, start: float, end: float,
    rests: list[tuple[float, float]], info: VideoInfo, cfg: Config,
) -> None:
    """Cut [start, end] but play `rests` (absolute seconds) at cfg.rest_speedup
    and everything else at 1x, via a trim/setpts/concat filtergraph. Audio is
    sped with the video (atempo)."""
    if not rests:
        return cut_clip(src, out_path, start, end, info, cfg)

    dur = end - start
    # Burn-relative, clamped, non-overlapping rest spans.
    rel = sorted((max(0.0, a - start), min(dur, b - start)) for a, b in rests)

    # Alternating segments covering [0, dur]: (a, b, speed).
    segs, cur = [], 0.0
    for a, b in rel:
        a = max(a, cur)
        if a > cur + 0.05:
            segs.append((cur, a, 1.0))
        if b > a + 0.05:
            segs.append((a, b, cfg.rest_speedup))
        cur = max(cur, b)
    if cur < dur - 0.05:
        segs.append((cur, dur, 1.0))

    n_rest = sum(1 for *_, s in segs if s != 1.0)
    marker = cfg.ramp_marker and n_rest > 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    badge = out_path.parent / "_badge.png"
    if marker:  # corner "8x" badge overlaid on sped sections
        _make_badge(badge, cfg.rest_speedup)
        bw = max(160, info.width // 6)
        mx = max(16, info.width // 50)            # right margin
        my = max(48, info.width // 18)            # top margin (lower, off the edge)

    tmp_dir = out_path.parent / f".{out_path.stem}.parts"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    try:
        part_paths = []
        for i, (a, b, s) in enumerate(segs):
            part = tmp_dir / f"part_{i:03d}.mp4"
            part_paths.append(part)
            part_start = start + a
            part_dur = b - a
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{part_start:.3f}", "-t", f"{part_dur:.3f}",
                "-i", str(src),
            ]
            if marker and s != 1.0:
                fc = (
                    f"[0:v]setpts=(PTS-STARTPTS)/{s}[vr];"
                    f"[1:v]scale={bw}:-1,format=rgba[bd];"
                    f"[vr][bd]overlay=W-w-{mx}:{my}[v];"
                    f"[0:a]asetpts=PTS-STARTPTS{_atempo_chain(s)}[a]"
                )
                cmd += [
                    "-i", str(badge),
                    "-filter_threads", "1", "-filter_complex_threads", "1",
                    "-filter_complex", fc,
                    "-map", "[v]", "-map", "[a]",
                ]
            else:
                cmd += [
                    "-filter_threads", "1", "-filter_complex_threads", "1",
                    "-filter_complex",
                    f"[0:v]setpts=(PTS-STARTPTS)/{s}[v];"
                    f"[0:a]asetpts=PTS-STARTPTS{_atempo_chain(s)}[a]",
                    "-map", "[v]", "-map", "[a]",
                ]
            cmd += [
                *_encode_args(info, cfg),
                "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
                str(part),
            ]
            _run_logged(cmd)

        concat_file = tmp_dir / "concat.txt"
        concat_file.write_text("".join(f"file '{p}'\n" for p in part_paths))
        # Concat the parts and copy the original's date + GPS location across
        # from a metadata-only second input (the parts dropped it on re-encode).
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-i", str(src),
            "-map", "0", "-map_metadata", "1", "-c", "copy",
            "-movflags", "use_metadata_tags+faststart", str(out_path),
        ]
        _run_logged(cmd)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
