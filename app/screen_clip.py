"""Quick check (CLIP variant): does a video contain climbing?

A different paradigm from `screen.py`: instead of detecting people and reasoning
about elevation above an estimated ground band, this asks a zero-shot CLIP model
"does this frame look like rock climbing?" directly. That sidesteps the cases
where the geometric signal is weakest — a framed-up shot with no base in view,
or a climber too small for a confident body-height — at the cost of a heavier
model and a less interpretable score.

Optional dependency: `open_clip_torch`. Install with:
    uv sync --extra clip
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

import numpy as np

from . import classify, clip
from .config import Config

# Zero-shot prompt sets. CLIP scores each frame against both groups; the
# climbing score is the summed positive probability after a softmax over *all*
# prompts, so a frame only reads as climbing when it beats the negatives too.
_POSITIVE = [
    "a person rock climbing on a cliff",
    "a rock climber high on a steep rock face",
    "a person bouldering on rock",
    "someone climbing a tall outdoor rock wall",
]
_NEGATIVE = [
    "an empty rock cliff with no people",
    "people standing around on the ground",
    "a landscape photo of mountains",
    "a person hiking on a trail",
    "a close-up of trees and grass",
]


def _load_clip(cfg: Config):
    try:
        import open_clip
        import torch
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "CLIP screening needs open_clip_torch. Install with: uv sync --extra clip"
        ) from e

    from .detect import _resolve_device

    device = _resolve_device(cfg.device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.clip_model, pretrained=cfg.clip_pretrained
    )
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(cfg.clip_model)

    prompts = _POSITIVE + _NEGATIVE
    with torch.no_grad():
        text = tokenizer(prompts).to(device)
        text_feat = model.encode_text(text)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
    return model, preprocess, text_feat, device, len(_POSITIVE)


def _frame_scores(video_path: Path, info: clip.VideoInfo, cfg: Config,
                  progress: Callable[[float], None] | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, climbing_score) per sampled frame. Score is in 0..1."""
    import torch
    from PIL import Image

    model, preprocess, text_feat, device, n_pos = _load_clip(cfg)
    total = max(1, int(info.duration * cfg.clip_fps))
    times, scores = [], []
    for i, (t, frame) in enumerate(clip.sample_frames(video_path, info, cfg)):
        # sample_frames yields BGR (OpenCV); CLIP wants RGB PIL.
        img = Image.fromarray(frame[:, :, ::-1])
        with torch.no_grad():
            x = preprocess(img).unsqueeze(0).to(device)
            feat = model.encode_image(x)
            feat /= feat.norm(dim=-1, keepdim=True)
            logits = (100.0 * feat @ text_feat.T).softmax(dim=-1)[0]
            scores.append(float(logits[:n_pos].sum().cpu()))
        times.append(t)
        if progress:
            progress(min(1.0, (i + 1) / total))
    return np.asarray(times), np.asarray(scores)


def screen_video_clip(
    video_path: Path, cfg: Config = Config(),
    progress: Callable[[float], None] | None = None,
) -> dict:
    """Return {present, peak_score, climbing_seconds, longest_stretch_seconds, duration}.

    Mirrors `screen.screen_video`'s contract (minus the body-height field) so the
    CLI can swap between them."""
    video_path = Path(video_path)
    info = clip.probe(video_path)
    # CLIP needs only a modest resolution; reuse the fast sampling cadence.
    scfg = dataclasses.replace(cfg, analysis_fps=cfg.clip_fps, analysis_width=640)
    times, scores = _frame_scores(video_path, info, scfg, progress)

    if len(times) < 2:
        return {"present": False, "peak_score": 0.0, "climbing_seconds": 0.0,
                "longest_stretch_seconds": 0.0, "duration": round(info.duration, 1)}

    dt = float(np.median(np.diff(times)))
    high = scores >= cfg.clip_threshold
    longest = max((times[en] - times[s] + dt for s, en in classify._runs(high)), default=0.0)
    return {
        "present": bool(longest >= cfg.min_climb_seconds),
        "peak_score": round(float(scores.max()), 3),
        "climbing_seconds": round(float(high.sum()) * dt, 1),
        "longest_stretch_seconds": round(float(longest), 1),
        "duration": round(info.duration, 1),
    }
