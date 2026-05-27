"""Tunable configuration for the climbing-cam pipeline.

Everything that might need calibration on real footage lives here so the
detection/segmentation logic stays declarative. Defaults are tuned for the
target case: outdoor sport climbing, fixed camera, base of the route in frame.

The key idea behind the elevation signal: a climber's vertical position is
measured relative to an estimated "ground band" and expressed in *body-heights*
(the climber's own pixel height near the ground). Body-height units make the
thresholds independent of how far back the phone is or whether you shot 1080p
or 4K.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- Paths -------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
WEB_DIR = PROJECT_DIR / "web"


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    # --- Frame sampling (analysis only; clips are cut from the original) -----
    # 2 fps = 0.5s resolution, ample given the 8s/15s burn thresholds, and ~33%
    # faster than 3 fps. Bump up if you need finer burn boundaries.
    analysis_fps: float = env_float("CLIMBING_CAM_ANALYSIS_FPS", 2.0)
    # Width we downscale frames to before detection. Higher preserves a small,
    # distant climber high on a tall route at the cost of speed. Key tuning knob:
    # 1920 reliably catches a distant outdoor climber (~70% frame recall on a 4K
    # portrait route); drop to 1280 for ~2x faster runs if your climber is large
    # in frame.
    analysis_width: int = env_int("CLIMBING_CAM_ANALYSIS_WIDTH", 1920)

    # --- Detection / tracking ------------------------------------------------
    model: str = env("CLIMBING_CAM_MODEL", "yolo11m.pt")
    screen_model: str = env("CLIMBING_CAM_SCREEN_MODEL", "yolo11s.pt")
    device: str = env("CLIMBING_CAM_DEVICE", "mps")
    person_class: int = env_int("CLIMBING_CAM_PERSON_CLASS", 0)
    conf: float = env_float("CLIMBING_CAM_CONF", 0.20)
    imgsz: int = env_int("CLIMBING_CAM_IMGSZ", 1920)
    tracker: str = env("CLIMBING_CAM_TRACKER", "bytetrack.yaml")

    # --- Pose variant (experimental) ----------------------------------------
    # Opt-in: use a YOLO *pose* model and derive each detection's bottom/height
    # from keypoints (ankles -> nose) instead of the raw box. The box bottom is
    # inflated by raised arms / dangling rope; an anatomical span gives a steadier
    # elevation signal. Falls back to the box per-frame when keypoints are too
    # low-confidence (small/distant/occluded climber). Same framework, same
    # Detection downstream — classify.py is untouched. Toggle and re-detect.
    pose: bool = False
    pose_model: str = "yolo11m-pose.pt"  # auto-downloads on first use
    kpt_conf: float = 0.30               # min keypoint confidence to trust it

    # --- Ground band + elevation --------------------------------------------
    # Ground line estimated as this percentile of all person box-bottoms (image
    # y grows downward, so the ground band sits at a high y value).
    ground_percentile: float = 85.0
    # A box-bottom counts as "near the ground" (for body-height calibration)
    # when within this many body-heights of the ground line.
    near_ground_bh: float = 0.5
    # Elevation thresholds (body-heights):
    #  - leave_bh: feet off the ground / starting to climb. Defines where a burn
    #    *starts* (catches the first low moves) and its on-wall extent.
    #  - enter_bh: clearly a real climb. A burst must reach this to count (filters
    #    a belayer/standing noise), and the clip *ends* at the last enter_bh moment
    #    (so the lower-off is trimmed).
    #  - exit_bh: below this counts as "at the base" for ground-people counting.
    leave_bh: float = 0.5
    enter_bh: float = 1.0
    exit_bh: float = 0.4
    # An enter_bh crossing must persist at least this long to count as on-wall
    # (vs. a one-frame detection spike on a person-shaped rock feature). At the
    # default 2 fps this requires ~2+ consecutive elevated frames. Guards the
    # burn end especially under the higher-recall pose detector.
    enter_persist_seconds: float = 1.0

    # --- Burn segmentation ---------------------------------------------------
    # Splitting one video into separate attempts. Consecutive on-wall stretches
    # merge through short gaps (dropouts, on-wall hangs); a gap longer than
    # merge_gap_seconds means you came down and rested => a new attempt. (On real
    # footage a hang is ~minutes while a between-go rest is much longer, and the
    # belayer/you are often hidden at the base, so an explicit gap is far more
    # reliable than counting people on the ground.) A sustained 2+ people-at-base
    # stretch also forces a boundary when it *is* visible.
    merge_gap_seconds: float = 300.0
    # A climber whose box top is within this fraction of the frame's top edge has
    # "climbed out the top of frame". Such an exit suppresses the long-gap split
    # (you're still on the same ascent, just out of view) unless you actually come
    # back down to the base — so a route climbed up out of frame stays one attempt.
    top_exit_frac: float = 0.06
    min_ground_people: int = 2
    ground_rest_seconds: float = 25.0
    min_burn_seconds: float = 15.0    # discard on-wall runs shorter than this
    # When extending a burn start back over the first low moves, bridge lulls up
    # to this long (you momentarily undetected near the ground); stop at the
    # first longer gap so we don't grab pre-climb faffing at the base.
    start_link_seconds: float = 8.0
    # A burn's start anchors at the first enter_bh run that lasts at least this
    # long (sustained climbing). Leading shorter blips — near-camera setup, base
    # staging — are skipped so the clip starts when you actually got on route.
    start_anchor_seconds: float = 15.0
    # Hard cap on the start-extension: the lead-in is the *first low moves*, so it
    # should never reach more than this far before the confirmed climb. Guards the
    # near-continuous-activity case (back-to-back goes / other parties on the wall)
    # where the leave_bh signal never drops and the start would otherwise walk back
    # across the whole video, swallowing earlier attempts.
    max_lead_seconds: float = 60.0
    lost_timeout_seconds: float = 30.0  # framed-up fallback: merge gaps under this
    pad_lead_seconds: float = 3.0     # include this much before leaving the ground
    apex_tail_seconds: float = 3.0    # clip ends this long after the last on-wall moment

    # --- Climber selection ---------------------------------------------------
    # A track must rise at least this high (body-heights) to count as an
    # "ascending" track / climbing candidate.
    ascend_bh: float = 1.0
    # If two or more tracks ascend and the runner-up reaches at least this
    # fraction of the best track's peak elevation, selection is ambiguous and
    # we ask the user to confirm.
    ambiguous_ratio: float = 0.6

    # --- Climbing screening (is there climbing in this video?) --------------
    # Someone elevated >= ascend_bh for a contiguous stretch this long = climbing.
    min_climb_seconds: float = 4.0

    # --- CLIP screening variant (experimental; --check --clip) --------------
    # Zero-shot "does this frame look like climbing?" — no people/elevation
    # logic. A frame counts when its summed positive-prompt probability clears
    # clip_threshold; the video counts when a contiguous run exceeds
    # min_climb_seconds. Needs the optional `clip` extra (open_clip_torch).
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    clip_fps: float = 1.0
    clip_threshold: float = 0.5

    # --- Phase 2: speed-ramp the rests --------------------------------------
    # A "rest" is a hang: a sustained stretch where your height stays flat — you
    # make no net vertical progress. Measured on the (dropout-interpolated)
    # elevation, so it catches both visible hangs and undetected ones. Active
    # climbing gains height and stays 1x. (We rejected optical-flow body motion:
    # on real footage resting/shaking and small climbing moves overlap too much.)
    speed_ramp: bool = env_bool("CLIMBING_CAM_SPEED_RAMP", True)
    rest_speedup: float = env_float("CLIMBING_CAM_REST_SPEEDUP", 8.0)
    min_rest_seconds: float = env_float("CLIMBING_CAM_MIN_REST_SECONDS", 45.0)
    rest_band_bh: float = env_float("CLIMBING_CAM_REST_BAND_BH", 0.2)
    rest_smooth_seconds: float = env_float("CLIMBING_CAM_REST_SMOOTH_SECONDS", 5.0)
    rest_inset_seconds: float = env_float("CLIMBING_CAM_REST_INSET_SECONDS", 3.0)
    ramp_marker: bool = env_bool("CLIMBING_CAM_RAMP_MARKER", True)

    # --- Encoding (clip cutting) --------------------------------------------
    # Target video bitrate by output height (h264_videotoolbox, in Mbps).
    audio: bool = True

    def bitrate_for_height(self, height: int) -> str:
        if height >= 2000:      # ~4K
            return "45M"
        if height >= 1300:      # ~1440p
            return "22M"
        if height >= 1000:      # ~1080p
            return "14M"
        return "8M"


DEFAULT = Config()
