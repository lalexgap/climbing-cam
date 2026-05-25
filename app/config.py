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

from dataclasses import dataclass
from pathlib import Path

# --- Paths -------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
WEB_DIR = PROJECT_DIR / "web"


@dataclass(frozen=True)
class Config:
    # --- Frame sampling (analysis only; clips are cut from the original) -----
    # 2 fps = 0.5s resolution, ample given the 8s/15s burn thresholds, and ~33%
    # faster than 3 fps. Bump up if you need finer burn boundaries.
    analysis_fps: float = 2.0
    # Width we downscale frames to before detection. Higher preserves a small,
    # distant climber high on a tall route at the cost of speed. Key tuning knob:
    # 1920 reliably catches a distant outdoor climber (~70% frame recall on a 4K
    # portrait route); drop to 1280 for ~2x faster runs if your climber is large
    # in frame.
    analysis_width: int = 1920

    # --- Detection / tracking ------------------------------------------------
    model: str = "yolo11m.pt"   # detection model; auto-downloads on first use
    device: str = "mps"          # Apple GPU; falls back to "cpu" if unavailable
    person_class: int = 0        # COCO "person"
    conf: float = 0.20           # low-ish to catch the small distant climber
    imgsz: int = 1920            # inference size; match analysis_width
    tracker: str = "bytetrack.yaml"

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

    # --- Burn segmentation ---------------------------------------------------
    # Splitting one video into separate attempts. Consecutive on-wall stretches
    # merge through short gaps (dropouts, on-wall hangs); a gap longer than
    # merge_gap_seconds means you came down and rested => a new attempt. (On real
    # footage a hang is ~minutes while a between-go rest is much longer, and the
    # belayer/you are often hidden at the base, so an explicit gap is far more
    # reliable than counting people on the ground.) A sustained 2+ people-at-base
    # stretch also forces a boundary when it *is* visible.
    merge_gap_seconds: float = 300.0
    min_ground_people: int = 2
    ground_rest_seconds: float = 25.0
    min_burn_seconds: float = 15.0    # discard on-wall runs shorter than this
    # When extending a burn start back over the first low moves, bridge lulls up
    # to this long (you momentarily undetected near the ground); stop at the
    # first longer gap so we don't grab pre-climb faffing at the base.
    start_link_seconds: float = 8.0
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

    # --- Phase 2: speed-ramp the rests --------------------------------------
    # A "rest" is a hang: a sustained stretch where your height stays flat — you
    # make no net vertical progress. Measured on the (dropout-interpolated)
    # elevation, so it catches both visible hangs and undetected ones. Active
    # climbing gains height and stays 1x. (We rejected optical-flow body motion:
    # on real footage resting/shaking and small climbing moves overlap too much.)
    speed_ramp: bool = True
    rest_speedup: float = 8.0          # playback speed during a rest
    min_rest_seconds: float = 45.0     # a flat stretch this long counts as a hang
    rest_band_bh: float = 0.2          # height stays within this (body-heights) = flat
    rest_smooth_seconds: float = 5.0   # smooth elevation before flatness test
    rest_inset_seconds: float = 3.0    # start the speed-up this much later and end
                                       # it earlier, so moves around a rest stay 1x
    ramp_marker: bool = True           # green border on sped sections (testing aid)

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
