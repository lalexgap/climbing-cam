"""Climber selection and burn-interval detection.

The functions here are deliberately pure (numpy in, intervals out) so the
segmentation logic can be unit-tested without running YOLO. See
tests/test_classify.py.

Coordinate note: image y grows downward, so the ground band sits at a *high* y
and elevation = (ground_y - box_bottom) / body_height grows as you climb.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .detect import DetectionResult


@dataclass
class Burn:
    start: float   # clip start (seconds, padded)
    end: float     # clip end (seconds, trimmed after apex)
    apex_t: float  # timestamp of highest point


@dataclass
class GroundEstimate:
    ground_y: float
    body_h: float
    reliable: bool  # False => no usable ground band (framed-up regime)


@dataclass
class ClimberPick:
    track_id: int | None
    ascending_ids: list[int]
    peaks: dict[int, float]  # track_id -> peak elevation (body-heights)
    ambiguous: bool


# --- Ground band + body-height calibration ----------------------------------

def estimate_ground(result: DetectionResult, cfg: Config) -> GroundEstimate:
    bottoms, heights = [], []
    for fd in result.frames:
        for d in fd.detections:
            bottoms.append(d.y2)
            heights.append(d.height)
    if not bottoms:
        return GroundEstimate(float(result.frame_height), result.frame_height * 0.3, False)

    bottoms = np.asarray(bottoms)
    heights = np.asarray(heights)
    body_h0 = float(np.median(heights))
    ground_y = float(np.percentile(bottoms, cfg.ground_percentile))

    # Refine body height using only detections sitting near the ground line.
    near = bottoms >= (ground_y - cfg.near_ground_bh * body_h0)
    body_h = float(np.median(heights[near])) if near.any() else body_h0
    body_h = max(body_h, 1.0)

    # A ground band is "reliable" only if a decent share of detections actually
    # sit near it (i.e. we routinely see the base). Otherwise we're framed up.
    near_frac = float(np.mean(bottoms >= (ground_y - cfg.near_ground_bh * body_h)))
    reliable = near_frac >= 0.15
    return GroundEstimate(ground_y, body_h, reliable)


def track_elevation(result: DetectionResult, track_id: int, est: GroundEstimate) -> np.ndarray:
    """Per-frame elevation (body-heights) for one track; NaN where absent."""
    elev = np.full(len(result.frames), np.nan)
    for i, fd in enumerate(result.frames):
        for d in fd.detections:
            if d.track_id == track_id:
                elev[i] = (est.ground_y - d.y2) / est.body_h
                break
    return elev


def track_presence(result: DetectionResult, track_id: int) -> np.ndarray:
    present = np.zeros(len(result.frames), dtype=bool)
    for i, fd in enumerate(result.frames):
        present[i] = any(d.track_id == track_id for d in fd.detections)
    return present


def frame_max_elevation(result: DetectionResult, est: GroundEstimate) -> np.ndarray:
    """Per-frame elevation of the *highest* person in frame (track-agnostic).

    This is the robust Phase-1 signal: a climber re-ID'd into many short tracks
    as they shrink high on the wall still registers, because we never depend on
    a single track surviving the whole ascent. The belayer and people at the
    base sit at elevation ~0, so the signal rises only when someone is on the
    wall. NaN where no person is detected."""
    elev = np.full(len(result.frames), np.nan)
    for i, fd in enumerate(result.frames):
        ys = [(est.ground_y - d.y2) / est.body_h for d in fd.detections]
        if ys:
            elev[i] = max(ys)
    return elev


def frame_presence(result: DetectionResult) -> np.ndarray:
    """True wherever any person is detected (framed-up fallback signal)."""
    return np.array([len(fd.detections) > 0 for fd in result.frames], dtype=bool)


def frame_ground_count(result: DetectionResult, est: GroundEstimate, cfg: Config) -> np.ndarray:
    """Per-frame count of people standing near the ground (elev < exit_bh).

    Used to tell a real attempt boundary (you come down -> belayer + you = 2+
    people at the base) from you merely hanging undetected high on the wall
    (just the belayer = 1)."""
    counts = np.zeros(len(result.frames), dtype=int)
    for i, fd in enumerate(result.frames):
        counts[i] = sum(
            1 for d in fd.detections
            if (est.ground_y - d.y2) / est.body_h < cfg.exit_bh
        )
    return counts


def frame_top_exit(result: DetectionResult, cfg: Config) -> np.ndarray:
    """True where the highest person's box touches the top edge of the frame.

    Distinguishes "climbed out the top of frame" (still on the wall, just no
    longer visible) from "came down": both leave only the belayer in view, so the
    elevation signal can't tell them apart, but a climber exiting the top has a
    box jammed against y=0. detect_burns uses this so a long out-of-frame stretch
    on one continuous ascent isn't mistaken for a between-attempt rest."""
    margin = cfg.top_exit_frac * result.frame_height
    out = np.zeros(len(result.frames), dtype=bool)
    for i, fd in enumerate(result.frames):
        if fd.detections:
            highest = min(fd.detections, key=lambda d: d.y2)  # smallest y2 = highest
            out[i] = highest.y1 <= margin
    return out


# --- Climbing screening (is there climbing in this video?) ------------------

def climbing_evidence(times: np.ndarray, elev: np.ndarray, cfg: Config) -> dict:
    """Decide whether a clip contains climbing, from the per-frame max elevation.

    Climbing is present if someone is clearly off the ground (elev >= ascend_bh)
    for a sustained contiguous stretch (>= min_climb_seconds). Returns the
    verdict plus evidence (peak height, total + longest elevated time)."""
    times = np.asarray(times, dtype=float)
    elev = np.asarray(elev, dtype=float)
    if len(times) < 2:
        return {"present": False, "peak_bh": 0.0,
                "climbing_seconds": 0.0, "longest_stretch_seconds": 0.0}

    dt = float(np.median(np.diff(times)))
    finite = np.isfinite(elev)
    peak = float(np.nanmax(elev)) if finite.any() else 0.0
    high = finite & (elev >= cfg.ascend_bh)
    longest = max((times[en] - times[s] + dt for s, en in _runs(high)), default=0.0)
    return {
        "present": bool(longest >= cfg.min_climb_seconds),
        "peak_bh": round(peak, 2),
        "climbing_seconds": round(float(high.sum()) * dt, 1),
        "longest_stretch_seconds": round(float(longest), 1),
    }


# --- Climber selection -------------------------------------------------------

def pick_climber(result: DetectionResult, est: GroundEstimate, cfg: Config) -> ClimberPick:
    """Choose the climbing track: the one that rises highest out of the ground
    band. Flags ambiguity when a second track ascends comparably."""
    track_ids = {d.track_id for fd in result.frames for d in fd.detections}
    peaks: dict[int, float] = {}
    for tid in track_ids:
        elev = track_elevation(result, tid, est)
        if np.isfinite(elev).any():
            peaks[tid] = float(np.nanmax(elev))

    if not peaks:
        return ClimberPick(None, [], {}, False)

    ascending = sorted(
        [tid for tid, p in peaks.items() if p >= cfg.ascend_bh],
        key=lambda t: peaks[t],
        reverse=True,
    )
    if not ascending:
        # No track clears the ascent bar — likely framed-up; fall back to the
        # most-present track and let presence-based segmentation handle it.
        best = max(peaks, key=lambda t: peaks[t])
        return ClimberPick(best, [], peaks, False)

    best = ascending[0]
    ambiguous = (
        len(ascending) >= 2
        and peaks[ascending[1]] >= cfg.ambiguous_ratio * peaks[best]
    )
    return ClimberPick(best, ascending, peaks, ambiguous)


# --- Burn segmentation (pure, tested) ----------------------------------------

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index runs where mask is True."""
    runs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def _despike(mask: np.ndarray, times: np.ndarray, min_seconds: float) -> np.ndarray:
    """Drop True-runs shorter than `min_seconds`. An isolated enter_bh frame is a
    detection spike (e.g. a person-shaped shadow/crack in the rock), not a real
    on-wall moment — and since a burn ends at its last enter_bh frame, one such
    spike can otherwise stretch a burn boundary by minutes. Matters most for the
    higher-recall pose detector, which trades a little precision for that recall."""
    if min_seconds <= 0:
        return mask
    out = mask.copy()
    for s, e in _runs(mask):
        if times[e] - times[s] < min_seconds:
            out[s:e + 1] = False
    return out


def _has_ground_rest(ground_count: np.ndarray, times: np.ndarray,
                     lo: int, hi: int, cfg: Config) -> bool:
    """True if the gap [lo, hi] contains a sustained stretch (>= ground_rest_seconds)
    with >= min_ground_people at the base — i.e. you actually came down to rest,
    a genuine attempt boundary (vs. just hanging undetected high)."""
    if hi < lo:
        return False
    best, start = 0.0, None
    for i in range(lo, hi + 1):
        if ground_count[i] >= cfg.min_ground_people:
            start = i if start is None else start
            best = max(best, times[i] - times[start])
        else:
            start = None
    return best >= cfg.ground_rest_seconds


def _build_burns(merged: list[list[int]], times: np.ndarray, elev: np.ndarray | None,
                 cfg: Config, duration: float) -> list[Burn]:
    burns: list[Burn] = []
    for s, e in merged:
        if times[e] - times[s] < cfg.min_burn_seconds:
            continue
        if elev is not None:
            seg = elev[s:e + 1]
            highs = np.where(np.isfinite(seg) & (seg >= cfg.enter_bh))[0]
            if len(highs) == 0:
                continue  # never clearly on the wall -> not a real climb
            apex_t = float(times[s + int(np.nanargmax(seg))])
            # End just after the *last* clearly-on-wall moment -> trims the
            # lower-off, and works for long attempts where the high point recurs
            # (we don't stop at the first apex).
            end_ref = float(times[s + int(highs[-1])])
        else:  # presence (framed-up) mode: no elevation to reason about
            apex_t = end_ref = float(times[e])
        # Start uses the *leave_bh* extent (set by the caller's mask), so the
        # first low moves are included.
        start = max(0.0, float(times[s]) - cfg.pad_lead_seconds)
        end = min(duration, end_ref + cfg.apex_tail_seconds)
        burns.append(Burn(start, max(end, start + cfg.min_burn_seconds), apex_t))
    return burns


def _extend_start(active: np.ndarray, times: np.ndarray, s: int, link: float,
                  floor: int = 0, max_lead: float = 0.0) -> int:
    """Walk the burn start back through leave_bh activity immediately preceding
    it, bridging lulls up to `link` seconds, to catch the first low moves.

    Stops at `floor` (don't cross into a previous burn) and never reaches further
    than `max_lead` seconds before `s` (the lead-in is the first low moves, not an
    arbitrarily long on-wall stretch — without this, near-continuous activity from
    back-to-back goes or other parties walks the start back across the whole video)."""
    new_s, last_active_t = s, times[s]
    i = s - 1
    while i >= floor:
        if max_lead > 0 and times[s] - times[i] > max_lead:
            break
        if active[i]:
            new_s, last_active_t = i, times[i]
        elif last_active_t - times[i] > link:
            break
        i -= 1
    return new_s


def detect_burns(times: np.ndarray, elev: np.ndarray, ground_count: np.ndarray,
                 cfg: Config, duration: float | None = None,
                 at_top: np.ndarray | None = None) -> list[Burn]:
    """Elevation-based burn detection (primary, base-in-frame path).

    Bursts are confirmed where elevation clearly reaches the wall (enter_bh);
    short gaps (detection dropouts, on-wall hangs) merge through, and burns split
    where on-wall activity stops for longer than merge_gap_seconds (you came down
    and rested) or where 2+ people sit at the base for a sustained stretch. Each
    burn's start is then extended back over the first low moves (leave_bh), and it
    ends at the last enter_bh moment so the lower-off is trimmed.

    `at_top` (optional, from frame_top_exit) suppresses the long-gap split when the
    climber exited the top of frame and never came down: that's one continuous
    ascent with the climber out of view, not two attempts. A genuine come-down
    (2+ people back at the base) still splits regardless."""
    times = np.asarray(times, dtype=float)
    elev = np.asarray(elev, dtype=float)
    ground_count = np.asarray(ground_count)
    if duration is None:
        duration = float(times[-1]) if len(times) else 0.0

    enter = np.isfinite(elev) & (elev >= cfg.enter_bh)
    high = _despike(enter, times, cfg.enter_persist_seconds)
    runs = _runs(high)
    if not runs:
        return []
    # Where *sustained* climbing happens (a longer persistence than `high`): used
    # only to anchor a burn's start, so leading non-sustained enter_bh blips —
    # setting up close to the camera (a too-big box reads as elevated), staging at
    # the base — don't drag the clip start back before you actually got on route.
    anchor = _despike(enter, times, cfg.start_anchor_seconds)

    merged = [list(runs[0])]
    for s, e in runs[1:]:
        e1 = merged[-1][1]
        gap = times[s] - times[e1]
        came_down = _has_ground_rest(ground_count, times, e1 + 1, s - 1, cfg)
        # Did the previous burst end with the climber exiting the top of frame?
        exited_top = at_top is not None and bool(at_top[max(0, e1 - 2):e1 + 1].any())
        if came_down or (gap >= cfg.merge_gap_seconds and not exited_top):
            merged.append([s, e])      # real boundary -> new burn
        else:
            merged[-1][1] = e          # dropout / hang / out-the-top -> merge

    active = np.isfinite(elev) & (elev >= cfg.leave_bh)
    floor = 0
    for burst in merged:
        s, e = burst
        # Anchor the start at the first *sustained* climb within the burst, so
        # leading short blips are skipped; fall back to the burst start if none.
        sustained = np.where(anchor[s:e + 1])[0]
        anchor_idx = s + int(sustained[0]) if len(sustained) else s
        burst[0] = _extend_start(active, times, anchor_idx, cfg.start_link_seconds,
                                 floor=floor, max_lead=cfg.max_lead_seconds)
        floor = e + 1          # next burn's start can't reach into this one
    return _build_burns(merged, times, elev, cfg, duration)


def _movavg(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    pad = w // 2
    return np.convolve(np.pad(x, pad, mode="edge"), np.ones(w) / w, mode="same")[pad:pad + len(x)]


def rest_intervals(times: np.ndarray, elev: np.ndarray, cfg: Config) -> list[tuple[float, float]]:
    """Find rest (hang) stretches within a burn for Phase 2 speed-ramping.

    A rest is a sustained stretch (>= min_rest_seconds) where your height stays
    flat — elevation stays within rest_band_bh body-heights (no net climbing
    progress). Works on the dropout-interpolated elevation, so a hang where you
    go undetected reads flat too, while climbing through a dropout reads as
    rising (not a rest). Returns absolute (start, end) times."""
    times = np.asarray(times, dtype=float)
    elev = np.asarray(elev, dtype=float)
    n = len(times)
    if n < 2:
        return []

    on_wall = np.isfinite(elev) & (elev >= cfg.leave_bh)
    if on_wall.sum() < 2:  # whole span undetected -> one long hang if long enough
        return [(float(times[0]), float(times[-1]))] if times[-1] - times[0] >= cfg.min_rest_seconds else []
    idx = np.arange(n)
    e = np.interp(idx, idx[on_wall], elev[on_wall])
    fps = 1.0 / float(np.median(np.diff(times)))
    e = _movavg(e, max(1, int(round(cfg.rest_smooth_seconds * fps))))

    # Greedy maximal runs where height stays within rest_band_bh.
    rests, i = [], 0
    while i < n:
        j, lo, hi = i, e[i], e[i]
        while j + 1 < n and max(hi, e[j + 1]) - min(lo, e[j + 1]) < cfg.rest_band_bh:
            j += 1
            lo, hi = min(lo, e[j]), max(hi, e[j])
        if times[j] - times[i] >= cfg.min_rest_seconds:
            rests.append((float(times[i]), float(times[j])))
            i = j + 1
        else:
            i += 1

    # Inset each rest so the speed-up starts a little later and ends earlier,
    # keeping the moves on either side of the hang at normal speed.
    ins = cfg.rest_inset_seconds
    return [(a + ins, b - ins) for a, b in rests if (b - ins) - (a + ins) > 0.5]


def detect_burns_presence(times: np.ndarray, present: np.ndarray, cfg: Config,
                          duration: float | None = None) -> list[Burn]:
    """Framed-up fallback: any time a person is in frame counts as on-wall.
    Merges runs separated by gaps shorter than the lost-timeout."""
    times = np.asarray(times, dtype=float)
    present = np.asarray(present, dtype=bool)
    if duration is None:
        duration = float(times[-1]) if len(times) else 0.0

    runs = _runs(present)
    if not runs:
        return []
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        if times[s] - times[merged[-1][1]] < cfg.lost_timeout_seconds:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return _build_burns(merged, times, None, cfg, duration)
