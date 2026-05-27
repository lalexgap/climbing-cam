"""Unit tests for the burn-segmentation logic.

These exercise the pure interval detection on synthetic elevation signals, so
they run without YOLO/ffmpeg. Coordinate reminder: elevation is in body-heights,
0 = on the ground, higher = further up the wall.
"""

import numpy as np
import pytest

from app.classify import (
    climbing_evidence,
    detect_burns,
    detect_burns_presence,
    estimate_ground,
    pick_climber,
    rest_intervals,
)
from app.config import Config
from app.detect import Detection, DetectionResult, FrameDetections

FPS = 3.0
CFG = Config()


def signal(segments):
    """Build an elevation array from (duration_s, start_elev, end_elev) segments.
    Use np.nan for both elevs to mark an 'absent' (undetected) stretch."""
    parts = []
    for dur, a, b in segments:
        n = round(dur * FPS)
        parts.append(np.linspace(a, b, n))
    elev = np.concatenate(parts)
    times = np.arange(len(elev)) / FPS
    return times, elev


def ground(elev, where=None):
    """Default ground_count: 1 person (the belayer) everywhere. `where` is an
    optional boolean mask of frames with a 2nd person at the base (you, resting)."""
    gc = np.ones(len(elev), dtype=int)
    if where is not None:
        gc[where] = 2
    return gc


# --- detect_burns (elevation path) ------------------------------------------

def test_single_clean_burn():
    # 10s ground, 30s climb to apex 3.0, 10s lower-off, 20s ground.
    times, elev = signal([(10, 0, 0), (30, 0, 3.0), (10, 3.0, 0), (20, 0, 0)])
    burns = detect_burns(times, elev, ground(elev), CFG)
    assert len(burns) == 1
    b = burns[0]
    # Leaves ground (elev>=leave_bh=0.5) ~15s in; clip starts a 3s pad earlier.
    assert b.start == pytest.approx(12, abs=1.0)
    # Clip ends ~3s after the last clearly-on-wall moment (elev drops below
    # enter_bh=1.0 on the lower-off at ~46.7s).
    assert b.end == pytest.approx(49.7, abs=1.0)


def test_short_attempt_is_dropped():
    # On the wall only ~6s -> below the 15s minimum.
    times, elev = signal([(10, 0, 0), (3, 0, 2.0), (3, 2.0, 0), (10, 0, 0)])
    assert detect_burns(times, elev, ground(elev), CFG) == []


def test_two_burns_split_only_when_you_rest_on_the_ground():
    seg = [(5, 0, 0), (30, 0, 2.5), (30, 0, 0), (30, 0, 2.5), (5, 2.5, 0)]
    times, elev = signal(seg)
    # 2 people at the base for 30s (> ground_rest_seconds) mid-video => boundary.
    rest = (times >= 35) & (times < 65)
    burns = detect_burns(times, elev, ground(elev, rest), CFG)
    assert len(burns) == 2
    assert burns[0].end < burns[1].start


def test_long_dropout_with_only_belayer_does_NOT_split():
    # The real-footage case: you hang high but go undetected for a long stretch
    # while only the belayer (1 person) is visible at the base -> one burn.
    seg = [(5, 0, 0), (30, 0, 2.5), (60, np.nan, np.nan), (30, 2.6, 2.6), (5, 2.6, 0)]
    times, elev = signal(seg)
    burns = detect_burns(times, elev, ground(elev), CFG)  # ground_count == 1 throughout
    assert len(burns) == 1


def test_long_activity_gap_splits_into_two_attempts():
    # On-wall, then a long stretch off the wall (you came down and rested),
    # then on-wall again. The gap exceeds merge_gap_seconds => two attempts,
    # even with only the belayer (1 person) ever visible.
    seg = [(5, 0, 0), (30, 0, 2.5), (360, 0, 0), (30, 0, 2.5), (5, 2.5, 0)]
    times, elev = signal(seg)
    burns = detect_burns(times, elev, ground(elev), CFG)
    assert len(burns) == 2
    assert burns[0].end < burns[1].start


def test_brief_ground_touch_does_not_split():
    seg = [(8, 0, 0), (12, 2.0, 2.0), (3, 0, 0), (15, 2.5, 2.5), (10, 0, 0)]
    times, elev = signal(seg)
    burns = detect_burns(times, elev, ground(elev), CFG)
    assert len(burns) == 1


def test_high_dropout_does_not_split():
    seg = [(8, 0, 0), (15, 2.0, 3.0), (5, np.nan, np.nan), (15, 3.0, 2.0), (10, 0, 0)]
    times, elev = signal(seg)
    burns = detect_burns(times, elev, ground(elev), CFG)
    assert len(burns) == 1


def test_climb_out_the_top_of_frame_stays_one_attempt():
    # Climb up and out the top of frame (at_top while high), gone for a long gap
    # (only the belayer visible at the base), then lowered back in. One attempt.
    seg = [(5, 0, 0), (30, 0, 3.0), (390, 0, 0), (20, 3.0, 0)]
    times, elev = signal(seg)
    at_top = elev >= 2.5  # box jammed against the top edge while high
    burns = detect_burns(times, elev, ground(elev), CFG, at_top=at_top)
    assert len(burns) == 1


def test_top_exit_still_splits_if_you_come_back_down_to_rest():
    # Same long gap, but 2 people sit at the base mid-gap (you came down to rest)
    # -> a real boundary, split despite the top exit.
    seg = [(5, 0, 0), (30, 0, 3.0), (390, 0, 0), (20, 0, 3.0), (5, 3.0, 0)]
    times, elev = signal(seg)
    at_top = elev >= 2.5
    rest = (times >= 60) & (times < 400)  # belayer + you at the base for >25s
    burns = detect_burns(times, elev, ground(elev, rest), CFG, at_top=at_top)
    assert len(burns) == 2


# --- detect_burns_presence (framed-up fallback) -----------------------------

def test_presence_fallback_splits_on_long_absence():
    present = np.concatenate([
        np.ones(round(30 * FPS), bool),
        np.zeros(round(35 * FPS), bool),   # > 30s lost timeout -> split
        np.ones(round(25 * FPS), bool),
    ])
    times = np.arange(len(present)) / FPS
    burns = detect_burns_presence(times, present, CFG)
    assert len(burns) == 2


# --- rest_intervals (Phase 2 speed-ramping) ---------------------------------
# A rest = a hang: height stays flat (within rest_band_bh) for >= min_rest_seconds.
# CFG.min_rest_seconds = 45s, rest_band_bh = 0.2.

def test_flat_hang_is_a_rest():
    # climb, hang at constant height for 70s, climb on.
    times, elev = signal([(20, 0.6, 2.6), (70, 2.6, 2.6), (20, 2.6, 4.0)])
    rests = rest_intervals(times, elev, CFG)
    assert len(rests) == 1
    dur = rests[0][1] - rests[0][0]
    assert 45 <= dur < 70   # detected ~70s flat, inset on both ends


def test_undetected_hang_at_same_height_is_a_rest():
    # 50s undetected, but you reappear at the same height -> flat -> a hang.
    times, elev = signal([(20, 0.6, 2.6), (50, np.nan, np.nan), (20, 2.6, 4.0)])
    assert len(rest_intervals(times, elev, CFG)) == 1


def test_climbing_through_a_dropout_is_not_a_rest():
    # 50s undetected, but you reappear much higher -> you were climbing.
    times, elev = signal([(20, 0.6, 2.6), (50, np.nan, np.nan), (20, 4.6, 6.0)])
    assert rest_intervals(times, elev, CFG) == []


def test_short_flat_is_not_a_rest():
    # only 20s flat (< min_rest_seconds).
    times, elev = signal([(20, 0.6, 2.6), (20, 2.6, 2.6), (20, 2.6, 4.0)])
    assert rest_intervals(times, elev, CFG) == []


def test_continuous_climb_has_no_rest():
    times, elev = signal([(60, 0.6, 5.0)])
    assert rest_intervals(times, elev, CFG) == []


# --- climbing_evidence (is there climbing in this video?) -------------------

def test_climbing_is_detected():
    times, elev = signal([(10, 0, 0), (10, 0, 3), (10, 3, 3), (10, 3, 0)])
    ev = climbing_evidence(times, elev, CFG)
    assert ev["present"] and ev["peak_bh"] >= 2.5


def test_no_climbing_on_the_ground():
    times, elev = signal([(30, 0.2, 0.2)])
    assert climbing_evidence(times, elev, CFG)["present"] is False


def test_brief_elevation_is_not_climbing():
    # a ~2s spike above ascend_bh (< min_climb_seconds) -> not climbing.
    times, elev = signal([(20, 0, 0), (2, 0, 2), (2, 2, 0), (20, 0, 0)])
    assert climbing_evidence(times, elev, CFG)["present"] is False


# --- estimate_ground + pick_climber -----------------------------------------

def _frames_from_tracks(tracks, frame_h=720):
    """tracks: dict[track_id] -> list of (frame_idx, y_bottom, height)."""
    n = max(idx for samples in tracks.values() for idx, *_ in samples) + 1
    frames = [FrameDetections(t=i / FPS) for i in range(n)]
    for tid, samples in tracks.items():
        for idx, y2, h in samples:
            frames[idx].detections.append(
                Detection(tid, x1=100, y1=y2 - h, x2=150, y2=y2, conf=0.9))
    return DetectionResult(frames=frames, frame_width=1280, frame_height=frame_h)


def test_pick_climber_prefers_the_one_who_rises():
    # Belayer stays on the ground (y2~700); climber rises (y2 700 -> 200).
    belayer = [(i, 700, 200) for i in range(60)]
    climber = [(i, 700 - (i * 8), 180) for i in range(60)]  # bottom moves up
    det = _frames_from_tracks({1: belayer, 2: climber})
    est = estimate_ground(det, CFG)
    pick = pick_climber(det, est, CFG)
    assert pick.track_id == 2
    assert not pick.ambiguous


def test_pick_climber_flags_ambiguity_when_two_rise():
    a = [(i, 700 - i * 8, 180) for i in range(60)]
    b = [(i, 700 - i * 8, 180) for i in range(60)]
    det = _frames_from_tracks({1: a, 2: b})
    est = estimate_ground(det, CFG)
    pick = pick_climber(det, est, CFG)
    assert pick.ambiguous
    assert set(pick.ascending_ids) == {1, 2}
