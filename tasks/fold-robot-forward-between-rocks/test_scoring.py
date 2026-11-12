"""Offline tests of the fold-forward scorer on synthetic states (no simulator):  python -m pytest test_scoring.py"""

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import task as T  # noqa: E402
import trailkit as tk  # noqa: E402


def rx(deg):
    t = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])


def dummy(raise_deg=0.0, dx=0.0, joints=None):
    """The dummy with its trunk raised ``raise_deg`` about the hip axis (x) from lying on its back."""
    R = rx(raise_deg) @ tk.rot_from_xyzw(T.START_ROT)
    pos = np.array(T.DUMMY.pos) + (dx, 0.0, 0.0)
    return tk.state_at(T.DUMMY, pos, tk.quat_from_rot(R), joints=joints)


def rocks(dx=0.0):
    return [tk.state_at(r, np.array(r.pos) + (dx if i == 0 else 0.0, 0, 0), r.spawn_xyzw) for i, r in enumerate(T.ROCKS)]


def rec(max_raise=130.0, rest=True, released=True, **flags):
    r = tk.Recorder()
    for _ in range(T.REST_STEPS):
        r.window("rest", rest, T.REST_STEPS)
        r.window("released", released, T.REST_STEPS)
    r.extreme("max_raise", max_raise)
    r.values["pelvis0"] = np.asarray(T.DUMMY.pos[:2], float)
    for k in ("pelvis_lifted", "rock_moved"):
        r.latch(k, flags.get(k, False))
    return r


def test_angles():
    assert abs(T.hatch_angle(dummy(130.0)) - 50.0) < 0.5
    assert abs(T.raise_angle(dummy(0.0))) < 0.5 and abs(T.raise_angle(dummy(-0.5))) < 1.0
    assert abs(T.raise_angle(dummy(90.0)) - 90.0) < 0.5
    assert abs(T.raise_angle(dummy(129.0)) - 51.0) < 0.5          # folded: the spine's elevation falls again


def test_sideways_swing_is_not_a_raise():
    st = tk.state_at(T.DUMMY, T.DUMMY.pos, tk.quat_from_rot(
        np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float).T @ np.eye(3) @ tk.rot_from_xyzw(T.START_ROT)))
    s = T.D.spine(st)
    assert abs(T.raise_angle(st) - math.degrees(math.asin(s[2]))) < 1e-6


def test_folded_scores_full():
    s, why = T.evaluate(dummy(129.0), rocks(), rec())
    assert s == 1.0, why


def test_sitting_or_fallen_back_is_quarter_after_a_raise():
    assert T.evaluate(dummy(90.0), rocks(), rec(max_raise=95.0))[0] == 0.25        # hatch 90 from up
    assert T.evaluate(dummy(0.0), rocks(), rec(max_raise=75.0))[0] == 0.25         # fell back


def test_never_raised_is_zero():
    assert T.evaluate(dummy(0.0), rocks(), rec(max_raise=30.0))[0] == 0.0


def test_rock_turned_in_place_counts_as_moved():
    r = T.ROCKS[0]
    st0 = tk.state_at(r, r.pos, r.spawn_xyzw)
    st1 = tk.state_at(r, r.pos, tk.quat_xyzw_from_axis_angle((0, 0, 1), math.radians(10.0)))
    assert float(np.linalg.norm(T.rock_corners(st1) - T.rock_corners(st0), axis=1).max()) > T.ROCK_FOUL


def test_pelvis_moved_or_out_of_the_rocks_is_zero():
    assert T.evaluate(dummy(129.0, dx=0.0), rocks(), rec())[0] == 1.0
    r = rec()
    r.values["pelvis0"] = np.asarray(T.DUMMY.pos[:2], float) + (0.0, 0.05)        # it started 5 cm away
    assert T.evaluate(dummy(129.0), rocks(), r)[0] == 0.0
    assert T.evaluate(dummy(129.0, dx=0.09), rocks(), rec())[0] == 0.0           # out past a rock's inner face


@pytest.mark.parametrize("kw", [dict(pelvis_lifted=True), dict(rock_moved=True), dict(rest=False), dict(released=False)])
def test_fouls_and_end_conditions_zero(kw):
    assert T.evaluate(dummy(129.0), rocks(), rec(**kw))[0] == 0.0


def test_limb_on_hatch_is_detected_and_costs_the_full_score():
    # the right arm raised over the head and swung back and across, so the forearm lies on the back (found by search)
    joints = {ln.name: ln.q0 for ln in T.DUMMY.links} | {"r_sh_pitch": math.radians(-170.0),
                                                          "r_sh_roll": math.radians(100.0), "r_elbow": math.radians(-60.0)}
    st = dummy(129.0, joints=joints)
    assert T.limbs_on_hatch(st) == ["r_elbow"]
    assert T.evaluate(st, rocks(), rec())[0] == 0.25


def test_folded_pose_has_nothing_on_the_hatch():
    assert T.limbs_on_hatch(dummy(129.0)) == []


def test_scoring_text_matches_constants():
    doc = T.score.__doc__
    doc = " ".join(doc.split())
    for text in ("3 cm above the pad", "corner of a rock", "1 cm", "70 degrees", "4 cm", "60 degrees", "2 s", "5 cm",
                 "3 cm outside", "above level"):
        assert text in doc
