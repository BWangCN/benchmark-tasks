"""Offline tests of the lay-it-on-its-side scorer on synthetic states (no Isaac Sim).

    python -m pytest test_scoring.py -q    # from this folder, in the Isaac Lab-Arena venv
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

for _m in ("dummy", "dummy_spec", "task"):      # other task folders carry their own copies
    sys.modules.pop(_m, None)

TASK = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TASK))

import dummy as D  # noqa: E402
import task as T  # noqa: E402
import trailkit as tk  # noqa: E402


PROPPED = {k: math.radians(v) for k, v in {        # the demo's settled pose in the simulator (degrees)
    "r_sh_pitch": -42, "r_sh_roll": 4, "r_elbow": 42, "r_hip_pitch": -37, "r_hip_roll": 7, "r_knee": 33,
    "l_sh_roll": 0}.items()}


def lying(rot, pose=None, tilt_deg=0.0, shift=(0.0, 0.0)):
    """The dummy resting on the ground (on its lowest point, limbs included) in orientation ``rot`` (turned ``tilt_deg``
    about world y), head toward -y."""
    pose = dict(T.START_POSE) | (pose or {})
    R = tk.rot_from_xyzw(rot)
    if tilt_deg:
        t = math.radians(tilt_deg)
        R = np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]]) @ R
    q = tk.quat_from_rot(R)
    x, y = T.DUMMY_X + shift[0], shift[1]
    st = tk.state_at(T.DUMMY, (x, y, 0.0), q, joints=pose)
    z = T.GROUND - D.lowest(T.DUMMY, st)                 # resting on its lowest point, limbs included
    return tk.state_at(T.DUMMY, (x, y, z), q, joints=pose)


SIDE = T.head_minus_y(D.on_left_side())        # left side down, chest toward the robot


def rec(rest=True, released=True, steps=None, lifted=False, stood=False, was_on_side=True):
    r = tk.Recorder()
    for _ in range(T.REST_STEPS if steps is None else steps):
        r.window("rest", rest, T.REST_STEPS)
        r.window("released", released, T.REST_STEPS)
    r.latch("lifted", lifted)
    r.latch("stood", stood)
    r.latch("was_on_side", was_on_side)
    return r


def test_start_on_its_back_head_to_minus_y_left_side_toward_robot():
    st = lying(T.START_ROT)
    assert math.isclose(T.side_angle(st), 90.0, abs_tol=1e-6)
    assert np.allclose(D.chest_normal(st), (0, 0, 1))
    assert np.allclose(D.shoulder_line(st), (-1, 0, 0))   # right shoulder to left points at the robot
    assert np.allclose(D.spine(st), (0, -1, 0))
    assert T.rock_gap(st) > T.ROCK_CLEAR


def test_idle_scores_zero():
    assert T.evaluate(lying(T.START_ROT), rec(was_on_side=False))[0] == 0.0


def test_full_score_on_left_side_released_at_rest():
    assert T.trunk_lift(lying(SIDE, PROPPED)) < T.LIFT_FOUL
    assert T.evaluate(lying(SIDE, PROPPED), rec()) == (1.0, "on its left side (0 degrees), released and at rest on its own")


def test_side_limit_25_degrees_either_way():
    for tilt, want in ((24.0, 1.0), (-24.0, 1.0), (26.0, 0.25), (-26.0, 0.25)):
        assert T.evaluate(lying(SIDE, PROPPED, tilt_deg=tilt), rec())[0] == want, tilt


def test_right_side_is_not_left_side():
    st = lying(T.head_minus_y(D.on_right_side()))
    assert T.side_angle(st) > 170.0
    assert T.evaluate(st, rec(was_on_side=False))[0] == 0.0


def test_rolled_back_after_being_on_its_side_is_quarter():
    s, why = T.evaluate(lying(T.START_ROT), rec())
    assert s == 0.25 and why.startswith("was on its left side for 1 s, but did not stay")


def test_still_held_is_quarter():
    s, why = T.evaluate(lying(SIDE, PROPPED), rec(released=False))
    assert s == 0.25 and "still held" in why


def test_not_at_rest_is_quarter_and_short_rest_window_is_not_enough():
    assert T.evaluate(lying(SIDE, PROPPED), rec(rest=False))[0] == 0.25
    assert T.evaluate(lying(SIDE, PROPPED), rec(steps=T.REST_STEPS - 1))[0] == 0.25    # less than 2 s


def test_release_must_hold_for_the_whole_last_two_seconds():
    r = rec()
    r.window("released", False, T.REST_STEPS)
    r.window("released", True, T.REST_STEPS)
    assert T.evaluate(lying(SIDE, PROPPED), r)[0] == 0.25


def test_on_side_without_the_1s_milestone_and_not_meeting_the_end_is_zero():
    assert T.evaluate(lying(SIDE, PROPPED), rec(released=False, was_on_side=False))[0] == 0.0


def test_fouls_latch_whatever_the_end():
    assert T.evaluate(lying(SIDE, PROPPED), rec(lifted=True))[0] == 0.0
    assert T.evaluate(lying(SIDE, PROPPED), rec(stood=True))[0] == 0.0
    r = rec(lifted=True)
    r.latch("lifted", False)
    assert r.flag("lifted")


def test_leaning_on_a_rock_is_not_on_its_own():
    # moved across the pad until its back touches the rock at (43, -21)
    st = lying(SIDE, PROPPED)
    for dx in np.arange(0.0, 0.25, 0.002):
        s2 = lying(SIDE, PROPPED, shift=(dx, -0.10))
        if T.rock_gap(s2) < T.ROCK_CLEAR:
            break
    assert T.rock_gap(s2) < T.ROCK_CLEAR
    s, why = T.evaluate(s2, rec())
    assert s == 0.25 and "rock" in why
    assert T.rock_gap(st) >= T.ROCK_CLEAR


def test_off_the_pad_is_not_full_score():
    st = lying(SIDE, PROPPED)
    assert T.on_pad(st)
    # slid toward the robot until part of it hangs past the pad's near edge (x 3 cm)
    for dx in np.arange(0.0, -0.30, -0.005):
        s2 = lying(SIDE, PROPPED, shift=(dx, 0.0))
        if not T.on_pad(s2):
            break
    assert not T.on_pad(s2)
    s, why = T.evaluate(s2, rec())
    assert s == 0.25 and "pad" in why


def test_resting_on_the_table_below_the_pad_is_not_on_the_pad():
    st = lying(SIDE, PROPPED, shift=(-0.25, 0.0))
    st = tk.state_at(T.DUMMY, st.pos - np.array([0, 0, T.PAD_H]), st.quat_xyzw, joints=st.joints)
    assert not T.on_pad(st)
    assert T.evaluate(st, rec())[0] == 0.25


def test_trunk_lift_and_spine_tilt():
    st = lying(T.START_ROT)
    up = tk.state_at(T.DUMMY, st.pos + np.array([0, 0, 0.05]), st.quat_xyzw, joints=st.joints)
    assert math.isclose(T.trunk_lift(up), 0.05, abs_tol=1e-6) and T.trunk_lift(up) > T.LIFT_FOUL
    for tilt in (0.0, 45.0, 90.0):
        assert T.spine_tilt(lying(T.START_ROT, tilt_deg=tilt)) < 1e-6        # a roll keeps the spine level


def test_recorder_milestone_needs_20_steps_in_a_row():
    r = tk.Recorder()
    for k in range(T.SIDE_STEPS - 1):
        w = r.window("side", True, T.SIDE_STEPS)
        r.latch("was_on_side", len(w) == T.SIDE_STEPS and all(w))
    assert not r.flag("was_on_side")
    w = r.window("side", True, T.SIDE_STEPS)
    r.latch("was_on_side", len(w) == T.SIDE_STEPS and all(w))
    assert r.flag("was_on_side")


def test_scoring_text_matches_constants():
    doc = T.score.__doc__
    assert T.SIDE_DEG == 25.0 and "25 degrees" in doc
    assert T.SIDE_STEPS == 20 and "20 control" in doc and "(1 s)" in doc
    assert T.LIFT_FOUL == 0.040 and "4 cm" in doc
    assert T.SPINE_FOUL_DEG == 30.0 and "30 degrees from level" in doc
    assert T.REST_STEPS == 40 and "40" in doc and "2 s" in doc
    assert T.RELEASE == 0.05 and "5 cm" in doc
    assert T.ROCK_CLEAR == 0.01 and "1 cm" in doc
    assert T.PAD_BELOW == 0.002 and "2 mm below the pad" in doc
    assert T.INSTRUCTION == "Lay the humanoid robot on the pad on its left side so that it stays there on its own."
