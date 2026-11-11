"""Offline tests of the turn-onto-its-back scorer on synthetic states (no Isaac Sim).

    python -m pytest test_scoring.py -q    # from this folder, in the Isaac Lab-Arena venv
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

for _m in ("dummy", "dummy_spec"):          # other task folders carry their own copies
    sys.modules.pop(_m, None)

TASK = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TASK))

import dummy as D  # noqa: E402
import task as T  # noqa: E402
import trailkit as tk  # noqa: E402

ARMS_OUT = {"l_sh_roll": 2.0, "r_sh_roll": -2.0}     # arms up beside the head: clear of the body


def lying(rot, pose=None, tilt_deg=0.0):
    """The dummy resting on the ground in orientation ``rot`` (optionally turned ``tilt_deg`` about its long axis)."""
    pose = dict(T.START_POSE if pose is None else pose)
    R = tk.rot_from_xyzw(rot)
    if tilt_deg:
        t = math.radians(tilt_deg)
        Ry = np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])
        R = Ry @ R
    q = tk.quat_from_rot(R)
    st = tk.state_at(T.DUMMY, (T.DUMMY_X, 0.0, 0.0), q, joints=pose)
    z = T.GROUND - D.lowest(T.DUMMY, st, which={"base"})
    return tk.state_at(T.DUMMY, (T.DUMMY_X, 0.0, z), q, joints=pose)


def rec(rest=True, released=True, lifted=False, steps=None, trapped=False):
    r = tk.Recorder()
    for _ in range(T.REST_STEPS if steps is None else steps):
        r.window("rest", rest, T.REST_STEPS)
        r.window("released", released, T.REST_STEPS)
    r.latch("lifted", lifted)
    r.latch("trapped", trapped)
    return r


def with_link(st, name, origin, R):
    links = dict(st.links)
    links[name] = (np.asarray(origin, float), np.asarray(R, float))
    return tk.BodyState(pos=st.pos, quat_xyzw=st.quat_xyzw, lin_vel=st.lin_vel, ang_vel=st.ang_vel, low=st.low,
                        high=st.high, joints=st.joints, links=links)


ACROSS = np.column_stack([(0, 0, 1), (0, 1, 0), (-1, 0, 0)])   # a forearm link frame whose segment runs along world x


def test_start_pose_face_down_nothing_under():
    st = lying(T.START_ROT)
    assert math.isclose(T.chest_angle(st), 180.0, abs_tol=1e-6)
    assert D.limb_under_body(T.DUMMY, st) == []
    assert T.trunk_lift(st) < 1e-9


def test_idle_scores_zero():
    assert T.evaluate(lying(T.START_ROT), rec())[0] == 0.0


def test_full_score_on_back_limbs_clear():
    st = lying(D.on_back(), ARMS_OUT)
    assert D.limb_under_body(T.DUMMY, st) == []
    assert T.evaluate(st, rec())[0] == 1.0


def test_chest_limit_12_degrees():
    assert T.evaluate(lying(D.on_back(), ARMS_OUT, tilt_deg=11.0), rec())[0] == 1.0
    assert T.evaluate(lying(D.on_back(), ARMS_OUT, tilt_deg=13.0), rec())[0] == 0.25


def test_quarter_score_tilted_up_to_45_degrees():
    assert T.evaluate(lying(D.on_back(), ARMS_OUT, tilt_deg=44.0), rec())[0] == 0.25
    assert T.evaluate(lying(D.on_back(), ARMS_OUT, tilt_deg=46.0), rec())[0] == 0.0


def test_quarter_score_limb_under_the_back():
    # the left arm swung in under the trunk (adducted across the body's back)
    st = lying(D.on_back(), {"l_sh_roll": -0.5, "r_sh_roll": -2.0})
    assert "l_sh_roll" in D.limb_under_body(T.DUMMY, st) or "l_elbow" in D.limb_under_body(T.DUMMY, st)
    assert T.evaluate(st, rec())[0] == 0.25


def test_on_its_side_scores_zero():
    assert T.evaluate(lying(D.on_left_side(), ARMS_OUT), rec())[0] == 0.0
    assert T.evaluate(lying(D.on_right_side(), ARMS_OUT), rec())[0] == 0.0


def test_lift_foul_latches_whatever_the_end():
    r = rec(lifted=True)
    assert T.evaluate(lying(D.on_back(), ARMS_OUT), r)[0] == 0.0
    r.latch("lifted", False)                  # a later step without the lift does not clear it
    assert r.flag("lifted")
    assert T.evaluate(lying(D.on_back(), ARMS_OUT), r)[0] == 0.0


def test_full_score_needs_rest_and_release():
    st = lying(D.on_back(), ARMS_OUT)
    assert T.evaluate(st, rec(rest=False))[0] == 0.0
    assert T.evaluate(st, rec(released=False))[0] == 0.0
    assert T.evaluate(st, rec(steps=T.REST_STEPS - 1))[0] == 0.0     # less than 1 s of rest


def test_partial_credit_also_needs_rest_and_release():
    st = lying(D.on_back(), ARMS_OUT, tilt_deg=30.0)
    assert T.evaluate(st, rec())[0] == 0.25
    assert T.evaluate(st, rec(released=False))[0] == 0.0


def test_trunk_lift_measures_the_lowest_trunk_point():
    st = lying(T.START_ROT)
    up = tk.state_at(T.DUMMY, st.pos + np.array([0, 0, 0.05]), st.quat_xyzw, joints=st.joints)
    assert math.isclose(T.trunk_lift(up), 0.05, abs_tol=1e-6)
    assert T.trunk_lift(up) > T.LIFT_FOUL


def test_trapped_during_the_turn_caps_at_quarter():
    st = lying(D.on_back(), ARMS_OUT)
    assert T.evaluate(st, rec(trapped=True)) == (0.25, "on its back, but an arm was trapped under it during the turn")


def test_release_must_hold_for_the_whole_last_second():
    st = lying(D.on_back(), ARMS_OUT)
    r = rec()
    r.window("released", False, T.REST_STEPS)          # a hand close to the body 1 step ago...
    r.window("released", True, T.REST_STEPS)           # ...gone on the last step
    assert T.evaluate(st, r)[0] == 0.0


def test_forearm_straight_across_under_the_back_is_detected():
    st = lying(D.on_back(), ARMS_OUT)
    trunk = st.pos + st.rotation() @ np.array([0.0, 0.0, 0.09])
    centre = np.array([trunk[0], trunk[1], T.GROUND + 0.011])
    st2 = with_link(st, "l_elbow", centre - ACROSS @ np.array([0.0, 0.0, -0.0525]), ACROSS)
    assert D.limb_under_body(T.DUMMY, st2) == ["l_elbow"]
    assert T.evaluate(st2, rec())[0] == 0.25


def test_the_head_is_not_judged():
    st = lying(D.on_back(), ARMS_OUT)
    head = st.pos + st.rotation() @ np.array([0.0, 0.0, 0.176])
    centre = np.array([head[0], head[1], T.GROUND + 0.011])
    st2 = with_link(st, "l_elbow", centre - ACROSS @ np.array([0.0, 0.0, -0.0525]), ACROSS)
    assert D.limb_under_body(T.DUMMY, st2) == []


def test_standing_it_on_end_is_a_foul():
    # turned end over end about world x, pivoting on the head: the lowest trunk point stays on the pad
    st = lying(T.START_ROT)
    for deg in (15, 30, 45):
        t = math.radians(deg)
        Rx = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
        R = Rx @ st.rotation()
        s2 = tk.state_at(T.DUMMY, st.pos, tk.quat_from_rot(R), joints=st.joints)
        s2 = tk.state_at(T.DUMMY, st.pos + np.array([0, 0, T.GROUND - D.lowest(T.DUMMY, s2, which={"base"})]),
                         tk.quat_from_rot(R), joints=st.joints)
        assert T.trunk_lift(s2) < 1e-6
        assert (T.spine_tilt(s2) > T.SPINE_FOUL_DEG) == (deg > 30)
    r = rec()
    r.latch("stood", True)
    assert T.evaluate(lying(D.on_back(), ARMS_OUT), r)[0] == 0.0


def test_a_roll_keeps_the_spine_level():
    for tilt in (0.0, 45.0, 90.0, 135.0):
        assert T.spine_tilt(lying(D.on_back(), ARMS_OUT, tilt_deg=tilt)) < 1e-6


def test_scoring_text_matches_constants():
    doc = T.score.__doc__
    assert T.CHEST_UP_DEG == 12.0 and "12 degrees" in doc
    assert T.CHEST_PART_DEG == 45.0 and "45 degrees" in doc
    assert T.LIFT_FOUL == 0.040 and "4 cm" in doc
    assert T.RELEASE == 0.05 and "5 cm" in doc
    assert T.REST_STEPS == 20 and "20" in doc
    assert T.TRAP_FROM_DEG == 135.0 and "45 degrees from face down" in doc
    assert T.SPINE_FOUL_DEG == 30.0 and "30 degrees from level" in doc
    assert "every 5 mm" in doc and "head is not judged" in doc
    assert T.INSTRUCTION == "The humanoid robot on the pad has fallen face down. Turn it onto its back."
