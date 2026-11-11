"""Scripted full-score demo for crossing the washout at an angle, and its near-miss variants.

    python demo.py task.py demo_out                               # full score
    python demo.py task.py out --variant square_on --no-record    # walk it straight at the gap: the front wheels drop in
    python demo.py task.py out --variant lift_the_nose --no-record  # lift the nose over the gap: lifting foul
    python demo.py task.py out --variant idle --no-record         # do nothing

1. The right arm grips the rear tow tab from above. The left arm grips the steering tiller and turns the front wheels
   to full lock to the right (closed loop on the measured angle).
2. The right arm walks the rover along its heading while the left arm keeps the tiller where it is: the rover turns
   right until it is about 33 degrees off square to the gap.
3. The left arm straightens the steering and lets go; the right arm walks the rover across the gap at that angle, one
   wheel over the gap at a time, until the tab reaches y 3 cm (or x 33 cm), and lets go.
4. The left arm, fingers closed and pointing down, pushes the same tab on by its end face until the rear tyres are
   past y 23.5 cm, then backs off.

The script reads only the rover's pose and steering angle and the arms' own joint state.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import demokit as dk  # noqa: E402
import trailkit as tk  # noqa: E402

GRIP_OPEN = 0.6
TURN_RATE = math.radians(12.0)
LOCK = math.radians(-29.0)          # right
TURNED = math.radians(40.0)         # heading off square before the steering is straightened (about 34 after it)
LEAD = 0.02                         # m: the walking hand leads the tab by this
RIGHT_REACH_Y = 0.03                # m: the right arm walks the tab no farther than this (y) ...
RIGHT_REACH_X = 0.33                # m: ... or (x): it hands over to the left arm there
GOAL_REAR_Y = 0.228                 # m: stop pushing once the rear tyres' lowest y passes this (the pad starts at 0.22; it coasts on 2 to 4 cm)


def _st(ctx) -> tk.BodyState:
    return ctx.states()["rover"]


def pad():
    from yam_arena.constants import PAD_TIP_PAST_TCP
    return PAD_TIP_PAST_TCP


def tcp_for_tip(point, rot):
    return np.asarray(point, float) - rot[:, 2] * pad()


def yaw(st) -> float:
    R = st.rotation()
    return math.atan2(R[1, 0], R[0, 0])


def tab_tip(task, st, end="rear"):
    """The grip point: the rear tow tab's middle, or (front) the front tow post, clear of the steering tiller."""
    rv = task.rv
    if end == "front":
        return st.to_world((rv.TOW_POST_X, 0.0, rv.S.BUMPER_HEIGHT + rv.TOW_TAB[2] / 2 + 0.012))
    return st.to_world((-(rv.S.LENGTH / 2 + 0.024), 0.0, rv.S.BUMPER_HEIGHT - rv.TOW_TAB[2] / 2 + 0.004))


def grip_rear_tab(task, ctx, A, end="rear"):
    st = _st(ctx)
    tip = tab_tip(task, st, end)
    rot = A.grip_rotation(tcp_for_tip(tip, dk.DOWN_X), (0, 0, -1), st.rotation()[:, 1])
    yield from A.transit(tcp_for_tip(tip, rot) + (0, 0, 0.03), rot, 3.0, grip=GRIP_OPEN)
    yield from A.move(tcp_for_tip(tip, rot), rot, 0.8)
    yield from A.set_grip(0.0, 0.6)
    mark_grip(ctx, A)


def grip_tiller(task, ctx, A):
    rv = task.rv
    st = _st(ctx)
    h = rv.TILLER_TOP - 0.018
    tip = rv.tiller_point(st, h=h)
    rot = A.grip_rotation(tcp_for_tip(tip, dk.DOWN_X), (0, 0, -1), st.rotation()[:, 1])
    yield from A.transit(tcp_for_tip(tip, rot) + (0, 0, 0.02), rot, 3.0, height=0.12, grip=GRIP_OPEN)
    yield from A.move(tcp_for_tip(tip, rot), rot, 0.8)
    yield from A.set_grip(0.0, 0.6)
    return h


_GRIP: dict = {}


def mark_grip(ctx, A):
    """Remember the hand's rotation and the rover's yaw when the hand closed on the tab."""
    _GRIP[A.side] = (A.rot.copy(), yaw(_st(ctx)))


def walk_step(task, ctx, A, lead=LEAD, end="rear"):
    """One control step of walking the gripped rear tab along the rover's heading; the hand turns with the rover, so
    the tab does not twist out of the pads."""
    st = _st(ctx)
    hd = st.rotation()[:, 0]
    hd = np.array([hd[0], hd[1], 0.0]) / max(np.hypot(hd[0], hd[1]), 1e-9)
    rot0, yaw0 = _GRIP.get(A.side, (A.rot, yaw(st)))
    a = yaw(st) - yaw0
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    rot = Rz @ rot0
    A.aim(tcp_for_tip(tab_tip(task, st, end) + hd * lead, rot), rot)


def release(A, up=0.06):
    yield from A.set_grip(GRIP_OPEN, 0.5)
    yield from A.move(A.pos + (0, 0, up), None, 0.6)


def set_steering(task, ctx, A, target):
    """Grip the tiller, carry it to ``target`` (closed loop on the measured angle), let go and go home."""
    rv = task.rv
    h = rv.TILLER_TOP - 0.018
    yield from grip_tiller(task, ctx, A)
    yield from rv.steer_with_tiller(lambda: _st(ctx), A, target, lambda p: tcp_for_tip(p, A.rot), h, TURN_RATE,
                                    timeout_s=20.0)
    yield from dk.wait(0.3)
    ctx.note(f"steering {math.degrees(rv.steer_angle(_st(ctx))):.1f} deg (target {math.degrees(target):.0f})")
    yield from release(A, 0.05)
    yield from A.go_home(2.0)


def turn_and_cross(task, ctx):
    """Hold the rover by its rear tab; full lock; walk it round on the locked steering until it is TURNED off
    square; straighten the steering. The steering holds its angle while the rover rolls (the servo's friction)."""
    L, R = ctx.left, ctx.right
    yield from grip_rear_tab(task, ctx, R)
    yield from set_steering(task, ctx, L, LOCK)
    yaw0 = yaw(_st(ctx))
    for _ in range(int(40 * dk.CONTROL_HZ)):
        st = _st(ctx)
        if yaw0 - yaw(st) >= TURNED:
            break
        walk_step(task, ctx, R, lead=0.025)
        yield
    R.aim(tcp_for_tip(tab_tip(task, _st(ctx)), R.rot), R.rot)
    ctx.note(f"turned {math.degrees(yaw0 - yaw(_st(ctx))):.1f} deg at y {_st(ctx).pos[1]:.3f}")
    yield from dk.wait(0.5)
    yield from set_steering(task, ctx, L, 0.0)
    ctx.note(f"{math.degrees(yaw0 - yaw(_st(ctx))):.1f} deg off square, steering straight")


TRACE = bool(__import__("os").environ.get("DEMO_TRACE"))


def cross(task, ctx, A, stop, end="rear"):
    """Walk the gripped tab along the heading until ``stop(st)``."""
    for k in range(int(40 * dk.CONTROL_HZ)):
        st = _st(ctx)
        if stop(st):
            break
        walk_step(task, ctx, A, end=end)
        if TRACE and k % 5 == 0:
            ctx.note(f"trace {A.side} k {k}: tab {np.round(tab_tip(task, st, end), 3)} hand {np.round(A.pos, 3)} "
                     f"yaw {math.degrees(yaw(st)):.1f} heading {np.round(st.rotation()[:2, 0], 2)}")
        yield


def rear_low_y(task, st) -> float:
    pts = task.tyre_points(st)
    return min(float(pts[n][:, 1].min()) for n in ("wheel_rl", "wheel_rr"))


def tab_end(task, st):
    """The middle of the rear tow tab's end face (the rover frame's -x end)."""
    rv = task.rv
    return st.to_world((-(rv.S.LENGTH / 2 + rv.TOW_TAB[0] - 0.002), 0.0, rv.S.BUMPER_HEIGHT - rv.TOW_TAB[2] / 2 + 0.004))


def push_on(task, ctx, A, stop, behind=0.012):
    """Closed fingers pointing down, against the rear tab's end face: push the rover along its heading until
    ``stop(st)``, pressing harder while it stands still, then back off and up."""
    st = _st(ctx)
    rot = A.grip_rotation(tcp_for_tip(tab_end(task, st), dk.DOWN_X), (0, 0, -1), st.rotation()[:, 1])
    hd = st.rotation()[:, 0]
    hd = np.array([hd[0], hd[1], 0.0]) / max(np.hypot(hd[0], hd[1]), 1e-9)
    yield from A.set_grip(0.0, 0.3)
    yield from A.transit(tcp_for_tip(tab_end(task, st) - hd * (behind + 0.03), rot), rot, 3.0)
    yield from A.move(tcp_for_tip(tab_end(task, st) - hd * behind, rot), rot, 0.8)
    boost, past = 0.0, [st.pos.copy()]
    for k in range(int(30 * dk.CONTROL_HZ)):
        st = _st(ctx)
        if stop(st):
            break
        hd = st.rotation()[:, 0]
        hd = np.array([hd[0], hd[1], 0.0]) / max(np.hypot(hd[0], hd[1]), 1e-9)
        past = (past + [st.pos.copy()])[-11:]
        moving = len(past) < 11 or float(np.linalg.norm(st.pos[:2] - past[0][:2])) > 0.002
        boost = max(boost - 0.002, 0.0) if moving else min(boost + 0.0004, 0.015)
        A.aim(tcp_for_tip(tab_end(task, st) - hd * behind + hd * (0.012 + 0.01 + boost), rot), rot)
        if TRACE and k % 10 == 0:
            ctx.note(f"trace push k {k}: tab end {np.round(tab_end(task, st), 3)} rear y {rear_low_y(task, st):.3f}")
        yield
    yield from A.move(A.pos - hd * 0.03 + np.array([0, 0, 0.03]), None, 0.6)


def script(task, ctx, variant: str = ""):
    L, R = ctx.left, ctx.right
    yield from dk.wait(0.5)
    if variant == "idle":
        yield from dk.wait(40.0)
        return
    if variant == "square_on":
        yield from grip_rear_tab(task, ctx, R)
        yield from cross(task, ctx, R, lambda st: tab_tip(task, st)[1] > RIGHT_REACH_Y)
        ctx.note(f"report {task.report()} beyond {task.beyond(_st(ctx))}")
        yield from release(R)
        yield from R.go_home(2.0)
        yield from dk.wait(2.0)
        return
    if variant == "lift_the_nose":
        rv = task.rv
        st = _st(ctx)
        tip = st.to_world((rv.S.LENGTH / 2 + 0.0165, 0.0, rv.S.BUMPER_HEIGHT - rv.TOW_TAB[2] / 2 + 0.004))
        rot = L.grip_rotation(tcp_for_tip(tip, dk.DOWN_X), (0, 0, -1), st.rotation()[:, 1])
        yield from L.transit(tcp_for_tip(tip, rot) + (0, 0, 0.03), rot, 3.0, grip=GRIP_OPEN)
        yield from L.move(tcp_for_tip(tip, rot), rot, 0.8)
        yield from L.set_grip(0.0, 0.6)
        yield from L.move(L.pos + (0, 0, 0.10), None, 2.0)     # try to lift the front axle over the gap
        yield from grip_rear_tab(task, ctx, R)
        yield from cross(task, ctx, R, lambda st: tab_tip(task, st)[1] > RIGHT_REACH_Y)
        yield from dk.together(release(R), release(L))
        yield from dk.together(R.go_home(2.0), L.go_home(2.0))
        yield from dk.wait(2.0)
        return
    yield from turn_and_cross(task, ctx)
    yield from cross(task, ctx, R, lambda st: tab_tip(task, st)[1] > RIGHT_REACH_Y or tab_tip(task, st)[0] > RIGHT_REACH_X)
    ctx.note(f"right arm let go: tyres beyond {sum(task.beyond(_st(ctx)).values())}, dip "
             f"{task.report()['deepest_dip_mm']} mm")
    yield from release(R)
    yield from R.go_home(2.0)
    # hand-over: the left arm pushes the same rear tab on by its end face, fingers closed and pointing down (gripping
    # it from above put the gripper's body against the rover's rear, and the arm shoved the rover back)
    yield from push_on(task, ctx, L, lambda st: rear_low_y(task, st) > GOAL_REAR_Y)
    ctx.note(f"left arm done: rear tyres' lowest y {rear_low_y(task, _st(ctx)):.3f}")
    yield from L.go_home(2.0)
    yield from dk.wait(2.5)
    ctx.note(f"end: tyres beyond {sum(task.beyond(_st(ctx)).values())}, on the pad {task.on_pad(_st(ctx))}")


def debug(task, ctx) -> str:
    st = _st(ctx)
    return (f"xy {np.round(st.pos[:2], 3)} yaw {math.degrees(yaw(st)):.1f} steer "
            f"{math.degrees(task.rv.steer_angle(st)):.1f} | {task.report()}")


if __name__ == "__main__":
    dk.main(script, debug)
