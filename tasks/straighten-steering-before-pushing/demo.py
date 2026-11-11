"""Scripted full-score demo for the steered-wheels task, and its near-miss variants.

    python demo.py task.py demo_out                                 # full score: MP4 per camera, setup.png, result.json
    python demo.py task.py out --variant push_only --no-record      # push it as it stands (it runs out of the lane)
    python demo.py task.py out --variant shove_nose --no-record     # shove the nose straight instead
    python demo.py task.py out --variant tiller_only --no-record    # straighten without holding the body, then walk it
    python demo.py task.py out --variant tow_by_tiller --no-record  # one arm tows it by the tiller like a wagon
    python demo.py task.py out --variant lift_front --no-record     # lift the nose by its tab and walk it (wheelbarrow)
    python demo.py task.py out --variant drive_with_tiller --no-record  # steer it while walking it (like a cart's drawbar)
    python demo.py task.py out --variant straighten_only --no-record
    python demo.py task.py out --variant idle --no-record           # do nothing

1. The right arm grips the rear tow tab from above and holds the body.
2. The left arm grips the steering tiller (the red pin ahead of the bonnet) and carries it along its arc until the
   wheels are straight (closed loop on the measured angle), then lets go and goes home.
3. Still holding the rear tow tab, the right arm walks the rover down the lane along its heading, lets go, and pushes
   it the rest of the way into the bay with closed fingers against the tab's end, then withdraws.

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
TURN_RATE = math.radians(12.0)     # rad/s of the steering while the tiller is carried
PUSH_SPEED = 0.06                  # m/s along the lane
SLOW_SPEED = 0.012                 # m/s over the last 6 cm
WALK_X = 0.49                      # m: walked by the gripped tab this far (if short, the fingers push the rest)
GOAL_X = 0.49                      # m: the rover's origin here puts all four tyres in the bay (0.36 to 0.64)


def _rover(ctx) -> tk.BodyState:
    return ctx.states()["rover"]


def pad():
    from yam_arena.constants import PAD_TIP_PAST_TCP
    return PAD_TIP_PAST_TCP


def tcp_for_tip(point, rot):
    return np.asarray(point, float) - rot[:, 2] * pad()


def grip_rear_tab(task, ctx, A):
    rv = task.rv
    st = _rover(ctx)
    tip = st.to_world((-(rv.S.LENGTH / 2 + 0.024), 0.0, rv.S.BUMPER_HEIGHT - rv.TOW_TAB[2] / 2 + 0.004))
    rot = A.grip_rotation(tcp_for_tip(tip, dk.DOWN_X), (0, 0, -1), st.rotation()[:, 1])
    yield from A.transit(tcp_for_tip(tip, rot) + (0, 0, 0.04), rot, 3.0, grip=GRIP_OPEN)
    yield from A.move(tcp_for_tip(tip, rot), rot, 0.8)
    yield from A.set_grip(0.0, 0.6)


def straighten(task, ctx, A, target: float = 0.0):
    """Grip the steering tiller and carry it along its arc to straight (the arc follows the body's measured pose)."""
    rv = task.rv
    st = _rover(ctx)
    h = rv.TILLER_TOP - 0.018
    tip = rv.tiller_point(st, h=h)
    rot = A.grip_rotation(tcp_for_tip(tip, dk.DOWN_X), (0, 0, -1), st.rotation()[:, 1])   # pads either side of the pin
    yield from A.transit(tcp_for_tip(tip, rot) + (0, 0, 0.02), rot, 3.0, height=0.10, grip=GRIP_OPEN)
    yield from A.move(tcp_for_tip(tip, rot), rot, 0.8)
    yield from A.set_grip(0.0, 0.6)
    yield from rv.steer_with_tiller(lambda: _rover(ctx), A, target, lambda p: tcp_for_tip(p, rot), h, TURN_RATE)
    yield from dk.wait(0.3)
    st = _rover(ctx)
    ctx.note(f"steering {math.degrees(rv.steer_angle(st)):.1f} deg; body yaw "
             f"{math.degrees(math.atan2(st.rotation()[1, 0], st.rotation()[0, 0])):.1f} deg, "
             f"moved {100 * float(np.linalg.norm(st.pos[:2] - task.ROVER0.pos[:2])):.1f} cm")
    yield from A.set_grip(GRIP_OPEN, 0.4)
    yield from A.move(A.pos + (0, 0, 0.06), None, 0.6)


def carry_into_bay(task, ctx, A, goal_x: float = GOAL_X):
    """With the rear tow tab gripped: walk the hand along the rover's heading (it rolls on its wheels), slowing over
    the last 6 cm, then let go and withdraw."""
    rv = task.rv
    rot0 = A.rot.copy()
    h0 = _rover(ctx).rotation()[:, 0]
    yaw0 = math.atan2(h0[1], h0[0])
    for _ in range(int(50 * dk.CONTROL_HZ)):
        st = _rover(ctx)
        left = goal_x - st.pos[0]
        if left <= 0.0:
            break
        h = st.rotation()[:, 0]
        h = np.array([h[0], h[1], 0.0]) / max(np.hypot(h[0], h[1]), 1e-9)
        v = PUSH_SPEED if left > 0.06 else SLOW_SPEED
        tip = st.to_world((-(rv.S.LENGTH / 2 + 0.024), 0.0, rv.S.BUMPER_HEIGHT - rv.TOW_TAB[2] / 2 + 0.004))
        a = math.atan2(h[1], h[0]) - yaw0                  # the hand turns with the rover: the tab stays in the pads
        rot = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]) @ rot0
        A.aim(tcp_for_tip(tip + h * (0.02 if v == PUSH_SPEED else 0.01), rot), rot)   # the hand leads the tab
        yield
    yield from dk.wait(0.5)
    yield from A.set_grip(GRIP_OPEN, 0.5)
    yield from A.move(A.pos + (0, 0, 0.06), None, 0.6)
    ctx.note(f"carried to x {_rover(ctx).pos[0]:.3f}")
    yield from A.go_home(2.0)


def finger_push(task, ctx, A, goal_x: float = GOAL_X):
    """Closed fingers against the end face of the rear tow tab, pushing along the rover's heading (closed loop on its
    pose), slowing over the last 6 cm, then withdrawing."""
    rv = task.rv
    end_x = -(rv.S.LENGTH / 2 + rv.TOW_TAB[0] - 0.002)

    def frame(st):
        h = st.rotation()[:, 0]
        h = np.array([h[0], h[1], 0.0]) / max(np.hypot(h[0], h[1]), 1e-9)
        rot = dk.tool_rotation((h[0], h[1], -1.0), (-h[1], h[0], 0))
        return h, rot, st.to_world((end_x, 0.0, rv.S.BUMPER_HEIGHT - 0.002))

    h, rot, face = frame(_rover(ctx))
    yield from A.set_grip(0.0, 0.3)
    yield from A.transit(tcp_for_tip(face - h * 0.03, rot), rot, 3.0)
    yield from A.move(tcp_for_tip(face, rot), rot, 0.8)
    for _ in range(int(30 * dk.CONTROL_HZ)):
        st = _rover(ctx)
        left = goal_x - st.pos[0]
        if left <= 0.0:
            break
        h, rot, face = frame(st)
        A.aim(tcp_for_tip(face + h * (0.012 if left > 0.04 else 0.005), rot), rot)
        yield
    yield from A.move(A.pos - h * 0.03 + (0, 0, 0.02), None, 0.6)
    ctx.note(f"pushed to x {_rover(ctx).pos[0]:.3f}")
    yield from A.go_home(2.0)


def push_into_bay(task, ctx, A, goal_x: float = GOAL_X):
    """Grip the rear tow tab and walk the rover down the lane (used as it stands by the push_only variant)."""
    yield from grip_rear_tab(task, ctx, A)
    yield from carry_into_bay(task, ctx, A, goal_x)


def shove_nose(task, ctx, A):
    """Push the nose sideways with closed fingers until the heading points down the lane rotated by the steering."""
    rv = task.rv
    st = _rover(ctx)
    rot = dk.tool_rotation((0, -1, -0.4), (1, 0, 0))
    contact = st.to_world((rv.S.LENGTH / 2 - 0.02, rv.S.BODY_WIDTH / 2, 0.06))
    yield from A.set_grip(0.0, 0.3)
    yield from A.transit(tcp_for_tip(contact + np.array([0, 0.03, 0]), rot), rot, 3.0)
    for _ in range(int(4 * dk.CONTROL_HZ)):
        st = _rover(ctx)
        if rv.steer_angle(st) + math.atan2(st.rotation()[1, 0], st.rotation()[0, 0]) < math.radians(1):
            break
        A.aim(A.pos + np.array([0, -0.010 / dk.CONTROL_HZ * 2, 0]), rot)
        yield
    yield from A.move(A.pos + np.array([0, 0.04, 0.03]), rot, 0.6)
    yield from A.go_home(2.0)


def tow_by_tiller(task, ctx, A, goal_x: float = GOAL_X):
    """One arm only: grip the tiller and pull the rover down the lane by it, like a wagon's handle."""
    rv = task.rv
    st = _rover(ctx)
    h = rv.TILLER_TOP - 0.018
    tip = rv.tiller_point(st, h=h)
    rot = A.grip_rotation(tcp_for_tip(tip, dk.DOWN_X), (0, 0, -1), st.rotation()[:, 1])
    yield from A.transit(tcp_for_tip(tip, rot) + (0, 0, 0.02), rot, 3.0, height=0.10, grip=GRIP_OPEN)
    yield from A.move(tcp_for_tip(tip, rot), rot, 0.8)
    yield from A.set_grip(0.0, 0.6)
    for _ in range(int(50 * dk.CONTROL_HZ)):
        st = _rover(ctx)
        if st.pos[0] >= goal_x:
            break
        A.aim(tcp_for_tip(rv.tiller_point(st, h=h) + np.array([0.04, 0.0, 0.0]), rot), rot)
        yield
    ctx.note(f"towed by the tiller to x {_rover(ctx).pos[0]:.3f}, steering {math.degrees(rv.steer_angle(_rover(ctx))):.1f}")
    yield from A.set_grip(GRIP_OPEN, 0.5)
    yield from A.move(A.pos + (0, 0, 0.06), None, 0.6)
    yield from A.go_home(2.0)


def lift_front(task, ctx, A):
    """Grip the front tow tab and lift the nose (a 10 cm lift command): a wheelbarrow."""
    rv = task.rv
    st = _rover(ctx)
    tip = st.to_world(((rv.S.LENGTH / 2 + 0.024), 0.0, rv.S.BUMPER_HEIGHT - rv.TOW_TAB[2] / 2 + 0.004))
    rot = A.grip_rotation(tcp_for_tip(tip, dk.DOWN_X), (0, 0, -1), st.rotation()[:, 1])
    yield from A.transit(tcp_for_tip(tip, rot) + (0, 0, 0.03), rot, 3.0, grip=GRIP_OPEN)
    yield from A.move(tcp_for_tip(tip, rot), rot, 0.8)
    yield from A.set_grip(0.0, 0.6)
    yield from A.move(A.pos + (0, 0, 0.10), None, 2.0)


def drive_with_tiller(task, ctx, L, R):
    """Steer while moving: the right arm walks the rover from the start while the left arm works the tiller, counter-
    steering and then holding it straight (a cart steered by its drawbar)."""
    rv = task.rv
    st = _rover(ctx)
    h = rv.TILLER_TOP - 0.018
    tip = rv.tiller_point(st, h=h)
    rot = L.grip_rotation(tcp_for_tip(tip, dk.DOWN_X), (0, 0, -1), st.rotation()[:, 1])
    yield from L.transit(tcp_for_tip(tip, rot) + (0, 0, 0.02), rot, 3.0, height=0.10, grip=GRIP_OPEN)
    yield from L.move(tcp_for_tip(tip, rot), rot, 0.8)
    yield from L.set_grip(0.0, 0.6)
    for k in range(int(30 * dk.CONTROL_HZ)):
        st = _rover(ctx)
        if st.pos[0] >= 0.33:
            break
        hd = st.rotation()[:, 0]
        hd = np.array([hd[0], hd[1], 0.0]) / max(np.hypot(hd[0], hd[1]), 1e-9)
        tab = st.to_world((-(rv.S.LENGTH / 2 + 0.024), 0.0, rv.S.BUMPER_HEIGHT - rv.TOW_TAB[2] / 2 + 0.004))
        R.aim(tcp_for_tip(tab + hd * 0.03, R.rot), R.rot)
        yaw = math.atan2(hd[1], hd[0])
        target = max(min(-2.0 * yaw - 3.0 * st.pos[1], math.radians(25)), -math.radians(25))   # back to the centre line
        q = rv.steer_angle(st)
        L.aim(tcp_for_tip(rv.tiller_point(st, steer=q + max(min(target - q, 0.6), -0.6), h=h), rot), rot)
        yield
    ctx.note(f"drove with the tiller to x {_rover(ctx).pos[0]:.3f}, report {task.report()}")
    yield from L.set_grip(GRIP_OPEN, 0.4)
    yield from L.move(L.pos + (0, 0, 0.06), None, 0.6)
    yield from L.go_home(2.0)
    yield from carry_into_bay(task, ctx, R)


def script(task, ctx, variant: str = ""):
    L, R = ctx.left, ctx.right
    yield from dk.wait(0.5)
    if variant == "idle":
        yield from dk.wait(30.0)
        return
    if variant == "shove_nose":
        yield from shove_nose(task, ctx, R)
        yield from push_into_bay(task, ctx, R)
        return
    if variant == "drive_with_tiller":
        yield from grip_rear_tab(task, ctx, R)
        yield from drive_with_tiller(task, ctx, L, R)
        yield from dk.wait(1.0)
        return
    if variant == "tow_by_tiller":
        yield from tow_by_tiller(task, ctx, L)
        yield from dk.wait(1.0)
        return
    if variant == "tiller_only":
        yield from straighten(task, ctx, L)
        yield from L.go_home(2.0)
        yield from grip_rear_tab(task, ctx, R)
        yield from carry_into_bay(task, ctx, R, goal_x=WALK_X)
        if _rover(ctx).pos[0] < GOAL_X - 0.005:
            yield from finger_push(task, ctx, R)
        yield from dk.wait(1.0)
        return
    if variant == "lift_front":
        yield from lift_front(task, ctx, L)
        yield from push_into_bay(task, ctx, R)
        yield from L.set_grip(GRIP_OPEN, 0.5)
        yield from L.go_home(2.0)
        yield from dk.wait(1.0)
        return
    if variant == "push_only":
        yield from push_into_bay(task, ctx, R)
        yield from dk.wait(1.0)
        return
    yield from grip_rear_tab(task, ctx, R)
    yield from straighten(task, ctx, L)
    yield from L.go_home(2.0)
    if variant == "straighten_only":
        yield from R.set_grip(GRIP_OPEN, 0.4)
        yield from R.move(R.pos + (0, 0, 0.06), None, 0.6)
        yield from R.go_home(2.0)
        yield from dk.wait(2.0)
        return
    yield from carry_into_bay(task, ctx, R, goal_x=WALK_X)
    if _rover(ctx).pos[0] < GOAL_X - 0.005:                  # short of the bay: push the rest with the fingers
        yield from finger_push(task, ctx, R)
    yield from dk.wait(1.0)


def debug(task, ctx) -> str:
    st = _rover(ctx)
    return (f"x {st.pos[0]:.3f} y {st.pos[1]:.3f} steer {math.degrees(task.rv.steer_angle(st)):.1f} "
            f"| {task.report()['fouls']} {task.report()['last']}")


if __name__ == "__main__":
    dk.main(script, debug)
