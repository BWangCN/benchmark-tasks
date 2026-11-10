"""Scripted full-score demo for parking the rover across the slope, and its near-miss variants.

    python demo.py task.py demo_out                                # full score: MP4 per camera, setup.png, result.json
    python demo.py task.py out --variant remove_first --no-record   # take the stop block away as the rover stands
    python demo.py task.py out --variant off_contour --no-record    # park it 6 degrees off the contour
    python demo.py task.py out --variant idle --no-record           # do nothing

1. The right arm takes the rear tow bar from above and lifts the tail 1.5 cm off the slope: it now holds the rover.
2. The left arm takes the stop block off the slope by its outer end and sets it on the table.
3. The right arm swings the rover 90 degrees about its front axle (the front wheels roll as it turns) until it stands
   across the slope, trims its heading to within 0.15 degrees (closed loop), sets the tail down and lets go.

The script reads only object poses and the arms' own joint state.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import demokit as dk  # noqa: E402
import trailkit as tk  # noqa: E402

REAR_TOOL = dk.tool_rotation((0.5, 0, -1), (0, -1, 0))      # from above, leaning 27 degrees forward, across the bar
PICK_LEFT = dk.tool_rotation((0, 0, -1), (-1, 0, 0))        # straight down, closing along x (left hand)
BAR_GRIP_X = 0.011
TAIL_UP = 0.015
GRIP_PART = 0.40
GRIP_OPEN = 0.6


def axis_rot(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def tilt_y(rot, angle: float):
    return axis_rot((0, 1, 0), angle) @ rot


def take_tail(task, ctx, A):
    st = ctx.states()["rover"]
    rot = tilt_y(REAR_TOOL, -task.SLOPE)
    bar = st.to_world((-(task.TAB_X + BAR_GRIP_X), 0.0, task.TAB_Z))
    p = bar + rot[:, 2] * (task.TAB_SIZE[2] / 2 + 0.002 - 0.013)
    yield from A.via(p - rot[:, 2] * 0.05, rot, 2.5, lift=0.05, grip=GRIP_PART)
    yield from A.move(p, rot, 0.8)
    yield from A.set_grip(0.0, 0.5)
    yield from A.move(A.pos + task.UP * TAIL_UP, rot, 0.8)


def swing_across(task, ctx, A, turn: float, seconds: float = 4.0):
    """Swing the held tail about the front axle until the rover stands across the slope, trim the heading closed loop
    to within 0.15 degrees, set the tail down and let go."""
    st = ctx.states()["rover"]
    pivot = st.to_world((task.rv.AXLE_X, 0.0, 0.0))
    p0, r0 = A.pos.copy(), A.rot.copy()
    n = int(seconds * dk.CONTROL_HZ)
    for k in range(1, n + 1):
        R = axis_rot(task.UP, turn * dk.smooth(k / n))
        A.aim(pivot + R @ (p0 - pivot), R @ r0)
        yield
    for _ in range(int(8 * dk.CONTROL_HZ)):
        f = ctx.states()["rover"].rotation()[:, 0]
        err = math.atan2(f[1], f[0]) - turn
        if abs(err) < math.radians(0.15):
            break
        R = axis_rot(task.UP, -0.5 * err)
        A.aim(pivot + R @ (A.pos - pivot), R @ A.rot)
        yield
    yield from A.move(A.pos - task.UP * (TAIL_UP - 0.002), None, 1.0)
    yield from dk.wait(1.0)                              # let the tail settle on its tyres before opening
    yield from A.set_grip(1.0, 0.4)
    yield from A.move(A.pos - A.rot[:, 2] * 0.06, None, 0.8)
    f = ctx.states()["rover"].rotation()[:, 0]
    ctx.note(f"set down across the slope: heading {math.degrees(math.atan2(f[1], f[0])):.1f} deg")
    yield from A.go_home(2.0)


def remove_stop(task, ctx, A):
    st = ctx.states()["stop_block"]
    rot = tilt_y(PICK_LEFT, -task.SLOPE)
    p = st.to_world((0.0, 0.026, 0.0)) + rot[:, 2] * 0.006 - rot[:, 2] * 0.013
    yield from A.via(p - rot[:, 2] * 0.05, rot, 2.5, lift=0.05, grip=GRIP_OPEN)
    yield from A.move(p, rot, 0.8)
    yield from A.set_grip(0.0, 0.5)
    yield from A.move(A.pos + (0, 0, 0.06), rot, 0.8)
    dest = np.array([0.24, 0.22, task.TOP + 0.03 / 2 + 0.002]) + PICK_LEFT[:, 2] * 0.006 - PICK_LEFT[:, 2] * 0.013
    yield from A.move(dest + (0, 0, 0.05), PICK_LEFT, 2.0)
    yield from A.move(dest, PICK_LEFT, 0.8)
    yield from A.set_grip(GRIP_OPEN, 0.4)
    yield from A.move(A.pos + (0, 0, 0.06), PICK_LEFT, 0.6)
    ctx.note("stop block away")
    yield from A.go_home(2.0)


def script(task, ctx, variant: str = ""):
    L, R = ctx.left, ctx.right
    yield from dk.wait(0.5)
    if variant == "idle":
        yield from dk.wait(40.0)
        return
    if variant == "remove_first":
        yield from remove_stop(task, ctx, L)
        yield from dk.wait(2.0)
        return
    yield from take_tail(task, ctx, R)                 # the right hand holds the rover by its tail
    yield from remove_stop(task, ctx, L)               # the left takes the stop block away meanwhile
    turn = math.pi / 2 - (math.radians(6.0) if variant == "off_contour" else 0.0)
    yield from swing_across(task, ctx, R, turn)         # rear to the right: the rover across the slope, nose to the left
    yield from dk.wait(3.0)
    p0 = ctx.states()["rover"].pos.copy()
    yield from dk.wait(3.0)
    ctx.note(f"creep over 3 s after letting go: {1000 * np.linalg.norm(ctx.states()['rover'].pos - p0):.1f} mm")


def debug(task, ctx) -> str:
    st = ctx.states()["rover"]
    f = st.rotation()[:, 0]
    return (f"rover x {100 * st.pos[0]:.1f} y {100 * st.pos[1]:.1f} heading {math.degrees(math.atan2(f[1], f[0])):.0f} | "
            f"{task.report()['fouls']} {task.report()['last']}")


if __name__ == "__main__":
    dk.main(script, debug)
