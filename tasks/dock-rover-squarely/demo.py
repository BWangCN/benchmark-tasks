"""Scripted full-score demo for docking the rover, and its near-miss variants.

    python demo.py task.py demo_out                                # full score: MP4 per camera, setup.png, result.json
    python demo.py task.py out --variant push_straight --no-record  # push it in as it stands (it arrives wide)
    python demo.py task.py out --variant idle --no-record           # do nothing

1. The left arm takes the rear tow bar from above and lifts the tail 1.5 cm (the rear wheels clear, the front ones
   stay down), swings it round the front axle until the rover is square to the dock, and sets it down.
2. The right arm, fingers closed and pointing forward, pushes the rover by the end of its tow bar straight into the
   dock (closed loop on its measured pose), and withdraws.

The script reads only the rover's pose and the arms' own joint state.
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
PUSH_TOOL = dk.tool_rotation((1, 0, -0.2), (0, -1, 0))      # pointing forward, fingers closed
BAR_GRIP_X = 0.011
TAIL_UP = 0.015
GRIP_PART = 0.40


def _rover(ctx) -> tk.BodyState:
    return ctx.states()["rover"]


def rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def bar_point(task, st: tk.BodyState, rot) -> np.ndarray:
    centre = st.to_world((-(task.TAB_X + BAR_GRIP_X), 0.0, task.TAB_Z))
    return centre + rot[:, 2] * (task.TAB_SIZE[2] / 2 + 0.002 - 0.013)


def square_up(task, ctx, A, seconds: float = 3.0):
    st = _rover(ctx)
    h0 = task.heading(st)
    rot = rz(h0) @ REAR_TOOL
    p = bar_point(task, st, rot)
    yield from A.via(p - rot[:, 2] * 0.05, rot, 2.5, lift=0.05, grip=GRIP_PART)
    yield from A.move(p, rot, 0.8)
    yield from A.set_grip(0.0, 0.5)
    yield from A.move(A.pos + (0, 0, TAIL_UP), rot, 0.8)
    pivot = st.to_world((task.rv.AXLE_X, 0.0, 0.0))          # the front axle, on the ground
    p0, r0 = A.pos.copy(), rot.copy()
    n = int(seconds * dk.CONTROL_HZ)
    for k in range(1, n + 1):
        turn = -h0 * dk.smooth(k / n)
        R = rz(turn)
        q = pivot + R @ (p0 - pivot)
        q[2] = p0[2]
        A.aim(q, R @ r0)
        yield
    yield from A.move(A.pos - (0, 0, TAIL_UP - 0.002), None, 0.8)
    yield from A.set_grip(1.0, 0.4)
    yield from A.move(A.pos - A.rot[:, 2] * 0.06, None, 0.8)
    ctx.note(f"squared: heading {math.degrees(task.heading(_rover(ctx))):.1f} deg, front axle y "
             f"{100 * _rover(ctx).to_world((task.rv.AXLE_X, 0, 0))[1]:.1f} cm")
    yield from A.go_home(2.0)


def push_in(task, ctx, A, speed: float = 0.02, square: bool = True):
    def face(st):
        return st.to_world((-task.TAB_X - task.TAB_SIZE[0] / 2, 0.0, task.TAB_Z))
    st = _rover(ctx)
    yield from A.via(face(st) - PUSH_TOOL[:, 2] * 0.023, PUSH_TOOL, 3.0, lift=0.06, grip=0.0)
    plane = task.DOCK_X - task.PAD_SIZE[0]                  # the pads' face
    for _ in range(int(20 * dk.CONTROL_HZ)):
        st = _rover(ctx)
        corners = task.bumper_face(st)[:, 0]
        # square: on until both bumper corners touch the pads; as it stands: until the leading corner reaches them
        if (corners.min() if square else corners.max()) >= plane - 0.0005:
            break
        A.aim(face(st) - PUSH_TOOL[:, 2] * 0.010 + np.array([speed / dk.CONTROL_HZ, 0, 0]), PUSH_TOOL)
        yield
    if square:
        hold = A.pos.copy() + np.array([0.002, 0.0, 0.0])    # press 2 mm on for 0.5 s so the rover comes to rest there
        for _ in range(int(0.5 * dk.CONTROL_HZ)):
            A.aim(hold, PUSH_TOOL)
            yield
    st = _rover(ctx)
    ctx.note(f"pushed: corners {np.round(1000 * (task.bumper_face(st)[:, 0] - plane), 1)} mm from the pads, bumper "
             f"centre y {1000 * task.bumper_face(st)[:, 1].mean():.1f} mm, heading {math.degrees(task.heading(st)):.2f} deg")
    yield from A.move(A.pos - PUSH_TOOL[:, 2] * 0.05, PUSH_TOOL, 1.5)          # withdraw slowly
    yield from A.go_home(2.0)


def script(task, ctx, variant: str = ""):
    L, R = ctx.left, ctx.right
    yield from dk.wait(0.5)
    if variant == "idle":
        yield from dk.wait(40.0)
        return
    if variant != "push_straight":
        yield from square_up(task, ctx, L)
    yield from push_in(task, ctx, R, square=variant != "push_straight")
    yield from dk.wait(1.5)


def debug(task, ctx) -> str:
    st = _rover(ctx)
    return (f"rover x {100 * st.pos[0]:.1f} y {100 * st.pos[1]:.1f} heading {math.degrees(task.heading(st)):.1f} | "
            f"{task.docked(st)} | {task.report()['fouls']} {task.report()['last']}")


if __name__ == "__main__":
    dk.main(script, debug)
