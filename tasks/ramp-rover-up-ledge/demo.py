"""Scripted full-score demo for the ledge task, and its near-miss variants.

    python demo.py task.py demo_out                              # full score: MP4 per camera, setup.png, result.json
    python demo.py task.py out --variant push_only --no-record   # push the rover straight at the ledge, no ramp
    python demo.py task.py out --variant ramp_only --no-record   # set the ramp rocks and stop
    python demo.py task.py out --variant wheelie --no-record     # press the tail down so the nose rises (a foul)
    python demo.py task.py out --variant pulse_wheelie --no-record   # the same, in a 0.4 s press
    python demo.py task.py out --variant idle --no-record        # do nothing

1. The left arm takes a ramp rock by its high step, sets it on the rover's left wheel line, high end toward the ledge,
   and taps it tight against the ledge's face with its closed fingers; the right arm does the same on the right line.
2. The right arm, fingers closed and pointing forward, puts its fingertips to the middle of the rear bumper and
   pushes the rover slowly forward (closed loop on the rover's measured pose): the front wheels roll up the steps
   onto the ledge, then the rear wheels.
3. It withdraws.

The script reads only object poses (the rover, the rocks) and the arms' own joint state.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import demokit as dk  # noqa: E402
import trailkit as tk  # noqa: E402

PUSH_TOOL = dk.tool_rotation((1, 0, -0.2), (0, -1, 0))   # pointing forward, a little down; fingers closed
ROCK_TOOL = dk.tool_rotation((0.35, 0, -1), (0, -1, 0))  # tilted forward, closing along -y: both arms reach the
#                                                          ledge's foot with it, so the rock keeps its pose in hand
PUSH_SPEED = 0.025        # m/s
GRIP_X = 0.017            # m: grip a ramp rock this far toward its high end from its centre (on the high step)


def _st(ctx, name) -> tk.BodyState:
    return ctx.states()[name]


def set_ramp(task, ctx, A, name: str, line_y: float):
    """Pick a ramp rock by its high step (the rock lies with its high end toward +x or -x) and set it tight to the
    ledge's face on a wheel line, high end against the ledge."""
    st = _st(ctx, name)
    R = st.rotation()
    high = 1.0 if R[0, 0] > 0 else -1.0                              # which world direction its high end faces
    grip = st.pos + R @ np.array([GRIP_X, 0.0, 0.0])
    grip[2] = task.TOP + 0.016
    rot = ROCK_TOOL if high > 0 else dk.tool_rotation((-0.35, 0, -1), (0, 1, 0))
    yield from A.via(grip + (0, 0, 0.05), rot, 3.0, lift=0.06, grip=1.0)
    yield from A.move(grip, rot, 1.0)
    yield from A.set_grip(0.0, 0.6)
    yield from A.move(grip + (0, 0, 0.14), rot, 1.0)                  # above the rover's wings (8.5 cm)
    if high < 0:                                                     # turn it round in the air, high end to +x
        mid = grip + (0.05, 0, 0.10)
        yield from A.move(mid, ROCK_TOOL, 2.0)
    centre = np.array([task.LEDGE_X[0] - 1.5 * task.STEP_L - 0.001, line_y, task.TOP])
    target = centre + np.array([GRIP_X, 0.0, 0.016])
    yield from A.move(target + (0, 0, 0.14), ROCK_TOOL, 2.0)
    yield from A.move(target + (0, 0, 0.004), ROCK_TOOL, 1.5)
    yield from A.set_grip(1.0, 0.5)
    yield from A.move(A.pos + (0, 0, 0.08), ROCK_TOOL, 0.8)
    # tap it tight: closed fingers behind its low end, pushed toward the ledge until its high end meets the face
    st = _st(ctx, name)
    low_end = st.to_world((-1.5 * task.STEP_L, 0.0, 0.0))
    tap = np.array([low_end[0] - 0.012, st.pos[1], task.TOP + 0.022])
    yield from A.set_grip(0.0, 0.4)
    yield from A.move(tap + (0, 0, 0.05), ROCK_TOOL, 1.0)
    yield from A.move(tap, ROCK_TOOL, 0.6)
    x = tap[0]
    for _ in range(int(3 * dk.CONTROL_HZ)):
        st = _st(ctx, name)
        pts = tk.body_points(task.ROCKS[0] if name == "ramp_rock_1" else task.ROCKS[2], st.pos, st.rotation())
        if pts[:, 0].max() >= task.LEDGE_X[0] - 0.001:
            break
        x += 0.015 / dk.CONTROL_HZ
        A.aim((x, tap[1], tap[2]), ROCK_TOOL)
        yield
    yield from A.move(A.pos + (-0.01, 0, 0.08), ROCK_TOOL, 0.8)
    ctx.note(f"{name} set under the wheel line y {100 * line_y:.1f} cm")


def push_up(task, ctx, A, until_x: float):
    st = _st(ctx, "rover")
    face = st.to_world((-0.131, 0.0, 0.050))
    tcp = face - PUSH_TOOL[:, 2] * (0.013 + 0.010)
    yield from A.via(tcp, PUSH_TOOL, 3.0, lift=0.06, grip=0.0)
    for _ in range(int(25 * dk.CONTROL_HZ)):
        st = _st(ctx, "rover")
        if st.pos[0] >= until_x:
            break
        face = st.to_world((-0.131, 0.0, 0.050))
        # fingertips 3 mm into the bumper face, moving forward at the push speed
        tcp = face - PUSH_TOOL[:, 2] * (0.013 - 0.003) + np.array([PUSH_SPEED / dk.CONTROL_HZ, 0.0, 0.0])
        A.aim(tcp, PUSH_TOOL)
        yield
    ctx.note(f"pushed: rover at x {100 * _st(ctx, 'rover').pos[0]:.1f} cm, tilt "
             f"{tk.tilt_deg(_st(ctx, 'rover').rotation()):.0f}")
    yield from A.move(A.pos - PUSH_TOOL[:, 2] * 0.05, PUSH_TOOL, 0.8)
    yield from A.go_home(2.0)


def script(task, ctx, variant: str = ""):
    L, R = ctx.left, ctx.right
    yield from dk.wait(0.5)
    if variant == "idle":
        yield from dk.wait(60.0)
        return
    if variant in ("wheelie", "pulse_wheelie"):
        # the forbidden shortcut: press the rear bumper down so the nose rises (as for a wheelchair at a kerb)
        st = _st(ctx, "rover")
        top = st.to_world((-0.122, 0.0, 0.053))
        down = dk.tool_rotation((0, 0, -1), (0, -1, 0))
        yield from R.via(top + (0, 0, 0.03), down, 3.0, lift=0.06, grip=0.0)
        yield from R.move(top + (0, 0, 0.013 - 0.030), down, 2.0 if variant == "wheelie" else 0.4)        # pad tips 3 cm into the bumper's top
        yield from dk.wait(1.5 if variant == "wheelie" else 0.0)
        yield from R.move(R.pos + (0, 0, 0.08), down, 1.0 if variant == "wheelie" else 0.4)
        yield from R.go_home(2.0)
        return
    if variant != "push_only":
        yield from set_ramp(task, ctx, L, "ramp_rock_1", task.rv.WHEEL_Y)
        yield from L.go_home(2.0)
        yield from set_ramp(task, ctx, R, "ramp_rock_3", -task.rv.WHEEL_Y)
        yield from R.go_home(2.0)
    if variant == "ramp_only":
        yield from dk.wait(1.0)
        return
    yield from push_up(task, ctx, R, task.LEDGE_X[0] + task.rv.AXLE_X + task.rv.TYRE_R + 0.012)
    yield from dk.wait(1.0)


def debug(task, ctx) -> str:
    st = ctx.states()["rover"]
    lows = task.tyre_lows(st)
    z = " ".join(f"{k[-2:]}:{100 * (v[2] - task.TOP):.1f}" for k, v in lows.items())
    rk = []
    for r in task.ROCKS[:4:2]:
        rs = ctx.states()[r.name]
        pts = tk.body_points(r, rs.pos, rs.rotation())
        rk.append(f"{r.name[-1]}: xmax {100 * pts[:, 0].max():.1f} y {100 * rs.pos[1]:.1f} yaw "
                  f"{math.degrees(tk.heading(rs.rotation())):.0f}")
    return (f"rover x {100 * st.pos[0]:.1f} tilt {tk.tilt_deg(st.rotation()):.0f} tyre z(cm) {z} | {' '.join(rk)} | "
            f"{task.report()['last']}")


if __name__ == "__main__":
    dk.main(script, debug)
