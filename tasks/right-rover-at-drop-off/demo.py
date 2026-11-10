"""Scripted full-score demo for the rover-righting task, and its near-miss variants.

    python demo.py task.py demo_out                         # full score: MP4 per camera, setup.png, result.json
    python demo.py task.py out --variant tip_in_place --no-record    # tip it over where it lies (lands off the edge)
    python demo.py task.py out --variant drag_only --no-record       # slide it back from the edge, never right it
    python demo.py task.py out --variant idle --no-record            # do nothing

1. The right arm, fingers closed and pointing down, sets them against the rover's chassis on the drop-off side and
   pushes it along the track, away from the edge, until its treads are 18 cm from the edge (closed loop on the
   rover's measured pose; the push is aimed through its centre of mass, corrected for any turn).
2. The right arm withdraws. The left arm, fingers closed, presses on the upper part of the rover's roof, square to it,
   and follows it round the line of its lower tyres until it passes its balance point and drops onto its wheels.
3. Both arms withdraw.

The script reads only the rover's pose (what a camera-based pose estimate gives) and the arms' own joint state.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import demokit as dk  # noqa: E402
import trailkit as tk  # noqa: E402

PUSH_BACK_TO = 0.18        # m: treads this far from the edge before tipping
PUSH_SPEED = 0.03          # m/s along the track's width
PUSH_Z = 0.024             # m: tool point above the deck while pushing (the pad tips are 13 mm lower)
YAW_GAIN = 0.15            # m of push-point shift along the rover per rad of heading error
TOOL = dk.tool_rotation((0, 0, -1), (0, -1, 0))   # pointing down, fingers closing along -y: both arms reach it
ROOF_POINT = (0.0, -0.030, 0.117)  # rover frame: on the roof, 3 cm below its upper edge as it lies on its side
FINGER_OFF = 0.010         # m: tool point to the closed fingers' outer face
PRESS_RATE = 0.015         # m/s: how fast the press deepens while the rover resists


def _rover(ctx) -> tk.BodyState:
    return ctx.states()["rover"]


def push_back(task, ctx, target_gap: float):
    """Right arm slides the rover, on its side, away from the edge until its treads are ``target_gap`` from it."""
    A = ctx.right
    st = _rover(ctx)
    z = task.DECK_TOP + PUSH_Z
    start = np.array([st.pos[0], st.pos[1] - 0.035, z])
    yield from A.via(start, TOOL, 3.0, lift=0.08, grip=0.0)
    y = start[1]
    for _ in range(int(25 * dk.CONTROL_HZ)):
        st = _rover(ctx)
        if st.pos[1] - task.EDGE_Y >= target_gap:
            break
        # aim through the centre of mass, shifted along the rover to turn it back square
        yaw = tk.heading(st.rotation())
        x = st.pos[0] - float(np.clip(YAW_GAIN * yaw, -0.06, 0.06))
        y += PUSH_SPEED / dk.CONTROL_HZ
        A.aim((x, y, z), TOOL)
        yield
    st = _rover(ctx)
    ctx.note(f"pushed back: treads {100 * (st.pos[1] - task.EDGE_Y):.1f} cm from the edge, "
             f"yaw {math.degrees(tk.heading(st.rotation())):.0f} deg, tilt {tk.tilt_deg(st.rotation()):.0f} deg")
    yield from A.move(A.pos + (0, -0.04, 0), TOOL, 0.8)
    yield from A.move(A.pos + (-0.02, -0.16, 0.08), TOOL, 1.5)    # in toward its own side before climbing
    yield from A.go_home(2.0)


def tip_over(task, ctx):
    """Left arm presses on the upper part of the roof, square to it, until the rover passes its balance point over
    its lower tyres and drops onto its wheels."""
    A = ctx.left
    st = _rover(ctx)
    out = lambda st, d: st.to_world((ROOF_POINT[0], ROOF_POINT[1], ROOF_POINT[2] + FINGER_OFF - d)) + (0, 0, 0.005)
    yield from A.via(out(st, -0.03), TOOL, 3.0, lift=0.06, grip=0.0)
    yield from A.move(out(_rover(ctx), 0.0), TOOL, 0.8)
    d = 0.0
    for _ in range(int(10 * dk.CONTROL_HZ)):
        st = _rover(ctx)
        if tk.tilt_deg(st.rotation()) < 45:            # past its balance point: let it drop
            break
        d = min(d + PRESS_RATE / dk.CONTROL_HZ, 0.04)
        A.aim(out(st, d), TOOL)
        yield
    ctx.note(f"tipped: rover tilt {tk.tilt_deg(_rover(ctx).rotation()):.0f} deg")
    yield from A.move(A.pos + (0, 0.06, 0.02), TOOL, 0.6)
    yield from A.go_home(2.0)


def script(task, ctx, variant: str = ""):
    yield from dk.wait(0.5)
    if variant == "idle":                       # the arms hold their ready pose for the whole run
        yield from dk.wait(60.0)
        return
    if variant != "tip_in_place":
        yield from push_back(task, ctx, PUSH_BACK_TO)
    if variant != "drag_only":
        yield from tip_over(task, ctx)
    yield from dk.wait(1.0)


def debug(task, ctx) -> str:
    st = ctx.states()["rover"]
    return (f"rover ({100 * st.pos[0]:.1f}, {100 * st.pos[1]:.1f}, {100 * (st.pos[2] - task.DECK_TOP):.1f}) cm "
            f"tilt {tk.tilt_deg(st.rotation()):.0f} yaw {math.degrees(tk.heading(st.rotation())):.0f} "
            f"gap {100 * task.edge_gap(st):.1f} cm v {st.speed():.3f} | {task.report()['last']}")


if __name__ == "__main__":
    dk.main(script, debug)
