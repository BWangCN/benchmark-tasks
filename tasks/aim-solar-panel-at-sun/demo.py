"""Scripted full-score demo for the panel-aiming task, and its near-miss variants.

    python demo.py task.py demo_out                               # full score: MP4 per camera, setup.png, result.json
    python demo.py task.py out --variant half_way --no-record     # stop the swing a third short (about 15 deg off)
    python demo.py task.py out --variant turn_rover --no-record   # aim by turning the whole rover by its bumper
    python demo.py task.py out --variant idle --no-record         # do nothing

0. The left arm steadies the rover: closed fingers rest on its bonnet, clear of the panel's sweep.
1. The right arm grips the panel's aiming handle, which sticks out of the panel's near edge, fingers closing across
   it.
2. It swings the panel to face the lamp: turning it about the vertical and tilting it up together, along a smooth
   path. Each step it reads the panel's pose (as a camera would estimate it), computes where the panel should face
   next, and moves its grip there; the direction to the lamp comes from the lamp's position on the table.
3. It opens, backs out along the panel's face and withdraws.

The script reads only object poses (the rover, the panel, the lamp) and the arm's own joint state.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import demokit as dk  # noqa: E402
import trailkit as tk  # noqa: E402

GRIP_ON = 0.022        # m: tool point this far along the handle from its free end (pad tips 13 mm further in)
SWING_S = 6.0          # seconds for the swing
GRIP_OPEN = 0.35       # gripper command while approaching (0 closed, 1 fully open: 94 mm)
SWING_RATE = 0.5       # rad/s at most on each axis of the mount
PRESS = -0.004         # m: the steadying fingertips press this far into the roof


def _st(ctx) -> tk.BodyState:
    return ctx.states()["rover"]


def target_angles(task, st: tk.BodyState) -> tuple[float, float]:
    """(turn, tilt) of the mount that face the panel at the lamp, from the rover's pose and the lamp's position."""
    c, _ = task.panel_frame(st)
    d = task.LAMP - c
    rover_heading = tk.heading(st.rotation())
    turn = math.atan2(d[1], d[0]) - rover_heading
    tilt = math.pi / 2 - math.atan2(d[2], math.hypot(d[0], d[1]))
    return turn, tilt


def measured_angles(task, st: tk.BodyState) -> tuple[float, float]:
    """(turn, tilt) read off the panel's measured orientation relative to the rover body."""
    _, R = task.panel_frame(st)
    Rl = st.rotation().T @ R                   # panel in the rover frame
    x = Rl[:, 0]                               # the panel's x axis: (cos tilt cos turn, cos tilt sin turn, -sin tilt)
    tilt = math.atan2(-x[2], math.hypot(x[0], x[1]))
    turn = math.atan2(x[1], x[0])
    return turn, tilt


def grip_pose(task, st: tk.BodyState, turn: float, tilt: float):
    """Tool pose holding the near edge of the panel when the mount is at (turn, tilt)."""
    s = tk.state_at(task.ROVER, st.pos, st.quat_xyzw, joints={"turntable": turn, "panel": tilt})
    p, R = task.panel_frame(s)
    pos = p + R @ np.array([-task.PANEL[0] / 2 - task.HANDLE_L + GRIP_ON, 0.0, 0.0])
    return pos, dk.tool_rotation(R[:, 0], R[:, 2])      # along the handle, clamping its flat faces


def script(task, ctx, variant: str = ""):
    A = ctx.right
    yield from dk.wait(0.5)
    if variant == "idle":
        yield from dk.wait(60.0)
        return
    st = _st(ctx)
    if variant == "turn_rover":
        # the wrong way: turn the whole rover by pushing its rear bumper sideways
        rear = st.to_world((-0.131, 0.0, 0.045))
        yield from A.via(rear + (0.0, 0.06, 0.0), dk.tool_rotation((0, 0, -1), (1, 0, 0)), 3.0, grip=0.0)
        yield from A.move(rear + (0.0, -0.08, 0.0), None, 3.0)
        yield from A.move(A.pos + (0, 0.04, 0.08), None, 1.0)
        yield from A.go_home(2.0)
        return
    st0 = st                                       # the rover's pose at the start: it is held still, not chased
    turn0, tilt0 = measured_angles(task, st)
    turn1, tilt1 = target_angles(task, st0)
    if variant == "half_way":
        turn1, tilt1 = turn0 + 0.67 * (turn1 - turn0), tilt0 + 0.67 * (tilt1 - tilt0)

    # 1. the left arm steadies the rover: closed fingers resting on its bonnet, well clear of the
    #    panel's sweep
    L = ctx.left
    roof = st0.to_world((0.080, 0.0, 0.085))
    down = dk.tool_rotation((0, 0, -1), (1, 0, 0))
    yield from L.via(roof + (0, 0, 0.04), down, 3.0, lift=0.04, grip=0.0)
    yield from L.move(roof + (0, 0, PRESS + 0.013), down, 1.0)       # pad tips 13 mm below the tool point

    # 2. the right arm grips the handle, coming in from behind it, low and along its axis
    pos, rot = grip_pose(task, st0, turn0, tilt0)
    behind = pos - rot[:, 2] * 0.06
    yield from A.move((0.16, -0.16, behind[2] + 0.10), rot, 2.0, grip=GRIP_OPEN)
    yield from A.move(behind + (0, 0, 0.04), rot, 1.2)
    yield from A.move(behind, rot, 0.8)
    yield from A.move(pos, rot, 1.5)
    yield from A.set_grip(0.0, 0.8)
    st = _st(ctx)
    turn0, tilt0 = measured_angles(task, st)       # start the swing from where the panel really is
    ctx.note(f"gripped the panel at turn {math.degrees(turn0):.0f} tilt {math.degrees(tilt0):.0f}")

    # 3. swing along the straight path in (turn, tilt) from where the panel is to where it faces the lamp; the
    #    clamped handle makes the panel follow the grip. Then trim the remaining error from the measured angles.
    n = int(SWING_S * dk.CONTROL_HZ)
    for k in range(1, n + 1):
        s = dk.smooth(k / n)
        A.aim(*grip_pose(task, st0, turn0 + s * (turn1 - turn0), tilt0 + s * (tilt1 - tilt0)))
        yield
    ct, ca = turn1, tilt1
    for _ in range(int(3 * dk.CONTROL_HZ)):
        st = _st(ctx)
        t, a = measured_angles(task, st)
        if abs(t - turn1) < math.radians(0.7) and abs(a - tilt1) < math.radians(0.7):
            break
        ct += 0.3 * (turn1 - t)
        ca += 0.3 * (tilt1 - a)
        A.aim(*grip_pose(task, st0, ct, ca))
        yield
    yield from dk.wait(0.5)
    st = _st(ctx)
    ctx.note(f"swung: aim error {task.aim_error_deg(st):.1f} deg; rover moved {100 * task.rover_moved(st)[0]:.1f} cm")
    yield from A.set_grip(GRIP_OPEN, 0.5)
    yield from A.move(A.pos - A.rot[:, 2] * 0.06, None, 1.0)          # straight back out along the handle
    yield from A.move(A.pos + (-0.04, -0.04, 0.05), None, 0.8)
    yield from dk.together(A.go_home(2.0), L.move(L.pos + (0, 0, 0.06), None, 0.8))
    yield from L.go_home(2.0)
    yield from dk.wait(1.0)


def debug(task, ctx) -> str:
    st = ctx.states()["rover"]
    t, a = measured_angles(task, st)
    d, turn = task.rover_moved(st)
    return (f"turn {math.degrees(t):.1f} tilt {math.degrees(a):.1f} aim err {task.aim_error_deg(st):.1f} | rover moved "
            f"{100 * d:.1f} cm turned {turn:.1f} | {task.report()['last']}")


if __name__ == "__main__":
    dk.main(script, debug)
