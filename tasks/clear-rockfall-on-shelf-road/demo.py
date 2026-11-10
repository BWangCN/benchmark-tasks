"""Scripted full-score demo for the rockfall task, and its near-miss variants.

    python demo.py task.py demo_out                               # full score: MP4 per camera, setup.png, result.json
    python demo.py task.py out --variant push_over --no-record    # push the big rock off the far edge (the easy way)
    python demo.py task.py out --variant small_only --no-record   # clear only the two small rocks
    python demo.py task.py out --variant idle --no-record         # do nothing

1. The right arm reaches over the big rock, sets its closed fingers against the rock's far face, low down, and drags
   it toward the robot across the road until it tips off the near edge into the ditch (closed loop on the rock's
   measured pose).
2. The right arm picks up the small rock on the near track and lays it in the ditch.
3. The left arm picks up the small rock on the far track and lays it in the ditch.
4. Both arms withdraw.

The script reads only the rocks' poses (what a camera-based pose estimate gives) and the arms' own joint state.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import demokit as dk  # noqa: E402
import trailkit as tk  # noqa: E402

DOWN_Y = dk.tool_rotation((0, 0, -1), (0, -1, 0))   # fingers closing along -y: a flat face toward +-x when closed
DOWN_X = dk.tool_rotation((0, 0, -1), (1, 0, 0))
DRAG_SPEED = 0.04          # m/s
DITCH_X = 0.17             # where small rocks are laid in the ditch (on the table)


def _st(ctx, name) -> tk.BodyState:
    return ctx.states()[name]


def drag_toward(task, ctx, A, name, direction: float):
    """Drag (direction -1, toward the robot) or push (+1, away) a rock along x with closed fingers on its face."""
    st = _st(ctx, name)
    pts = task.footprint(task.ROCKS[name][0], st)
    face = pts[:, 0].max() if direction < 0 else pts[:, 0].min()
    z = task.DECK_TOP + 0.026                       # tool point; the pad tips are 13 mm lower
    x = face - direction * 0.022                    # fingers 2.2 cm beyond the face
    y = st.pos[1]
    yield from A.via((x, y, z), DOWN_Y, 3.0, lift=0.10, grip=0.0)
    for _ in range(int(12 * dk.CONTROL_HZ)):
        st = _st(ctx, name)
        if direction < 0 and task.cleared(task.ROCKS[name][0], st):    # wholly off the near edge, in the ditch
            break
        if direction > 0 and st.pos[0] > task.FAR_X + 0.05:
            break
        x += direction * DRAG_SPEED / dk.CONTROL_HZ
        A.aim((x, st.pos[1], z), DOWN_Y)
        yield
    ctx.note(f"{name} {'dragged' if direction < 0 else 'pushed'} to x {100 * _st(ctx, name).pos[0]:.1f} cm")
    yield from A.move(A.pos + (0, 0, 0.08), DOWN_Y, 1.0)


def carry_to_ditch(task, ctx, A, name, drop_y: float):
    st = _st(ctx, name)
    top = task.footprint(task.ROCKS[name][0], st)[:, 2].max()
    grasp = np.array([st.pos[0], st.pos[1], st.pos[2] + 0.020])
    yaw = tk.heading(st.rotation())
    rot = dk.tool_rotation((0, 0, -1), (-np.sin(yaw), np.cos(yaw), 0))   # close across the rock's x side
    yield from A.via(grasp + (0, 0, 0.06), rot, 3.0, lift=0.06, grip=1.0)
    yield from A.move(grasp, rot, 1.0)
    yield from A.set_grip(0.0, 0.6)
    carry_z = max(grasp[2] + 0.06, top + 0.04, task.TOP + 0.17)   # above the waiting rover's roof (11.7 cm + deck)
    yield from A.move((grasp[0], grasp[1], carry_z), rot, 1.0)
    drop = np.array([DITCH_X, drop_y, task.TOP + 0.035])
    yield from A.move((drop[0], drop[1], carry_z), rot, 2.0)
    yield from A.move(drop + (0, 0, 0.04), rot, 1.0)
    yield from A.move(drop, rot, 0.8)
    yield from A.set_grip(1.0, 0.5)
    yield from A.move((drop[0], drop[1], task.TOP + 0.20), rot, 1.2)    # straight up over the ditch, clear of the
    ctx.note(f"{name} laid in the ditch")                                 # waiting rover's roof before going home


def script(task, ctx, variant: str = ""):
    R, L = ctx.right, ctx.left
    yield from dk.wait(0.5)
    if variant == "idle":
        yield from dk.wait(60.0)
        return
    if variant != "small_only":
        yield from drag_toward(task, ctx, R, "rock_a", +1.0 if variant == "push_over" else -1.0)
    if variant == "push_over":
        yield from R.go_home(2.0)
        return
    yield from carry_to_ditch(task, ctx, R, "rock_c", -0.25)
    yield from R.go_home(2.0)
    yield from carry_to_ditch(task, ctx, L, "rock_b", 0.05)
    yield from L.go_home(2.0)
    yield from dk.wait(1.0)


def debug(task, ctx) -> str:
    s = ctx.states()
    parts = [f"{n} ({100 * s[n].pos[0]:.1f},{100 * s[n].pos[1]:.1f},{100 * (s[n].pos[2] - task.TOP):.1f})"
             for n in task.ROCKS]
    return " ".join(parts) + f" | {task.report()['last']}"


if __name__ == "__main__":
    dk.main(script, debug)
