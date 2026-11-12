"""Scripted full-score demo for the fold-forward task, and its near-miss variants.

    python demo.py task.py demo_out                                       # full score: MP4 per camera, setup.png
    python demo.py task.py out --variant sit_and_release --no-record      # raise it to 75 degrees and let go
    python demo.py task.py out --variant roll_between_rocks --no-record   # try to roll it over by one shoulder
    python demo.py task.py out --variant roll_in_place --no-record        # roll it over about its own spine
    python demo.py task.py out --variant legs_up --no-record              # fold both legs up onto the trunk
    python demo.py task.py out --variant idle --no-record                 # do nothing

The robot lies on its back along y, head to the robot's left, between the rocks; its wrists lie against the rocks,
so there is no room for fingers there.

1. The left hand takes the head across its width, just above its base (fingers closing along x, the head free to
   turn in the grip like a pin), and carries it round the hip axis: the trunk rises through sitting and on to 112
   degrees, past its balance point (about 95).
2. It lets go and backs off: the trunk folds on over the thighs and rests there, back up.

The script reads only the robot's pose and joint angles (what a camera-based pose estimate gives) and the arms'
own joint state.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import demokit as dk  # noqa: E402
import dummy as D  # noqa: E402
import trailkit as tk  # noqa: E402

HEAD_GRIP_Z = 0.152       # m: grip the head this far up the spine from the hips (just above its base, clear of the
#                           shoulders: the lower the grip, the lower the arc and the better the arm reaches it)
ARC_SLACK = 0.0           # m: none (a 1 cm slack left the trunk short of its balance point, 2 runs)
CARRY_DEG = 112.0         # the head is carried this far round the hips: past the balance point (about 95)
GRIP_OPEN = 0.80          # fingers about 7.5 cm apart: the head is 5.2 cm wide
TOOL = dk.tool_rotation((0, 0, -1), (-1, 0, 0))    # pointing down, fingers closing along x (across the head)


def _st(ctx) -> tk.BodyState:
    return ctx.states()["dummy"]


def head_grip(task, st) -> np.ndarray:
    return st.to_world((0.0, 0.0, HEAD_GRIP_Z))


def carry_head(task, ctx, to_deg=CARRY_DEG, seconds=5.0):
    """Left hand: take the head across its width and carry it round the hip axis (x through the hips) to ``to_deg``;
    the head turns in the grip like a pin."""
    A = ctx.left
    st = _st(ctx)
    hip = np.asarray(st.pos, float)
    g = head_grip(task, st)
    yield from A.transit(g + (0, 0, 0.04), TOOL, 3.0, height=0.10, grip=GRIP_OPEN)
    yield from A.move(g, None, 0.8)
    yield from A.set_grip(0.0, 0.6)
    rel = g - hip
    r, th0 = float(np.hypot(rel[1], rel[2])), math.degrees(math.atan2(rel[2], rel[1]))
    r -= ARC_SLACK                      # just inside the head's own circle: it turns in the grip and is not dragged
    ctx.note(f"head taken at {np.round(100 * g, 1)} cm, {100 * r:.1f} cm from the hip axis")
    n = int(seconds * dk.CONTROL_HZ)
    for k in range(1, n + 1):
        t = math.radians(th0 + (to_deg - th0) * dk.smooth(k / n))
        A.aim(np.array([g[0], hip[1] + r * math.cos(t), hip[2] + r * math.sin(t)]), TOOL)
        yield
    ctx.note(f"carried: trunk raised {task.raise_angle(_st(ctx)):.0f} deg")


def let_go(task, ctx):
    A = ctx.left
    yield from A.set_grip(GRIP_OPEN, 0.4)
    yield from A.move(A.pos + (0, 0, 0.05), None, 0.6)
    yield from A.go_home(2.0)


def legs_up(task, ctx):
    """Wrong method: the right arm takes each shin in turn and swings the leg up about the hip and over onto the trunk
    (the trunk stays down)."""
    B = ctx.right
    for s in "rl":                                        # its right leg (nearer the robot), then its left
        st = _st(ctx)
        hip = np.asarray(st.pos, float)
        p, R = D.frames(task.DUMMY, st)[f"{s}_knee"]
        g = p + R @ np.array([0.0, 0.0, -0.05])
        g[2] = max(g[2], task.GROUND + 0.012)
        yield from B.transit(g + (0, 0, 0.04), TOOL, 2.5, height=0.10, grip=0.6)
        yield from B.move(g, None, 0.8)
        yield from B.set_grip(0.0, 0.6)
        rel = g - hip
        r = float(np.hypot(rel[1], rel[2])) - 0.01
        a0 = math.atan2(rel[2], rel[1])
        n = int(4.0 * dk.CONTROL_HZ)
        for k in range(1, n + 1):
            a = a0 + (math.radians(70.0) - a0) * dk.smooth(k / n)     # from pointing -y round over the hips toward +y
            B.aim(np.array([g[0], hip[1] + r * math.cos(a), max(hip[2] + r * math.sin(a), task.GROUND + 0.02)]), TOOL)
            yield
        yield from B.set_grip(0.6, 0.4)
        yield from B.move(B.pos + (0, 0, 0.05), None, 0.6)
    yield from B.go_home(2.0)


def roll_between_rocks(task, ctx):
    """Wrong method: the left arm takes the far shoulder by its upper arm, lifts it and pushes it over toward the near
    rock, as one would roll a body onto its front."""
    A = ctx.left
    fr = D.frames(task.DUMMY, _st(ctx))
    p = fr["l_sh_roll"][0] + fr["l_sh_roll"][1] @ np.array([0.0, 0.0, -0.03])
    p[2] = max(p[2], task.GROUND + 0.012)
    yield from A.transit(p + (0, 0, 0.04), TOOL, 2.5, height=0.10, grip=0.6)
    yield from A.move(p, None, 0.8)
    yield from A.set_grip(0.0, 0.6)
    yield from A.move(p + (-0.10, 0.0, 0.08), None, 3.0)
    yield from A.set_grip(0.6, 0.4)
    yield from A.move(A.pos + (0, 0, 0.05), None, 0.6)
    yield from A.go_home(2.0)


def roll_in_place(task, ctx):
    """Wrong method: roll the body over about its own spine where it lies. The left hand takes the far (left) upper arm
    near the shoulder and carries it round the spine's axis, up and over toward the near rock, 170 degrees."""
    A = ctx.left
    st = _st(ctx)
    fr = D.frames(task.DUMMY, st)
    p = fr["l_sh_roll"][0] + fr["l_sh_roll"][1] @ np.array([0.0, 0.0, -0.025])
    p[2] = max(p[2], task.GROUND + 0.012)
    axis = np.array([float(st.pos[0]), p[1], float(st.pos[2])])       # the spine's line, at the shoulder's y
    yield from A.transit(p + (0, 0, 0.04), TOOL, 2.5, height=0.10, grip=0.6)
    yield from A.move(p, None, 0.8)
    yield from A.set_grip(0.0, 0.6)
    rel = p - axis
    r, a0 = float(np.hypot(rel[0], rel[2])), math.atan2(rel[2], rel[0])
    n = int(6.0 * dk.CONTROL_HZ)
    for k in range(1, n + 1):
        a = a0 + math.radians(170.0) * dk.smooth(k / n)
        A.aim(np.array([axis[0] + r * math.cos(a), p[1], max(axis[2] + r * math.sin(a), task.GROUND + 0.012)]), TOOL)
        yield
    ctx.note(f"rolled: hatch {task.hatch_angle(_st(ctx)):.0f} deg from up")
    yield from let_go(task, ctx)


def script(task, ctx, variant: str = ""):
    yield from dk.wait(0.5)
    if variant == "idle":
        yield from dk.wait(40.0)
        return
    if variant == "legs_up":
        yield from legs_up(task, ctx)
        yield from dk.wait(3.0)
        return
    if variant == "roll_in_place":
        yield from roll_in_place(task, ctx)
        yield from dk.wait(3.0)
        return
    if variant == "roll_between_rocks":
        yield from roll_between_rocks(task, ctx)
        yield from dk.wait(3.0)
        return
    if variant == "sit_and_release":
        yield from carry_head(task, ctx, to_deg=75.0)
        yield from let_go(task, ctx)
    elif variant == "":
        yield from carry_head(task, ctx, seconds=7.0)
        yield from let_go(task, ctx)
    else:
        raise SystemExit(f"variant {variant!r} not written")
    yield from dk.wait(3.0)


def debug(task, ctx) -> str:
    st = _st(ctx)
    return (f"raise {task.raise_angle(st):.0f} hatch {task.hatch_angle(st):.0f} pelvis lift "
            f"{100 * task.pelvis_lift(st):.1f} cm pos ({100 * st.pos[0]:.1f}, {100 * st.pos[1]:.1f}) | {task.report()} | "
            f"L cmd {np.round(100 * ctx.left.pos, 1)} act {np.round(100 * tk.grippers(ctx.emb._env)['left']['points'][4], 1)} "
            f"R cmd {np.round(100 * ctx.right.pos, 1)}")


if __name__ == "__main__":
    dk.main(script, debug)
