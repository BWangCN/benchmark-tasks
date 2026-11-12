"""Scripted full-score demo for taking out the spare prop, and its wrong-method variants.

    python demo.py task.py demo_out                                  # full score
    python demo.py task.py out --variant take_nearest --no-record    # take the right arm's nearest prop untested
    python demo.py task.py out --variant press_corners --no-record   # press each slab corner to find the loose one
    python demo.py task.py out --variant idle --no-record            # do nothing

1. Each prop in turn is nudged sideways (along x, outward) by closed fingers aimed 5 mm into its inner face for
   1.5 s, the same command for all four: a prop that carries the slab is held by the slab's weight and moves under
   half a millimetre; the spare one slides several millimetres.
2. The arm on the spare prop's side grips its outer half (fingers across it), draws it out along y, lifts it,
   carries it to the pad, steering it on its measured position while it hangs a few millimetres above, and lets go.

The script reads only the poses of the props, the slab, and the arms' own joint state. SPARE_CORNER
(0 to 3) in the environment fixes the spare prop's corner for a run; unset, every reset picks one at random.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import demokit as dk  # noqa: E402
import trailkit as tk  # noqa: E402

HOVER = 0.006
NUDGE_MM = 1.5            # a prop that moved more than this under the nudge carries nothing


def tcp_for_tip(point, rot):
    from yam_arena.constants import PAD_TIP_PAST_TCP
    return np.asarray(point, float) - rot[:, 2] * PAD_TIP_PAST_TCP


def _s(ctx, name) -> tk.BodyState:
    return ctx.states()[name]


def corner_of(task, ctx, k: int) -> tuple[int, int]:
    p = _s(ctx, f"prop_{k}").pos
    return (1 if p[0] > task.X0 else -1, 1 if p[1] > task.Y0 else -1)


def nudge(task, ctx, k: int, moved_mm: dict):
    """Push prop k outward along x for 1.5 s; record in ``moved_mm`` how far it moved."""
    sx, sy = corner_of(task, ctx, k)
    A = ctx.left if sy > 0 else ctx.right
    p0 = np.array(_s(ctx, f"prop_{k}").pos)
    d = np.array([sx, 0.0, 0.0])
    rot = dk.tool_rotation((0, 0, -1), (1, 0, 0))
    face = np.array([p0[0] - sx * task.PROP[0] / 2, task.Y0 + sy * (task.SLAB[1] / 2 + 0.028), task.GROUND + 0.02])
    yield from A.set_grip(0.0, 0.3)
    yield from A.transit(tcp_for_tip(face - d * 0.025, rot), rot, 3.0, height=0.12)
    yield from A.move(tcp_for_tip(face - d * 0.012, rot), rot, 0.6)
    for _ in range(int(1.5 * dk.CONTROL_HZ)):
        A.aim(tcp_for_tip(face + d * 0.005, rot), rot)
        yield
    moved = 1000 * float(np.linalg.norm((np.array(_s(ctx, f"prop_{k}").pos) - p0)[:2]))
    yield from A.move(tcp_for_tip(face - d * 0.03, rot), rot, 0.5)
    yield from A.move(A.pos + (0, 0, 0.08), None, 0.5)
    ctx.note(f"prop_{k} at corner {(sx, sy)} nudged: moved {moved:.2f} mm")
    moved_mm[k] = moved


def take(task, ctx, k: int):
    """Grip prop k by its outer half, draw it out along y, carry it to the pad and let go."""
    sx, sy = corner_of(task, ctx, k)
    A = ctx.left if sy > 0 else ctx.right
    p0 = np.array(_s(ctx, f"prop_{k}").pos)
    rot = A.grip_rotation(tcp_for_tip(p0, dk.DOWN_X), (0, 0, -1), (1, 0, 0))   # fingers across it, along x
    tip = np.array([p0[0], task.Y0 + sy * (task.SLAB[1] / 2 + 0.03), task.GROUND + 0.015])
    yield from A.transit(tcp_for_tip(tip, rot) + (0, 0, 0.04), rot, 3.0, height=0.12, grip=1.0)
    yield from A.move(tcp_for_tip(tip, rot), rot, 0.8)
    yield from A.set_grip(0.0, 0.6)
    yield from A.move(A.pos + (0, sy * 0.08, 0), rot, 2.0)              # out from under, along the ground
    ctx.note(f"prop_{k} drawn out: {np.round(_s(ctx, f'prop_{k}').pos, 3)}")
    yield from A.move(A.pos + (0, 0, 0.05), rot, 0.8)
    held = A.rot.T @ (np.array(_s(ctx, f"prop_{k}").pos) - A.pos)
    goal = np.array([(task.PAD[0][0] + task.PAD[0][1]) / 2, (task.PAD[1][0] + task.PAD[1][1]) / 2, task.PAD_TOP + task.PROP[2] / 2])
    over = goal + (0, 0, HOVER)
    yield from A.transit(over - rot @ held + (0, 0, 0.04), rot, 4.0, height=0.12)
    yield from A.move(over - rot @ held, rot, 0.8)
    for _ in range(int(3.0 * dk.CONTROL_HZ)):
        e = np.array(_s(ctx, f"prop_{k}").pos) - over
        if np.linalg.norm(e) < 0.0015:
            break
        A.aim(A.pos - np.clip(e, -0.002, 0.002), rot)
        yield
    yield from A.move(A.pos - (0, 0, HOVER), rot, 0.5)
    yield from A.set_grip(1.0, 0.4)
    yield from A.move(A.pos + (0, 0, 0.06), rot, 0.6)
    ctx.note(f"prop_{k} let go at {np.round(_s(ctx, f'prop_{k}').pos, 3)} (goal {np.round(goal, 3)})")
    yield from A.go_home(2.5)


def press_corners(task, ctx):
    """The way to find a short foot under a rocking base: press each corner of the slab down and feel which rocks."""
    hx, hy = task.SLAB[0] / 2, task.SLAB[1] / 2
    rot = dk.tool_rotation((0, 0, -1), (1, 0, 0))
    for sx, sy in task.CORNERS:
        A = ctx.left if sy > 0 else ctx.right
        top = np.array([task.X0 + sx * (hx - 0.015), task.Y0 + sy * (hy - 0.015), task.SLAB_Z + task.SLAB[2]])
        yield from A.set_grip(0.0, 0.3)
        yield from A.transit(tcp_for_tip(top + (0, 0, 0.02), rot), rot, 3.0, height=0.14)
        yield from A.move(tcp_for_tip(top - (0, 0, 0.004), rot), rot, 1.0)       # pressed 4 mm into the top
        yield from dk.wait(0.5)
        yield from A.move(tcp_for_tip(top + (0, 0, 0.03), rot), rot, 0.5)
        ctx.note(f"pressed corner {(sx, sy)}")
    yield from dk.together(ctx.left.go_home(2.0), ctx.right.go_home(2.0))


def script(task, ctx, variant: str = ""):
    yield from dk.wait(0.5)
    if variant == "idle":
        yield from dk.wait(30.0)
        return
    if variant == "take_nearest":
        k = min(range(4), key=lambda i: float(np.linalg.norm(_s(ctx, f"prop_{i}").pos[:2] - np.array([0.20, -0.15]))))
        yield from take(task, ctx, k)
        yield from dk.wait(3.0)
        return
    if variant == "press_corners":
        yield from press_corners(task, ctx)
        k = min(range(4), key=lambda i: float(np.linalg.norm(_s(ctx, f"prop_{i}").pos[:2] - np.array([0.20, -0.15]))))
        yield from take(task, ctx, k)
        yield from dk.wait(3.0)
        return
    moved_mm: dict = {}
    order = sorted(range(4), key=lambda i: corner_of(task, ctx, i)[::-1])     # the right arm's two, then the left's
    for k in order:
        yield from nudge(task, ctx, k, moved_mm)
        if moved_mm[k] > NUDGE_MM:
            break
    yield from dk.together(ctx.left.go_home(2.0), ctx.right.go_home(2.0))
    spare = max(moved_mm, key=moved_mm.get)
    ctx.note(f"spare: prop_{spare} ({moved_mm[spare]:.2f} mm)")
    yield from take(task, ctx, spare)
    yield from dk.wait(3.0)


def debug(task, ctx) -> str:
    s = ctx.states()
    return " ".join(f"{n} {np.round(s[n].pos, 3)}" for n in ("prop_0", "prop_1", "prop_2", "prop_3", "slab"))


if __name__ == "__main__":
    dk.main(script, debug)
