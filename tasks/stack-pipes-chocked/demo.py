"""Scripted full-score demo for stacking the pipes, and its near-miss variants.

    python demo.py task.py demo_out                                  # full score
    python demo.py task.py out --variant no_chocks --no-record       # stack them with no chocks
    python demo.py task.py out --variant one_chock --no-record       # chock one side only
    python demo.py task.py out --variant chocks_after --no-record    # set the top pipe, then both chocks
    python demo.py task.py out --variant idle --no-record            # do nothing

1. The left arm lays both chocks along the pad, one on each side, before anything else is there.
2. The right arm lays each bottom pipe on the pad 1.2 cm inside its chock (room for the fingers) and rolls it out
   against the chock (closed fingers on its side).
3. The right arm sets the third pipe into the groove between them and lets go.

Every piece is steered on its measured position while it is held 6 mm above its place (the loaded arm sags), then
lowered and let go. The script reads only the poses of the pipes and chocks and the arms' own joint state.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import demokit as dk  # noqa: E402
import trailkit as tk  # noqa: E402

HOVER = 0.006
SERVO_S = 3.0
BOTTOM_Y = 0.03                       # the bottom pipes' axes at y -3 and +3 cm


def pad():
    from yam_arena.constants import PAD_TIP_PAST_TCP
    return PAD_TIP_PAST_TCP


def tcp_for_tip(point, rot):
    return np.asarray(point, float) - rot[:, 2] * pad()


def _s(ctx, name) -> tk.BodyState:
    return ctx.states()[name]


def carry(task, ctx, A, name: str, goal, grip_open: float, below: float, take_open: float = 1.0):
    """Take body ``name`` from above at its middle (tip ``below`` under its centre, fingers closing along y), carry it
    so that its centre is at ``goal`` (world), steering it on its measured position at HOVER above, lower, let go."""
    rot = A.grip_rotation(tcp_for_tip(_s(ctx, name).pos, dk.DOWN_X), (0, 0, -1), (0, 1, 0))
    tip = _s(ctx, name).pos - np.array([0, 0, below])
    yield from A.transit(tcp_for_tip(tip, rot) + (0, 0, 0.04), rot, 3.0, height=0.14, grip=take_open)
    yield from A.move(tcp_for_tip(tip, rot), rot, 0.8)
    for _ in range(int(1.5 * dk.CONTROL_HZ)):                 # the hand's real tip (its own joints) onto the middle
        real = np.asarray(tk.grippers(ctx.emb._env)[A.side]["points"][-1], float)
        e = real - tip
        if np.linalg.norm(e) < 0.002:
            break
        A.aim(A.pos - np.clip(e, -0.003, 0.003), rot)
        yield
    yield from A.set_grip(0.0, 0.6)
    ctx.note(f"{name} taken: {1000 * float(np.linalg.norm(_s(ctx, name).pos[:2] - tip[:2])):.1f} mm from the aim")
    yield from A.move(A.pos + (0, 0, 0.05), rot, 0.8)
    goal = np.asarray(goal, float)
    over = goal + (0, 0, HOVER)
    held = A.rot.T @ (_s(ctx, name).pos - A.pos)
    ctx.note(f"{name} lifted: {np.round(_s(ctx, name).pos, 3)}, in the hand at {np.round(1000 * held, 1)} mm")
    yield from A.transit(over - rot @ held + (0, 0, 0.04), rot, 4.0, height=0.14)
    ctx.note(f"{name} carried: {np.round(_s(ctx, name).pos, 3)}, in the hand at "
             f"{np.round(1000 * (A.rot.T @ (_s(ctx, name).pos - A.pos)), 1)} mm")
    yield from A.move(over - rot @ held, rot, 0.8)
    for _ in range(int(SERVO_S * dk.CONTROL_HZ)):
        e = _s(ctx, name).pos - over
        if np.linalg.norm(e) < 0.0015:
            break
        A.aim(A.pos - np.clip(e, -0.002, 0.002), rot)
        yield
    ctx.note(f"{name} steered: {1000 * float(np.linalg.norm(_s(ctx, name).pos - over)):.1f} mm off, held at "
             f"{np.round(_s(ctx, name).pos, 3)}, hand {np.round(A.pos, 3)}")
    yield from A.move(A.pos - (0, 0, HOVER), rot, 0.5)
    ctx.note(f"{name} lowered: {np.round(_s(ctx, name).pos, 3)}")
    yield from A.set_grip(grip_open, 0.4)
    yield from A.move(A.pos + (0, 0, 0.06), rot, 0.6)
    ctx.note(f"{name} at {np.round(_s(ctx, name).pos, 3)} (goal {np.round(goal, 3)})")


CHOCK_OPEN = 0.48                      # released: about 45 mm between the fingers for the 40 mm chock
PIPE_OPEN = 0.57                       # released: about 54 mm for the 50 mm pipe
FINGER_ROOM = 0.003 + 0.006 + 0.002    # a finger's inner face 3 mm off the piece, about 6 mm thick, 2 mm of air
CHOCK_A_Y = -0.0765                    # chock A's place (mirrored to +y): its face 5.65 cm out from the pad's middle


def chock_beside(task, ctx, pipe: str, side: int) -> np.ndarray:
    """A chock's place beside a pipe's outer side, as close as the fingers allow."""
    p = _s(ctx, pipe).pos
    return np.array([p[0], p[1] + side * (task.R + FINGER_ROOM + task.CHOCK[1] / 2), task.GROUND + task.CHOCK[2] / 2])


def roll(task, ctx, A, name: str, direction: int, limit=0.06):
    """Closed fingers pointing down, tip against the pipe's side at its axis height: push it along y in
    ``direction`` until it stops (against a chock) or has gone ``limit``."""
    rot = dk.tool_rotation((0, 0, -1), (1, 0, 0))
    p = _s(ctx, name).pos
    y0 = p[1]
    tip = lambda q: np.array([q[0], q[1] - direction * (task.R + 0.006), task.PAD_TOP + task.R])   # noqa: E731
    yield from A.set_grip(0.0, 0.3)
    yield from A.transit(tcp_for_tip(tip(p) - (0, direction * 0.02, 0), rot), rot, 3.0, height=0.12)
    yield from A.move(tcp_for_tip(tip(p), rot), rot, 0.6)
    past = []
    for _ in range(int(8 * dk.CONTROL_HZ)):
        q = _s(ctx, name).pos
        past = (past + [q[1]])[-11:]
        if abs(q[1] - y0) > limit or (len(past) == 11 and abs(past[-1] - past[0]) < 0.0005 and abs(q[1] - y0) > 0.003):
            break
        A.aim(tcp_for_tip(tip(q) + (0, direction * 0.006, 0), rot), rot)
        yield
    yield from A.move(A.pos - (0, direction * 0.02, 0) + (0, 0, 0.04), rot, 0.6)
    ctx.note(f"{name} rolled to y {_s(ctx, name).pos[1]:.4f}")


CHOCK_Y = 0.0835                       # the chocks' places: their inner faces 6.35 cm from the pad's middle line


def build(task, ctx, L, R, chock_a=True, chock_b=True, top_first=False):
    """Chocks first (nothing beside them yet), then each bottom pipe set between them and rolled out against its
    chock (room for the fingers), then the top pipe into the groove."""
    x = 0.25
    if chock_a:
        yield from carry(task, ctx, L, "chock_0", (x, -CHOCK_Y, task.GROUND + task.CHOCK[2] / 2), 1.0, 0.008)
    if chock_b and not top_first:
        yield from carry(task, ctx, L, "chock_1", (x, CHOCK_Y, task.GROUND + task.CHOCK[2] / 2), 1.0, 0.008)
    yield from L.go_home(2.0)
    fa = _s(ctx, "chock_0").pos[1] + task.CHOCK[1] / 2 if chock_a else -0.0635
    fb = _s(ctx, "chock_1").pos[1] - task.CHOCK[1] / 2 if (chock_b and not top_first) else 0.0635
    yield from carry(task, ctx, R, "pipe_0", (x, fa + task.R + 0.012, task.PAD_TOP + task.R), PIPE_OPEN, 0.005)
    if chock_a:
        yield from roll(task, ctx, R, "pipe_0", -1, limit=0.03)
    yield from carry(task, ctx, R, "pipe_1", (x, fb - task.R - 0.012, task.PAD_TOP + task.R), PIPE_OPEN, 0.005)
    if chock_b and not top_first:
        yield from roll(task, ctx, R, "pipe_1", 1, limit=0.03)
    ctx.note(f"bottom pair at y {_s(ctx, 'pipe_0').pos[1]:.4f}, {_s(ctx, 'pipe_1').pos[1]:.4f}")
    yield from top(task, ctx, R)
    if top_first and chock_b:
        yield from carry(task, ctx, L, "chock_1", chock_beside(task, ctx, "pipe_1", 1), CHOCK_OPEN, 0.008)
        yield from L.go_home(2.0)


def top(task, ctx, R):
    a, b = _s(ctx, "pipe_0").pos, _s(ctx, "pipe_1").pos
    d = abs(b[1] - a[1])
    h = task.R + math.sqrt(max((2 * task.R) ** 2 - (d / 2) ** 2, 0.0))
    yield from carry(task, ctx, R, "pipe_2", ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, task.PAD_TOP + h + 0.002),
                     PIPE_OPEN, 0.005)
    yield from R.go_home(2.0)


def script(task, ctx, variant: str = ""):
    L, R = ctx.left, ctx.right
    yield from dk.wait(0.5)
    if variant == "idle":
        yield from dk.wait(30.0)
        return
    if variant == "no_chocks":
        yield from build(task, ctx, L, R, chock_a=False, chock_b=False)
    elif variant == "one_chock":
        yield from build(task, ctx, L, R, chock_b=False)
    elif variant == "chocks_after":
        yield from build(task, ctx, L, R, chock_a=False, chock_b=False)    # top pipe on, then both chocks
        yield from carry(task, ctx, L, "chock_0", chock_beside(task, ctx, "pipe_0", -1), CHOCK_OPEN, 0.008)
        yield from carry(task, ctx, L, "chock_1", chock_beside(task, ctx, "pipe_1", 1), CHOCK_OPEN, 0.008)
        yield from L.go_home(2.0)
    else:
        yield from build(task, ctx, L, R)
    yield from dk.wait(6.0)
    ctx.note("speeds: " + " ".join(f"{n} {1000 * _s(ctx, n).speed():.1f}mm/s {_s(ctx, n).spin():.3f}rad/s" for n in ("pipe_0", "pipe_1", "pipe_2")))
    ctx.note("end positions: " + " ".join(f"{n} {np.round(_s(ctx, n).pos[1:], 3)}" for n in ("pipe_0", "pipe_1", "pipe_2", "chock_0", "chock_1")))
    ctx.note(f"end: {task.stacked([_s(ctx, f'pipe_{i}') for i in range(3)], [_s(ctx, f'chock_{i}') for i in range(2)])}")


def debug(task, ctx) -> str:
    s = ctx.states()
    return " ".join(f"{n} {np.round(s[n].pos, 3)}" for n in ("pipe_0", "pipe_1", "pipe_2", "chock_0", "chock_1"))


if __name__ == "__main__":
    dk.main(script, debug)
