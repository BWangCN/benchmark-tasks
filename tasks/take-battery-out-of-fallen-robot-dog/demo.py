"""Scripted full-score demo for the robot-dog battery task, and its near-miss variants.

    python demo.py task.py demo_out                                  # full score: MP4 per camera, setup.png
    python demo.py task.py out --variant push_trunk --no-record      # push the trunk over toward its back
    python demo.py task.py out --variant lift_clear --no-record      # lift the whole robot by its handles
    python demo.py task.py out --variant roll_only --no-record       # turn it onto its back, leave the battery
    python demo.py task.py out --variant pull_propped --no-record    # draw the battery out as it lies, propped
    python demo.py task.py out --variant pull_propped_level --no-record   # the same, the hand level from the arm side
    python demo.py task.py out --variant jolt --no-record            # lift the high side 5 cm and drop it, twice
    python demo.py task.py out --variant idle --no-record            # do nothing

The robot lies along y, head to the robot's left, fallen on its right side and propped on its two right legs, back
up and tilted away from the arms; the belly (and the battery bay in it) faces down.

1. Both hands take the two hip handles of the high (left) side, pointing down, fingers closing across each rung, so
   that the rung turns in the grip like a pin: the left hand the front one, the right hand the rear one.
2. They carry the handles up and over a pivot at the robot's low outer edge, together: the robot rolls across its
   propping legs onto its back. Once it is past its balance point they let go.
3. One hand takes the battery's grip tab (now facing the arms), coming level along the bay's axis, draws it straight
   out of the bay against its catch,
   carries it to the charging pad and sets it down there.

The script reads only the robot's and the battery's poses and joint angles (what a camera-based pose estimate gives)
and the arms' own joint state.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import demokit as dk  # noqa: E402
import quadruped as Q  # noqa: E402
import quadruped_spec as S  # noqa: E402
import trailkit as tk  # noqa: E402

TOOL = dk.tool_rotation((0, 0, -1), (-1, 0, 0))     # pointing down, fingers closing along x
GRIP_OPEN = 0.55
ROLL_DEG = 120.0          # the handles are carried this far round the pivot
LET_GO_BELLY_UP = 70.0    # let go once the belly is within this of straight up


def _st(ctx):
    return ctx.states()["dog"]


def roll_by_handles(task, ctx, roll_deg=ROLL_DEG, seconds=6.0):
    A, B = ctx.left, ctx.right
    st = _st(ctx)
    hp = Q.handle_points(task.DOG, st)
    # the high side's handles: the two with the larger height
    side = "l" if hp["fl"][2] + hp["rl"][2] > hp["fr"][2] + hp["rr"][2] else "r"
    front, rear = hp[f"f{side}"].copy(), hp[f"r{side}"].copy()
    pts = Q.points(task.DOG, st, which={"base"}, tags=("trunk",))
    pivot = np.array([float(pts[:, 0].max()), 0.0, task.GROUND])            # the trunk's low outer edge, on the ground
    ctx.note(f"handles {side}: front {np.round(100 * front, 1)}, rear {np.round(100 * rear, 1)} cm; pivot x "
             f"{100 * pivot[0]:.1f} cm; belly {Q.belly_up_deg(st):.0f} deg from up")
    yield from dk.together(A.transit(front + (0, 0, 0.04), TOOL, 3.0, height=0.12, grip=GRIP_OPEN),
                           B.transit(rear + (0, 0, 0.04), TOOL, 3.0, height=0.12, grip=GRIP_OPEN))
    yield from dk.together(A.move(front, None, 0.8), B.move(rear, None, 0.8))
    yield from dk.together(A.set_grip(0.0, 0.6), B.set_grip(0.0, 0.6))
    arcs = []
    for p in (front, rear):
        rel = p - pivot
        arcs.append((p, float(np.hypot(rel[0], rel[2])), math.atan2(rel[2], rel[0])))
    n = int(seconds * dk.CONTROL_HZ)
    for k in range(1, n + 1):
        d = -math.radians(roll_deg) * dk.smooth(k / n)                       # over toward +x
        for arm, (p, r, a0) in zip((A, B), arcs):
            arm.aim(np.array([pivot[0] + r * math.cos(a0 + d), p[1],
                              max(pivot[2] + r * math.sin(a0 + d), task.GROUND + 0.02)]), TOOL)
        yield
        if Q.belly_up_deg(_st(ctx)) < LET_GO_BELLY_UP:
            break
    ctx.note(f"rolled: belly {Q.belly_up_deg(_st(ctx)):.0f} deg from up; ik misses L {len(A.misses)} R {len(B.misses)}")
    yield from dk.together(A.set_grip(GRIP_OPEN, 0.4), B.set_grip(GRIP_OPEN, 0.4))
    yield from dk.together(A.move(A.pos + (-0.03, 0, 0.06), None, 0.8), B.move(B.pos + (-0.03, 0, 0.06), None, 0.8))
    yield from dk.together(A.go_home(2.0), B.go_home(2.0))
    yield from dk.wait(1.5)


def _seg_dist(pts, a, b):
    ab = b - a
    t = np.clip(((pts - a) @ ab) / float(ab @ ab), 0.0, 1.0)
    return np.linalg.norm(pts - (a + np.outer(t, ab)), axis=1)


def blocking_leg(task, st, tab, out, corridor=0.05):
    """The leg whose thigh or calf comes nearest the line along which the tab is drawn out (within ``corridor``)."""
    best = None
    for leg in Q.LEGS:
        pts = Q.points(task.DOG, st, which={f"{leg}_thigh", f"{leg}_calf"})
        d = float(_seg_dist(pts, tab, tab + 0.15 * out).min())
        if d < corridor and (best is None or d < best[1]):
            best = (leg, d)
    return best


def hold_leg_aside(task, ctx, leg, shared):
    """Left hand: take the blocking leg's calf at its middle and swing it 8 cm along the body, away from the tab; hold
    it there until the battery is out, then let go."""
    A = ctx.left
    st = _st(ctx)
    p, R = Q.frames(task.DOG, st)[f"{leg}_calf"]
    g = p + R @ (Q.CALF_DIR * S.CALF_LEN * 0.5)
    away = np.sign(g[1] - float(st.pos[1])) or 1.0
    rot = A.grip_rotation(g, (0, 0, -1), R @ np.cross(Q.CALF_DIR, (0, 1, 0)))
    yield from A.transit(g + (0, 0, 0.04), rot, 2.5, height=0.14, grip=0.4)
    yield from A.move(g, None, 0.8)
    yield from A.set_grip(0.0, 0.5)
    yield from A.move(g + (0, 0.08 * away, 0.02), None, 1.5)
    shared["leg_held"] = True
    while not shared.get("battery_out"):
        yield
    yield from A.set_grip(0.4, 0.4)
    yield from A.move(A.pos + (0, 0, 0.05), None, 0.6)
    yield from A.go_home(2.0)


def take_battery(task, ctx):
    """If a leg lies across the bay, the left hand holds it aside. The right hand approaches the tab along the bay's
    axis, fingers closing across the tab, draws the battery straight out, turns to point down (the battery hangs
    below), carries it to the charging pad and sets it down there."""
    B = ctx.right
    st, bs = _st(ctx), ctx.states()["battery"]
    tab = bs.to_world((0.0, 0.0, -S.BATTERY_SIZE[2] / 2 - S.TAB_SIZE[2] / 2))
    out = bs.rotation() @ np.array([0.0, 0.0, -1.0])                          # the battery's outward direction
    across = bs.rotation()[:, 1]                                               # the tab is thin along the battery's y
    blk = blocking_leg(task, st, tab, out)
    ctx.note(f"tab at {np.round(100 * tab, 1)} cm, outward {np.round(out, 2)}; belly {Q.belly_up_deg(st):.0f} deg; "
             f"blocking leg {blk}")
    shared = {"leg_held": blk is None, "battery_out": False}

    def battery():
        while not shared["leg_held"]:
            yield
        b_now = ctx.states()["battery"]
        t = b_now.to_world((0.0, 0.0, -S.BATTERY_SIZE[2] / 2 - S.TAB_SIZE[2] / 2))
        o = b_now.rotation() @ np.array([0.0, 0.0, -1.0])
        rot = B.grip_rotation(t, -o, b_now.rotation()[:, 1])
        far = t + 0.15 * o                                     # level with the tab, out along the bay's axis
        yield from B.transit(far, rot, 3.0, height=max(0.04, float(far[2]) - task.TOP), grip=0.4)
        yield from B.move(t + 0.002 * o, None, 1.5)
        yield from B.set_grip(0.0, 0.6)
        yield from B.move(B.pos + 0.06 * o, None, 2.0)
        shared["battery_out"] = True
        yield from B.move(B.pos + (0, 0, 0.04), dk.tool_rotation((0, 0, -1), (1, 0, 0)), 1.5)   # battery along y
        c = np.array([(task.CHARGER[0][0] + task.CHARGER[0][1]) / 2, (task.CHARGER[1][0] + task.CHARGER[1][1]) / 2])
        hang = S.BATTERY_SIZE[2] + S.TAB_SIZE[2] / 2                             # tool point above the battery's bottom
        set_z = task.GROUND + task.CHARGER_H + hang + 0.004
        yield from B.move((c[0], c[1], max(B.pos[2], set_z + 0.05)), None, 3.0)
        yield from B.move((c[0], c[1], set_z), None, 1.5)
        yield from B.set_grip(0.6, 0.5)
        yield from B.move(B.pos + (0, 0, 0.06), None, 0.8)
        yield from B.go_home(2.0)

    if blk is None:
        yield from battery()
    else:
        yield from dk.together(hold_leg_aside(task, ctx, blk[0], shared), battery())


def pull_level(task, ctx):
    """Wrong method: go for the tab as the robot lies propped, the hand level and pointing at the robot (+x), fingers
    across the tab, as far under the belly as it goes, then pull back along the bay's axis."""
    B = ctx.right
    bs = ctx.states()["battery"]
    tab = bs.to_world((0.0, 0.0, -S.BATTERY_SIZE[2] / 2 - S.TAB_SIZE[2] / 2))
    out = bs.rotation() @ np.array([0.0, 0.0, -1.0])
    rot = B.grip_rotation(tab, (1.0, 0.0, -0.2), bs.rotation()[:, 1])
    ctx.note(f"level approach to the tab at {np.round(100 * tab, 1)} cm, outward {np.round(out, 2)}")
    yield from B.transit(tab + (-0.10, 0.0, 0.0), rot, 3.0, height=max(0.04, float(tab[2]) - task.TOP), grip=0.4)
    yield from B.move(tab, None, 2.0)
    yield from B.set_grip(0.0, 0.6)
    yield from B.move(B.pos + 0.06 * out, None, 2.0)
    yield from B.set_grip(0.6, 0.4)
    yield from B.move(B.pos + (-0.05, 0, 0.05), None, 0.8)
    yield from B.go_home(2.0)


def jolt(task, ctx):
    """Wrong method: lift the high side 5 cm by its two handles (under the lift foul) and let it drop, to shake the
    battery out of its downward-facing bay."""
    A, B = ctx.left, ctx.right
    hp = Q.handle_points(task.DOG, _st(ctx))
    side = "l" if hp["fl"][2] + hp["rl"][2] > hp["fr"][2] + hp["rr"][2] else "r"
    f, r = hp[f"f{side}"], hp[f"r{side}"]
    yield from dk.together(A.transit(f + (0, 0, 0.04), TOOL, 3.0, height=0.12, grip=GRIP_OPEN),
                           B.transit(r + (0, 0, 0.04), TOOL, 3.0, height=0.12, grip=GRIP_OPEN))
    yield from dk.together(A.move(f, None, 0.8), B.move(r, None, 0.8))
    yield from dk.together(A.set_grip(0.0, 0.6), B.set_grip(0.0, 0.6))
    for _ in range(2):
        yield from dk.together(A.move(A.pos + (0, 0, 0.05), None, 1.0), B.move(B.pos + (0, 0, 0.05), None, 1.0))
        yield from dk.together(A.set_grip(1.0, 0.1), B.set_grip(1.0, 0.1))
        yield from dk.wait(1.0)
        hp = Q.handle_points(task.DOG, _st(ctx))
        f, r = hp[f"f{side}"], hp[f"r{side}"]
        yield from dk.together(A.move(f + (0, 0, 0.03), None, 1.0), B.move(r + (0, 0, 0.03), None, 1.0))
        yield from dk.together(A.move(f, None, 0.6), B.move(r, None, 0.6))
        yield from dk.together(A.set_grip(0.0, 0.5), B.set_grip(0.0, 0.5))
    yield from dk.together(A.set_grip(GRIP_OPEN, 0.3), B.set_grip(GRIP_OPEN, 0.3))
    yield from dk.together(A.go_home(2.0), B.go_home(2.0))


def push_trunk(task, ctx):
    """Wrong method: the right hand pushes the trunk's top edge over toward the robot's back (+x), 10 s."""
    B = ctx.right
    t = math.radians(15.0)
    tool = dk.tool_rotation((math.sin(t), 0.0, -math.cos(t)), (0, 1, 0))

    def target(d):
        st = _st(ctx)
        pts = Q.points(task.DOG, st, which={"base"}, tags=("trunk",))
        top = pts[np.argmax(pts[:, 2])]
        return np.array([top[0] - 0.012 + d, float(st.pos[1]), max(top[2] - 0.015, task.GROUND + 0.02)])

    yield from B.transit(target(-0.03), tool, 3.0, height=0.14, grip=0.0)
    d = -0.03
    for _ in range(int(10 * dk.CONTROL_HZ)):
        d = min(d + 0.02 / dk.CONTROL_HZ, 0.12)
        B.aim(target(d), tool)
        yield
    yield from B.move(B.pos + (-0.05, 0, 0.05), None, 0.8)
    yield from B.go_home(2.0)


def lift_clear(task, ctx):
    """Wrong method: both hands lift the robot clear by two handles, 6 cm, and set it down again."""
    A, B = ctx.left, ctx.right
    hp = Q.handle_points(task.DOG, _st(ctx))
    side = "l" if hp["fl"][2] + hp["rl"][2] > hp["fr"][2] + hp["rr"][2] else "r"
    f, r = hp[f"f{side}"], hp[f"r{side}"]
    yield from dk.together(A.transit(f + (0, 0, 0.04), TOOL, 3.0, height=0.12, grip=GRIP_OPEN),
                           B.transit(r + (0, 0, 0.04), TOOL, 3.0, height=0.12, grip=GRIP_OPEN))
    yield from dk.together(A.move(f, None, 0.8), B.move(r, None, 0.8))
    yield from dk.together(A.set_grip(0.0, 0.6), B.set_grip(0.0, 0.6))
    yield from dk.together(A.move(A.pos + (0, 0, 0.10), None, 3.0), B.move(B.pos + (0, 0, 0.10), None, 3.0))
    yield from dk.together(A.move(A.pos + (0, 0, -0.10), None, 3.0), B.move(B.pos + (0, 0, -0.10), None, 3.0))
    yield from dk.together(A.set_grip(GRIP_OPEN, 0.4), B.set_grip(GRIP_OPEN, 0.4))
    yield from dk.together(A.go_home(2.0), B.go_home(2.0))


def script(task, ctx, variant: str = ""):
    yield from dk.wait(2.0)                                  # it settles onto its legs
    if variant == "idle":
        yield from dk.wait(40.0)
        return
    if variant == "push_trunk":
        yield from push_trunk(task, ctx)
    elif variant == "jolt":
        yield from jolt(task, ctx)
    elif variant == "lift_clear":
        yield from lift_clear(task, ctx)
    elif variant == "pull_propped":
        yield from take_battery(task, ctx)
    elif variant == "pull_propped_level":
        yield from pull_level(task, ctx)
    elif variant in ("", "roll_only"):
        yield from roll_by_handles(task, ctx)
        if variant == "":
            yield from take_battery(task, ctx)
    else:
        raise SystemExit(f"variant {variant!r} not written")
    yield from dk.wait(2.0)


def debug(task, ctx) -> str:
    st, bs = _st(ctx), ctx.states()["battery"]
    return (f"belly {Q.belly_up_deg(st):.0f} pos ({100 * st.pos[0]:.1f}, {100 * st.pos[1]:.1f}, {100 * (st.pos[2] - task.GROUND):.1f}) "
            f"bat out {1000 * Q.battery_out(st, bs):.0f} mm at ({100 * bs.pos[0]:.1f}, {100 * bs.pos[1]:.1f}, "
            f"{100 * (bs.pos[2] - task.GROUND):.1f}) | {task.report()} | L {np.round(100 * ctx.left.pos, 1)} "
            f"R {np.round(100 * ctx.right.pos, 1)}")


if __name__ == "__main__":
    dk.main(script, debug)
