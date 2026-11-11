"""Scripted full-score demo for the turn-onto-its-back task, and its near-miss variants.

    python demo.py task.py demo_out                                  # full score: MP4 per camera, setup.png
    python demo.py task.py out --variant roll_only --no-record       # roll it over without moving its arm first
    python demo.py task.py out --variant lift_and_flip --no-record   # pick the body up and turn it in the air
    python demo.py task.py out --variant idle --no-record            # do nothing

The body lies face down along y, head to the robot's left; it is rolled toward the robot, about its near edge, so
the arm on that side (the dummy's left arm) is the one that would be trapped.

1. Clear the near arm: the left hand takes the near forearm (fingers about 4 cm apart, closing across it), carries it
   round the shoulder just off the ground from 20 to 80 degrees out from the side and lays it down; the right hand
   takes it there and carries it on to 120 degrees, beside the head. Each carry runs on an arc about the shoulder,
   just inside full stretch, as a limp arm moves.
2. Roll: the left hand grips the head across its width (the head is rigid with the trunk) and carries it 68 degrees
   round the body's near edge, turning with it. From 30 degrees on, the right hand comes onto the trunk's highest far
   edge, fingers closed, and pushes; once it is pressing, the left hand lets go and backs out along its fingers, and
   the right hand pushes the body past its side onto its back.
3. Both hands withdraw.

The script reads only the dummy's pose and joint angles (what a camera-based pose estimate gives) and the arms' own
joint state.
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

FINGER_OFF = 0.012        # m: tool point to the closed fingers' outer face
PUSH_MIN = 0.020          # m: tool point at least this far above the ground (fingertips clear of it)
PRESS_RATE = 0.02         # m/s: how fast the push deepens while the body resists
PRESS_MAX = 0.05          # m
PUSH_BELOW = 0.008        # m: tool point this far below the trunk's highest far edge


def _st(ctx) -> tk.BodyState:
    return ctx.states()["dummy"]


def trunk_box(st: tk.BodyState, tag: str = "trunk") -> np.ndarray:
    """World corners of the trunk (or pelvis) box."""
    s = next(s for s in D.base_shapes() if s.tag == tag)
    c, h = np.asarray(s.centre), np.asarray(s.size) / 2
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c
    return st.pos + corners @ st.rotation().T


HEAD_GRIP_Y = 0.012       # m: grip the head this far from its top end (the pads clear the shoulders)
HEAD_GRIP_Z = 0.022       # m: tool point above the ground at the grip
CARRY_DEG = 68.0          # the left hand carries the head this far round the body's near edge
CARRY_S = 6.0             # s
PUSH_FROM_DEG = 30.0      # the right hand comes in once the body has turned this far
PUSH_Y = 0.04             # m: the right hand pushes the trunk here along the body (15 cm from the left hand)


def push_target(task, st, d):
    """Right hand: on the trunk's highest far edge, ``d`` metres in from it."""
    tb = trunk_box(st)
    c = tb[np.argsort(tb[:, 2])[-4:]]                   # the four highest corners: the edge that leads the turn
    corner = c[np.argmax(c[:, 0])]                      # on the far side
    z = max(task.GROUND + PUSH_MIN, corner[2] - PUSH_BELOW)
    return np.array([corner[0] + FINGER_OFF - d, PUSH_Y, z])


def push_tool(x: float) -> np.ndarray:
    """Right hand: pointing down, tilted 15 degrees toward the robot, fingers closing along x."""
    t = math.radians(15.0)
    return dk.tool_rotation((-math.sin(t), 0.0, -math.cos(t)), (1, 0, 0))


def carry_tool(phi: float) -> np.ndarray:
    t = math.radians(phi)
    return dk.tool_rotation((math.sin(t), 0.0, -math.cos(t)), (math.cos(t), 0.0, math.sin(t)))


def roll(task, ctx, seconds: float = 16.0):
    """Turn the body toward the robot about its near edge. The left hand grips the head across its width and carries
    it round (the head is rigid with the trunk). Once the body has turned 30 degrees the right hand comes onto the
    trunk's highest far edge and pushes; when it is pressing, the left hand lets go and backs out along its fingers,
    and the right hand pushes the body past its side onto its back."""
    A, R = ctx.left, ctx.right
    st = _st(ctx)
    head = D.points(task.DUMMY, st, which={"base"}, tags=("head",))
    grip = np.array([float(head[:, 0].mean()), float(head[:, 1].max()) - HEAD_GRIP_Y - 0.0135,
                     task.GROUND + HEAD_GRIP_Z])
    pivot = np.array([float(trunk_box(st)[:, 0].min()), grip[1], task.GROUND])
    ctx.note(f"head grip at {np.round(100 * grip, 1)} cm, turning about x {100 * pivot[0]:.1f} cm")
    yield from A.via(grip, carry_tool(0.0), 3.0, lift=0.06, grip=1.0, ceiling=task.GROUND + 0.10)
    yield from A.set_grip(0.0, 0.8)
    shared = {"pressing": False, "done": False}

    def carry():
        rel = grip - pivot
        n = int(CARRY_S * dk.CONTROL_HZ)
        for k in range(1, n + 1):
            phi = CARRY_DEG * dk.smooth(k / n)
            c, s_ = math.cos(math.radians(phi)), math.sin(math.radians(phi))
            A.aim(pivot + np.array([rel[0] * c - rel[2] * s_, 0.0, rel[2] * c + rel[0] * s_]), carry_tool(phi))
            yield
        while not shared["pressing"] and not shared["done"]:
            yield
        ctx.note(f"left hand lets go (chest {task.chest_angle(_st(ctx)):.0f} deg from up)")
        yield from A.set_grip(1.0, 0.25)
        yield from A.move(A.pos + 0.06 * -carry_tool(CARRY_DEG)[:, 2], None, 0.5)

    def push():
        while task.chest_angle(_st(ctx)) > 180.0 - PUSH_FROM_DEG:
            yield
        p = push_target(task, _st(ctx), -0.015)
        yield from R.via(p, push_tool(p[0]), 1.5, lift=0.03, grip=0.0, ceiling=p[2] + 0.03)
        d = -0.015
        for _ in range(int(seconds * dk.CONTROL_HZ)):
            st = _st(ctx)
            if task.chest_angle(st) < 50.0:
                break
            d = min(d + PRESS_RATE / dk.CONTROL_HZ, PRESS_MAX)
            shared["pressing"] = shared["pressing"] or d > 0.008
            p = push_target(task, st, d)
            R.aim(p, push_tool(p[0]))
            yield
        shared["done"] = True

    yield from dk.together(carry(), push())
    ctx.note(f"rolled: chest {task.chest_angle(_st(ctx)):.0f} deg from up, ik misses L {len(A.misses)} R {len(R.misses)}")
    yield from dk.together(A.move(A.pos + (0, 0.03, 0.06), None, 0.6), R.move(R.pos + (0.03, 0, 0.04), None, 0.6))
    yield from dk.together(A.go_home(2.0), R.go_home(2.0))


SWEEP_R = 0.10            # m: grip the near forearm this far from the shoulder (4 cm past the elbow)
SWEEP_SLACK = 0.006       # m: the grip runs this far inside full stretch, so it does not pull the body
SWEEP_LIFT = 0.012        # m: carry the forearm this far off the ground
SWEEP_HANDOVER = 80.0     # deg out from the side: the left hand reaches to here, the right hand from here
SWEEP_END = 120.0         # deg: the arm ends beside the head


def sweep_point(sh, ab, r, z):
    a = math.radians(ab)
    d = np.array([-math.sin(a), -math.cos(a), 0.0])
    return np.array([sh[0], sh[1], z]) + r * d, d


def sweep_tool(d, tilt_deg, flip=False):
    """Pointing down, tilted along the arm by ``tilt_deg`` (negative: fingertips toward the shoulder), fingers
    closing across the forearm; ``flip`` turns the hand half round (the right arm's wrist needs it)."""
    t = math.radians(tilt_deg)
    R = dk.tool_rotation(np.array([0.0, 0.0, -math.cos(t)]) + math.sin(t) * d, (d[1], -d[0], 0.0))
    return R @ np.diag([-1.0, -1.0, 1.0]) if flip else R


def sweep_arm(task, ctx, arm, a0, a1, tilt0, tilt1, seconds, flip=False):
    """One hand takes the near forearm at ``a0`` degrees out from the side, carries it round the shoulder, just off
    the ground, to ``a1`` and lays it down."""
    st = _st(ctx)
    sh = D.frames(task.DUMMY, st)["l_sh_roll"][0]
    z = task.GROUND + 0.020
    p, d = sweep_point(sh, a0, SWEEP_R, z)
    yield from arm.via(p + (0, 0, 0.04), sweep_tool(d, 0.0, flip), 2.5, lift=0.05, grip=GRIP_HALF,
                       ceiling=task.GROUND + 0.10)
    yield from arm.move(p, sweep_tool(d, tilt0, flip), 0.8)
    yield from arm.set_grip(0.0, 0.6)
    n = max(1, int(seconds * dk.CONTROL_HZ))
    for k in range(1, n + 1):
        s = dk.smooth(k / n)
        lift = SWEEP_LIFT * math.sin(math.pi * min(1.0, 1.5 * s)) if s < 2 / 3 else 0.0
        p, d = sweep_point(sh, a0 + (a1 - a0) * s, SWEEP_R - SWEEP_SLACK * math.sin(math.pi * s), z + lift)
        arm.aim(p, sweep_tool(d, tilt0 + (tilt1 - tilt0) * s, flip))
        yield
    yield from arm.set_grip(GRIP_HALF, 0.5)
    yield from arm.move(arm.pos + (0, 0, 0.05), None, 0.6)


def arm_overhead(task, ctx):
    """Move the near arm up beside the head first, so the body does not roll onto it."""
    yield from sweep_arm(task, ctx, ctx.left, ARM_START_DEG, SWEEP_HANDOVER, 0.0, 0.0, 3.0)
    yield from ctx.left.go_home(1.5)
    yield from sweep_arm(task, ctx, ctx.right, SWEEP_HANDOVER, SWEEP_END, -30.0, -15.0, 2.5, flip=True)
    yield from ctx.right.go_home(1.5)
    ctx.note(f"near arm moved: {np.round(np.degrees([_st(ctx).joints[k] for k in ('l_sh_pitch', 'l_sh_roll')]), 0)} deg")


GRIP_HALF = 0.42          # fingers about 4 cm apart
ARM_START_DEG = 20.0      # the near arm's start angle out from the side (task.ARMS_OUT)


LIFT_THIGH_ALONG = 0.030  # m: lift_and_flip grips the far thigh this far from the hip joint
LIFT_H = 0.07             # m: lift_and_flip raises the body this far


def lift_and_flip(task, ctx):
    """Wrong method: pick the body up (left hand on the head, right hand on the far thigh by the hip), raise it 7 cm,
    turn the head round in the air, and let go."""
    A, R = ctx.left, ctx.right
    st = _st(ctx)
    head = D.points(task.DUMMY, st, which={"base"}, tags=("head",))
    hg = np.array([float(head[:, 0].mean()), float(head[:, 1].max()) - HEAD_GRIP_Y - 0.0135, task.GROUND + HEAD_GRIP_Z])
    hp, hR = D.frames(task.DUMMY, st)["r_hip_roll"]
    tg = hp + hR @ np.array([0.0, 0.0, -LIFT_THIGH_ALONG])
    tg[2] = max(tg[2], task.GROUND + 0.020)
    yield from A.via(hg, carry_tool(0.0), 3.0, lift=0.06, grip=1.0, ceiling=task.GROUND + 0.10)
    yield from R.via(tg + (0, 0, 0.04), dk.DOWN_X, 2.5, lift=0.05, grip=GRIP_HALF, ceiling=task.GROUND + 0.10)
    yield from R.move(tg, dk.DOWN_X, 0.8)
    yield from dk.together(A.set_grip(0.0, 0.8), R.set_grip(0.0, 0.8))
    yield from dk.together(A.move(A.pos + (0, 0, LIFT_H), None, 2.0), R.move(R.pos + (0, 0, LIFT_H), None, 2.0))
    ctx.note(f"lifted: trunk {100 * task.trunk_lift(_st(ctx)):.1f} cm off the ground")
    yield from R.set_grip(GRIP_HALF, 0.4)
    yield from R.move(R.pos + (0, 0, 0.04), None, 0.6)
    pivot = np.array([float(trunk_box(_st(ctx))[:, 0].min()), hg[1], task.GROUND + LIFT_H])
    rel = A.pos - pivot
    for k in range(1, 81):
        phi = 60.0 * dk.smooth(k / 80)
        c, s_ = math.cos(math.radians(phi)), math.sin(math.radians(phi))
        A.aim(pivot + np.array([rel[0] * c - rel[2] * s_, 0.0, rel[2] * c + rel[0] * s_]), carry_tool(phi))
        yield
    yield from A.set_grip(1.0, 0.3)
    yield from A.move(A.pos + 0.06 * -carry_tool(60.0)[:, 2], None, 0.5)
    yield from dk.together(A.go_home(2.0), R.go_home(2.0))


def script(task, ctx, variant: str = ""):
    yield from dk.wait(0.5)
    if variant == "idle":
        yield from dk.wait(40.0)
        return
    if variant == "roll_only":
        yield from roll(task, ctx)
        yield from dk.wait(2.0)
        return
    if variant == "lift_and_flip":
        yield from lift_and_flip(task, ctx)
        yield from dk.wait(2.0)
        return
    if variant == "":
        yield from arm_overhead(task, ctx)
        yield from roll(task, ctx)
        yield from dk.wait(2.0)
        return
    raise SystemExit(f"variant {variant!r} not written yet")


def debug(task, ctx) -> str:
    st = ctx.states()["dummy"]
    return (f"dummy ({100 * st.pos[0]:.1f}, {100 * st.pos[1]:.1f}, {100 * (st.pos[2] - task.GROUND):.1f}) cm "
            f"chest {task.chest_angle(st):.0f} lift {100 * task.trunk_lift(st):.1f} cm "
            f"under {D.limb_under_body(task.DUMMY, st)} v {st.speed():.3f} | {task.report()['last']} | "
            f"L cmd {np.round(100 * ctx.left.pos, 1)} act {np.round(100 * tk.grippers(ctx.emb._env)['left']['points'][4], 1)} "
            f"R cmd {np.round(100 * ctx.right.pos, 1)} act {np.round(100 * tk.grippers(ctx.emb._env)['right']['points'][4], 1)} "
            f"Rmiss {len(ctx.right.misses)}")


if __name__ == "__main__":
    dk.main(script, debug)
