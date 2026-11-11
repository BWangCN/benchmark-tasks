"""Scripted full-score demo for the lay-it-on-its-side task, and its near-miss variants.

    python demo.py task.py demo_out                                     # full score: MP4 per camera, setup.png
    python demo.py task.py out --variant roll_and_release --no-record   # carry it onto its side by the head, let go
    python demo.py task.py out --variant hold_to_the_end --no-record    # the full method, but never let go
    python demo.py task.py out --variant idle --no-record               # do nothing

The body lies on its back along y, head to the robot's right (-y), its left side toward the robot; it is rolled
toward the robot, about its near edge, onto its left side. Its right (upper, far) arm and leg become the props.

The first-aid recovery roll, with the far limbs as handles:
1. The right hand takes the far forearm 4.5 cm past the elbow; the left hand takes the far thigh by the knee.
2. Together they turn the body 85 degrees round its near edge. The right hand moves as if fixed to the forearm,
   raising the arm as the body turns, so that on its side the arm points forward out of the chest (the shoulder has
   no twist joint: a hand that kept pointing down would twist the whole body). The left hand carries the knee up,
   across the near leg and down in front.
3. Both hands let go and withdraw; the body settles forward onto its knee and forearm.

The script reads only the dummy's pose and joint angles (what a camera-based pose estimate gives) and the arms' own
joint state.
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import demokit as dk  # noqa: E402
import dummy as D  # noqa: E402
import trailkit as tk  # noqa: E402

GRIP_HALF = 0.42          # fingers about 4 cm apart
UP = np.array([0.0, 0.0, 1.0])


def _f(name, default):
    return float(os.environ.get(name, default))


def _st(ctx) -> tk.BodyState:
    return ctx.states()["dummy"]


def trunk_box(st: tk.BodyState, tag: str = "trunk") -> np.ndarray:
    s = next(s for s in D.base_shapes() if s.tag == tag)
    c, h = np.asarray(s.centre), np.asarray(s.size) / 2
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c
    return st.pos + corners @ st.rotation().T


def across_tool(axis, max_tilt_deg: float = 40.0) -> np.ndarray:
    """Pointing down at a limb segment along ``axis``, square to it (tilted at most ``max_tilt_deg`` from straight
    down), the fingers closing across it."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    z = -UP + (UP @ a) * a
    z /= np.linalg.norm(z)
    tilt = math.degrees(math.acos(float(np.clip(-z[2], -1, 1))))
    if tilt > max_tilt_deg:                                   # blend back toward straight down
        h = np.array([z[0], z[1], 0.0])
        h /= np.linalg.norm(h)
        t = math.radians(max_tilt_deg)
        z = math.sin(t) * h - math.cos(t) * UP
    x = np.cross(a, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    return dk.tool_rotation(z, x / np.linalg.norm(x))


def limb_point(task, st0, joints: dict, link: str, along: float):
    """World point ``along`` metres down ``link``'s segment, and the segment's direction, for the body held where
    ``st0`` has it with these joint angles."""
    st = tk.state_at(task.DUMMY, st0.pos, st0.quat_xyzw, joints=dict(st0.joints) | joints)
    p, R = D.frames(task.DUMMY, st)[link]
    return p + R @ np.array([0.0, 0.0, -along]), -R[:, 2]


def blend(a: dict, b: dict, s: float) -> dict:
    return {k: a.get(k, 0.0) + (b[k] - a.get(k, 0.0)) * s for k in b}


LEG_PROP = {"r_hip_pitch": -1.6, "r_hip_roll": 0.52, "r_knee": 1.2}
FORE_ALONG = 0.045        # m: grip the far forearm 4.5 cm past the elbow
THIGH_ALONG = 0.09        # m: grip the far thigh 9 cm from the hip (2 cm short of the knee)
SLACK = 0.006             # m: the grip runs this far inside the limb's reach, so it does not drag the body
KNEE_TILT = 0.0           # deg: the knee grip points straight down (reach)


HEAD_GRIP_Y = 0.012       # m: grip the head this far from its top end
HEAD_GRIP_Z = 0.022       # m: tool point above the ground at the grip
CARRY_DEG = _f("CARRY_DEG", 85.0)   # the body is carried this far round its near edge
CARRY_S = 7.0             # s
KNEE_LIFT = 0.035         # m: extra height of the knee path at mid-carry (over the near leg)


def carry_tool(phi: float) -> np.ndarray:
    t = math.radians(phi)
    return dk.tool_rotation((math.sin(t), 0.0, -math.cos(t)), (math.cos(t), 0.0, math.sin(t)))


def turn(pivot, p, phi):
    """``p`` turned ``phi`` degrees toward -x about the line along y through ``pivot``."""
    rel = np.asarray(p, float) - pivot
    c, s = math.cos(math.radians(phi)), math.sin(math.radians(phi))
    return pivot + np.array([rel[0] * c - rel[2] * s, rel[1], rel[2] * c + rel[0] * s])


def turn_dir(d, phi):
    c, s = math.cos(math.radians(phi)), math.sin(math.radians(phi))
    return np.array([d[0] * c - d[2] * s, d[1], d[2] * c + d[0] * s])


def roll(task, ctx, knee: bool = True, hold_s: float = 0.0, let_go: bool = True):
    """Turn the body toward the robot onto its left side: the right hand carries the head round the near edge;
    with ``knee`` the left hand carries the far knee up, across and down in front as the body turns."""
    A, R_ = ctx.left, ctx.right
    st0 = _st(ctx)
    head = D.points(task.DUMMY, st0, which={"base"}, tags=("head",))
    grip = np.array([float(head[:, 0].mean()), float(head[:, 1].min()) + HEAD_GRIP_Y + 0.0135,
                     task.GROUND + HEAD_GRIP_Z])
    pivot = np.array([float(trunk_box(st0)[:, 0].min()), 0.0, task.GROUND])
    ctx.note(f"head grip at {np.round(100 * grip, 1)} cm, turning about x {100 * pivot[0]:.1f} cm")
    legs0 = {k: st0.joints[k] for k in LEG_PROP}
    moves = [R_.via(grip, carry_tool(0.0), 3.0, lift=0.06, grip=1.0, ceiling=task.GROUND + 0.10)]
    if knee:
        kp, ka = limb_point(task, st0, legs0, "r_hip_roll", THIGH_ALONG)
        kp = kp + (0, 0, 0.002)
        moves.append(A.via(kp + (0, 0, 0.04), across_tool(ka, KNEE_TILT), 3.0, lift=0.05, grip=GRIP_HALF,
                           ceiling=task.GROUND + 0.10))
    yield from dk.together(*moves)
    grips = [R_.set_grip(0.0, 0.8)]
    if knee:
        grips = [A.move(kp, across_tool(ka, KNEE_TILT), 0.8), R_.set_grip(0.0, 0.8)]
    yield from dk.together(*grips)
    if knee:
        yield from A.set_grip(0.0, 0.6)
    hip = D.frames(task.DUMMY, st0)["r_hip_roll"][0]
    n = int(CARRY_S * dk.CONTROL_HZ)
    for k in range(1, n + 1):
        s = dk.smooth(k / n)
        phi = CARRY_DEG * s
        R_.aim(turn(pivot, grip, phi), carry_tool(phi))
        if knee:
            p, a = limb_point(task, st0, blend(legs0, LEG_PROP, s), "r_hip_roll", THIGH_ALONG)
            p = p - SLACK * math.sin(math.pi * s) * (p - hip) / np.linalg.norm(p - hip)
            p = turn(pivot, p, phi) + (0, 0, KNEE_LIFT * math.sin(math.pi * s))
            A.aim(p, across_tool(turn_dir(a, phi), KNEE_TILT))
        yield
    st = _st(ctx)
    ctx.note(f"carried: side {task.side_angle(st):.0f} deg from on-its-left-side, lift {100 * task.trunk_lift(st):.1f} cm,"
             f" joints {np.round(np.degrees([st.joints[k] for k in (*ARM_UP, *LEG_PROP)]), 0)}, "
             f"ik misses L {len(A.misses)} R {len(R_.misses)}")
    if hold_s:
        yield from dk.wait(hold_s)
    if not let_go:
        while True:
            yield
    lets = [R_.set_grip(1.0, 0.3)] + ([A.set_grip(1.0, 0.3)] if knee else [])
    yield from dk.together(*lets)
    backs = [R_.move(R_.pos + 0.06 * -carry_tool(CARRY_DEG)[:, 2], None, 0.5)]
    if knee:
        backs.append(A.move(A.pos + (0, 0, 0.05), None, 0.5))
    yield from dk.together(*backs)
    yield from dk.together(A.go_home(2.0), R_.go_home(2.0))


ARM_UP = {"r_sh_pitch": -1.6, "r_sh_roll": -0.1, "r_elbow": -0.3}   # far arm pointing up from the chest (forward)
ROLL_DEG = _f("ROLL_DEG", 85.0)    # the body is turned this far round its near edge
ROLL_S = 8.0                      # s


def recovery_roll(task, ctx, props: bool = True, leg: bool = True):
    """The first-aid recovery roll, toward the robot. The right hand takes the far forearm, the left hand the far
    thigh by the knee; together they turn the body round its near edge while the far arm is raised to point out of
    the chest and the far knee is brought up and across, so that on its side the arm and knee point forward. The right
    hand moves as if fixed to the forearm (the shoulder has no twist joint); the left hand points down. Without
    ``props`` the same two holds turn the body but leave the arm and leg straight (the naive roll)."""
    A, R_ = ctx.left, ctx.right
    st0 = _st(ctx)
    pivot = np.array([float(trunk_box(st0)[:, 0].min()), 0.0, task.GROUND])
    arm0 = {k: st0.joints[k] for k in ARM_UP}
    legs0 = {k: st0.joints[k] for k in LEG_PROP}
    arm1, legs1 = (ARM_UP, LEG_PROP if leg else legs0) if props else (arm0, legs0)

    def fore(s):
        st = tk.state_at(task.DUMMY, st0.pos, st0.quat_xyzw, joints=dict(st0.joints) | blend(arm0, arm1, s))
        return D.frames(task.DUMMY, st)["r_elbow"]

    def ry(phi):
        c, s_ = math.cos(math.radians(phi)), math.sin(math.radians(phi))
        return np.array([[c, 0.0, -s_], [0.0, 1.0, 0.0], [s_, 0.0, c]])

    p0, L0 = fore(0.0)
    G = L0.T @ across_tool(-L0[:, 2], 0.0)                    # the grip, in the forearm's frame
    fg = p0 + L0 @ np.array([0.0, 0.0, -FORE_ALONG]) + (0, 0, -0.004)
    kp, ka = limb_point(task, st0, legs0, "r_hip_roll", THIGH_ALONG)
    kp = kp + (0, 0, 0.002)
    hip = D.frames(task.DUMMY, st0)["r_hip_roll"][0]
    ctx.note(f"forearm grip {np.round(100 * fg, 1)} cm, knee grip {np.round(100 * kp, 1)} cm, pivot x {100 * pivot[0]:.1f}")
    yield from R_.via(fg + (0, 0, 0.04), L0 @ G, 2.5, lift=0.05, grip=GRIP_HALF, ceiling=task.GROUND + 0.10)
    yield from R_.move(fg, L0 @ G, 0.8)
    yield from R_.set_grip(0.0, 0.6)
    yield from A.via(kp + (0, 0, 0.04), across_tool(ka, KNEE_TILT), 2.5, lift=0.05, grip=GRIP_HALF,
                     ceiling=task.GROUND + 0.10)
    yield from A.move(kp, across_tool(ka, KNEE_TILT), 0.8)
    yield from A.set_grip(0.0, 0.6)
    n = int(ROLL_S * dk.CONTROL_HZ)
    for k in range(1, n + 1):
        s = dk.smooth(k / n)
        phi = ROLL_DEG * s
        p, L = fore(s)
        R_.aim(turn(pivot, p + L @ np.array([0.0, 0.0, -FORE_ALONG]), phi) + (0, 0, -0.004), ry(phi) @ L @ G)
        q, a = limb_point(task, st0, blend(legs0, legs1, s), "r_hip_roll", THIGH_ALONG)
        q = q - SLACK * math.sin(math.pi * s) * (q - hip) / np.linalg.norm(q - hip)
        A.aim(turn(pivot, q, phi) + (0, 0, (KNEE_LIFT if props and leg else 0.0) * math.sin(math.pi * s)), across_tool(turn_dir(a, phi), KNEE_TILT))
        yield
    st = _st(ctx)
    ctx.note(f"rolled: side {task.side_angle(st):.0f} deg from on-its-left-side, lift {100 * task.trunk_lift(st):.1f} cm,"
             f" r arm {np.round(np.degrees([st.joints[k] for k in ARM_UP]), 0)} r leg "
             f"{np.round(np.degrees([st.joints[k] for k in LEG_PROP]), 0)}, ik misses L {len(A.misses)} R {len(R_.misses)}")


def let_go(task, ctx):
    A, R_ = ctx.left, ctx.right
    yield from dk.together(A.set_grip(1.0, 0.4), R_.set_grip(1.0, 0.4))
    yield from dk.together(A.move(A.pos + (0, 0, 0.05), None, 0.6), R_.move(R_.pos + (0, 0, 0.05), None, 0.6))
    yield from dk.together(A.go_home(2.0), R_.go_home(2.0))


def script(task, ctx, variant: str = ""):
    yield from dk.wait(0.5)
    if variant == "idle":
        yield from dk.wait(40.0)
        return
    if variant == "roll_and_release":         # carry it by the head onto its side, limbs as they lie, and let go
        yield from roll(task, ctx, knee=False, hold_s=1.5)
        yield from dk.wait(6.0)
        return
    if variant == "knee_only":                # head carried round, far knee brought forward, far arm left as it lies
        yield from roll(task, ctx, knee=True, hold_s=1.5)
        yield from dk.wait(6.0)
        return
    if variant == "arm_only":                 # the recovery roll with the far leg kept straight
        yield from recovery_roll(task, ctx, props=True, leg=False)
        yield from dk.wait(1.5)
        yield from let_go(task, ctx)
        yield from dk.wait(6.0)
        return
    if variant == "hold_to_the_end":
        yield from recovery_roll(task, ctx)
        while True:
            yield
    if variant == "":
        yield from recovery_roll(task, ctx)
        yield from let_go(task, ctx)
        yield from dk.wait(_f("END_WAIT", 6.0))
        return
    raise SystemExit(f"unknown variant {variant!r}")


def debug(task, ctx) -> str:
    st = ctx.states()["dummy"]
    return (f"dummy ({100 * st.pos[0]:.1f}, {100 * st.pos[1]:.1f}, {100 * (st.pos[2] - task.GROUND):.1f}) cm "
            f"side {task.side_angle(st):.0f} chest {np.round(D.chest_normal(st), 2)} lift {100 * task.trunk_lift(st):.1f} cm "
            f"spine {task.spine_tilt(st):.0f} v {st.speed():.3f} | {task.report()['last']} | "
            f"L cmd {np.round(100 * ctx.left.pos, 1)} R cmd {np.round(100 * ctx.right.pos, 1)} "
            f"Lmiss {len(ctx.left.misses)} Rmiss {len(ctx.right.misses)} | spine {np.round(D.spine(st), 2)} "
            f"r arm {np.round(np.degrees([st.joints[k] for k in ARM_UP]), 0)} r leg "
            f"{np.round(np.degrees([st.joints[k] for k in LEG_PROP]), 0)} gap L "
            f"{100 * tk.grippers(ctx.emb._env)['left']['opening']:.1f} R {100 * tk.grippers(ctx.emb._env)['right']['opening']:.1f}")


if __name__ == "__main__":
    dk.main(script, debug)
