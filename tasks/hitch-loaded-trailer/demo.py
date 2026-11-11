"""Scripted full-score demo for hitching the trailer, and its near-miss variants.

    python demo.py task.py demo_out                                   # full score
    python demo.py task.py out --variant back_up_only --no-record     # back the rover up, the tongue left on the ground
    python demo.py task.py out --variant lift_and_drop --no-record    # hold the tongue high, back up, let it fall
    python demo.py task.py out --variant small_drop --no-record       # let it fall from the ball's top (about 1 cm)
    python demo.py task.py out --variant tiny_drop --no-record        # let it fall from 6 mm above the ball's top
    python demo.py task.py out --variant overshoot_drop --no-record   # back the rover 5 cm too far, drop the tongue on its bar
    python demo.py task.py out --variant idle --no-record             # do nothing

1. The left arm grips the trailer's tongue near its tip and lifts it until the socket's rim is 1.5 cm above the
   rover's tow ball.
2. The right arm, fingers closed against the end of the rover's front tow tab, pushes it backward until the ball is
   9 mm short of the socket (closed loop on the measured poses; the socket swings that far toward it as it is
   lowered), then withdraws.
3. The left arm lowers the tongue slowly until the socket rests on the ball, opens and withdraws.

The script reads only the rover's and the trailer's poses and the arms' own joint state.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import demokit as dk  # noqa: E402
import trailkit as tk  # noqa: E402

GRIP_OPEN = 0.6
GRIP_BACK = 0.06          # m: the left arm grips the tongue this far behind the socket's centre
LIFT_CLEAR = 0.015        # m: the socket's rim this far above the ball's top while the rover backs under it
LOWER_SPEED = 0.01        # m/s
TRACE = bool(__import__('os').environ.get('DEMO_TRACE'))
SWING = 0.009             # m: the socket moves this far toward the rover as it is lowered onto the ball (measured)


def _s(ctx, name):
    return ctx.states()[name]


def pad():
    from yam_arena.constants import PAD_TIP_PAST_TCP
    return PAD_TIP_PAST_TCP


def tcp_for_tip(point, rot):
    return np.asarray(point, float) - rot[:, 2] * pad()


def tongue_grip_point(task, tr):
    """GRIP_BACK behind the socket's centre (trailer frame): the pad tips 10 mm below the tongue's top face (it is 14 mm deep)."""
    return tr.to_world((0.0, -task.TONGUE_L + GRIP_BACK, task.T_WHEEL_R + 0.02 + 0.007 - 0.010))


def lift_tongue(task, ctx, A, height_above_ball):
    tr = _s(ctx, "trailer")
    tip = tongue_grip_point(task, tr)
    rot = A.grip_rotation(tcp_for_tip(tip, dk.DOWN_X), (0, 0, -1), (1, 0, 0))
    yield from A.transit(tcp_for_tip(tip, rot) + (0, 0, 0.03), rot, 3.0, grip=GRIP_OPEN)
    yield from A.move(tcp_for_tip(tip, rot), rot, 0.8)
    yield from A.set_grip(0.0, 0.6)
    ball = task.ball_centre(_s(ctx, "rover"))
    want = ball[2] + task.BALL_R + height_above_ball            # the socket's rim (its centre point) this high
    for _ in range(int(6 * dk.CONTROL_HZ)):
        s = task.socket_centre(_s(ctx, "trailer"))
        err = want - s[2]
        if abs(err) < 0.002:
            break
        A.aim(A.pos + np.array([0, 0, float(np.clip(err, -0.003, 0.003))]), rot)
        yield
    ctx.note(f"socket rim {1000 * (task.socket_centre(_s(ctx, 'trailer'))[2] - ball[2]):.1f} mm above the ball's centre")


def back_under(task, ctx, A, align=True, short=0.0):
    """Closed fingers against the end face of the rover's front tow tab: push the rover backward along its heading
    until the ball is under the socket (or 12 cm, without align), slowing as it nears, then withdraw."""
    rv = task.rv
    end_x = rv.S.LENGTH / 2 + rv.TOW_TAB[0] - 0.002

    def frame(st):
        hd = st.rotation()[:, 0]
        hd = np.array([hd[0], hd[1], 0.0]) / max(np.hypot(hd[0], hd[1]), 1e-9)
        rot = dk.tool_rotation((-hd[0], -hd[1], -1.0), (hd[1], -hd[0], 0))      # pointing backward along the rover
        return hd, rot, st.to_world((end_x, 0.0, rv.S.BUMPER_HEIGHT - 0.002))

    st = _s(ctx, "rover")
    hd, rot, face = frame(st)
    yield from A.set_grip(0.0, 0.3)
    yield from A.transit(tcp_for_tip(face + hd * 0.03, rot), rot, 3.0)
    yield from A.move(tcp_for_tip(face, rot), rot, 0.8)
    y0 = st.pos[1]
    boost, past = 0.0, [st.pos.copy()]
    for _ in range(int(30 * dk.CONTROL_HZ)):
        st = _s(ctx, "rover")
        hd, rot, face = frame(st)
        past = (past + [st.pos.copy()])[-11:]
        moving = len(past) < 11 or float(np.linalg.norm(st.pos[:2] - past[0][:2])) > 0.002   # 4 mm/s over 0.5 s
        boost = max(boost - 0.002, 0.0) if moving else min(boost + 0.0004, 0.015)  # press harder while it stalls
        if align:
            d = float((task.socket_centre(_s(ctx, "trailer")) - task.ball_centre(st)) @ (-hd)) - short   # still to go
            if d < 0.001:
                break
            press = (0.012 if d > 0.03 else (0.005 if d > 0.01 else 0.003)) + boost
        else:
            if abs(st.pos[1] - y0) > 0.12:
                break
            press = 0.012 + boost
        A.aim(tcp_for_tip(face - hd * press, rot), rot)
        yield
    yield from A.move(A.pos + hd * 0.03 + np.array([0, 0, 0.02]), None, 0.6)
    yield from dk.wait(0.5)
    b, s = task.ball_centre(_s(ctx, "rover")), task.socket_centre(_s(ctx, "trailer"))
    ctx.note(f"backed under: ball {1000 * float(np.hypot(*(b[:2] - s[:2]))):.1f} mm off the socket's axis")
    yield from A.go_home(2.0)


def hover_over_ball(task, ctx, A, clear=0.003):
    """Lower the held tongue until the socket's rim is ``clear`` above the ball's top."""
    rot = A.rot
    for _ in range(int(6 * dk.CONTROL_HZ)):
        b, s = task.ball_centre(_s(ctx, "rover")), task.socket_centre(_s(ctx, "trailer"))
        err = (b[2] + task.BALL_R + clear) - s[2]
        if abs(err) < 0.0015:
            break
        A.aim(A.pos + np.array([0, 0, float(np.clip(err, -0.002, 0.002))]), rot)
        yield


def lower_onto_ball(task, ctx, A):
    """Lower the tongue at LOWER_SPEED until the socket stops on the ball, then open and withdraw."""
    rot = A.rot
    last, still = None, 0
    for _ in range(int(10 * dk.CONTROL_HZ)):
        A.aim(A.pos + np.array([0, 0, -LOWER_SPEED / dk.CONTROL_HZ]), rot)
        yield
        s = task.socket_centre(_s(ctx, "trailer"))
        if TRACE:
            b = task.ball_centre(_s(ctx, "rover"))
            ctx.note(f"trace hand z {1000 * A.pos[2]:.1f} rim-ball {1000 * (s[2] - b[2]):.1f} off "
                     f"{1000 * (s[1] - b[1]):.1f},{1000 * (s[0] - b[0]):.1f} mm")
        low = s[2] - task.ball_centre(_s(ctx, "rover"))[2] < 0.002        # the rim within 2 mm of the ball's centre: seated or nearly
        still = still + 1 if low and last is not None and abs(s[2] - last) < 0.0003 else 0
        last = s[2]
        if still > 6:
            break
    yield from A.move(A.pos + np.array([0, 0, -0.004]), rot, 0.4)    # let the tongue's weight settle on the ball
    yield from A.set_grip(GRIP_OPEN, 0.6)
    yield from A.move(A.pos + (0, 0, 0.06), None, 0.8)
    ok, dxy, dz = task.seated(_s(ctx, "rover"), _s(ctx, "trailer"))
    ctx.note(f"lowered: seated {ok} ({1000 * dxy:.1f} mm, rim {1000 * dz:.1f} mm above the ball's centre)")
    yield from A.go_home(2.0)


def script(task, ctx, variant: str = ""):
    L, R = ctx.left, ctx.right
    yield from dk.wait(0.5)
    if variant == "idle":
        yield from dk.wait(30.0)
        return
    if variant == "back_up_only":
        yield from back_under(task, ctx, R, align=False)
        yield from dk.wait(3.0)
        return
    if variant in ("small_drop", "tiny_drop"):
        # let go with the socket's rim level with the ball's top (a drop of about a centimetre), or 6 mm above it
        yield from lift_tongue(task, ctx, L, LIFT_CLEAR)
        yield from back_under(task, ctx, R, short=SWING)
        yield from hover_over_ball(task, ctx, L, clear=0.0 if variant == "small_drop" else 0.006)
        yield from L.set_grip(GRIP_OPEN, 0.2)
        yield from L.move(L.pos + (0, 0, 0.06), None, 0.6)
        yield from L.go_home(2.0)
        yield from dk.wait(3.0)
        return
    if variant == "overshoot_drop":
        # hold the tongue's tip 5 cm above the ball (its bar clear of it), back the rover 5 cm past the socket, let go:
        # the bar falls onto the ball
        yield from lift_tongue(task, ctx, L, 0.05)
        yield from back_under(task, ctx, R, short=-0.05)
        yield from L.set_grip(GRIP_OPEN, 0.2)
        yield from L.move(L.pos + (0, 0, 0.06), None, 0.6)
        yield from L.go_home(2.0)
        yield from dk.wait(3.0)
        return
    if variant == "lift_and_drop":
        yield from lift_tongue(task, ctx, L, 0.05)
        yield from back_under(task, ctx, R)
        yield from L.set_grip(GRIP_OPEN, 0.2)                      # let the tongue fall onto the ball
        yield from L.move(L.pos + (0, 0, 0.06), None, 0.6)
        yield from L.go_home(2.0)
        yield from dk.wait(3.0)
        return
    yield from lift_tongue(task, ctx, L, LIFT_CLEAR)
    yield from back_under(task, ctx, R, short=SWING)                 # the socket swings toward the rover as it comes
    yield from lower_onto_ball(task, ctx, L)                         # down: stop the ball that far short of it
    yield from dk.wait(2.5)


def debug(task, ctx) -> str:
    b, s = task.ball_centre(_s(ctx, "rover")), task.socket_centre(_s(ctx, "trailer"))
    return f"ball {np.round(b, 3)} socket {np.round(s, 3)} | {task.report()}"


if __name__ == "__main__":
    dk.main(script, debug)
