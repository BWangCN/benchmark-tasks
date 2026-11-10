"""Scripted full-score demo for the log-crossing task, and its near-miss variants.

    python demo.py task.py demo_out                              # full score: MP4 per camera, setup.png, result.json
    python demo.py task.py out --variant push_only --no-record   # push the rover straight into the bog
    python demo.py task.py out --variant logs_only --no-record   # lay the logs and stop
    python demo.py task.py out --variant idle --no-record        # do nothing

1. The arms lay the four logs across the bog one by one, far side first, each tight against the last (the left arm
   takes the logs on the left verge, the right arm those on the right), gripping each log across its diameter with the
   fingers part-open and letting go with them part-open, so the neighbouring logs are not disturbed.
2. The right arm tamps the row tight toward the far bank; a log riding on its neighbours is pressed down into its
   slot, and the row tamped again.
3. The right arm, fingers closed and pointing forward, pushes the rover by the middle of its rear bumper (closed
   loop on its measured pose) across the logs onto the far deck, then withdraws.

The script reads only object poses (the rover and the logs) and the arms' own joint state.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import demokit as dk  # noqa: E402
import trailkit as tk  # noqa: E402

PUSH_TOOL = dk.tool_rotation((1, 0, -0.2), (0, 0, 1))      # pointing forward, fingers closed one above the other
PUSH_SPEED = 0.025
GRIP_PART = 0.45           # part-open: about 42 mm between the fingers, for a 20 mm log


def _st(ctx, name) -> tk.BodyState:
    return ctx.states()[name]


def lay_log(task, ctx, A, name: str, x: float):
    st = _st(ctx, name)
    if not (0.0 < st.pos[0] < 0.8 and abs(st.pos[1]) < 0.5):  # knocked off the table: nothing to lay
        ctx.note(f"{name} is not on the table; skipped")
        return
    pick_rot = dk.tool_rotation((0, 0, -1), (0, -1, 0))        # across a log lying along x (both arms reach it)
    place_rot = dk.tool_rotation((0, 0, -1), (-1, 0, 0) if A.side == "left" else (1, 0, 0))  # across a log along y
    grip = np.array([st.pos[0], st.pos[1], task.TOP + 0.0135])    # pad tips just on the turf, pads across the log
    yield from A.via(grip + (0, 0, 0.05), pick_rot, 2.5, lift=0.06, grip=GRIP_PART)
    yield from A.move(grip, pick_rot, 0.8)
    yield from A.set_grip(0.0, 0.5)
    yield from A.move(grip + (0, 0, 0.07), pick_rot, 0.6)
    # drop it into its slot from just above the decks: the fingers cannot fit beside a log in a full-width mat
    target = np.array([x, 0.0, task.TOP + 0.046])
    yield from A.move(target + (0, 0, 0.05), place_rot, 2.0)
    yield from A.move(target, place_rot, 0.8)
    yield from A.set_grip(GRIP_PART, 0.3)
    yield from dk.wait(0.4)
    yield from A.move(target + (0, 0, 0.05), place_rot, 0.6)
    ctx.note(f"{name} dropped at x {100 * x:.1f} cm")


def row_gap(task, ctx) -> float:
    """The widest gap in the row of logs across the bog (between neighbours, or at a deck), from their poses: what the
    demo tamps until it is small. Logs not in the bog count as a gap the bog's whole width."""
    s = ctx.states()
    xs = sorted(s[lg.name].pos[0] for lg in task.LOGS if task.BOG[0] - 0.002 <= s[lg.name].pos[0] <= task.BOG[1] + 0.002)
    if len(xs) < task.N_LOGS:
        return task.BOG[1] - task.BOG[0]
    edges = [task.BOG[0]] + [e for x in xs for e in (x - task.LOG_R, x + task.LOG_R)] + [task.BOG[1]]
    return max(edges[i + 1] - edges[i] for i in range(0, len(edges), 2))


def tamp(task, ctx, A):
    """Push the row of logs toward the far bank with closed fingers on the nearest log, until the mat is tight."""
    s = ctx.states()
    near = min(task.LOGS, key=lambda lg: s[lg.name].pos[0])
    st = s[near.name]
    rot = dk.tool_rotation((0, 0, -1), (0, -1, 0))
    x = st.pos[0] - task.LOG_R - 0.010
    z = task.TOP + 0.034                                   # pad tips 21 mm up: on the log's upper half, above the deck
    yield from A.via((x, st.pos[1], z + 0.04), rot, 2.0, lift=0.05, grip=0.0)
    yield from A.move((x, st.pos[1], z), rot, 0.6)
    for _ in range(int(2 * dk.CONTROL_HZ)):
        s_ = ctx.states()
        if row_gap(task, ctx) <= 0.0035 and all(s_[lg.name].pos[2] < task.TOP + task.LOG_R + 0.003 for lg in task.LOGS):
            break
        x += 0.008 / dk.CONTROL_HZ
        A.aim((x, st.pos[1], z), rot)
        yield
    yield from A.move(A.pos + (-0.01, 0, 0.06), rot, 0.8)
    ctx.note(f"tamped: widest gap {1000 * row_gap(task, ctx):.1f} mm")


def seat(task, ctx, A) -> bool:
    """Press down, with the closed fingertips on its middle, any log that rides on its neighbours. True if any did."""
    flat = task.TOP + task.LOG_R * np.sqrt(3) / 2                # a six-sided log's centre lying on a face
    rot = dk.tool_rotation((0, 0, -1), (1, 0, 0))
    pressed = False
    for lg in task.LOGS:
        st = ctx.states()[lg.name]
        if st.pos[2] < flat + 0.003:
            continue
        top = np.array([st.pos[0], st.pos[1], flat + task.LOG_R * np.sqrt(3) / 2 + 0.013])   # tool point: tips on top
        yield from A.via(top + (0, 0, 0.03), rot, 2.0, lift=0.04, grip=0.0)
        yield from A.move(top - (0, 0, 0.004), rot, 0.8)
        yield from dk.wait(0.3)
        yield from A.move(top + (0, 0, 0.04), rot, 0.5)
        ctx.note(f"{lg.name} pressed down from {1000 * (st.pos[2] - task.TOP):.0f} mm, now "
                 f"{1000 * (ctx.states()[lg.name].pos[2] - task.TOP):.0f} mm")
        pressed = True
    return pressed


def push_across(task, ctx, A, until_x: float):
    st = _st(ctx, "rover")
    face = st.to_world((-0.131, 0.0, 0.050))
    yield from A.via(face - PUSH_TOOL[:, 2] * 0.023, PUSH_TOOL, 3.0, lift=0.06, grip=0.0)
    for _ in range(int(25 * dk.CONTROL_HZ)):
        st = _st(ctx, "rover")
        if st.pos[0] >= until_x:
            break
        face = st.to_world((-0.131, 0.0, 0.050))
        A.aim(face - PUSH_TOOL[:, 2] * 0.010 + np.array([PUSH_SPEED / dk.CONTROL_HZ, 0, 0]), PUSH_TOOL)
        yield
    ctx.note(f"pushed: rover at x {100 * _st(ctx, 'rover').pos[0]:.1f} cm")
    yield from A.move(A.pos - PUSH_TOOL[:, 2] * 0.05, PUSH_TOOL, 0.8)
    yield from A.go_home(2.0)


def script(task, ctx, variant: str = ""):
    L, R = ctx.left, ctx.right
    yield from dk.wait(0.5)
    if variant == "idle":
        yield from dk.wait(60.0)
        return
    if variant != "push_only":
        pitch = (task.BOG[1] - task.BOG[0]) / task.N_LOGS
        xs = [task.BOG[1] - pitch * (i + 0.5) for i in range(task.N_LOGS)]      # far side first
        order = [("log_1", L), ("log_3", R), ("log_2", L), ("log_4", R)]
        for (name, arm), x in zip(order, xs):
            yield from lay_log(task, ctx, arm, name, x)
            other = R if arm is L else L
            yield from arm.move(arm.pos + (0, 0.08 if arm is L else -0.08, 0.04), None, 0.6)   # out of the other's way
        yield from dk.together(L.go_home(2.0), R.go_home(2.0))
        yield from tamp(task, ctx, R)
        for _ in range(2):                                   # a log riding on its neighbours: press it in, tamp again
            if not (yield from seat(task, ctx, R)):
                break
            yield from tamp(task, ctx, R)
        yield from R.go_home(2.0)
    if variant == "logs_only":
        yield from dk.wait(1.0)
        return
    yield from push_across(task, ctx, R, task.FAR[0] + task.rv.AXLE_X + task.rv.TYRE_R + 0.015)
    yield from dk.wait(1.0)


def debug(task, ctx) -> str:
    s = ctx.states()
    st = s["rover"]
    logs = " ".join(f"{lg.name[-1]}:{100 * s[lg.name].pos[0]:.1f}" for lg in task.LOGS)
    ly = " ".join(f"{100 * s[lg.name].pos[1]:.1f}" for lg in task.LOGS)
    return (f"rover x {100 * st.pos[0]:.1f} v {st.speed():.3f} tilt {tk.tilt_deg(st.rotation()):.0f} | logs x {logs} y {ly} | "
            f"moving {task.report()['moving']} rest {task.report()['rest_window']} steps {task.report()['steps']} | {task.report()['last']}")


if __name__ == "__main__":
    dk.main(script, debug)
