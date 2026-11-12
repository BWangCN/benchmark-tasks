"""Scripted full-score demo for sliding the slab off the rover, and its wrong-method variants.

    python demo.py task.py demo_out                                       # full score
    python demo.py task.py out --variant push_along --no-record           # push it straight away (along the rover)
    python demo.py task.py out --variant brace_and_push_along --no-record # hold the rover's rear tow tab, push along
    python demo.py task.py out --variant idle --no-record                 # do nothing

1. The right arm sets its closed fingers, pointing down, against the slab's right-hand side (its -y edge) and pushes
   it across the rover (+y), pressing a little harder whenever it stops, until the slab has slid off the far side
   onto the ground.
2. If the slab has come to lean against the rover's side, the arm pushes its raised edge on over until it lies flat.
3. The arm lifts clear and goes home.

The script reads only the poses of the rover and the slab and the arms' own joint state.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import demokit as dk  # noqa: E402
import trailkit as tk  # noqa: E402


def tcp_for_tip(point, rot):
    from yam_arena.constants import PAD_TIP_PAST_TCP
    return np.asarray(point, float) - rot[:, 2] * PAD_TIP_PAST_TCP


def _s(ctx, name) -> tk.BodyState:
    return ctx.states()[name]


def push(task, ctx, A, d, local_edge, travel: float, seconds: float):
    """Closed fingers against the slab's edge at ``local_edge`` (slab frame), pushing along ``d`` (world) until the
    edge has gone ``travel`` or ``seconds`` have passed; a little harder (up to 6 cm past the edge) while it stands."""
    d = np.asarray(d, float)
    rot = dk.tool_rotation((0, 0, -1), tuple(d))
    edge = lambda: np.array(_s(ctx, "slab").to_world(local_edge))            # noqa: E731
    e0 = edge()
    yield from A.set_grip(0.0, 0.3)
    yield from A.transit(tcp_for_tip(e0 - d * 0.04, rot), rot, 3.0, height=0.20)
    yield from A.move(tcp_for_tip(e0 - d * 0.012, rot), rot, 0.8)
    boost, past = 0.0, []
    for _ in range(int(seconds * dk.CONTROL_HZ)):
        e = edge()
        if float((e - e0) @ d) > travel:
            break
        past = (past + [e])[-11:]
        moving = len(past) < 11 or float(np.linalg.norm(e - past[0])) > 0.002
        boost = max(boost - 0.002, 0.0) if moving else min(boost + 0.0008, 0.06)
        A.aim(tcp_for_tip(e - d * 0.012 + d * (0.02 + boost), rot), rot)
        yield
    ctx.note(f"pushed: slab at {np.round(_s(ctx, 'slab').pos, 3)}, rover at {np.round(_s(ctx, 'rover').pos, 3)}")
    yield from A.move(A.pos - d * 0.03 + (0, 0, 0.06), None, 0.8)
    yield from A.go_home(2.5)


def hold_tab(task, ctx, A):
    """Take the rover's rear tow tab (toward the robot) from above: the bracing of a load before pushing what is on it."""
    st = _s(ctx, "rover")
    rv = task.rv
    tip = st.to_world((-(rv.S.LENGTH / 2 + 0.024), 0.0, rv.S.BUMPER_HEIGHT - rv.TOW_TAB[2] / 2 + 0.004))
    rot = A.grip_rotation(tcp_for_tip(tip, dk.DOWN_X), (0, 0, -1), st.rotation()[:, 1])
    yield from A.transit(tcp_for_tip(tip, rot) + (0, 0, 0.03), rot, 3.0, grip=1.0)
    yield from A.move(tcp_for_tip(tip, rot), rot, 0.8)
    yield from A.set_grip(0.0, 0.6)


def script(task, ctx, variant: str = ""):
    L, R = ctx.left, ctx.right
    yield from dk.wait(0.5)
    hx, hy = task.SLAB[0] / 2, task.SLAB[1] / 2
    if variant == "idle":
        yield from dk.wait(30.0)
        return
    if variant in ("push_along", "brace_and_push_along"):
        if variant == "brace_and_push_along":
            yield from hold_tab(task, ctx, L)
        yield from push(task, ctx, R, (1.0, 0.0, 0.0), (-hx, 0.0, 0.0), 0.15, 30.0)
        if variant == "brace_and_push_along":
            yield from L.set_grip(1.0, 0.4)
            yield from L.move(L.pos + (0, 0, 0.06), None, 0.6)
            yield from L.go_home(2.0)
        yield from dk.wait(3.0)
        return
    yield from push(task, ctx, R, (0.0, 1.0, 0.0), (0.0, -hy, 0.0), 0.24, 60.0)   # past the front tyres
    yield from dk.wait(1.5)
    st = _s(ctx, "slab")
    if task.slab_low(st) > task.GROUND + 0.004:                         # resting on a tyre or leaning on the side
        ctx.note(f"slab not down (lowest {1000 * (task.slab_low(st) - task.GROUND):.0f} mm up): pushed on")
        yield from push(task, ctx, R, (0.0, 1.0, 0.0), (0.0, -hy, 0.0), 0.06, 15.0)
    yield from dk.wait(3.0)
    st, rv_ = _s(ctx, "slab"), _s(ctx, "rover")
    ctx.note(f"end: slab {np.round(st.pos, 4)}, lowest {1000 * (task.slab_low(st) - task.GROUND):.1f} mm up, "
             f"gap to the rover {1000 * task.rover_gap(rv_, st):.1f} mm, over it {task.over_rover(st, rv_)}")


def debug(task, ctx) -> str:
    s = ctx.states()
    return f"rover {np.round(s['rover'].pos, 3)} slab {np.round(s['slab'].pos, 3)}"


if __name__ == "__main__":
    dk.main(script, debug)
