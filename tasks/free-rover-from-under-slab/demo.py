"""Scripted full-score demo for getting the rover out from under the rubble, and its near-miss variants.

    python demo.py task.py demo_out                                   # full score
    python demo.py task.py out --variant timber_first --no-record     # lift the timber out of the way first
    python demo.py task.py out --variant pull_only --no-record        # draw the rover out with the timber in place
    python demo.py task.py out --variant idle --no-record             # do nothing

1. The right arm takes the bent bar on the slab's top near the drum end and holds the slab's end where it is.
2. The left arm lifts the fallen timber at its middle, carries it over to the far side and sets it down out of the
   rover's way. The timber was what held the drum on its face: the drum rolls to its lip, and the slab's far end
   settles while the right hand keeps its bar at height.
3. The left arm takes the rover's front tow tab from above and draws it out along its wheels onto the pad.
4. The right arm lowers the slab's end onto what is left under it and lets go.

The script reads only the poses of the rover, the slab and the timber, and the arms' own joint state.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import demokit as dk  # noqa: E402
import trailkit as tk  # noqa: E402

LEAD = 0.02
TILT = math.radians(15.0)
TRACE = bool(__import__("os").environ.get("DEMO_TRACE"))


def pad():
    from yam_arena.constants import PAD_TIP_PAST_TCP
    return PAD_TIP_PAST_TCP


def tcp_for_tip(point, rot):
    return np.asarray(point, float) - rot[:, 2] * pad()


def _s(ctx, name) -> tk.BodyState:
    return ctx.states()[name]


def heading(st):
    hd = st.rotation()[:, 0]
    return np.array([hd[0], hd[1], 0.0]) / max(math.hypot(hd[0], hd[1]), 1e-9)


def best_rotation(A, tip, tilt, close):
    """The tool tilted ``tilt`` from straight down, toward whichever side the arm reaches best, closing along
    ``close`` (made square to the approach)."""
    from yam_arena import ik
    best = None
    for d in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        a = np.array([d[0] * math.sin(tilt), d[1] * math.sin(tilt), -math.cos(tilt)])
        c = np.asarray(close, float) - a * float(np.asarray(close, float) @ a)
        for cc in (c, -c):
            rot = dk.tool_rotation(a, cc)
            q, pe, re = ik.solve(tcp_for_tip(tip, rot), ik.matrix_to_quat_xyzw(rot), A.q, base_pos=A.base, iters=120)
            margin = min(q[5] - ik.LOWER[5], ik.UPPER[5] - q[5])
            key = (pe > 0.003 or re > 0.05, pe, -margin)
            if best is None or key < best[0]:
                best = (key, rot)
    return best[1]


def hold_slab(task, ctx, R):
    tip = task.tab_point(_s(ctx, "slab")) - np.array([0, 0, 0.004])
    rot = best_rotation(R, tip, TILT, (1, 0, 0))
    yield from R.transit(tcp_for_tip(tip, rot) + (0, 0, 0.04), rot, 3.0, height=0.20, grip=1.0)
    yield from R.move(tcp_for_tip(tip, rot), rot, 0.8)
    yield from R.set_grip(0.0, 0.6)
    ctx.note(f"slab held: tab at {np.round(task.tab_point(_s(ctx, 'slab')), 3)}, hand {np.round(R.pos, 3)}")


def lift_timber(task, ctx, L):
    """Take the timber at its middle from above (fingers across it), lift it 5 cm, carry it away from the rover along
    +y and then over to the far side along +x, set it down and let go."""
    t = _s(ctx, "timber")
    grip = t.pos + np.array([0, 0, task.TIMBER_H / 2 - 0.012])
    rot = L.grip_rotation(tcp_for_tip(grip, dk.DOWN_X), (0, 0, -1), (0, 1, 0))
    yield from L.transit(tcp_for_tip(grip, rot) + (0, 0, 0.04), rot, 3.0, height=0.14, grip=1.0)
    yield from L.move(tcp_for_tip(grip, rot), rot, 0.8)
    yield from L.set_grip(0.0, 0.6)
    yield from L.move(L.pos + (0, 0, 0.05), rot, 1.2)
    ctx.note(f"timber lifted: {np.round(_s(ctx, 'timber').pos, 3)}; drum {np.round(_s(ctx, 'drum').pos, 3)}")
    yield from L.move(L.pos + (0, 0.135, 0), rot, 1.5)
    yield from L.move(L.pos + (0.20, 0, 0), rot, 2.0)
    yield from L.move(L.pos - (0, 0, 0.045), rot, 1.0)
    yield from L.set_grip(1.0, 0.4)
    yield from L.move(L.pos + (0, 0, 0.05), None, 0.6)
    ctx.note(f"timber set down at {np.round(_s(ctx, 'timber').pos, 3)}")
    yield from L.go_home(2.0)


def front_tab(task, st):
    rv = task.rv
    return st.to_world((rv.S.LENGTH / 2 + 0.024, 0.0, rv.S.BUMPER_HEIGHT - rv.TOW_TAB[2] / 2 + 0.004))


def rear_low_y(task, st) -> float:
    pts = task.tyre_points(st)
    return min(float(pts[n][:, 1].min()) for n in ("wheel_rl", "wheel_rr"))


def draw_out(task, ctx, A, until_rear_y: float):
    st = _s(ctx, "rover")
    tip = front_tab(task, st)
    rot = A.grip_rotation(tcp_for_tip(tip, dk.DOWN_X), (0, 0, -1), st.rotation()[:, 1])
    yield from A.transit(tcp_for_tip(tip, rot) + (0, 0, 0.03), rot, 3.0, grip=1.0)
    yield from A.move(tcp_for_tip(tip, rot), rot, 0.8)
    yield from A.set_grip(0.0, 0.6)
    for k in range(int(40 * dk.CONTROL_HZ)):
        st = _s(ctx, "rover")
        if rear_low_y(task, st) >= until_rear_y:
            break
        A.aim(tcp_for_tip(front_tab(task, st) + heading(st) * LEAD, rot), rot)
        yield
    ctx.note(f"drawn out: rear tyres' lowest y {rear_low_y(task, _s(ctx, 'rover')):.3f}")
    yield from A.set_grip(1.0, 0.5)
    yield from A.move(A.pos + (0, 0, 0.06), None, 0.6)
    yield from A.go_home(2.0)


def steady(task, ctx, R, gen, z0, base, rot):
    """Run the other arm's moves while R keeps the held tab at height z0: the loaded arm sags 1 to 2 cm as the slab's
    weight comes onto it (when the drum rolls), so its hand is raised by what the tab is measured to have sunk."""
    lift = 0.0
    for _ in gen:
        err = z0 - task.tab_point(_s(ctx, "slab"))[2]
        lift = float(np.clip(lift + 0.4 * err, 0.0, 0.03))
        R.aim(base + np.array([0.0, 0.0, lift]), rot)
        yield


def lower_slab(task, ctx, R):
    """Lower the held end at 2 cm/s until it stops on what is under it, then let go."""
    last, still = None, 0
    for _ in range(int(10 * dk.CONTROL_HZ)):
        R.aim(R.pos - np.array([0, 0, 0.02 / dk.CONTROL_HZ]), R.rot)
        yield
        z = task.tab_point(_s(ctx, "slab"))[2]
        still = still + 1 if last is not None and abs(z - last) < 0.0003 else 0
        last = z
        if still > 8:
            break
    yield from R.set_grip(1.0, 0.5)
    yield from R.move(R.pos + (0, 0, 0.05), None, 0.6)
    ctx.note(f"slab let go: tab at {np.round(task.tab_point(_s(ctx, 'slab')), 3)}")
    yield from R.go_home(2.0)


def script(task, ctx, variant: str = ""):
    L, R = ctx.left, ctx.right
    yield from dk.wait(0.5)
    if variant == "idle":
        yield from dk.wait(30.0)
        return
    goal = task.PAD_Y0 + 0.012
    if variant == "timber_first":
        yield from lift_timber(task, ctx, L)
        yield from draw_out(task, ctx, L, goal)
        yield from dk.wait(2.5)
        return
    if variant == "pull_only":
        yield from draw_out(task, ctx, L, goal)
        yield from dk.wait(2.5)
        return
    yield from hold_slab(task, ctx, R)
    z0, base, rot = task.tab_point(_s(ctx, "slab"))[2] + 0.002, R.pos.copy(), R.rot.copy()
    yield from steady(task, ctx, R, lift_timber(task, ctx, L), z0, base, rot)
    yield from steady(task, ctx, R, dk.wait(1.0), z0, base, rot)
    ctx.note(f"drum now {np.round(_s(ctx, 'drum').pos, 3)}; slab drop so far {task.report()['slab_drop_mm']} mm")
    yield from steady(task, ctx, R, draw_out(task, ctx, L, goal), z0, base, rot)
    yield from lower_slab(task, ctx, R)
    yield from dk.wait(2.5)


def debug(task, ctx) -> str:
    s = ctx.states()
    return f"rover {np.round(s['rover'].pos[:2], 3)} drum {np.round(s['drum'].pos, 3)} | {task.report()}"


if __name__ == "__main__":
    dk.main(script, debug)
