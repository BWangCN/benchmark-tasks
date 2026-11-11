"""Scripted full-score demo for the ledge-descent task, and its near-miss variants.

    python demo.py task.py demo_out                               # full score: MP4 per camera, setup.png, result.json
    python demo.py task.py out --variant push_off --no-record     # push the rover off the edge (the front drops)
    python demo.py task.py out --variant lift_down --no-record    # carry it down by both tabs (lifted clear)
    python demo.py task.py out --variant front_then_push --no-record   # lower the front, then push it off (the rear drops)
    python demo.py task.py out --variant hold_then_drop --no-record    # hold the front 6 mm under deck height, then let go
    python demo.py task.py out --variant idle --no-record         # do nothing

1. The right arm takes the front tow tab from above, rolls the rover forward on its rear wheels until the front
   wheels are past the edge, and lowers the nose onto the lower track, turning the hand with the rover's pitch.
2. The left arm takes the rear tow tab before the right lets go (the rover, nose down on free wheels, would roll).
3. The left arm rolls the rover on until the rear wheels are past the edge, lowers the rear axle, and lets go.

The script reads only the rover's pose and the arms' own joint state.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import demokit as dk  # noqa: E402
import trailkit as tk  # noqa: E402

GRIP_PART = 0.35                       # part-open: about 33 mm between the fingers, for the 20 mm tall bar
CARRY_SPEED = 0.03                     # m/s
TAB_GRIP_X = 0.011                     # m: grip the bar this far beyond its centre (fingers clear of the bumper)
TIP_PAST = 0.004                       # m: pad tips this far past the bar's far face


def _rover(ctx) -> tk.BodyState:
    return ctx.states()["rover"]


def pitched(rot, pitch: float):
    """A tool rotation turned about the world y axis by the rover's pitch (positive: nose down)."""
    c, s = math.cos(pitch), math.sin(pitch)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return Ry @ rot


def rover_pitch(st: tk.BodyState) -> float:
    fwd = st.rotation()[:, 0]
    return math.atan2(-fwd[2], fwd[0])


def side_tool(side: str, pitch: float = 0.0):
    """Reaching across the bar from the arm's own side, tilted down a little, the fingers closing on its top and
    bottom; turned with the rover's pitch."""
    s = 1.0 if side == "left" else -1.0
    if side == "left":   # the rear bar: from above, leaning 27 degrees forward (the housing over the robot's side)
        return pitched(dk.tool_rotation((0.5, 0, -1), (0, -1, 0)), pitch)
    return pitched(dk.tool_rotation((0, -s, -0.3), (0, 0, -1)), pitch)     # 17 degrees down: the housing clears the deck


def tool_offset(task, rot) -> np.ndarray:
    """From the bar's middle to the tool point: the pad tips just past the bar's far face, 13 mm past the tool point."""
    across = task.TAB_SIZE[2] if abs(rot[2, 2]) > 0.7 else task.TAB_SIZE[1]     # from above: the bar's height
    return rot[:, 2] * (across / 2 + TIP_PAST - 0.013)


def tab_point(task, st: tk.BodyState, end: int, side: str) -> np.ndarray:
    """World point to put the tool at, for the front (end=+1) or rear (end=-1) tow bar."""
    local = np.array([end * (task.TAB_X + TAB_GRIP_X), 0.0, task.TAB_Z])
    return st.to_world(local) + tool_offset(task, side_tool(side, rover_pitch(st)))


def take_tab(task, ctx, A, end: int):
    st = _rover(ctx)
    rot = side_tool(A.side, rover_pitch(st))
    p = tab_point(task, st, end, A.side)
    yield from A.via(p - rot[:, 2] * 0.06, rot, 2.5, lift=0.05, grip=GRIP_PART)    # in line with the bar, then onto it
    yield from A.move(p, rot, 0.8)
    yield from A.set_grip(0.0, 0.5)
    ctx.note(f"{'front' if end > 0 else 'rear'} tab taken at x {100 * p[0]:.1f} cm")


def carry_axle(task, ctx, A, end: int, axle_to: float, lower: bool, lower_s: float = 3.0):
    """Roll the held end forward until its axle is at ``axle_to``, then (if ``lower``) set it down on the lower
    track, turning the hand with the rover's pitch. The other axle rolls along on its wheels."""
    task_rv = task.rv
    for _ in range(int(12 * dk.CONTROL_HZ)):                 # closed loop on the held axle's measured position
        axle = _rover(ctx).to_world((end * task_rv.AXLE_X, 0.0, task_rv.TYRE_R))
        if axle[0] >= axle_to:
            break
        A.aim(A.pos + (CARRY_SPEED / dk.CONTROL_HZ, 0, 0), A.rot)
        yield
    if not lower:
        return
    # where the tab ends up: the held axle on the lower track, the other axle where it stands
    st = _rover(ctx)
    other = st.to_world((-end * task_rv.AXLE_X, 0.0, task_rv.TYRE_R))
    held = np.array([axle_to, 0.0, task.TOP + task_rv.TYRE_R])
    base = task_rv.S.WHEELBASE
    if end > 0:                                   # front down, rear on the deck: nose down
        held[0] = other[0] + math.sqrt(max(base ** 2 - (other[2] - held[2]) ** 2, 0.0))
        pitch = math.asin((other[2] - held[2]) / base)
    else:                                         # rear down beside the front: level
        held[0] = other[0] - base
        pitch = 0.0
    held[2] += 0.003                              # set the tyres down, not pressed in
    # move the hand as the held axle must move (wherever on the bar it holds): turn by the change of pitch about
    # the axle, then carry the axle to its place
    axle = st.to_world((end * task_rv.AXLE_X, 0.0, task_rv.TYRE_R))
    turn = pitch - rover_pitch(st)
    target = held + pitched(np.eye(3), turn) @ (A.pos - axle)
    rot = pitched(A.rot, turn)
    yield from A.move(target, rot, lower_s)
    ctx.note(f"{'front' if end > 0 else 'rear'} axle set down, pitch {math.degrees(rover_pitch(_rover(ctx))):.1f} deg")


def let_go(A, out: float = 0.07):
    yield from A.set_grip(1.0, 0.4)                                   # fully open: no finger catches the bar
    yield from A.move(A.pos - A.rot[:, 2] * out, None, 0.8)          # back out the way the hand came in
    yield from A.move(A.pos + (0, 0, 0.05), None, 0.5)


def push_rover(task, ctx, A, until_x: float):
    """Push the rover by the end of its rear tow bar, fingers closed and pointing forward, until it is at until_x."""
    tool = dk.tool_rotation((1, 0, -0.2), (0, -1, 0))
    face = _rover(ctx).to_world((-task.TAB_X - task.TAB_SIZE[0] / 2, 0.0, task.TAB_Z))     # the rear bar's end
    yield from A.via(face - tool[:, 2] * 0.023, tool, 3.0, lift=0.06, grip=0.0)
    for _ in range(int(20 * dk.CONTROL_HZ)):
        st = _rover(ctx)
        if st.pos[0] > until_x:
            break
        face = st.to_world((-task.TAB_X - task.TAB_SIZE[0] / 2, 0.0, task.TAB_Z))
        A.aim(face - tool[:, 2] * 0.010 + np.array([0.03 / dk.CONTROL_HZ, 0, 0]), tool)
        yield
    yield from A.move(A.pos - tool[:, 2] * 0.05, tool, 0.8)
    yield from A.go_home(2.0)


def script(task, ctx, variant: str = ""):
    L, R = ctx.left, ctx.right
    yield from dk.wait(0.5)
    if variant == "idle":
        yield from dk.wait(60.0)
        return
    edge = task.DECK_X[1]
    if variant == "push_off":
        yield from push_rover(task, ctx, R, edge + 0.16)
        yield from dk.wait(2.0)
        return
    if variant == "lift_down":
        yield from dk.together(take_tab(task, ctx, R, +1), take_tab(task, ctx, L, -1))
        yield from dk.together(L.move(L.pos + (0, 0, 0.05), None, 1.0), R.move(R.pos + (0, 0, 0.05), None, 1.0))
        yield from dk.together(L.move(L.pos + (0.22, 0, 0), None, 3.0), R.move(R.pos + (0.22, 0, 0), None, 3.0))
        yield from dk.together(L.move(L.pos + (0, 0, -0.088), None, 2.0), R.move(R.pos + (0, 0, -0.088), None, 2.0))
        yield from dk.together(let_go(L), let_go(R))
        yield from dk.together(L.go_home(2.0), R.go_home(2.0))
        yield from dk.wait(2.0)
        return
    if variant == "hold_then_drop":                  # hold the front just under deck height for 1 s, then let it fall
        yield from take_tab(task, ctx, R, +1)
        yield from carry_axle(task, ctx, R, +1, edge + 0.045, lower=False)
        yield from R.move(R.pos + (0, 0, -0.006), None, 1.0)
        yield from dk.wait(1.0)
        yield from let_go(R)
        yield from R.go_home(2.0)
        yield from dk.wait(2.0)
        return
    # 1. the front axle over the edge and down
    yield from take_tab(task, ctx, R, +1)
    yield from carry_axle(task, ctx, R, +1, edge + 0.045, lower=True)
    yield from dk.wait(0.5)
    if variant == "front_then_push":                 # lower the front properly, then push the rear off the edge
        yield from let_go(R)
        yield from R.go_home(2.0)
        yield from push_rover(task, ctx, R, edge + 0.16)
        yield from dk.wait(2.0)
        return
    # 2. the left arm holds the rear before the right lets go
    yield from take_tab(task, ctx, L, -1)
    yield from let_go(R)
    yield from R.go_home(2.0)
    # 3. the rear axle over the edge and down
    yield from carry_axle(task, ctx, L, -1, edge + 0.045, lower=True)
    yield from dk.wait(0.5)
    yield from let_go(L)
    yield from L.go_home(2.0)
    yield from dk.wait(1.5)


def debug(task, ctx) -> str:
    st = _rover(ctx)
    lows = task.tyre_lows(st)
    lz = " ".join(f"{n[-2:]}:{100 * (p[2] - task.TOP):.1f}" for n, p in lows.items())
    rep = task.report()
    gs = tk.grippers(ctx.emb._env)
    grip = " ".join(f"{side[0].upper()} tip-bar {np.round(100 * (gs[side]['points'][-1] - st.to_world((e * task.TAB_X, 0.0, task.TAB_Z))), 1)} "
                    f"open {1000 * gs[side]['opening']:.0f}mm" for side, e in (("right", 1), ("left", -1)))
    return (f"{grip} | rover x {100 * st.pos[0]:.1f} pitch {math.degrees(rover_pitch(st)):.1f} v {st.speed():.3f} | "
            f"tyre z {lz} | first {rep['first_down']} fouls {rep['fouls']} | {rep['last']}")


if __name__ == "__main__":
    dk.main(script, debug)
