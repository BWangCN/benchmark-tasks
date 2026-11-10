"""Scripted full-score demo for the high-centred rover, and its near-miss variants.

    python demo.py task.py demo_out                               # full score: MP4 per camera, setup.png, result.json
    python demo.py task.py out --variant pull_rock --no-record     # pull the rock out from under the rover
    python demo.py task.py out --variant fast_pull --no-record     # the same pull, three times as fast
    python demo.py task.py out --variant hold_and_pull --no-record # hold the rear tow bar where it is and pull the rock
    python demo.py task.py out --variant idle --no-record          # do nothing

1. The left arm takes the rear tow bar from above and lifts the rear about 9 cm: the rover tips onto its front tyres
   and its chassis rises off the rock.
2. The right arm grips the end of the rock that sticks out past the rover's right side and slides the rock out
   sideways, clear of the rover, and lets go.
3. The left arm lowers the rear slowly onto the track and lets go.

The script reads only the rover's and the rock's poses and the arms' own joint state.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import demokit as dk  # noqa: E402
import trailkit as tk  # noqa: E402

REAR_TOOL = dk.tool_rotation((0.5, 0, -1), (0, -1, 0))      # from above, leaning 27 degrees forward, across the bar
ROCK_TOOL = dk.tool_rotation((0, 0, -1), (1, 0, 0))         # straight down, closing along x across the rock
BAR_GRIP_X = 0.011
REAR_LIFT = 0.090        # m: the rover pivots onto its front tyres and its chassis rises clear of the rock
GRIP_PART = 0.40


def _st(ctx, name) -> tk.BodyState:
    return ctx.states()[name]


def pitched(rot, pitch: float):
    c, s_ = math.cos(pitch), math.sin(pitch)
    return np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]]) @ rot


def rover_pitch(st: tk.BodyState) -> float:
    f = st.rotation()[:, 0]
    return math.atan2(-f[2], f[0])


def move_rear(task, ctx, A, dz: float, seconds: float):
    """Raise (dz > 0) or lower the rear by its bar, the hand turning with the rover's measured pitch."""
    z0, p0 = A.pos[2], A.pos.copy()
    n = int(seconds * dk.CONTROL_HZ)
    for k in range(1, n + 1):
        target = p0.copy()
        target[2] = z0 + dz * dk.smooth(k / n)
        A.aim(target, pitched(REAR_TOOL, rover_pitch(_st(ctx, "rover"))))
        if getattr(ctx, "trace", False) and k % 4 == 0:
            st = _st(ctx, "rover")
            rl = min(p[2] for n, p in task.tyre_lows(st).items() if n.startswith("wheel_r")) - task.GROUND
            ctx.note(f"rear {100 * rl:.1f} cm, pitch {math.degrees(rover_pitch(st)):.1f}")
        yield


def bar_point(task, st: tk.BodyState) -> np.ndarray:
    """Tool point on the rear bar from above: pad tips 2 mm below its underside."""
    centre = st.to_world((-(task.TAB_X + BAR_GRIP_X), 0.0, task.TAB_Z))
    return centre + REAR_TOOL[:, 2] * (task.TAB_SIZE[2] / 2 + 0.002 - 0.013)


def lift_rear(task, ctx, A):
    p = bar_point(task, _st(ctx, "rover"))
    yield from A.via(p - REAR_TOOL[:, 2] * 0.05, REAR_TOOL, 2.5, lift=0.05, grip=GRIP_PART)
    yield from A.move(p, REAR_TOOL, 0.8)
    yield from A.set_grip(0.0, 0.5)
    yield from move_rear(task, ctx, A, REAR_LIFT, 3.0)


def lower_rear(task, ctx, A):
    yield from move_rear(task, ctx, A, -(REAR_LIFT - 0.02), 3.5)
    for _ in range(int(4 * dk.CONTROL_HZ)):                 # on down, closed loop, until the rear tyres touch
        st = _st(ctx, "rover")
        if min(p[2] for n, p in task.tyre_lows(st).items() if n.startswith("wheel_r")) - task.GROUND < 0.002:
            break
        A.aim(A.pos - (0, 0, 0.001), pitched(REAR_TOOL, rover_pitch(st)))
        yield
    yield from dk.wait(0.3)
    yield from A.set_grip(1.0, 0.4)
    yield from A.move(A.pos - A.rot[:, 2] * 0.06, None, 0.8)
    yield from A.go_home(2.0)


def slide_rock_out(task, ctx, A, distance: float = 0.125, seconds: float = 3.0):
    rock = _st(ctx, "rock")
    end = rock.to_world((0.0, -task.ROCK_SIZE[1] / 2 + 0.015, task.ROCK_SIZE[2] / 2))   # 1.5 cm in from its end
    p = end + np.array([0.0, 0.0, 0.013 - 0.010])                                     # pad tips 1 cm below its middle
    yield from A.via(p + (0, 0, 0.05), ROCK_TOOL, 2.5, lift=0.05, grip=0.7)
    yield from A.move(p, ROCK_TOOL, 0.8)
    yield from A.set_grip(0.0, 0.6)
    yield from A.move(A.pos + (0, -distance, 0.004), ROCK_TOOL, seconds)
    yield from A.set_grip(0.7, 0.4)
    yield from A.move(A.pos + (0, 0, 0.06), ROCK_TOOL, 0.6)
    ctx.note(f"rock slid out to y {100 * _st(ctx, 'rock').pos[1]:.1f} cm")
    yield from A.go_home(2.0)


def script(task, ctx, variant: str = ""):
    L, R = ctx.left, ctx.right
    yield from dk.wait(0.5)
    if variant == "idle":
        yield from dk.wait(60.0)
        return
    if variant in ("pull_rock", "fast_pull"):
        yield from slide_rock_out(task, ctx, R, seconds=3.0 if variant == "pull_rock" else 1.0)
        yield from dk.wait(2.0)
        return
    if variant == "hold_and_pull":
        p = bar_point(task, _st(ctx, "rover"))
        yield from L.via(p - REAR_TOOL[:, 2] * 0.05, REAR_TOOL, 2.5, lift=0.05, grip=GRIP_PART)
        yield from L.move(p, REAR_TOOL, 0.8)
        yield from L.set_grip(0.0, 0.5)                 # hold the bar where it is: no lift
        yield from slide_rock_out(task, ctx, R)
        yield from dk.wait(1.0)
        yield from L.set_grip(1.0, 0.4)
        yield from L.move(L.pos - L.rot[:, 2] * 0.06, None, 0.8)
        yield from L.go_home(2.0)
        yield from dk.wait(2.0)
        return
    yield from lift_rear(task, ctx, L)
    ctx.note(f"rear lifted: chassis {100 * (task.chassis_low(_st(ctx, 'rover')) - task.GROUND):.1f} cm up, "
             f"pitch {math.degrees(-math.asin(_st(ctx, 'rover').rotation()[2, 0])):.1f} deg")
    yield from slide_rock_out(task, ctx, R)
    ctx.trace = True
    yield from lower_rear(task, ctx, L)
    yield from dk.wait(1.5)


def debug(task, ctx) -> str:
    st = _st(ctx, "rover")
    lz = " ".join(f"{n[-2:]}:{100 * (p[2] - task.GROUND):.1f}" for n, p in task.tyre_lows(st).items())
    return (f"rover x {100 * st.pos[0]:.1f} y {100 * st.pos[1]:.1f} chassis {100 * (task.chassis_low(st) - task.GROUND):.1f} "
            f"| tyres {lz} | rock y {100 * _st(ctx, 'rock').pos[1]:.1f} | {task.report()['fouls']} {task.report()['last']}")


if __name__ == "__main__":
    dk.main(script, debug)
