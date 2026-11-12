"""On the station table: three smooth (greased) steel pipe sections lie loose on an earth-brown sheet beside a grey
pad that is too narrow for them to lie side by side; two rubber-soled steel chocks lie on the other side. The robot
must stack the three pipes on the pad. Two pipes on the ground with the third on them stand only with the bottom pair
chocked: between pipes this smooth the top pipe's weight drives the bottom pair apart.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import rubble as rb  # noqa: E402
import trailkit as tk  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Stack the three pipes on the pad, lengthwise along it."

TOP = tk.TOP
TEX = HERE / "textures"

# -- layout (world frame: x forward from the arm bases, y to the robot's left, metres) ---------------------------
SHEET_T = 0.0006
GROUND = TOP + SHEET_T
PAD = ((0.14, 0.36), (-0.06, 0.06))     # x, y: 22 x 12 cm: three pipes side by side (15 cm) would overhang it
PAD_T = 0.002
PAD_TOP = GROUND + PAD_T
R, L, PIPE_MASS = 0.025, 0.12, 0.25           # short, thin-walled steel pipe sections
PIPE_MU = 0.15                          # greased steel on steel (published static values 0.11 to 0.16)
PIPE_STARTS = ((0.22, -0.22), (0.22, -0.31), (0.22, -0.40))   # 9 cm apart: room for the fingers
CHOCK = (0.12, 0.04, 0.04)
CHOCK_MASS = 7800 * CHOCK[0] * CHOCK[1] * CHOCK[2]      # steel, rubber-soled: 1.5 kg
CHOCK_MU = 0.9                          # rubber soles
CHOCK_STARTS = ((0.20, 0.20), (0.20, 0.29))
EARTH = (0.42, 0.31, 0.21)
PAD_GREY = (0.62, 0.62, 0.60)

SHEET = tk.body("sheet", pos=(0.27, 0.0, TOP), shapes=[
    tk.Shape.span((-0.17, -0.45, 0.0), (0.17, 0.45, SHEET_T), EARTH, friction=(0.5, 0.45)),
    tk.Shape.tile((0, 0, SHEET_T + 0.0002), (0.34, 0.90), str(TEX / "earth.png"), texture_m=0.12),
    tk.Shape.span((PAD[0][0] - 0.27, PAD[1][0], SHEET_T), (PAD[0][1] - 0.27, PAD[1][1], SHEET_T + PAD_T), PAD_GREY,
                  friction=(0.5, 0.45), tag="pad")])
PIPES = [rb.pipe(f"pipe_{i}", (x, y, GROUND + R + 0.0003), R, L, "x", PIPE_MASS, (PIPE_MU, PIPE_MU))
         for i, (x, y) in enumerate(PIPE_STARTS)]
CHOCKS = [rb.block(f"chock_{i}", (x, y, GROUND + CHOCK[2] / 2 + 0.0003), CHOCK, mass=CHOCK_MASS,
                   friction=(CHOCK_MU, CHOCK_MU)) for i, (x, y) in enumerate(CHOCK_STARTS)]
OBJECTS = [SHEET, *PIPES, *CHOCKS]

# -- scoring -------------------------------------------------------------------------------------------------------
ON_PAD = 0.004            # m: a pipe's axis within this of R above the pad's top: it lies on the pad
CONTACT = 0.003           # m: the top pipe rests on a bottom pipe when their axes are within 2R + this
TOP_UP = 0.040            # m: the third pipe's axis at least this far above the pad: it rests on the other two
#                           (6.8 cm with the bottom pair touching, 5.4 cm with it 3 cm apart; on the pad 2.5 cm)
REST_MOVE = 0.002         # m: no pipe's centre moved more than this over the last REST_STEPS (positions, not speeds:
#                           a stack of round pipes can keep turning in place)
REST_STEPS = 40
RELEASE = 0.05
ON_CHOCK = 0.003          # m: a chock point this close under the top pipe, along its length: it rests on the chock

REC = tk.Recorder()


def axis_ends(st: tk.BodyState) -> np.ndarray:
    """The two ends of a pipe's axis (world)."""
    return np.array([st.to_world((0, 0, -L / 2)), st.to_world((0, 0, L / 2))])


def inside_pad(st: tk.BodyState) -> bool:
    e = axis_ends(st)
    return bool(np.all((e[:, 0] >= PAD[0][0]) & (e[:, 0] <= PAD[0][1]) & (e[:, 1] >= PAD[1][0]) & (e[:, 1] <= PAD[1][1])))


def axis_height(st: tk.BodyState) -> float:
    return float(axis_ends(st)[:, 2].mean() - PAD_TOP)


def lying(st: tk.BodyState) -> bool:
    """A pipe lying with its axis along x (within about 18 degrees)."""
    e = axis_ends(st)
    a = (e[1] - e[0]) / max(float(np.linalg.norm(e[1] - e[0])), 1e-9)
    return abs(float(a[0])) >= 0.95


def on_pad(st: tk.BodyState) -> bool:
    return inside_pad(st) and lying(st) and abs(axis_height(st) - R) <= ON_PAD


def under_top(top: tk.BodyState, chocks: list | None) -> bool:
    """Is any chock point within ON_CHOCK of the top pipe's surface, below its axis and along its length (a chock
    holding it up)?"""
    for spec, st in zip(CHOCKS, chocks or []):
        pts = rb.sample_points(spec, st, 0.005)
        local = (pts - np.asarray(top.pos, float)) @ top.rotation()          # the pipe's axis is its local z
        radial = np.hypot(local[:, 0], local[:, 1])
        world_below = pts[:, 2] < float(top.pos[2])
        if np.any((np.abs(local[:, 2]) < L / 2 - 0.002) & (radial <= R + ON_CHOCK) & world_below):   # not the end faces
            return True
    return False


def stacked(pipes: list, chocks: list | None = None) -> tuple[int, str]:
    """2: a pyramid on the pad; 1: two pipes lying on the pad and no full pyramid; 0 otherwise."""
    low = [p for p in pipes if on_pad(p)]
    if len(low) >= 2:
        for top in pipes:
            others = [p for p in pipes if p is not top]
            if not all(on_pad(p) for p in others) or not inside_pad(top):
                continue
            ys = sorted(float(p.pos[1]) for p in others)
            touching = all(math.hypot(float(top.pos[1] - p.pos[1]), float(top.pos[2] - p.pos[2])) <= 2 * R + CONTACT
                           for p in others)
            over = all(abs(float(top.pos[0] - p.pos[0])) <= L / 2 for p in others)
            if (all(lying(p) for p in pipes) and axis_height(top) >= TOP_UP and ys[0] < float(top.pos[1]) < ys[1]
                    and touching and over and not under_top(top, chocks)):
                return 2, f"stacked: top pipe's axis {100 * axis_height(top):.1f} cm above the pad"
        return 1, "two pipes on the pad, no full pyramid"
    return 0, f"{len(low)} pipe(s) lying on the pad"


def released_all(pipes: list, grips: dict) -> bool:
    return all(tk.released(spec, st, grips, RELEASE) for spec, st in zip(PIPES, pipes))


@tk.on_reset
def _reset(environment=None) -> None:
    REC.reset()


@tk.on_step
def _record(environment) -> None:
    REC.steps += 1
    pipes = [tk.body_state(environment, p.name) for p in PIPES]
    hist = REC.values.setdefault("centres", [])
    hist.append(np.array([p.pos for p in pipes]))
    del hist[:-REST_STEPS]
    REC.window("rest", len(hist) == REST_STEPS and float(np.max(np.linalg.norm(hist[-1] - hist[0], axis=1))) < REST_MOVE,
               REST_STEPS)
    REC.window("released", released_all(pipes, tk.grippers(environment)), REST_STEPS)


def evaluate(pipes: list, rec: tk.Recorder = REC, chocks: list | None = None) -> tuple[float, str]:
    if not (rec.full("rest", REST_STEPS) and all(rec.windows["rest"])):
        return 0.0, "the pipes not at rest for 2 s"
    if not (rec.full("released", REST_STEPS) and all(rec.windows["released"])):
        return 0.0, "a gripper or an arm within 5 cm of a pipe in the last 2 s"
    level, why = stacked(pipes, chocks)
    return {2: 1.0, 1: 0.25, 0: 0.0}[level], why


_LAST: dict = {}


def score(objects) -> float:
    """At the end, after 2 s with the three pipes at rest (no pipe's centre moved more than 2 mm over the last 2 s)
    and released (no point of either gripper, and no point along either arm from its shoulder joint to its palm,
    within 5 cm of any pipe): 1.0 the three pipes' axes wholly inside the pad's outline, two lying on the pad (axis
    2.5 cm above it, within 4 mm) and the third resting on both (all three lying along x, within about 18 degrees;
    its axis between theirs along y, at least 4 cm above the pad, within 5.3 cm of each of theirs so that it touches
    both, its middle within 6 cm of theirs along x, and no chock within 3 mm under it along its length: it rests on
    the pipes, not on a chock); 0.25 two pipes lying on the pad (lying along x, within about 18 degrees, their axes
    inside the pad's outline and 2.5 cm above it, within 4 mm) and no full pyramid; 0 otherwise. A full score ends
    the trial once it has held for 1 s."""
    s, why = evaluate([objects[p.name] for p in PIPES], REC, [objects[c.name] for c in CHOCKS])
    _LAST.update(score=s, why=why)
    return s


def success(objects) -> bool:
    return evaluate([objects[p.name] for p in PIPES], REC, [objects[c.name] for c in CHOCKS])[0] == 1.0


def report() -> dict:
    return {"last": dict(_LAST), "steps": REC.steps}
