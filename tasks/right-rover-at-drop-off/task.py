"""On the station table: a raised earth track (a 5 cm deck) runs away from the robot. Its right-hand long edge,
seen from the robot, is a bare rock drop-off to the table below; a strip of turf lines its left-hand side. A small
yellow-and-white trail crawler (a 1:18 rover, 26 x 12 x 12 cm, 0.5 kg) lies on its left side on the track, nose away
from the robot, its tyres toward the drop-off and about 4 cm from the edge. The robot must get the rover back on its
four wheels on the track.

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

import rover as rv  # noqa: E402
import rover_spec as RS  # noqa: E402
import trailkit as tk  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "The rover has rolled onto its side on the track. Get it back on its wheels."

TOP = tk.TOP
TEX = HERE / "textures"

# -- layout (world frame: x forward from the arm bases, y to the robot's left, metres) ---------------------------
DECK_H = 0.050                       # track deck height above the table
DECK_TOP = TOP + DECK_H
DECK_X = (0.08, 0.46)                # near and far ends of the track
EDGE_Y = -0.06                       # the drop-off: the track's right-hand long edge
DECK_Y1 = 0.30                       # the track's left-hand side
VERGE_Y = 0.24                       # turf verge from here to the left-hand side
ROVER_X = 0.24                       # rover origin (midway between its axles), on its side
START_GAP = 0.04                     # tyre treads to the edge at the start
ROVER_Y = EDGE_Y + START_GAP
ROCK = (0.46, 0.45, 0.43)
EARTH = (0.46, 0.34, 0.23)

TRACK = tk.body("track", pos=((DECK_X[0] + DECK_X[1]) / 2, (EDGE_Y + DECK_Y1) / 2, TOP), shapes=[
    # the deck: painted board, its exposed long edge the rock face of the drop-off
    tk.Shape.span((-(DECK_X[1] - DECK_X[0]) / 2, -(DECK_Y1 - EDGE_Y) / 2, 0.0),
                  ((DECK_X[1] - DECK_X[0]) / 2, (DECK_Y1 - EDGE_Y) / 2, DECK_H), ROCK, friction=(0.6, 0.5)),
    # earth look on the track's top, and a turf strip along its left-hand side (visual)
    tk.Shape.tile((0.0, ((VERGE_Y - EDGE_Y) / 2) - (DECK_Y1 - EDGE_Y) / 2, DECK_H + 0.0004),
                  (DECK_X[1] - DECK_X[0], VERGE_Y - EDGE_Y), str(TEX / "earth.png"), texture_m=0.12),
    tk.Shape.tile((0.0, (DECK_Y1 - EDGE_Y) / 2 - (DECK_Y1 - VERGE_Y) / 2, DECK_H + 0.0006),
                  (DECK_X[1] - DECK_X[0], DECK_Y1 - VERGE_Y), str(TEX / "turf.png"), texture_m=0.08),
])
ROVER = rv.rover_l1("rover", pos=(ROVER_X, ROVER_Y, DECK_TOP + rv.side_origin_height()), rot_xyzw=rv.on_left_side(0.0))
OBJECTS = [TRACK, ROVER]

# -- scoring -------------------------------------------------------------------------------------------------------
FELL_BELOW = 0.010      # m: the rover has left the deck when its lowest point is this far below the deck top
UPRIGHT_DEG = 10.0      # deck (roof) within this of level
TYRE_ON = 0.010         # m: a tyre is on the deck when its lowest point is within this of the deck top
EDGE_CLEAR = 0.020      # m: tyres at least this far from the edge for full score
MOVED_BACK = 0.080      # m: farther from the edge than at the start, for 0.25
ON_SIDE_DEG = 30.0      # lying on a side: its axles (body y axis) within this of vertical
REST_SPEED, REST_SPIN, REST_STEPS = 0.01, 0.5, 20   # at rest: under 1 cm/s and 0.5 rad/s for 1 s (20 steps)
RELEASE = 0.05          # m: no gripper within this of the rover

REC = tk.Recorder()


def edge_gap(st: tk.BodyState) -> float:
    """Distance (m) from the drop-off edge to the rover's nearest point, seen from above (negative past it)."""
    return float(tk.body_points(ROVER, st.pos, st.rotation())[:, 1].min() - EDGE_Y)


def tyres_on_deck(st: tk.BodyState) -> bool:
    """All four tyres rest on the deck: each tyre's lowest point within 1 cm of the deck top and inside its outline."""
    R = st.rotation()
    for tag in rv.TYRE_TAGS:
        pts = tk.body_points(ROVER, st.pos, R, tags=(tag,))
        low = pts[np.argmin(pts[:, 2])]
        inside = DECK_X[0] <= low[0] <= DECK_X[1] and EDGE_Y <= low[1] <= DECK_Y1
        if not (inside and abs(low[2] - DECK_TOP) <= TYRE_ON):
            return False
    return True


def tyre_edge_gap(st: tk.BodyState) -> float:
    return min(lo for lo, _ in rv.tyre_box_ys(st).values()) - EDGE_Y


@tk.on_reset
def _reset(environment=None) -> None:
    REC.reset()
    if environment is not None:
        REC.values["start_gap"] = edge_gap(tk.body_state(environment, "rover"))


@tk.on_step
def _record(environment) -> None:
    st = tk.body_state(environment, "rover")
    REC.steps += 1
    REC.values.setdefault("start_gap", edge_gap(st))
    REC.latch("fell", tk.lowest_z(ROVER, st) < DECK_TOP - FELL_BELOW, "the rover left the deck")
    REC.extreme("max_speed", st.speed())
    REC.window("rest", st.speed() < REST_SPEED and st.spin() < REST_SPIN, REST_STEPS)
    REC.values["released"] = tk.released(ROVER, st, tk.grippers(environment), RELEASE)


def at_rest() -> bool:
    return REC.full("rest", REST_STEPS) and all(REC.windows["rest"])


def evaluate(st: tk.BodyState, rec: tk.Recorder = REC) -> tuple[float, str]:
    """(score, reason) from the recorder and the rover's final state."""
    if rec.flag("fell"):
        return 0.0, "the rover left the deck"
    if not (rec.full("rest", REST_STEPS) and all(rec.windows["rest"])):
        return 0.0, "not at rest"
    if not rec.values.get("released", False):
        return 0.0, "a gripper is within 5 cm of the rover"
    upright = tk.tilt_deg(st.rotation()) <= UPRIGHT_DEG
    if upright and tyres_on_deck(st):
        if tyre_edge_gap(st) >= EDGE_CLEAR:
            return 1.0, "upright on the track, clear of the edge"
        return 0.5, "upright on the track, a tyre within 2 cm of the edge"
    on_side = abs(st.rotation()[2, 1]) >= math.cos(math.radians(ON_SIDE_DEG))
    if on_side and edge_gap(st) - rec.values.get("start_gap", START_GAP) >= MOVED_BACK:
        return 0.25, "still on its side, moved 8 cm back from the edge"
    return 0.0, "not righted"


_LAST: dict = {}


def score(objects) -> float:
    """0 if at any time the rover left the deck (its lowest point more than 1 cm below the deck top). Otherwise, at
    the end, at rest and released: 1.0 upright (roof within 10 degrees of level) with all four tyres on the deck and
    at least 2 cm from the edge; 0.5 upright on the deck with at least one tyre closer than 2 cm to the edge; 0.25 still on a
    side (axles within 30 degrees of vertical) but at least 8 cm farther from the edge than at the start."""
    s, why = evaluate(objects["rover"])
    _LAST.update(score=s, why=why)
    return s


def success(objects) -> bool:
    return evaluate(objects["rover"])[0] == 1.0


def report() -> dict:
    return {"last": dict(_LAST), "fouls": {k: REC.why[k] for k in REC.flags if REC.flags[k]},
            "start_gap_cm": round(100 * REC.values.get("start_gap", float("nan")), 1),
            "max_speed": round(REC.extremes.get("max_speed", 0.0), 3)}
