"""On the station table: a fixed earth-brown deck tilted 10 degrees, rising away from the robot. A small
yellow-and-white trail crawler (a 1:18 rover, driveshafts removed so its wheels roll freely, a tow bar on its rear
bumper) stands on it nose uphill, held only by a stop block behind its left front tyre. The robot must take the stop
block away and leave the rover parked on the slope.

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
import trailkit as tk  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Take the stop block away from the rover's wheel and leave the rover parked on the slope."

TOP = tk.TOP
TEX = HERE / "textures"

# -- layout (world frame: x forward from the arm bases, y to the robot's left, metres) ---------------------------
SLOPE_DEG = 10.0
SLOPE = math.radians(SLOPE_DEG)
UP = np.array([-math.sin(SLOPE), 0.0, math.cos(SLOPE)])     # the deck's surface normal (it rises toward +x)
ALONG = np.array([math.cos(SLOPE), 0.0, math.sin(SLOPE)])   # up the slope
DECK_L, DECK_W, DECK_T = 0.40, 0.40, 0.020          # wide enough for the rover turned across it
DECK_X0 = 0.16                                              # the deck's lower edge, on the table
TILT_Q = tk.quat_xyzw_from_axis_angle((0, 1, 0), -SLOPE)    # nose up
ROVER_S = 0.17                                              # the rover's origin, this far up the slope from the lower edge
P0 = (np.array([DECK_X0, 0.0, TOP]) + UP * DECK_T / math.cos(SLOPE) * math.cos(SLOPE) + ALONG * ROVER_S)
BLOCK = (0.030, 0.060, 0.030)                               # the stop block's section (along x, z); see STOP_SIZE
STOP_MASS = 0.190                          # rubber-soled aluminium, 3 x 8 x 3 cm (2.7 g/cm3 x 72 cm3): its friction
#                                            (0.9 x 1.83 N = 1.65 N) holds the rover's 0.87 N plus its own 0.32 N downhill
STOP_RED = (0.80, 0.15, 0.12)

_deck_centre = np.array([DECK_X0, 0.0, TOP]) + ALONG * DECK_L / 2 + UP * DECK_T / 2
DECK = tk.body("slope", pos=tuple(_deck_centre), rot_xyzw=TILT_Q, shapes=[
    tk.Shape.box((0, 0, 0), (DECK_L, DECK_W, DECK_T), (0.40, 0.31, 0.22), friction=(0.9, 0.8)),
    tk.Shape.tile((0, 0, DECK_T / 2 + 0.0004), (DECK_L, DECK_W), str(TEX / "earth.png"), texture_m=0.12),
])
TAB_SIZE = (0.048, 0.016, 0.020)
TAB_X = rv.S.LENGTH / 2 + TAB_SIZE[0] / 2 - 0.002
TAB_Z = rv.S.BUMPER_HEIGHT
TABS = (tk.Shape.box((-TAB_X, 0.0, TAB_Z), TAB_SIZE, (0.20, 0.20, 0.22), friction=(1.0, 0.9), tag="tow_r"),)
ROVER = rv.rover_l3("rover", pos=tuple(P0 + UP * 0.0008), rot_xyzw=TILT_Q, extra_shapes=TABS)
_rover0 = tk.state_at(ROVER, ROVER.pos, ROVER.rot_xyzw)


def _on_deck(local_xy) -> tuple:
    """World centre of a block resting on the deck at rover-frame (x, y)."""
    p = _rover0.to_world((local_xy[0], local_xy[1], 0.0)) - UP * 0.0008 + UP * (BLOCK[2] / 2 + 0.0005)
    return tuple(p)


STOP_SIZE = (BLOCK[0], 0.080, BLOCK[2])                      # long: its outer end lies well out past the rover's side
STOP_LOCAL = (rv.AXLE_X - rv.TYRE_R - BLOCK[0] / 2 - 0.001, 0.039 + STOP_SIZE[1] / 2)   # behind the left front tyre
STOP = tk.body("stop_block", pos=_on_deck(STOP_LOCAL), rot_xyzw=TILT_Q,
               shapes=[tk.Shape.box((0, 0, 0), STOP_SIZE, STOP_RED, friction=(0.9, 0.8))], mass=STOP_MASS)
OBJECTS = [DECK, ROVER, STOP]

# -- scoring -------------------------------------------------------------------------------------------------------
ON_BOARD = 0.008         # m: a tyre on the board: its lowest point within this of the board's surface, inside its outline
STILL_STEPS = 60         # control steps (3 s at 20 Hz) over which the rover must stay still
STILL_MOVE = 0.005       # m: ... its origin moving less than this in that time (a creep under 1.7 mm/s: within
#                          about 1.1 degrees of the contour; 3 degrees off it creeps 13 mm in 3 s)
STILL_SPIN = 0.5         # rad/s: ... and the stop block and the rover turning slower than this
AWAY = 0.03              # m: the stop block at least this far from every part of the rover
RELEASE = 0.05

REC = tk.Recorder()


def above_deck(p) -> float:
    return float(np.dot(np.asarray(p) - (np.array([DECK_X0, 0.0, TOP]) + UP * DECK_T), UP))


def inside_deck(p) -> bool:
    s_ = float(np.dot(np.asarray(p) - np.array([DECK_X0, 0.0, TOP]), ALONG))
    return 0.0 <= s_ <= DECK_L and abs(float(p[1])) <= DECK_W / 2


def box_grid(spec, st: tk.BodyState, n: int = 5) -> np.ndarray:
    """Points over the whole surface of a box body (a grid on each face), world frame."""
    h = np.asarray(spec.shapes[0].size) / 2
    g = np.linspace(-1.0, 1.0, n)
    pts = []
    for ax in range(3):
        for sgn in (-1.0, 1.0):
            for a in g:
                for b in g:
                    q = np.empty(3)
                    q[ax] = sgn
                    q[(ax + 1) % 3], q[(ax + 2) % 3] = a, b
                    pts.append(q * h)
    return st.pos + np.asarray(pts) @ st.rotation().T


def stop_away(rover: tk.BodyState, stop: tk.BodyState) -> bool:
    """The stop block taken away: every point of its surface at least 3 cm from the rover (tyres, body, tow bar)."""
    body = tk.distance_to_body(ROVER, rover, box_grid(STOP, stop))             # chassis, shell, tow bar
    tyres = np.vstack(list(rv.l3_tyre_points(ROVER, rover).values()))          # the tyres (on the wheel links)
    half = np.asarray(STOP.shapes[0].size) / 2
    tyre = float(tk.point_box_distance(tyres, stop.pos, stop.rotation(), half).min())
    return min(body, tyre) >= AWAY


def still(rec: tk.Recorder) -> bool:
    """The rover's origin moved less than 5 mm over the last 3 s, and nothing turned faster than 0.5 rad/s."""
    if not rec.full("pos", STILL_STEPS):
        return False
    p = np.asarray(rec.windows["pos"])
    return float(np.linalg.norm(p - p[-1], axis=1).max()) < STILL_MOVE and all(rec.windows["spin"])


def tyres_on_board(st: tk.BodyState) -> bool:
    lows = {n: pts[np.argmin(pts[:, 2])] for n, pts in rv.l3_tyre_points(ROVER, st).items()}
    return all(inside_deck(p) and abs(above_deck(p)) <= ON_BOARD for p in lows.values())


def tyre_off_board(st: tk.BodyState) -> bool:
    """A tyre's lowest point outside the board's outline (rolled or carried off the slope)."""
    lows = {n: pts[np.argmin(pts[:, 2])] for n, pts in rv.l3_tyre_points(ROVER, st).items()}
    return any(not inside_deck(p) for p in lows.values())


@tk.on_reset
def _reset(environment=None) -> None:
    REC.reset()


@tk.on_step
def _record(environment) -> None:
    REC.steps += 1
    rover = tk.body_state(environment, "rover")
    stop = tk.body_state(environment, "stop_block")
    REC.latch("off", tyre_off_board(rover), "a tyre left the board (the rover rolled off)")
    REC.window("pos", np.array(rover.pos, float), STILL_STEPS)
    REC.window("spin", rover.spin() < STILL_SPIN and stop.spin() < STILL_SPIN and stop.speed() < 0.01, STILL_STEPS)
    grips = tk.grippers(environment)
    REC.window("released", tk.released(ROVER, rover, grips, RELEASE) and tk.released(STOP, stop, grips, RELEASE),
               STILL_STEPS)
    REC.values["stop_away"] = stop_away(rover, stop)


def evaluate(rover: tk.BodyState, rec: tk.Recorder = REC) -> tuple[float, str]:
    if rec.flag("off"):
        return 0.0, rec.why.get("off", "off")
    if not still(rec):
        return 0.0, "the rover is not still (it moved 5 mm or more in the last 3 s)"
    if not (rec.full("released", STILL_STEPS) and all(rec.windows["released"])):
        return 0.0, "a gripper or an arm came within 5 cm of the rover or the stop block in the last 3 s"
    if not tyres_on_board(rover):
        return 0.0, "a tyre is off the board's surface"
    if not rec.values.get("stop_away", False):
        return 0.0, "the stop block is within 3 cm of the rover"
    return 1.0, "parked on the slope, stop block away"


_LAST: dict = {}


def score(objects) -> float:
    """0 if at any time a tyre leaves the board (the rover rolled off). Otherwise, at the end, with the rover still
    (its origin moving less than 5 mm over the last 3 s) and released over those 3 s: 1.0 with all four tyres on the
    board and every point of the stop block at least 3 cm from the rover; otherwise 0."""
    s, why = evaluate(objects["rover"])
    _LAST.update(score=s, why=why)
    return s


def success(objects) -> bool:
    return evaluate(objects["rover"])[0] == 1.0


def report() -> dict:
    p = np.asarray(REC.windows.get("pos", [np.zeros(3)]))
    return {"last": dict(_LAST), "fouls": {k: REC.why[k] for k in REC.flags if REC.flags[k]}, "steps": REC.steps,
            "still_window_mm": round(1000 * float(np.linalg.norm(p - p[-1], axis=1).max()), 2)}
