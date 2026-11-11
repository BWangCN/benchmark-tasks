"""On the station table: an earth track 57 cm wide runs across the table from the robot's right to its left. Across
it runs a washout, a gap 5 cm wide and 3 cm deep; beyond it, a marked pad. A loaded trail crawler (a 1:18 rover
built heavy, 14 kg in all; driveshafts removed, so its wheels roll freely; steering servo
unpowered, so the front wheels stay where they are put) stands on the near side, wheels straight, square to the gap.
The robot must push it across the washout onto the pad.

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

INSTRUCTION = "The rover has lost power. Push it across the washout to the pad without lifting it."

TOP = tk.TOP
TEX = HERE / "textures"

# -- layout (world frame: x forward from the arm bases, y to the robot's left, metres) ---------------------------
TRACK_X = (0.13, 0.70)           # 57 cm wide: crossed at about 34 degrees, the rover drifts
#                                  0.5 cm sideways for each cm it advances, and it must end wholly on the pad
TRACK_Y = (-0.45, 0.56)
DECK_T = 0.032                   # the track's slab: the gap is cut 3 cm into it
GAP_Y = (0.15, 0.20)             # the washout: 5 cm wide, across the track
GAP_D = 0.030
PAD_Y = (0.22, 0.56)             # the pad: the far end of the track
GROUND = TOP + DECK_T            # the track's surface
ROVER_XY = (0.20, -0.12)         # start: heading +y (toward the robot's left), square to the gap
EARTH = (0.42, 0.31, 0.21)
DARK = (0.24, 0.18, 0.13)
LINE = (0.93, 0.93, 0.90)


def _track():
    cx, cy = (TRACK_X[0] + TRACK_X[1]) / 2, (TRACK_Y[0] + TRACK_Y[1]) / 2
    hx = (TRACK_X[1] - TRACK_X[0]) / 2
    y0, y1 = TRACK_Y[0] - cy, TRACK_Y[1] - cy
    g0, g1 = GAP_Y[0] - cy, GAP_Y[1] - cy
    p0, p1 = PAD_Y[0] - cy, PAD_Y[1] - cy
    fr = (0.9, 0.8)
    shapes = [tk.Shape.span((-hx, y0, 0), (hx, g0, DECK_T), EARTH, fr, "near"),
              tk.Shape.span((-hx, g1, 0), (hx, y1, DECK_T), EARTH, fr, "far"),
              tk.Shape.span((-hx, g0, 0), (hx, g1, DECK_T - GAP_D), DARK, fr, "gap_floor"),
              tk.Shape.tile((0, (y0 + g0) / 2, DECK_T + 0.0004), (2 * hx, g0 - y0), str(TEX / "earth.png"), texture_m=0.12),
              tk.Shape.tile((0, (g1 + y1) / 2, DECK_T + 0.0004), (2 * hx, y1 - g1), str(TEX / "earth.png"), texture_m=0.12),
              tk.Shape("box", (0, p0 + 0.003, DECK_T + 0.0006), size=(2 * hx, 0.006, 0.0004), color=LINE, collide=False)]
    return tk.body("track", pos=(cx, cy, TOP), shapes=shapes)


TRACK = _track()
ROVER = rv.rover_l4("rover", pos=(ROVER_XY[0], ROVER_XY[1], GROUND + 0.0008), rot_xyzw=rv.upright(math.pi / 2))
ROVER0 = tk.state_at(ROVER, ROVER.pos, ROVER.spawn_xyzw)
OBJECTS = [TRACK, ROVER]

# -- scoring -------------------------------------------------------------------------------------------------------
LIFT_CLEAR = 0.010       # m: both tyres of an axle this far above the ground under them at once: lifted
ON_GROUND = 0.005        # m: a tyre on the pad: its lowest point within this of the surface
REST_SPEED, REST_SPIN = 0.005, 0.05
REST_STEPS = 40          # 2 s at 20 Hz
RELEASE = 0.05

REC = tk.Recorder()


def ground_under(p) -> float:
    """Height of the surface under world point ``p``: the gap's floor over the gap (and off the track: the table)."""
    if not (TRACK_X[0] <= p[0] <= TRACK_X[1] and TRACK_Y[0] <= p[1] <= TRACK_Y[1]):
        return TOP
    if GAP_Y[0] < p[1] < GAP_Y[1]:
        return GROUND - GAP_D
    return GROUND


def tyre_points(st: tk.BodyState) -> dict[str, np.ndarray]:
    return rv.l3_tyre_points(ROVER, st)


def tyre_clearance(st: tk.BodyState) -> dict[str, float]:
    """Each tyre's lowest point's height above the surface directly under it (over the gap: above the near edge's
    level, so a tyre dipping into the gap reads negative and one carried over it reads ~0)."""
    out = {}
    for n, pts in tyre_points(st).items():
        p = pts[int(np.argmin(pts[:, 2]))]
        under = ground_under(p)
        out[n] = float(p[2] - (GROUND if under == GROUND - GAP_D else under))
    return out


def beyond(st: tk.BodyState) -> dict[str, bool]:
    """A tyre is beyond the gap when its whole tread is past the gap's far edge."""
    return {n: bool(pts[:, 1].min() > GAP_Y[1]) for n, pts in tyre_points(st).items()}


def on_pad(st: tk.BodyState) -> bool:
    """All four tyres on the pad: wholly past its line, inside the track, resting on the surface."""
    for pts in tyre_points(st).values():
        low = pts[:, 2].min()
        if pts[:, 1].min() < PAD_Y[0] or pts[:, 1].max() > PAD_Y[1]:
            return False
        if pts[:, 0].min() < TRACK_X[0] or pts[:, 0].max() > TRACK_X[1]:
            return False
        if low - GROUND > ON_GROUND:
            return False
    return True


AXLES = {"front": ("wheel_fl", "wheel_fr"), "rear": ("wheel_rl", "wheel_rr")}


def over_gap(st: tk.BodyState) -> dict[str, bool]:
    """A tyre is over the washout when the middle of its tread's lowest line lies between the gap's edges."""
    out = {}
    for n, pts in tyre_points(st).items():
        low = pts[pts[:, 2] <= pts[:, 2].min() + 0.002]
        out[n] = bool(GAP_Y[0] < float(low[:, 1].mean()) < GAP_Y[1])
    return out


def track_step(rec: tk.Recorder, st: tk.BodyState) -> None:
    c = tyre_clearance(st)
    lifted = [ax for ax, (a, b) in {"front": ("wheel_fl", "wheel_fr"), "rear": ("wheel_rl", "wheel_rr")}.items()
              if c[a] > LIFT_CLEAR and c[b] > LIFT_CLEAR]
    rec.latch("lifted", bool(lifted), f"both {lifted[0] if lifted else ''} tyres more than 1 cm off the ground at once")
    over = over_gap(st)
    both = [ax for ax, (a, b) in AXLES.items() if over[a] and over[b]]
    rec.latch("both_over", bool(both), f"both {both[0] if both else ''} tyres over the washout at once")
    rec.extreme("dip", -min(c.values()))


@tk.on_reset
def _reset(environment=None) -> None:
    REC.reset()


@tk.on_step
def _record(environment) -> None:
    REC.steps += 1
    st = tk.body_state(environment, "rover")
    track_step(REC, st)
    REC.window("rest", st.speed() < REST_SPEED and st.spin() < REST_SPIN, REST_STEPS)
    REC.window("released", rv.l4_released(ROVER, st, tk.grippers(environment), RELEASE), REST_STEPS)


def evaluate(st: tk.BodyState, rec: tk.Recorder = REC) -> tuple[float, str]:
    for foul in ("lifted", "both_over"):
        if rec.flag(foul):
            return 0.0, rec.why.get(foul, foul)
    if not (rec.full("rest", REST_STEPS) and all(rec.windows["rest"])):
        return 0.0, "not at rest for 2 s"
    if not (rec.full("released", REST_STEPS) and all(rec.windows["released"])):
        return 0.0, "a gripper or an arm within 5 cm of the rover in the last 2 s"
    b = beyond(st)
    if all(b.values()) and on_pad(st):
        return 1.0, "across the washout, on the pad"
    if any(b.values()):
        return 0.25, f"{sum(b.values())} of 4 tyres beyond the washout"
    return 0.0, "no tyre beyond the washout"


_LAST: dict = {}


def score(objects) -> float:
    """0 if at any time both tyres of an axle are more than 1 cm above the ground under them at once (the rover
    lifted), or both tyres of an axle are over the washout at once (the middle of each tread's lowest line between its
    edges: taken square, or carried across level). Otherwise, at the end, after 2 s at rest with the arms released (5 cm from the rover): 1.0 all four tyres
    beyond the washout and on the pad (inside the track, resting on it); 0.25 at least one tyre beyond the washout;
    0 otherwise."""
    s, why = evaluate(objects["rover"])
    _LAST.update(score=s, why=why)
    return s


def success(objects) -> bool:
    return evaluate(objects["rover"])[0] == 1.0


def report() -> dict:
    return {"last": dict(_LAST), "fouls": {k: REC.why[k] for k in REC.flags if REC.flags[k]},
            "deepest_dip_mm": round(1000 * REC.extremes.get("dip", 0.0), 1), "steps": REC.steps}
