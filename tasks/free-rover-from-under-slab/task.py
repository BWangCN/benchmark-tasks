"""On the station table: a loaded rover (14 kg, free-rolling wheels, steering unpowered) stands on an earth track under
a collapsed concrete slab. The slab bridges over it, 1.3 cm above its roof: its near end on a square concrete block, its
far end on a long smooth steel pipe lying on its side on a debris face that falls toward the rover. A fallen timber lies
across the rover's way out; its far end is wedged against the pipe and keeps it on its face. A bent bar stands out of
the slab's top near the pipe end. The pad is ahead of the rover. The robot must get the rover out from under the rubble.

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
import rubble as rb  # noqa: E402
import trailkit as tk  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Get the rover out from under the rubble onto the pad."

TOP = tk.TOP
TEX = HERE / "textures"

# -- layout (world frame: x forward from the arm bases, y to the robot's left, metres) ---------------------------
FOIL = 0.0006
GROUND = TOP + FOIL                     # the track's surface
TRACK_X = (0.10, 0.42)
TRACK_Y = (-0.30, 0.56)
PAD_Y0 = 0.22                           # the pad: the track beyond y 22 cm (its tape line's near edge)
PILE = np.array([0.26, 0.0])            # the pile's middle (and the rover's start)
H = 0.130                               # the slab's underside above the track (the rover is 11.7 cm tall)
SLAB_SIZE = (0.32, 0.16, 0.012)
SLAB_MASS = 1.47                        # concrete
BLOCK_SIZE = (0.04, 0.08, H)
BLOCK_X = PILE[0] - 0.12
DRUM_R, DRUM_L, DRUM_MASS = 0.050, 0.28, 1.0
DRUM_X = PILE[0] + 0.155                # under the slab's far end
DRUM_ROLL = 0.02                        # m: once free, the drum rolls this far down its face to a fixed lip
DRUM_Y = PILE[1] + 0.09                 # a long pipe section beside the rover, its far end past the rover's front
SLOPE_DEG = -40.0                       # the debris face under the drum falls toward the rover (-x)
TAB = (0.010, 0.030, 0.030)             # a bent bar standing out of the slab's top near the drum end: a gripper takes it
TAB_X = 0.10                            # its middle, from the slab's middle along x (x 36 cm)
TIMBER_Y = 0.215                        # the fallen timber lies across the rover's way out, along x
TIMBER_T, TIMBER_H = 0.04, 0.06         # its thickness (y) and height
STOP_X = (PILE[0] - 0.16, PILE[0] - 0.12)   # the fixed rock at its near end
TIMBER_DENSITY = 600.0                  # wood
EARTH = (0.42, 0.31, 0.21)
LINE = (0.93, 0.93, 0.90)

_zc = H - DRUM_R                                                     # the drum's centre above the track
_phi = math.radians(SLOPE_DEG)
_n = np.array([math.sin(_phi), 0.0, math.cos(_phi)])
_contact = np.array([DRUM_X - PILE[0], DRUM_Y - PILE[1], _zc]) - DRUM_R * _n
_t_z = TIMBER_H * 0.75                                               # the timber bears on the drum this high
_t_x1 = DRUM_X - math.sqrt(DRUM_R ** 2 - (_zc - _t_z) ** 2) - 0.001  # the drum's side there (world x)
_t_len = _t_x1 - STOP_X[1]
_lip_x1 = DRUM_X - DRUM_ROLL - DRUM_R                              # the lip's face: the drum's side after its roll


def _track():
    cx, cy = (TRACK_X[0] + TRACK_X[1]) / 2, (TRACK_Y[0] + TRACK_Y[1]) / 2
    hx, hy = (TRACK_X[1] - TRACK_X[0]) / 2, (TRACK_Y[1] - TRACK_Y[0]) / 2
    fr = (0.8, 0.7)
    shapes = [tk.Shape.span((-hx, -hy, 0), (hx, hy, FOIL), EARTH, fr, "track"),
              tk.Shape.tile((0, 0, FOIL + 0.0002), (2 * hx, 2 * hy), str(TEX / "earth.png"), texture_m=0.12),
              tk.Shape("box", (0, PAD_Y0 + 0.003 - cy, FOIL + 0.0004), size=(2 * hx, 0.006, 0.0004), color=LINE,
                       collide=False)]
    return tk.body("track", pos=(cx, cy, TOP), shapes=shapes)


TRACK = _track()
_face = rb.debris_face_shapes(_contact, SLOPE_DEG, width=DRUM_L + 0.02)
DEBRIS = tk.body("debris", pos=(PILE[0], PILE[1], GROUND), shapes=_face + [
    tk.Shape.span((STOP_X[0] - PILE[0], TIMBER_Y - TIMBER_T / 2 - 0.01, 0.0), (STOP_X[1] - PILE[0], TIMBER_Y + TIMBER_T / 2 + 0.01,
                  TIMBER_H), rb.ROCK, friction=rb.CONCRETE_FR, tag="fixed_rock"),
    tk.Shape.span((_lip_x1 - 0.007 - PILE[0], DRUM_Y - DRUM_L / 2 - PILE[1], 0.0), (_lip_x1 - PILE[0], TIMBER_Y - TIMBER_T / 2 - 0.012 - PILE[1],
                  _zc), rb.ROCK, friction=rb.CONCRETE_FR, tag="lip")])
ROVER = rv.rover_l4("rover", pos=(PILE[0], PILE[1], GROUND + 0.0008), rot_xyzw=rv.upright(math.pi / 2))
ROVER0 = tk.state_at(ROVER, ROVER.pos, ROVER.spawn_xyzw)
BLOCK = rb.block("block", (BLOCK_X, PILE[1], GROUND + H / 2 + 0.0005), BLOCK_SIZE, mass=0.9)
DRUM = rb.drum("drum", (DRUM_X, DRUM_Y, GROUND + _zc + 0.0005), DRUM_R, DRUM_L, DRUM_MASS)
SLAB = tk.body("slab", pos=(PILE[0], PILE[1], GROUND + H + SLAB_SIZE[2] / 2 + 0.001), mass=SLAB_MASS, shapes=[
    tk.Shape.box((0, 0, 0), SLAB_SIZE, rb.CONCRETE, friction=rb.CONCRETE_FR),
    tk.Shape.box((TAB_X, 0, SLAB_SIZE[2] / 2 + TAB[2] / 2), TAB, rb.RUST, friction=(1.0, 0.9), tag="tab")])
SLAB0 = tk.state_at(SLAB, SLAB.pos, (0.0, 0.0, 0.0, 1.0))
TIMBER = rb.block("timber", ((STOP_X[1] + _t_x1) / 2 + 0.0002, TIMBER_Y, GROUND + TIMBER_H / 2 + 0.0005),
                  (_t_len - 0.0004, TIMBER_T, TIMBER_H), color=(0.55, 0.42, 0.28),
                  mass=TIMBER_DENSITY * _t_len * TIMBER_T * TIMBER_H, friction=(0.6, 0.5))
PIECES = (SLAB, DRUM, BLOCK, TIMBER)
# the slab's gap over the rover where the scene puts them (13 mm)
GAP0 = tk.distance_to_body(ROVER, tk.state_at(ROVER, ROVER.pos, ROVER.rot_xyzw), rb.sample_points(SLAB, SLAB0))
OBJECTS = [TRACK, DEBRIS, ROVER, BLOCK, DRUM, SLAB, TIMBER]
if __import__("os").environ.get("PREMISE_NO_TIMBER"):    # premise probe only: the timber taken away at the start
    PIECES = (SLAB, DRUM, BLOCK)
    OBJECTS = [TRACK, DEBRIS, ROVER, BLOCK, DRUM, SLAB]


def tab_point(st: tk.BodyState) -> np.ndarray:
    return st.to_world((TAB_X, 0.0, SLAB_SIZE[2] / 2 + TAB[2] / 2))


def slab_held(st: tk.BodyState, grips: dict) -> bool:
    """A gripper closed on the slab's bar: the point between its pad tips within 2.5 cm of the bar's middle and its
    fingers shut on the 1 cm bar (under 1.5 cm apart)."""
    t = tab_point(st)
    return any(g["opening"] < 0.015 and float(np.linalg.norm(np.asarray(g["points"][-1]) - t)) < 0.025
               for g in grips.values())


# -- scoring -------------------------------------------------------------------------------------------------------
SLAB_DROP = 0.010        # m: the slab's lowest point this far below where it started: it came down
TOUCH = 0.002            # m: a piece of the pile this close to the rover touches it
OUT_Y = PILE[1] + SLAB_SIZE[1] / 2      # the rover is out from under the slab when its rear tyres are past this
ON_GROUND = 0.005
REST_SPEED, REST_SPIN = 0.005, 0.05
REST_STEPS = 40
RELEASE = 0.05

REC = tk.Recorder()


def slab_low(st: tk.BodyState) -> float:
    return float(min(st.to_world(p)[2] for s in SLAB.shapes for p in s.surface_points()))


def tyre_points(st: tk.BodyState) -> dict[str, np.ndarray]:
    return rv.l3_tyre_points(ROVER, st)


def on_pad(st: tk.BodyState) -> bool:
    for pts in tyre_points(st).values():
        if pts[:, 1].min() < PAD_Y0 or pts[:, 1].max() > TRACK_Y[1]:
            return False
        if pts[:, 0].min() < TRACK_X[0] or pts[:, 0].max() > TRACK_X[1]:
            return False
        if pts[:, 2].min() - GROUND > ON_GROUND:
            return False
    return True


def out_from_under(st: tk.BodyState) -> bool:
    pts = tyre_points(st)
    return min(float(pts[n][:, 1].min()) for n in ("wheel_rl", "wheel_rr")) > OUT_Y


def track_step(rec: tk.Recorder, rover: tk.BodyState, pieces: dict, held: bool = False) -> None:
    low = slab_low(pieces["slab"])
    out = out_from_under(rover)
    if "slab_ref" not in rec.values:
        rec.values["slab_ref"] = slab_low(SLAB0)         # from where the scene put it: a collapse while the scene
    #                                                      settles (the pile wrongly built) counts too
    if not out:
        # while the rover is under it: the slab may not come down toward the rover's roof, held or not (the held
        # far end may sag about the grip as the drum rolls; what counts is the gap over the rover)
        gap = tk.distance_to_body(ROVER, rover, rb.sample_points(SLAB, pieces["slab"]))
        closer = GAP0 - gap
        rec.extreme("slab_closer", closer)
        rec.latch("slab_down", closer > SLAB_DROP, f"the slab came {1000 * closer:.0f} mm closer to the rover under it")
    else:
        if held:
            rec.values["slab_ref"] = low                 # once the rover is out, the held slab may be lowered
        drop = rec.values["slab_ref"] - low
        rec.extreme("slab_drop", drop)
        rec.latch("slab_down", drop > SLAB_DROP and not held, f"the slab came down {1000 * drop:.0f} mm")
    for spec in PIECES:
        d = tk.distance_to_body(ROVER, rover, rb.sample_points(spec, pieces[spec.name]))
        rec.extreme(f"near_{spec.name}", -d)
        rec.latch("touched", d < TOUCH, f"the {spec.name} touched the rover")


@tk.on_reset
def _reset(environment=None) -> None:
    REC.reset()


@tk.on_step
def _record(environment) -> None:
    REC.steps += 1
    rover = tk.body_state(environment, "rover")
    pieces = {s.name: tk.body_state(environment, s.name) for s in PIECES}
    grips = tk.grippers(environment)
    track_step(REC, rover, pieces, slab_held(pieces["slab"], grips))
    REC.window("rest", rover.speed() < REST_SPEED and rover.spin() < REST_SPIN, REST_STEPS)
    REC.window("released", rv.l4_released(ROVER, rover, grips, RELEASE) and not slab_held(pieces["slab"], grips),
               REST_STEPS)


def evaluate(st: tk.BodyState, rec: tk.Recorder = REC) -> tuple[float, str]:
    for foul in ("slab_down", "touched"):
        if rec.flag(foul):
            return 0.0, rec.why.get(foul, foul)
    if not (rec.full("rest", REST_STEPS) and all(rec.windows["rest"])):
        return 0.0, "not at rest for 2 s"
    if not (rec.full("released", REST_STEPS) and all(rec.windows["released"])):
        return 0.0, "a gripper or an arm within 5 cm of the rover in the last 2 s"
    if on_pad(st):
        return 1.0, "out from under the rubble, on the pad"
    if out_from_under(st):
        return 0.5, "out from under the slab, not on the pad"
    return 0.0, "still under the slab"


_LAST: dict = {}


def score(objects) -> float:
    """0 if at any time, while the rover is under the slab (any point of its rear treads short of the slab's edge at
    y 8 cm), the slab comes more than 1 cm closer to the rover than where the scene put them (its nearest point to
    the rover's body, 13 mm at the start), held or not; or, once the rover is out, more than 1 cm below its
    reference while no gripper holds its bar (the reference: where the slab was at the last step at which a gripper
    held its bar with the rover out; where the scene put it if there was none) (the point between the pad tips
    within 2.5 cm of the bar's middle and the fingers shut on it, under 1.5 cm apart); or any piece of the pile
    (slab, pipe, block, timber) comes within 2 mm of the rover's body. Otherwise, at the end, after 2 s with the
    rover at rest (under 5 mm/s and 0.05 rad/s) and released (no point of either gripper, and no point along either
    arm from its shoulder joint to its palm, within 5 cm of the rover, its wheels and steering; no gripper holding
    the slab's bar): 1.0 all four tyres on the pad (every point of each tread past y 22 cm, inside the track, the
    lowest point within 5 mm of it); 0.5 the rover out from under the slab (every point of both rear treads past y 8
    cm); 0 otherwise. A full score ends the trial once it has held for 1 s."""
    s, why = evaluate(objects["rover"])
    _LAST.update(score=s, why=why)
    return s


def success(objects) -> bool:
    return evaluate(objects["rover"])[0] == 1.0


def report() -> dict:
    return {"last": dict(_LAST), "fouls": {k: REC.why[k] for k in REC.flags if REC.flags[k]},
            "slab_drop_mm": round(1000 * REC.extremes.get("slab_drop", 0.0), 1),
            "slab_closer_mm": round(1000 * REC.extremes.get("slab_closer", 0.0), 1),
            "nearest_mm": {s.name: round(-1000 * REC.extremes.get(f"near_{s.name}", -1.0), 1) for s in PIECES},
            "steps": REC.steps}
