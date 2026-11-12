"""On the station table: a loaded rover (14 kg, free-rolling wheels) stands on an earth track, facing away from the
robot (+x), with a flat concrete slab lying on its roof. The robot must get the slab off the rover and leave the rover where it
is. The slab's friction on the roof is more than what the wheels resist: the obvious push, away from the robot, runs
along the rover and carries it away. Across the wheels the tyres hold, and a push across (sideways, along y) slides
the slab off the side.

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

INSTRUCTION = "Get the slab off the rover and onto the ground without lifting it. The rover must stay where it is."

TOP = tk.TOP
TEX = HERE / "textures"

# -- layout (world frame: x forward from the arm bases, y to the robot's left, metres) ---------------------------
FOIL = 0.0006
GROUND = TOP + FOIL
TRACK_X, TRACK_Y = (0.10, 0.50), (-0.40, 0.40)
X0, Y0 = 0.28, -0.05                    # the rover's start (its origin: on the ground between the axles), facing +x
SLAB = (0.20, 0.10, 0.015)              # x (along the rover), y (across it), thickness
SLAB_MASS = 0.8                         # concrete-faced board, 2.7 g/cm3 x 300 cm3
ROOF = GROUND + rv.S.HEIGHT             # the rover's roof
WHEEL_HOLD = 0.02                       # N m per wheel: about 0.7 N at the tyre (stops a free rover creeping)
EARTH = (0.42, 0.31, 0.21)


def _track():
    cx, cy = (TRACK_X[0] + TRACK_X[1]) / 2, (TRACK_Y[0] + TRACK_Y[1]) / 2
    hx, hy = (TRACK_X[1] - TRACK_X[0]) / 2, (TRACK_Y[1] - TRACK_Y[0]) / 2
    return tk.body("track", pos=(cx, cy, TOP), shapes=[
        tk.Shape.span((-hx, -hy, 0), (hx, hy, FOIL), EARTH, friction=(0.8, 0.7), tag="track"),
        tk.Shape.tile((0, 0, FOIL + 0.0002), (2 * hx, 2 * hy), str(TEX / "earth.png"), texture_m=0.12)])


TRACK = _track()
ROVER = rv.rover_l4("rover", pos=(X0, Y0, GROUND + 0.0008), rot_xyzw=rv.upright(0.0), wheel_hold=WHEEL_HOLD)   # facing +x
SLAB_BODY = rb.slab("slab", (X0, Y0, ROOF + SLAB[2] / 2 + 0.001), SLAB, SLAB_MASS)
OBJECTS = [TRACK, ROVER, SLAB_BODY]

# -- scoring -------------------------------------------------------------------------------------------------------
ROVER_SHIFT = 0.03        # m: the rover's origin this far from its start along the ground: it did not stay
LIFT = 0.005              # m: the slab's lowest point this far above the roof: lifted, not slid (the demo's: 0.0)
ON_GROUND = 0.005         # m: the slab's lowest point within this of the ground: on it
CLEAR = 0.002             # m: no slab point within this of the rover's colliders: not touching it
REST_MOVE = 0.002         # m: the rover's and the slab's middles moved under this over the last REST_STEPS
REST_STEPS = 40
RELEASE = 0.05

REC = tk.Recorder()


def slab_low(st: tk.BodyState) -> float:
    return tk.lowest_z(SLAB_BODY, st)


def over_rover(slab: tk.BodyState, rover: tk.BodyState) -> bool:
    """Is the slab's middle over the rover's outline, seen from above?"""
    local = rover.to_local(slab.pos)
    return abs(float(local[0])) <= rv.S.LENGTH / 2 and abs(float(local[1])) <= rv.S.WIDTH / 2


class _Pose:
    """A link's pose in the shape ``tk.distance_to_body`` reads (``pos`` and ``rotation()``)."""

    def __init__(self, pos, rot):
        self.pos, self._rot = np.asarray(pos, float), np.asarray(rot, float)

    def rotation(self):
        return self._rot


def rover_gap(rover: tk.BodyState, slab: tk.BodyState) -> float:
    """The smallest distance from the slab (sampled every 4 mm) to the rover's colliders: its body and every link (the
    wheels, the front axle, the steering), each measured exactly in its own pose."""
    pts = rb.sample_points(SLAB_BODY, slab, 0.004)
    gap = _exact_distance(ROVER.shapes, rover, pts)
    for ln in ROVER.links:
        if ln.name in rover.links:
            lp, lr = rover.links[ln.name]
            gap = min(gap, _exact_distance(ln.shapes, _Pose(lp, lr), pts))
    return gap


_AXIS = {"X": np.array([1.0, 0.0, 0.0]), "Y": np.array([0.0, 1.0, 0.0]), "Z": np.array([0.0, 0.0, 1.0])}


def _exact_distance(shapes, pose, pts) -> float:
    """Smallest distance from points (world) to colliders in a pose; cylinders as cylinders (trailkit's own measure
    treats them as capsules, whose bulging ends reach a wheel's radius past its sides)."""
    R = pose.rotation()
    best = np.inf
    for s in shapes:
        if not s.collide:
            continue
        c = np.asarray(pose.pos) + R @ np.asarray(s.centre)
        r = R @ s.rotation()
        if s.kind == "box":
            d = tk.point_box_distance(pts, c, r, np.asarray(s.size) / 2)
        elif s.kind == "sphere":
            d = np.maximum(np.linalg.norm(pts - c, axis=1) - s.radius, 0.0)
        elif s.kind == "cylinder":
            a = r @ _AXIS[s.axis]
            v = pts - c
            ax = v @ a
            rad = np.linalg.norm(v - np.outer(ax, a), axis=1)
            d = np.hypot(np.maximum(np.abs(ax) - s.length / 2, 0.0), np.maximum(rad - s.radius, 0.0))
        else:                                                           # capsule
            d = np.asarray([tk.distance_to_body(tk.body("_", (0, 0, 0), [s]), pose, pts)])
        best = min(best, float(np.min(d)))
    return best


def track_step(rec: tk.Recorder, rover: tk.BodyState, slab: tk.BodyState) -> None:
    shift = float(np.hypot(rover.pos[0] - X0, rover.pos[1] - Y0))
    rec.extreme("rover_shift", shift)
    rec.latch("rover_moved", shift > ROVER_SHIFT, f"the rover moved {100 * shift:.1f} cm")
    up = slab_low(slab) - ROOF
    rec.extreme("slab_up", up)
    rec.latch("lifted", up > LIFT, f"the slab lifted {100 * up:.1f} cm above the roof")
    mid_up = float(slab.pos[2]) - (ROOF + SLAB[2] / 2)                 # its middle above where it lay: a lift by one end
    rec.extreme("slab_mid_up", mid_up)
    rec.latch("lifted", mid_up > LIFT, f"the slab's middle lifted {1000 * mid_up:.0f} mm")


def evaluate(rover: tk.BodyState, slab: tk.BodyState, rec: tk.Recorder = REC) -> tuple[float, str]:
    for k in ("rover_moved", "lifted"):
        if rec.flag(k):
            return 0.0, rec.why[k]
    if not (rec.full("rest", REST_STEPS) and all(rec.windows["rest"])):
        return 0.0, "the rover or the slab not at rest for 2 s"
    if not (rec.full("released", REST_STEPS) and all(rec.windows["released"])):
        return 0.0, "a gripper or an arm within 5 cm of the rover or the slab in the last 2 s"
    gap = rover_gap(rover, slab)
    if gap >= CLEAR and abs(slab_low(slab) - GROUND) <= ON_GROUND:
        return 1.0, "the slab on the ground, off the rover; the rover where it was"
    if not over_rover(slab, rover):
        return 0.5, "the slab's middle off the rover, the slab not lying on the ground clear of it"
    return 0.0, "the slab still on the rover"


@tk.on_reset
def _reset(environment=None) -> None:
    REC.reset()


@tk.on_step
def _record(environment) -> None:
    REC.steps += 1
    rover = tk.body_state(environment, "rover")
    slab = tk.body_state(environment, "slab")
    track_step(REC, rover, slab)
    hist = REC.values.setdefault("centres", [])
    hist.append(np.array([rover.pos, slab.pos]))
    del hist[:-REST_STEPS]
    REC.window("rest", len(hist) == REST_STEPS and float(np.max(np.linalg.norm(hist[-1] - hist[0], axis=1))) < REST_MOVE,
               REST_STEPS)
    grips = tk.grippers(environment)
    REC.window("released", rv.l4_released(ROVER, rover, grips, RELEASE) and tk.released(SLAB_BODY, slab, grips, RELEASE),
               REST_STEPS)


_LAST: dict = {}


def score(objects) -> float:
    """0 if at any time the rover's origin (on the ground between its axles) moves more than 3 cm from its start
    along the ground, or the slab is lifted, not slid: its lowest point more than 5 mm above the rover's roof, or
    its middle more than 5 mm above where it lay. Otherwise, at the end, after 2 s with the rover and the slab at
    rest (the rover's origin and the slab's middle moved under 2 mm over the last 2 s) and released (no point of
    either gripper, and no point along either arm from its shoulder joint to its palm, within 5 cm of the rover, its
    wheels and steering, or of the slab): 1.0 the slab on the ground (its lowest point within 5 mm of it) and clear
    of the rover (not touching it: no point within 2 mm of its body, wheels or steering); 0.5 the slab's middle off
    the rover's 26.2 x 12.4 cm footprint (tow tabs excluded) seen from above, but the slab not lying clear on the
    ground; 0 otherwise. A full score ends the trial once it has held for 1 s."""
    s, why = evaluate(objects["rover"], objects["slab"])
    _LAST.update(score=s, why=why)
    return s


def success(objects) -> bool:
    return evaluate(objects["rover"], objects["slab"])[0] == 1.0


def report() -> dict:
    return {"last": dict(_LAST), "steps": REC.steps,
            "rover_shift_mm": round(1000 * REC.extremes.get("rover_shift", 0.0), 1),
            "slab_up_mm": round(1000 * REC.extremes.get("slab_up", 0.0), 1),
            "slab_mid_up_mm": round(1000 * REC.extremes.get("slab_mid_up", 0.0), 1), "fouls": dict(REC.why)}
