"""On the station table: an earth track in front of the robot ends at a rock ledge 3.8 cm high, with more track on
top. A small yellow-and-white trail crawler (a 1:18 rover, motor disconnected so its wheels roll freely) stands on
the lower track facing the ledge, 9 cm short of it. Its tyres are smaller than the step. Four grey
stepped ramp rocks lie at the sides of the lower track. The robot must get the rover up onto the ledge without
lifting it.

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

INSTRUCTION = "The rover can't climb the ledge on its own. Get it up onto the ledge without lifting it."

TOP = tk.TOP
TEX = HERE / "textures"

# -- layout (world frame: x forward from the arm bases, y to the robot's left, metres) ---------------------------
LEDGE_H = 0.038                       # the step: well above the tyres' radius (2.8 cm), so a bare wheel jams
LEDGE_X = (0.44, 0.74)                # the ledge's face (toward the robot) and its far end
LEDGE_Y = (-0.17, 0.17)
LEDGE_TOP = TOP + LEDGE_H
LOW_X = (0.06, LEDGE_X[0])            # the lower track (an earth look on the table)
ROVER_X = 0.245                       # rover origin on the lower track, facing the ledge
ROCK_GREY = (0.52, 0.50, 0.47)

_lx = LEDGE_X[1] - LEDGE_X[0]
_ly = LEDGE_Y[1] - LEDGE_Y[0]
LEDGE = tk.body("ledge", pos=((LEDGE_X[0] + LEDGE_X[1]) / 2, 0.0, TOP), shapes=[
    tk.Shape.span((-_lx / 2, -_ly / 2, 0.0), (_lx / 2, _ly / 2, LEDGE_H), ROCK_GREY, friction=(0.8, 0.7)),
    tk.Shape.tile((0, 0, LEDGE_H + 0.0004), (_lx, _ly), str(TEX / "earth.png"), texture_m=0.12),
])
LOWER = tk.body("lower_track", pos=((LOW_X[0] + LOW_X[1]) / 2, 0.0, TOP), shapes=[
    tk.Shape.span((-(LOW_X[1] - LOW_X[0]) / 2, -_ly / 2, 0.0), ((LOW_X[1] - LOW_X[0]) / 2, _ly / 2, 0.0006),
                  (0.46, 0.34, 0.23), friction=(0.8, 0.7)),
    tk.Shape.tile((0, 0, 0.0008), (LOW_X[1] - LOW_X[0], _ly), str(TEX / "earth.png"), texture_m=0.12),
])
ROVER = rv.rover_l3("rover", pos=(ROVER_X, 0.0, TOP + 0.0012), rot_xyzw=rv.upright(0.0))

# stepped ramp rocks: three steps 1.27, 2.53 and 3.8 cm tall, each 2 cm long, 4 cm wide (x along the steps, high end at +x)
STEP_L, ROCK_W = 0.020, 0.040
ROCK_MASS = 0.15


def _ramp_rock(name, xy, yaw):
    """A stepped ramp rock lying on the table beside the track (z at the table top)."""
    shapes = [tk.Shape.span((-1.5 * STEP_L + i * STEP_L, -ROCK_W / 2, 0.0),
                            (-0.5 * STEP_L + i * STEP_L, ROCK_W / 2, LEDGE_H * (i + 1) / 3), ROCK_GREY, friction=(1.0, 0.9))
              for i in range(3)]
    return tk.body(name, pos=(xy[0], xy[1], TOP + 0.0004), shapes=shapes, mass=ROCK_MASS, yaw=yaw)


ROCKS = [_ramp_rock("ramp_rock_1", (0.20, 0.23), 0.0), _ramp_rock("ramp_rock_2", (0.31, 0.24), 0.4),
         _ramp_rock("ramp_rock_3", (0.20, -0.23), 0.0), _ramp_rock("ramp_rock_4", (0.31, -0.24), 2.8)]
OBJECTS = [LOWER, LEDGE, ROVER, *ROCKS]

# -- scoring -------------------------------------------------------------------------------------------------------
LIFT_CLEAR = 0.010       # m: a tyre is off the ground when its lowest point is this far above the surface under it
LIFT_STEPS = 10          # ... for this many control steps in a row (0.5 s): a lifted or tipped-up rover
TIPPED_DEG = 30.0
UPRIGHT_DEG = 10.0
ON_TOP = 0.010           # m: a tyre is on the ledge when its lowest point is within this of the ledge top, inside it
RAMP_TIGHT = 0.005       # m: a ramp rock's high end within this of the ledge face
LINE_HEADING_DEG = 15.0  # the wheel lines count only with the rover facing the ledge
HOP_CLEAR = 0.018        # m: a tyre this far above the surface under it, even for one step: lifted (a bare tyre must
#                          rise 2.8 cm to cross the face; on a ramp step it never rises more than one step, 1.3 cm)
RAMP_LINE = 0.015        # m: ... and its centre within this of a front wheel's line
REST_SPEED, REST_SPIN, REST_STEPS = 0.01, 0.5, 20
RELEASE = 0.05

REC = tk.Recorder()


def tyre_lows(st: tk.BodyState) -> dict[str, np.ndarray]:
    """Each tyre's lowest point (world)."""
    return {n: pts[np.argmin(pts[:, 2])] for n, pts in rv.l3_tyre_points(ROVER, st).items()}


def rock_on_table(spec: tk.Body, st: tk.BodyState) -> bool:
    """A ramp rock lying flat on the table (its base within 3 mm of the table, tilted under 10 degrees)."""
    low = tk.body_points(spec, st.pos, st.rotation())[:, 2].min()
    return low < TOP + 0.003 and tk.tilt_deg(st.rotation()) < 10.0


def support_height(p, rocks: dict) -> float:
    """Height of the highest surface under a point: table, lower track, ledge top or the step of a ramp rock that
    lies flat on the table (a rock held up or tipped is no support)."""
    h = TOP
    if LEDGE_X[0] <= p[0] <= LEDGE_X[1] and LEDGE_Y[0] <= p[1] <= LEDGE_Y[1]:
        h = max(h, LEDGE_TOP)
    for spec, st in rocks.values():
        if not rock_on_table(spec, st):
            continue
        for s in spec.shapes:
            c = st.pos + st.rotation() @ np.asarray(s.centre)
            local = st.rotation().T @ (np.asarray(p) - c)
            if abs(local[0]) <= s.size[0] / 2 + 0.003 and abs(local[1]) <= s.size[1] / 2 + 0.003:
                h = max(h, c[2] + s.size[2] / 2)
    return h


FOOTPRINT = 0.006        # m: a tyre bridging a small gap (ramp to ledge) rests on the higher edge within this


def support_near(p, rocks: dict) -> float:
    """The highest surface within FOOTPRINT of a point, horizontally."""
    d = FOOTPRINT
    return max(support_height(np.asarray(p) + (dx, dy, 0.0), rocks)
               for dx, dy in ((0, 0), (d, 0), (-d, 0), (0, d), (0, -d)))


def tyres_clear(st: tk.BodyState, rocks: dict) -> dict[str, bool]:
    """For each tyre: is its lowest point more than LIFT_CLEAR above the surface under it?"""
    return {n: bool(p[2] - support_near(p, rocks) > LIFT_CLEAR) for n, p in tyre_lows(st).items()}


def on_ledge(p) -> bool:
    return (LEDGE_X[0] <= p[0] <= LEDGE_X[1] and LEDGE_Y[0] <= p[1] <= LEDGE_Y[1]
            and abs(p[2] - LEDGE_TOP) <= ON_TOP)


def wheel_lines(st: tk.BodyState) -> list[float] | None:
    """The y of each front wheel's line (its tyre's centre), or None if the rover is not facing the ledge (heading
    within 15 degrees of straight at it)."""
    f = st.rotation()[:, 0]
    if math.degrees(abs(math.atan2(f[1], f[0]))) > LINE_HEADING_DEG:
        return None
    return [float(st.links[w][0][1]) for w in ("wheel_fl", "wheel_fr")]


def ramp_set(rocks: dict, st: tk.BodyState, wheels=("wheel_fl", "wheel_fr")) -> bool:
    """A ramp rock set under each named front wheel's line (taken from the rover's front tyres, the rover facing the
    ledge within 15 degrees): lying flat on the table (tilted under 10 degrees), its high step facing the ledge
    (within 20 degrees), the end of that step from 5 mm short of the ledge's face to 6 mm past it (pressed on the
    face), its centre within 1.5 cm of the wheel's line."""
    lines = wheel_lines(st)
    if lines is None:
        return False
    lines = [y for w, y in zip(("wheel_fl", "wheel_fr"), lines) if w in wheels]
    found = []
    for spec, st in rocks.values():
        R = st.rotation()
        if not rock_on_table(spec, st) or R[0, 0] < math.cos(math.radians(20)):
            continue
        high_end = st.to_world((1.5 * STEP_L, 0.0, 0.0))[0]
        if -RAMP_TIGHT <= high_end - LEDGE_X[0] <= 0.006:            # up to 6 mm "past": pressed on the face
            found.append(st.pos[1])
    return all(any(abs(y - line) <= RAMP_LINE for y in found) for line in lines)


def _rocks(environment) -> dict:
    return {r.name: (r, tk.body_state(environment, r.name)) for r in ROCKS}


@tk.on_reset
def _reset(environment=None) -> None:
    REC.reset()


@tk.on_step
def _record(environment) -> None:
    REC.steps += 1
    st = tk.body_state(environment, "rover")
    rocks = _rocks(environment)
    clear = tyres_clear(st, rocks)
    for n, c in clear.items():
        run = REC.values.get(f"clear_{n}", 0) + 1 if c else 0
        REC.values[f"clear_{n}"] = run
        REC.latch("lifted", run >= LIFT_STEPS, f"{n} held off the ground (the rover lifted or tipped up)")
    for n, p in tyre_lows(st).items():
        REC.latch("hopped", p[2] - support_near(p, rocks) > HOP_CLEAR, f"{n} lifted {100 * HOP_CLEAR:.1f} cm clear")
    lows = tyre_lows(st)
    for w in ("wheel_fl", "wheel_fr"):                      # each front wheel must come up by a ramp on its line
        if on_ledge(lows[w]) and not REC.values.get(f"up_{w}"):
            REC.values[f"up_{w}"] = True
            REC.latch("unramped", not ramp_set(rocks, st, (w,)), f"{w} reached the ledge without a ramp under it")
    REC.latch("tipped", tk.tilt_deg(st.rotation()) > TIPPED_DEG, "the rover tipped past 30 degrees")
    grips = tk.grippers(environment)
    rest = st.speed() < REST_SPEED and st.spin() < REST_SPIN and all(
        s.speed() < REST_SPEED and s.spin() < REST_SPIN for _, s in rocks.values())
    REC.window("rest", rest, REST_STEPS)
    REC.values["released"] = tk.released(ROVER, st, grips, RELEASE) and all(
        tk.released(spec, s, grips, RELEASE) for spec, s in rocks.values())
    REC.values["ramp_set"] = ramp_set(rocks, st)


def evaluate(st: tk.BodyState, rec: tk.Recorder = REC) -> tuple[float, str]:
    for foul in ("lifted", "hopped", "unramped", "tipped"):
        if rec.flag(foul):
            return 0.0, rec.why.get(foul, foul)
    if not (rec.full("rest", REST_STEPS) and all(rec.windows["rest"])):
        return 0.0, "not at rest"
    if not rec.values.get("released", False):
        return 0.0, "a gripper is within 5 cm of the rover or a ramp rock"
    lows = tyre_lows(st)
    upright = tk.tilt_deg(st.rotation()) <= UPRIGHT_DEG
    if upright and all(on_ledge(p) for p in lows.values()):
        return 1.0, "all four wheels on the ledge"
    if on_ledge(lows["wheel_fl"]) and on_ledge(lows["wheel_fr"]):
        return 0.5, "front axle on the ledge"
    if rec.values.get("ramp_set", False):
        return 0.25, "a ramp rock tight to the ledge under each front wheel"
    return 0.0, "the rover is below the ledge"


_LAST: dict = {}


def score(objects) -> float:
    """0 if at any time a tyre is held more than 1 cm off the surface under it for 0.5 s or 1.8 cm for one step (lifted,
    hopped or tipped up), a front tyre reaches the ledge with no ramp rock set on its line, or the rover tips past 30
    degrees. Otherwise, at the end, at rest and released: 1.0 all four wheels on the ledge, upright; 0.5 both front
    wheels on the ledge; 0.25 a ramp rock tight to the ledge under each front wheel's line."""
    s, why = evaluate(objects["rover"])
    _LAST.update(score=s, why=why)
    return s


def success(objects) -> bool:
    return evaluate(objects["rover"])[0] == 1.0


def report() -> dict:
    return {"last": dict(_LAST), "fouls": {k: REC.why[k] for k in REC.flags if REC.flags[k]}}
