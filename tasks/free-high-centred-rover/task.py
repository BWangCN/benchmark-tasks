"""On the station table: a small yellow-and-white trail crawler (a 1:18 rover, driveshafts removed so its wheels roll
freely) is high-centred on a flat grey rock on an earth track: its chassis rests on the rock and all four tyres hang
2.5 cm off the ground. The rock sticks out beyond the rover's right side. A short tow bar sticks out from the middle of
the rear bumper. The robot must get the rock out and set the rover back on its wheels.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 --scorer task_score ...
         (the drop test counts in 20 Hz control steps)
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

INSTRUCTION = "The rover is stuck on the rock. Get the rock out and set the rover back on its wheels where it stands."

TOP = tk.TOP
TEX = HERE / "textures"

# -- layout (world frame: x forward from the arm bases, y to the robot's left, metres) ---------------------------
TRACK_X, TRACK_Y = (0.08, 0.62), (-0.16, 0.16)
ROVER_XY = (0.31, 0.0)
ROCK_SIZE = (0.050, 0.140, 0.050)          # along x, y, z: under the chassis, sticking out 4 cm past the right tyres
ROCK_Y = -0.030                            # its centre: from y +4 cm (under the left rail) to -10 cm
ROCK_MASS = 0.30
ROCK_GREY = (0.52, 0.50, 0.47)
CLEARANCE = rv.S.GROUND_CLEARANCE          # the chassis' underside above the ground (2.5 cm)
LIFT0 = ROCK_SIZE[2] - CLEARANCE           # the rover's origin above the track at the start: tyres 2.5 cm up

TAB_SIZE = (0.048, 0.016, 0.020)
TAB_X = rv.S.LENGTH / 2 + TAB_SIZE[0] / 2 - 0.002
TAB_Z = rv.S.BUMPER_HEIGHT
TABS = (tk.Shape.box((-TAB_X, 0.0, TAB_Z), TAB_SIZE, (0.20, 0.20, 0.22), friction=(1.0, 0.9), tag="tow_r"),)  # rear only

_lx, _ly = TRACK_X[1] - TRACK_X[0], TRACK_Y[1] - TRACK_Y[0]
TRACK = tk.body("track", pos=((TRACK_X[0] + TRACK_X[1]) / 2, 0.0, TOP), shapes=[
    tk.Shape.span((-_lx / 2, -_ly / 2, 0.0), (_lx / 2, _ly / 2, 0.0006), (0.46, 0.34, 0.23), friction=(0.8, 0.7)),
    tk.Shape.tile((0, 0, 0.0008), (_lx, _ly), str(TEX / "earth.png"), texture_m=0.12),
])
ROCK = tk.body("rock", pos=(ROVER_XY[0], ROCK_Y, TOP + 0.0006),
               shapes=[tk.Shape.span((-ROCK_SIZE[0] / 2, -ROCK_SIZE[1] / 2, 0.0),
                                     (ROCK_SIZE[0] / 2, ROCK_SIZE[1] / 2, ROCK_SIZE[2]), ROCK_GREY, friction=(0.8, 0.7))],
               mass=ROCK_MASS)
ROVER = rv.rover_l3("rover", pos=(ROVER_XY[0], ROVER_XY[1], TOP + 0.0006 + LIFT0 + 0.0005), rot_xyzw=rv.upright(0.0),
                    extra_shapes=TABS)
OBJECTS = [TRACK, ROCK, ROVER]
HEADING0 = 0.0                             # the rover faces +x at the start

# -- scoring -------------------------------------------------------------------------------------------------------
DROP_STEP = 0.010        # m: a tyre's lowest point falling this far in one control step (0.20 m/s at 20 Hz): a drop
DROP_SPAN, DROP_WINDOW = 0.015, 3   # ... or this far over 3 control steps (0.15 s)
GROUND = TOP + 0.0006    # the track's top
DOWN_TOL = 0.008         # m: a tyre is down when its lowest point is within this of the track
DRAG_MAX = 0.050         # m: the rover carried this far sideways (across its heading, net) while its body rests on the
#                          rock: dragged with it. Pulled from under the full weight, the rock carries the rover about
#                          11 cm; lifted by its tail first, the rock rubs it about 1.7 cm sideways (the demo)
ON_ROCK = 0.004          # m: the chassis rests on the rock when its lowest point is within this of the rock's top
SHIFT_OK = 0.025         # m: ... and set down within this of where it stood (heading within 5 degrees) for full score
#                          (the rock rubs the lifted body about 1.7 cm sideways as it slides out)
ROLL_MAX, PITCH_MAX = 30.0, 45.0   # degrees: rolled onto its side, or stood on its nose or tail
UPRIGHT_DEG = 10.0
REST_SPEED, REST_SPIN, REST_STEPS = 0.01, 0.5, 20
RELEASE = 0.05

REC = tk.Recorder()


def tyre_lows(st: tk.BodyState) -> dict[str, np.ndarray]:
    return {n: pts[np.argmin(pts[:, 2])] for n, pts in rv.l3_tyre_points(ROVER, st).items()}


def chassis_low(st: tk.BodyState) -> float:
    """The lowest point of the rover's body (not its tyres)."""
    return float(tk.body_points(ROVER, st.pos, st.rotation())[:, 2].min())


def rock_outline(st: tk.BodyState) -> np.ndarray:
    """The rock's footprint corners (x, y), world."""
    hx, hy = ROCK_SIZE[0] / 2, ROCK_SIZE[1] / 2
    return np.array([st.to_world((sx * hx, sy * hy, z))[:2] for sx in (-1, 1) for sy in (-1, 1)
                     for z in (0.0, ROCK_SIZE[2])])


def rock_clear(rover: tk.BodyState, rock: tk.BodyState) -> bool:
    """No part of the rock's footprint (all eight corners, seen from above) inside the rover's outline seen from above (bumpers to bumpers, tyre to tyre)."""
    half_l, half_w = rv.S.LENGTH / 2, rv.S.WIDTH / 2
    corners = rock_outline(rock)
    local = (corners - rover.pos[:2]) @ rover.rotation()[:2, :2]
    # separating axis test between two rectangles: the rover's (axis-aligned in its frame) and the rock's
    if local[:, 0].max() < -half_l or local[:, 0].min() > half_l or local[:, 1].max() < -half_w or local[:, 1].min() > half_w:
        return True
    rv_c = np.array([[sx * half_l, sy * half_w] for sx in (-1, 1) for sy in (-1, 1)])
    R = rover.rotation()[:2, :2].T @ rock.rotation()[:2, :2]
    for ax in (R[:, 0], R[:, 1]):
        a, b = local @ ax, rv_c @ ax
        if a.max() < b.min() or a.min() > b.max():
            return True
    return False


def on_rock(rover: tk.BodyState, rock: tk.BodyState) -> bool:
    """The rover's body rests on the rock: some point of a grid over the rock's top face lies within 4 mm of it."""
    hx, hy = ROCK_SIZE[0] / 2, ROCK_SIZE[1] / 2
    grid = [(x, y, ROCK_SIZE[2]) for x in np.linspace(-hx, hx, 6) for y in np.linspace(-hy, hy, 15)]
    return tk.distance_to_body(ROVER, rover, rock.pos + np.asarray(grid) @ rock.rotation().T) <= ON_ROCK


def track_step(rec: tk.Recorder, rover: tk.BodyState, start_xy, held: bool, rock: tk.BodyState | None = None) -> None:
    lows = tyre_lows(rover)
    hist = rec.values.setdefault("hist", [])
    hist.append({w: float(p[2]) for w, p in lows.items()})
    del hist[:-(DROP_WINDOW + 1)]
    if len(hist) > 1:
        step = max(hist[-2][w] - hist[-1][w] for w in lows)
        span = max(hist[0][w] - hist[-1][w] for w in lows)
        if step > DROP_STEP or span > DROP_SPAN:
            rec.latch("drop", True, f"the rover dropped ({100 * max(step, span):.1f} cm in "
                                    f"{1 if step > DROP_STEP else len(hist) - 1} steps)")
    side = np.array([-math.sin(HEADING0), math.cos(HEADING0)])          # across the rover's heading at the start
    y = float(np.dot(rover.pos[:2], side))
    last = rec.values.get("last_y")
    if last is not None and rock is not None and on_rock(rover, rock):  # riding on the rock: sideways moves add up
        rec.values["carried_net"] = rec.values.get("carried_net", 0.0) + (y - last)   # signed: jitter cancels
    rec.values["last_y"] = y
    carried = abs(rec.values.get("carried_net", 0.0))
    rec.latch("dragged", carried > DRAG_MAX, f"the rover was carried {100 * carried:.1f} cm sideways on the rock (dragged)")
    rec.extreme("moved", abs(float(np.dot(rover.pos[:2] - np.asarray(start_xy), side))))
    rec.extreme("carried", carried)
    R = rover.rotation()
    roll, pitch = math.degrees(math.atan2(R[2, 1], R[2, 2])), math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0]))))
    rec.latch("tipped", abs(roll) > ROLL_MAX or abs(pitch) > PITCH_MAX,
              f"the rover tipped (roll {roll:.0f}, pitch {pitch:.0f} degrees)")


@tk.on_reset
def _reset(environment=None) -> None:
    REC.reset()


@tk.on_step
def _record(environment) -> None:
    REC.steps += 1
    rover = tk.body_state(environment, "rover")
    rock = tk.body_state(environment, "rock")
    REC.values.setdefault("start_xy", tuple(map(float, rover.pos[:2])))
    grips = tk.grippers(environment)
    released = tk.released(ROVER, rover, grips, RELEASE) and tk.released(ROCK, rock, grips, RELEASE)
    track_step(REC, rover, REC.values["start_xy"], held=not released, rock=rock)
    rest = all(b.speed() < REST_SPEED and b.spin() < REST_SPIN for b in (rover, rock))
    REC.window("rest", rest, REST_STEPS)
    REC.window("released", released, REST_STEPS)                   # released for the whole rest window
    REC.values["rock_clear"] = rock_clear(rover, rock)


def heading(st: tk.BodyState) -> float:
    f = st.rotation()[:, 0]
    return math.atan2(f[1], f[0])


def evaluate(rover: tk.BodyState, rec: tk.Recorder = REC) -> tuple[float, str]:
    for foul in ("drop", "dragged", "tipped"):
        if rec.flag(foul):
            return 0.0, rec.why.get(foul, foul)
    if not (rec.full("rest", REST_STEPS) and all(rec.windows["rest"])):
        return 0.0, "not at rest"
    if not (rec.full("released", REST_STEPS) and all(rec.windows["released"])):
        return 0.0, "a gripper is within 5 cm of the rover or the rock"
    if not rec.values.get("rock_clear", False):
        return 0.0, "the rock is still under the rover"
    lows = tyre_lows(rover)
    if not (tk.tilt_deg(rover.rotation()) <= UPRIGHT_DEG and all(p[2] - GROUND <= DOWN_TOL for p in lows.values())):
        return 0.0, "the rover is not on its wheels"
    start = np.asarray(rec.values.get("start_xy", ROVER_XY))
    moved = float(np.hypot(*(rover.pos[:2] - start)))
    rec.values["final_shift_cm"] = round(100 * moved, 2)
    rec.values["final_heading_deg"] = round(math.degrees(heading(rover)), 2)
    if moved <= SHIFT_OK and abs(heading(rover)) <= math.radians(5.0):
        return 1.0, "rock out, rover on its wheels where it stood"
    return 0.5, f"rock out, rover on its wheels but moved {100 * moved:.1f} cm or turned"


_LAST: dict = {}


def score(objects) -> float:
    """0 if at any time the rover drops (a tyre falls 1 cm in one control step or 1.5 cm in three), is moved more
    than 5 cm sideways while its body rests on the rock (dragged with it), or rolls past 30 or pitches past 45 degrees. Otherwise, at the end, at rest and released, the rock clear
    of the rover's outline and the rover upright on all four wheels: 1.0 within 2.5 cm and 5 degrees of where it stood;
    0.5 otherwise."""
    s, why = evaluate(objects["rover"])
    _LAST.update(score=s, why=why)
    return s


def success(objects) -> bool:
    return evaluate(objects["rover"])[0] == 1.0


def report() -> dict:
    return {"last": dict(_LAST), "fouls": {k: REC.why[k] for k in REC.flags if REC.flags[k]}, "steps": REC.steps,
            "max_moved_cm": round(100 * REC.extremes.get("moved", 0.0), 1),
            "carried_cm": round(100 * REC.extremes.get("carried", 0.0), 2),
            "final_shift_cm": REC.values.get("final_shift_cm"), "final_heading_deg": REC.values.get("final_heading_deg")}
