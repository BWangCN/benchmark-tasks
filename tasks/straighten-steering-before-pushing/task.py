"""On the station table: a painted earth lane runs away from the robot, 16 cm wide, with a marked bay at its far
end. A loaded rover (built heavy at the size of a 1:18 crawler, 14 kg in all, lead-filled; wheels rolling freely,
steering servo unpowered) stands in the lane facing the bay, its front wheels turned 25 degrees to the left. The robot must
move it into the bay with all four wheels kept in the lane.

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

INSTRUCTION = "Move the rover along the lane into the bay. Keep all four wheels in the lane."

TOP = tk.TOP
TEX = HERE / "textures"

# -- layout (world frame: x forward from the arm bases, y to the robot's left, metres) ---------------------------
LANE_X = (0.10, 0.72)
LANE_HALF = 0.080                     # a 16 cm lane: 18 mm each side of the rover's tyres when it runs straight
BAY_X = (0.36, 0.64)                  # the bay: the far half of the lane
ROVER_X = 0.25
STEER0 = math.radians(25.0)           # the front wheels at the start: turned 25 degrees to the left
LINE = (0.93, 0.93, 0.90)

_lx = LANE_X[1] - LANE_X[0]


def _paint(x0, x1, y0, y1):
    """A painted line: flat, no collision."""
    return tk.Shape("box", ((x0 + x1) / 2, (y0 + y1) / 2, 0.0012), size=(x1 - x0, y1 - y0, 0.0004), color=LINE,
                    collide=False)


LANE = tk.body("lane", pos=((LANE_X[0] + LANE_X[1]) / 2, 0.0, TOP), shapes=[
    tk.Shape.span((-_lx / 2, -0.14, 0.0), (_lx / 2, 0.14, 0.0006), (0.46, 0.34, 0.23), friction=(0.8, 0.7)),
    tk.Shape.tile((0, 0, 0.0008), (_lx, 0.28), str(TEX / "earth.png"), texture_m=0.12),
    _paint(-_lx / 2, _lx / 2, LANE_HALF, LANE_HALF + 0.006),                 # lane edges (inner edge at +-8 cm)
    _paint(-_lx / 2, _lx / 2, -LANE_HALF - 0.006, -LANE_HALF),
    _paint(BAY_X[0] - 0.006 - (LANE_X[0] + LANE_X[1]) / 2, BAY_X[0] - (LANE_X[0] + LANE_X[1]) / 2,
           -LANE_HALF, LANE_HALF),                                            # the bay's line across the lane
    _paint(BAY_X[1] - (LANE_X[0] + LANE_X[1]) / 2, BAY_X[1] + 0.006 - (LANE_X[0] + LANE_X[1]) / 2,
           -LANE_HALF, LANE_HALF),                                            # the bay's far end
])


STEERS = ("steer",)
ROVER = rv.rover_l4("rover", pos=(ROVER_X, 0.0, TOP + 0.0012), steer0=STEER0)


def steer_angles(st: tk.BodyState) -> dict[str, float]:
    return {k: float(st.joints.get(k, STEER0)) for k in STEERS}


OBJECTS = [LANE, ROVER]
ROVER0 = tk.state_at(ROVER, ROVER.pos, ROVER.spawn_xyzw)

# -- scoring -------------------------------------------------------------------------------------------------------
OUT_TOL = 0.005          # m: a tyre's lowest point this far past the lane's painted edge: out of the lane
SCRUB_MAX = 0.020        # m: the tyres' total sideways slide (each wheel centre's motion along its own axle)
LANE_TOP = TOP + 0.0006  # the lane sheet's surface
AXLE_CLEAR, LIFT_STEPS = 0.003, 10   # both tyres of an axle more than 3 mm above the lane for 0.5 s: lifted
TILT_MAX = 15.0          # degrees: any credit needs the rover upright
TURNED_TRAVEL_MAX = 0.05 # m: with the wheels turned the rover may be no farther than this from its start (straighten first)
STRAIGHT = math.radians(5.0)
HEADING_TOL = math.radians(8.0)
MOVED_MIN = 0.10         # m along the lane
REST_SPEED, REST_SPIN, REST_STEPS = 0.01, 0.5, 20
RELEASE = 0.05

REC = tk.Recorder()


def tyre_lows(st: tk.BodyState) -> dict[str, np.ndarray]:
    """Each tyre's lowest point; of its points within 2 mm of the lowest, the one farthest from the lane's centre line
    (the outer edge of the tread)."""
    out = {}
    for n, pts in rv.l3_tyre_points(ROVER, st).items():
        low = pts[pts[:, 2] <= pts[:, 2].min() + 0.002]
        out[n] = low[np.argmax(np.abs(low[:, 1]))]
    return out


def heading(st: tk.BodyState) -> float:
    f = st.rotation()[:, 0]
    return math.atan2(f[1], f[0])


def tyre_slide(prev: dict | None, st: tk.BodyState) -> tuple[float, dict]:
    """Sideways slide of all four tyres this step (each wheel centre's motion along its own axle, summed, whether or
    not the tyre touches the ground: a tyre held just clear and dragged counts too) and the wheel centres and axle
    directions for the next step."""
    now = {}
    slide = 0.0
    for n in ("wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"):
        p, R = st.links[n]
        now[n] = (np.asarray(p, float), np.asarray(R, float)[:, 1])
        if prev is not None:
            p0, ax0 = prev[n]
            ax = ax0 + now[n][1]
            ax[2] = 0.0
            ax /= max(np.linalg.norm(ax), 1e-9)
            slide += abs(float(np.dot(now[n][0] - p0, ax)))
    return slide, now


def track_step(rec: tk.Recorder, st: tk.BodyState) -> None:
    lows = tyre_lows(st)
    pts = rv.l3_tyre_points(ROVER, st)
    out = [n for n, p in lows.items() if abs(p[1]) > LANE_HALF + OUT_TOL
           or float(pts[n][:, 0].min()) < LANE_X[0] or float(pts[n][:, 0].max()) > LANE_X[1]]
    rec.latch("out", bool(out), f"{out[0] if out else ''} left the lane")
    slide, rec.values["wheels"] = tyre_slide(rec.values.get("wheels"), st)
    total = rec.values.get("scrub", 0.0) + slide
    rec.values["scrub"] = total
    rec.latch("scrub", total > SCRUB_MAX, f"the tyres slid sideways {100 * total:.1f} cm in all")
    up = {n: float(p[2] - LANE_TOP) for n, p in lows.items()}
    axle_up = any(up[a] > AXLE_CLEAR and up[b] > AXLE_CLEAR for a, b in (("wheel_fl", "wheel_fr"), ("wheel_rl", "wheel_rr")))
    run = rec.values.get("clear_run", 0) + 1 if axle_up else 0
    rec.values["clear_run"] = run
    rec.latch("lifted", run >= LIFT_STEPS, "both tyres of an axle held off the lane (the rover lifted)")
    rec.extreme("tilt", tk.tilt_deg(st.rotation()))
    away = float(np.linalg.norm(st.pos[:2] - ROVER0.pos[:2]))
    turned = any(abs(q) > STRAIGHT for q in steer_angles(st).values())
    if turned:
        rec.extreme("turned_away", away)
    rec.latch("moved_turned", turned and away > TURNED_TRAVEL_MAX,
              f"the rover was {100 * away:.1f} cm from its start with its front wheels turned more than 5 degrees")


@tk.on_reset
def _reset(environment=None) -> None:
    REC.reset()


@tk.on_step
def _record(environment) -> None:
    REC.steps += 1
    st = tk.body_state(environment, "rover")
    track_step(REC, st)
    REC.window("rest", st.speed() < REST_SPEED and st.spin() < REST_SPIN, REST_STEPS)
    REC.values["released"] = rv.l4_released(ROVER, st, tk.grippers(environment), RELEASE)


def evaluate(st: tk.BodyState, rec: tk.Recorder = REC) -> tuple[float, str]:
    for foul in ("out", "scrub", "lifted", "moved_turned"):
        if rec.flag(foul):
            return 0.0, rec.why.get(foul, foul)
    if tk.tilt_deg(st.rotation()) > TILT_MAX:
        return 0.0, "the rover is not upright"
    if not (rec.full("rest", REST_STEPS) and all(rec.windows["rest"])):
        return 0.0, "not at rest"
    if not rec.values.get("released", False):
        return 0.0, "a gripper is within 5 cm of the rover"
    turned = {k: q for k, q in steer_angles(st).items() if abs(q) > STRAIGHT}
    if turned:
        k, q = next(iter(turned.items()))
        return 0.0, f"front wheels turned {math.degrees(q):.0f} degrees"
    pts = rv.l3_tyre_points(ROVER, st)
    if all(BAY_X[0] <= float(v[:, 0].min()) and float(v[:, 0].max()) <= BAY_X[1] for v in pts.values()) \
            and abs(heading(st)) <= HEADING_TOL:
        return 1.0, "in the bay, wheels straight"
    if st.pos[0] - ROVER_X >= MOVED_MIN:
        return 0.5, "moved along the lane with the wheels straight, not wholly in the bay"
    return 0.25, "wheels straightened"


_LAST: dict = {}


def score(objects) -> float:
    """0 if at any time a tyre leaves the lane (the outer edge of its tread, its full width, more than 5 mm past the
    inner edge of a painted side line, or past either end of the lane), the tyres slide sideways more than 2 cm in all
    (each wheel centre's motion along its own axle, summed for all four wheels whether or not they touch the lane), both
    tyres of an axle are more than 3 mm above the lane for 0.5 s, or its front wheels are turned more than 5 degrees
    while the rover is more than 5 cm from its start. Otherwise, at the end, upright (tilt under 15 degrees), at rest,
    released, front wheels within 5 degrees of straight: 1.0 all four tyres wholly inside the bay (x 36 to 64) and
    heading within 8 degrees of the lane; 0.5 moved at least 10 cm along the lane; 0.25 not moved. 0 otherwise (tilted,
    not at rest, not released, or wheels turned)."""
    s, why = evaluate(objects["rover"])
    _LAST.update(score=s, why=why)
    return s


def success(objects) -> bool:
    return evaluate(objects["rover"])[0] == 1.0


def report() -> dict:
    return {"last": dict(_LAST), "fouls": {k: REC.why[k] for k in REC.flags if REC.flags[k]},
            "scrub_cm": round(100 * REC.values.get("scrub", 0.0), 2),
            "turned_away_cm": round(100 * REC.extremes.get("turned_away", 0.0), 2), "steps": REC.steps}
