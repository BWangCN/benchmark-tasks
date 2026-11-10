"""On the station table: an earth track runs away from the robot, raised 2 cm on two decks with an 8.3 cm bog between
them (dark wet mud at table level). A small yellow-and-white trail crawler (a 1:18 rover, motor disconnected so its
wheels roll freely) stands on the near deck facing the bog. Four six-sided logs (2 cm across the corners, 18 cm long)
lie on the turf beside the track. The robot must lay the logs across the bog, then push the rover over them.

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

INSTRUCTION = "The track ahead is a bog. Lay the logs, then push the rover across onto the far bank."

TOP = tk.TOP
TEX = HERE / "textures"

# -- layout (world frame: x forward from the arm bases, y to the robot's left, metres) ---------------------------
DECK_H = 0.020                        # both decks; the logs' tops lie flush with them
DECK_TOP = TOP + DECK_H
NEAR = (0.06, 0.370)                  # near deck x range (the rover starts here)
BOG = (0.370, 0.453)                  # the bog between the decks: 8.3 cm, room for four 2 cm logs and 3 mm of play
FAR = (0.453, 0.74)
TRACK_Y = (-0.12, 0.12)
LOG_R, LOG_L, LOG_MASS = 0.010, 0.180, 0.042
LOG_ROLL_DAMPING = 4.0                # stands in for a rough log's rolling resistance on turf and mud
N_LOGS = 4
ROVER_X = 0.235
MUD = (0.20, 0.17, 0.12)


def _deck(name, x0, x1):
    lx, ly = x1 - x0, TRACK_Y[1] - TRACK_Y[0]
    return tk.body(name, pos=((x0 + x1) / 2, 0.0, TOP), shapes=[
        tk.Shape.span((-lx / 2, -ly / 2, 0.0), (lx / 2, ly / 2, DECK_H), (0.40, 0.31, 0.22), friction=(0.8, 0.7)),
        tk.Shape.tile((0, 0, DECK_H + 0.0004), (lx, ly), str(TEX / "earth.png"), texture_m=0.12),
    ])


NEAR_DECK = _deck("near_deck", *NEAR)
FAR_DECK = _deck("far_deck", *FAR)
# the bog: wet mud at table level, a thin fixed sheet (its own look) between the decks
BOG_SHEET = tk.body("bog", pos=((BOG[0] + BOG[1]) / 2, 0.0, TOP), shapes=[
    tk.Shape.span((-(BOG[1] - BOG[0]) / 2, TRACK_Y[0], 0.0), ((BOG[1] - BOG[0]) / 2, TRACK_Y[1], 0.0008), MUD,
                  friction=(0.6, 0.5)),
])
# turf verges beside the track, where the logs lie
VERGE = tk.body("verge", pos=(0.33, 0.0, TOP), shapes=[
    tk.Shape.span((-0.15, 0.12, 0.0), (0.15, 0.26, 0.0004), (0.25, 0.45, 0.20), friction=(0.9, 0.8)),
    tk.Shape.span((-0.15, -0.26, 0.0), (0.15, -0.12, 0.0004), (0.25, 0.45, 0.20), friction=(0.9, 0.8)),
    tk.Shape.tile((0.0, 0.19, 0.0006), (0.30, 0.14), str(TEX / "turf.png"), texture_m=0.08),
    tk.Shape.tile((0.0, -0.19, 0.0006), (0.30, 0.14), str(TEX / "turf.png"), texture_m=0.08),
])
ROVER = rv.rover_l3("rover", pos=(ROVER_X, 0.0, DECK_TOP + 0.0006), rot_xyzw=rv.upright(0.0), wheel_drag=0.006)


LOG_FLAT = LOG_R * math.sqrt(3.0)    # a log's six-sided section: 2 cm across its corners, 1.73 cm across its flats


def _log(name, xy):
    """A log lying along x on the verge, resting on a flat (it is laid across the track, along y, in the bog). Its
    six-sided section (three boxes turned 60 degrees apart) stands in for a real log's uneven bark: a perfect cylinder
    rolls away at the slightest touch in a rigid-body simulator."""
    shapes = [tk.Shape.box((0, 0, 0), (LOG_L, LOG_R, LOG_FLAT), (0.55, 0.38, 0.22), friction=(0.9, 0.8),
                           rot_xyzw=tk.quat_xyzw_from_axis_angle((1, 0, 0), k * math.pi / 3)) for k in range(3)]
    return tk.body(name, pos=(xy[0], xy[1], TOP + LOG_FLAT / 2 + 0.0008), shapes=shapes, mass=LOG_MASS,
                   angular_damping=LOG_ROLL_DAMPING)


LOGS = [_log(f"log_{i + 1}", xy) for i, xy in enumerate([(0.31, 0.20), (0.37, 0.20), (0.31, -0.20), (0.37, -0.20)])]
OBJECTS = [NEAR_DECK, FAR_DECK, BOG_SHEET, VERGE, ROVER, *LOGS]

# -- scoring -------------------------------------------------------------------------------------------------------
BOG_TOP = TOP + 0.0008   # the bog sheet's top
LOG_TOP = BOG_TOP + LOG_R * math.sqrt(3.0)   # top of a log lying on a flat in the bog (17 mm up; the decks are 20 mm)
BOG_TOUCH = 0.004        # m: a tyre touches the bog when its lowest point is within this of the bog sheet, over it
SPEED_MAX = 0.30         # m/s: faster, the rover could jump the 8.3 cm bog between two control steps
ON_LOG = 0.004           # m: over the bog, a tyre rides on a log when its lowest point is within this of a log's top
LIFT_CLEAR = 0.008
CROSSWISE_DEG = 10.0     # a log lies across the track: its axis within 10 degrees of the track's width (y)
GAP_MAX = 0.005          # m: neighbouring logs at most 5 mm apart, and the end logs at most 5 mm from the decks. The bog
#                          leaves 3 mm of play for four logs, so four logs lying crosswise in it are a tight mat; the
#                          test catches a log missing, riding up, or lying askew
REST_SPEED, REST_SPIN, REST_STEPS = 0.01, 0.5, 20
RELEASE = 0.05

REC = tk.Recorder()
WHEEL_PATHS = [(-rv.WHEEL_Y - rv.S.TYRE_WIDTH / 2, -rv.WHEEL_Y + rv.S.TYRE_WIDTH / 2),
               (rv.WHEEL_Y - rv.S.TYRE_WIDTH / 2, rv.WHEEL_Y + rv.S.TYRE_WIDTH / 2)]


def tyre_lows(st: tk.BodyState) -> dict[str, np.ndarray]:
    return {n: pts[np.argmin(pts[:, 2])] for n, pts in rv.l3_tyre_points(ROVER, st).items()}


def in_bog(p) -> bool:
    return BOG[0] < p[0] < BOG[1] and TRACK_Y[0] <= p[1] <= TRACK_Y[1]


def on_a_log(p, logs: dict) -> bool:
    """Over the bog, is there a log under this tyre point, lying crosswise in the bog, with the tyre on its top?"""
    for st in logs.values():
        a = log_axis(st)
        if (abs(a[1]) < math.cos(math.radians(CROSSWISE_DEG)) or not BOG[0] - 0.002 <= st.pos[0] <= BOG[1] + 0.002
                or st.pos[2] > TOP + LOG_R + 0.004):                  # lying in the bog, not held up
            continue
        half = LOG_L / 2 * abs(float(a[1]))
        if (abs(p[0] - st.pos[0]) <= LOG_R + 0.006 and abs(p[1] - st.pos[1]) <= half
                and p[2] - (st.pos[2] + LOG_R * math.sqrt(3.0) / 2) <= ON_LOG):
            return True
    return False


def log_axis(st: tk.BodyState) -> np.ndarray:
    return st.rotation()[:, 0]


def mat(logs: dict) -> tuple[bool, str]:
    """The log mat is laid when all four logs lie in the bog crosswise (within 10 degrees of the track's width), resting
    on the bog, each spanning both wheel paths, with no gap wider than 5 mm between neighbours or at the decks."""
    xs = []
    for name, st in logs.items():
        a = log_axis(st)
        ang = math.degrees(math.acos(min(1.0, abs(float(a[1])) / max(1e-9, float(np.linalg.norm(a[:2]))))))
        if ang > CROSSWISE_DEG or abs(a[2]) > math.sin(math.radians(CROSSWISE_DEG)):
            return False, f"{name} not crosswise"
        if not (BOG[0] - 0.002 <= st.pos[0] <= BOG[1] + 0.002) or st.pos[2] > TOP + LOG_R + 0.004:   # resting in the bog
            return False, f"{name} not lying in the bog"
        ends = st.pos[1] + np.array([-1, 1]) * LOG_L / 2 * abs(float(a[1]))
        if not all(ends[0] <= p0 and ends[1] >= p1 for p0, p1 in WHEEL_PATHS):
            return False, f"{name} does not span both wheel paths"
        xs.append(float(st.pos[0]))
    xs.sort()
    edges = [BOG[0]] + [e for x in xs for e in (x - LOG_R, x + LOG_R)] + [BOG[1]]
    gaps = [edges[i + 1] - edges[i] for i in range(0, len(edges), 2)]
    if max(gaps) > GAP_MAX:
        return False, f"a gap of {1000 * max(gaps):.0f} mm"
    return True, "log mat laid"


@tk.on_reset
def _reset(environment=None) -> None:
    REC.reset()


@tk.on_step
def _record(environment) -> None:
    REC.steps += 1
    st = tk.body_state(environment, "rover")
    lows = tyre_lows(st)
    logs = {lg.name: tk.body_state(environment, lg.name) for lg in LOGS}
    REC.latch("bog", any(in_bog(p) and p[2] - BOG_TOP < BOG_TOUCH for p in lows.values()), "a tyre touched the bog")
    REC.latch("unsupported", any(in_bog(p) and not on_a_log(p, logs) for p in lows.values()),
              "a tyre went over the bog with no log under it")
    REC.latch("early", any(in_bog(p) for p in lows.values()) and not mat(logs)[0],
              "a tyre went over the bog before the mat was laid")
    REC.latch("off_track", any(not (TRACK_Y[0] <= p[1] <= TRACK_Y[1] and NEAR[0] <= p[0] <= FAR[1]) for p in lows.values()),
              "a tyre left the track (the rover went round the bog)")
    REC.latch("fast", st.speed() > SPEED_MAX, f"the rover moved faster than {SPEED_MAX} m/s (shoved across)")
    support = lambda p: DECK_TOP if (NEAR[0] <= p[0] <= NEAR[1] or FAR[0] <= p[0] <= FAR[1]) else (  # noqa: E731
        LOG_TOP if in_bog(p) else TOP)
    REC.latch("lifted", all(p[2] - support(p) > LIFT_CLEAR for p in lows.values()), "all four tyres off the ground")
    grips = tk.grippers(environment)
    moving = [n for n, b in [("rover", st), *logs.items()] if not (b.speed() < REST_SPEED and b.spin() < REST_SPIN)]
    REC.values["moving"] = moving
    rest = not moving
    REC.window("rest", rest, REST_STEPS)
    REC.values["released"] = tk.released(ROVER, st, grips, RELEASE) and all(
        tk.released(lg, logs[lg.name], grips, RELEASE) for lg in LOGS)
    REC.values["mat"] = mat(logs)


def across(st: tk.BodyState) -> bool:
    """All four tyres rest on the far deck."""
    return all(FAR[0] <= p[0] <= FAR[1] and abs(p[2] - DECK_TOP) < 0.008 for p in tyre_lows(st).values())


def evaluate(st: tk.BodyState, rec: tk.Recorder = REC) -> tuple[float, str]:
    for foul in ("bog", "unsupported", "early", "off_track", "fast", "lifted"):
        if rec.flag(foul):
            return 0.0, rec.why.get(foul, foul)
    if not (rec.full("rest", REST_STEPS) and all(rec.windows["rest"])):
        return 0.0, "not at rest"
    if not rec.values.get("released", False):
        return 0.0, "a gripper is within 5 cm of the rover or a log"
    laid, why = rec.values.get("mat", (False, ""))
    if not laid:
        return 0.0, why
    if across(st):
        return 1.0, "logs laid tight and the rover across"
    return 0.5, "logs laid tight; the rover not across"


_LAST: dict = {}


def score(objects) -> float:
    """0 if at any time a tyre touches the bog, is over the bog with no log lying under it, is over the bog before the
    mat is laid, or leaves the track; the rover moves faster than 0.3 m/s; or all four tyres leave the ground. Otherwise, at the end, at rest and released: 0.5 when all four logs lie in the bog crosswise, spanning both
    wheel paths, with no gap over 5 mm;
    1.0 when, in addition, all four tyres rest on the far deck."""
    s, why = evaluate(objects["rover"])
    _LAST.update(score=s, why=why)
    return s


def success(objects) -> bool:
    return evaluate(objects["rover"])[0] == 1.0


def report() -> dict:
    return {"last": dict(_LAST), "fouls": {k: REC.why[k] for k in REC.flags if REC.flags[k]},
            "moving": REC.values.get("moving", []),
            "rest_window": f"{sum(map(bool, REC.windows.get('rest', [])))}/{len(REC.windows.get('rest', []))}",
            "steps": REC.steps}
