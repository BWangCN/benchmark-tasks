"""On the station table: an earth track runs away from the robot to a charging dock (a back plate with two contact
pads, no guides). A small yellow-and-white trail crawler (a 1:18 rover, motor disconnected so its wheels roll freely,
steering fixed straight) stands on the track with its front axle on the dock's centreline, 13.5 cm short of it, but
turned 12 degrees off the dock's axis. A short tow bar sticks out from its rear bumper. The robot must put the rover
on its dock.

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

INSTRUCTION = "Put the rover on its charging dock."

TOP = tk.TOP
TEX = HERE / "textures"

# -- layout (world frame: x forward from the arm bases, y to the robot's left, metres) ---------------------------
TRACK_X, TRACK_Y = (0.08, 0.70), (-0.16, 0.16)
DOCK_X = 0.600                         # the dock's face (toward the robot)
PAD_Y, PAD_SIZE = 0.030, (0.006, 0.014, 0.014)
HEADING0 = math.radians(12.0)
FRONT_AXLE0 = (DOCK_X - 0.14 - (rv.S.LENGTH / 2 - rv.AXLE_X), 0.0)   # the front axle on the centreline, bumper 14 cm short
TAB_SIZE = (0.048, 0.016, 0.020)
TAB_X = rv.S.LENGTH / 2 + TAB_SIZE[0] / 2 - 0.002
TAB_Z = rv.S.BUMPER_HEIGHT
TABS = (tk.Shape.box((-TAB_X, 0.0, TAB_Z), TAB_SIZE, (0.20, 0.20, 0.22), friction=(1.0, 0.9), tag="tow_r"),)
GROUND = TOP + 0.0006

_lx, _ly = TRACK_X[1] - TRACK_X[0], TRACK_Y[1] - TRACK_Y[0]
TRACK = tk.body("track", pos=((TRACK_X[0] + TRACK_X[1]) / 2, 0.0, TOP), shapes=[
    tk.Shape.span((-_lx / 2, -_ly / 2, 0.0), (_lx / 2, _ly / 2, 0.0006), (0.46, 0.34, 0.23), friction=(0.8, 0.7)),
    tk.Shape.tile((0, 0, 0.0008), (_lx, _ly), str(TEX / "earth.png"), texture_m=0.12),
])
DOCK = tk.body("dock", pos=(DOCK_X, 0.0, TOP), shapes=[
    tk.Shape.span((0.0, -0.08, 0.0), (0.03, 0.08, 0.07), (0.25, 0.27, 0.30), friction=(0.6, 0.5)),       # back plate
    *[tk.Shape.box((-PAD_SIZE[0] / 2, s * PAD_Y, rv.S.BUMPER_HEIGHT), PAD_SIZE, (0.85, 0.65, 0.15), tag=f"pad{i}")
      for i, s in enumerate((-1, 1))],                                                                     # contacts
])
_c, _s = math.cos(HEADING0), math.sin(HEADING0)
ROVER_POS = (FRONT_AXLE0[0] - rv.AXLE_X * _c, FRONT_AXLE0[1] - rv.AXLE_X * _s, GROUND + 0.0005)
ROVER = rv.rover_l3("rover", pos=ROVER_POS, rot_xyzw=rv.upright(HEADING0), extra_shapes=TABS)
OBJECTS = [TRACK, DOCK, ROVER]

# -- scoring -------------------------------------------------------------------------------------------------------
FACE_TOL = 0.003         # m: the front bumper's face within this of the dock's contact pads
SQUARE_DEG = 2.0
CENTRE_TOL = 0.008       # m: the bumper's centre off the dock's centreline (the pads are 6 cm apart, 1.4 cm wide)
DOWN_TOL = 0.008
TIPPED_DEG = 30.0
REST_SPEED, REST_SPIN, REST_STEPS = 0.005, 0.5, 20   # free wheels: a slow roll-back must not count as rest
RELEASE = 0.05

REC = tk.Recorder()


def heading(st: tk.BodyState) -> float:
    f = st.rotation()[:, 0]
    return math.atan2(f[1], f[0])


def bumper_face(st: tk.BodyState) -> np.ndarray:
    """The front bumper's two outer front corners (world): both must touch the pads for the rover to dock."""
    bx, by, _ = rv.S.BUMPER_SIZE
    return np.array([st.to_world((rv.S.LENGTH / 2, s * by / 2, rv.S.BUMPER_HEIGHT)) for s in (-1, 1)])


def docked(st: tk.BodyState) -> tuple[bool, str]:
    corners = bumper_face(st)
    gap = DOCK_X - PAD_SIZE[0] - corners[:, 0]
    off = math.degrees(heading(st))
    centre = float(corners[:, 1].mean())
    if abs(off) > SQUARE_DEG:
        return False, f"turned {off:.1f} degrees off the dock's axis"
    if abs(centre) > CENTRE_TOL:
        return False, f"{100 * centre:.1f} cm off the dock's centre"
    if gap.max() > FACE_TOL:
        return False, f"{1000 * gap.max():.0f} mm short of the contacts"
    return True, "docked"


def tyres_down(st: tk.BodyState) -> bool:
    lows = {n: pts[np.argmin(pts[:, 2])] for n, pts in rv.l3_tyre_points(ROVER, st).items()}
    return all(p[2] - GROUND <= DOWN_TOL for p in lows.values())


@tk.on_reset
def _reset(environment=None) -> None:
    REC.reset()


@tk.on_step
def _record(environment) -> None:
    REC.steps += 1
    st = tk.body_state(environment, "rover")
    REC.latch("tipped", tk.tilt_deg(st.rotation()) > TIPPED_DEG, "the rover tipped past 30 degrees")
    REC.window("rest", st.speed() < REST_SPEED and st.spin() < REST_SPIN, REST_STEPS)
    REC.values["released"] = tk.released(ROVER, st, tk.grippers(environment), RELEASE)


def evaluate(st: tk.BodyState, rec: tk.Recorder = REC) -> tuple[float, str]:
    if rec.flag("tipped"):
        return 0.0, rec.why.get("tipped", "tipped")
    if not (rec.full("rest", REST_STEPS) and all(rec.windows["rest"])):
        return 0.0, "not at rest"
    if not rec.values.get("released", False):
        return 0.0, "a gripper or an arm is within 5 cm of the rover"
    if not tyres_down(st):
        return 0.0, "a tyre is off the track"
    ok, why = docked(st)
    return (1.0, why) if ok else (0.0, why)


_LAST: dict = {}


def score(objects) -> float:
    """0 if at any time the rover tips past 30 degrees. Otherwise, at the end, at rest (under 5 mm/s) and released, all
    four tyres on the track: 1.0 if docked (front bumper within 3 mm of the contact pads, heading within 2 degrees of the dock's
    axis, centred within 8 mm); otherwise 0."""
    s, why = evaluate(objects["rover"])
    _LAST.update(score=s, why=why)
    return s


def success(objects) -> bool:
    return evaluate(objects["rover"])[0] == 1.0


def report() -> dict:
    return {"last": dict(_LAST), "fouls": {k: REC.why[k] for k in REC.flags if REC.flags[k]}, "steps": REC.steps}
