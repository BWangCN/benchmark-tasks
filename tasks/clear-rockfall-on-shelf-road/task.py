"""On the station table: a raised earth road (a 4 cm deck) runs left to right across the table in front of the robot.
Its far edge is a bare rock drop-off; on the table below it a small orange rover is parked. On the near side, between
the robot and the road, is the ditch (the table). A yellow-and-white trail crawler (a 1:18 rover) waits at the left
end of the road, facing along it. Two pale tape strips mark its wheel tracks along the road. Three rocks have come
down onto the tracks: one big rock, too wide to grip, and two small ones. The robot must clear the wheel tracks.

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

INSTRUCTION = "Rocks have come down onto the shelf road. Clear the wheel tracks so the waiting rover can get through."

TOP = tk.TOP
TEX = HERE / "textures"

# -- layout (world frame: x forward from the arm bases, y to the robot's left, metres) ---------------------------
DECK_H = 0.040
DECK_TOP = TOP + DECK_H
NEAR_X, FAR_X = 0.25, 0.47            # the road's near edge (to the ditch) and far edge (the drop-off)
ROAD_Y = (-0.32, 0.36)
TRACK_X = 0.36                        # centre line of the wheel tracks
TRACK_HALF = rv.WHEEL_Y               # 5.1 cm: half the rover's track
STRIP_W = 0.030                       # tape strips 3 cm wide under each wheel line
STRIPS = [(TRACK_X - TRACK_HALF - STRIP_W / 2, TRACK_X - TRACK_HALF + STRIP_W / 2),
          (TRACK_X + TRACK_HALF - STRIP_W / 2, TRACK_X + TRACK_HALF + STRIP_W / 2)]
ROCK_GREY = (0.52, 0.50, 0.47)
EARTH = (0.46, 0.34, 0.23)
TAPE = (0.93, 0.88, 0.62)

_mid = ((NEAR_X + FAR_X) / 2, (ROAD_Y[0] + ROAD_Y[1]) / 2)
_lx, _ly = FAR_X - NEAR_X, ROAD_Y[1] - ROAD_Y[0]
ROAD = tk.body("road", pos=(_mid[0], _mid[1], TOP), shapes=[
    tk.Shape.span((-_lx / 2, -_ly / 2, 0.0), (_lx / 2, _ly / 2, DECK_H), ROCK_GREY, friction=(0.6, 0.5)),
    tk.Shape.tile((0.0, 0.0, DECK_H + 0.0004), (_lx, _ly), str(TEX / "earth.png"), texture_m=0.12),
    *[tk.Shape.span((x0 - _mid[0], -_ly / 2, DECK_H + 0.0006), (x1 - _mid[0], _ly / 2, DECK_H + 0.0010), TAPE)
      for x0, x1 in STRIPS],
])
# visual-only tape: the strips are thin boxes, 0.4 mm proud, part of the fixed road body

# the waiting rover at the left end of the road, facing along it (to the robot's right)
WAIT_POS = (TRACK_X, 0.21)
WAITING = rv.rover_l1("waiting_rover", pos=(WAIT_POS[0], WAIT_POS[1], DECK_TOP + 0.0005),
                      rot_xyzw=rv.upright(-math.pi / 2))
# the parked rover on the table below the drop-off
PARKED_POS = (0.62, -0.10)              # its rear bumper 1.9 cm clear of the road's far edge
PARKED = rv.rover_l1("parked_rover", pos=(PARKED_POS[0], PARKED_POS[1], TOP + 0.0005), rot_xyzw=rv.upright(0.0),
                     body_color=(0.95, 0.45, 0.10))


def _rock(name, centre_xy, size, mass, yaw=0.0, cap=0.6):
    """A blocky rock: a main box and a smaller tilted box on its top, so it does not read as a cube."""
    sx, sy, sz = size
    shapes = [tk.Shape.box((0, 0, sz / 2), (sx, sy, sz), ROCK_GREY, friction=(0.7, 0.6)),
              tk.Shape.box((0.1 * sx, -0.05 * sy, sz), (cap * sx, cap * sy, 0.35 * sz), (0.47, 0.45, 0.42),
                           rot_xyzw=tk.quat_xyzw_from_axis_angle((1, 1, 0), 0.25), friction=(0.7, 0.6))]
    return tk.body(name, pos=(centre_xy[0], centre_xy[1], DECK_TOP + 0.0005), shapes=shapes, mass=mass, yaw=yaw)


ROCK_A = _rock("rock_a", (TRACK_X, -0.10), (0.100, 0.120, 0.065), 0.90, yaw=0.15)
ROCK_B = _rock("rock_b", (TRACK_X + TRACK_HALF, 0.02), (0.050, 0.040, 0.040), 0.12, yaw=-0.3)
ROCK_C = _rock("rock_c", (TRACK_X - TRACK_HALF, -0.25), (0.045, 0.050, 0.040), 0.12, yaw=0.4)
ROCKS = {"rock_a": (ROCK_A, 0.7), "rock_b": (ROCK_B, 0.15), "rock_c": (ROCK_C, 0.15)}
OBJECTS = [ROAD, WAITING, PARKED, ROCK_A, ROCK_B, ROCK_C]

# -- scoring -------------------------------------------------------------------------------------------------------
TOUCH = 0.003            # m: a rock's surface closer than this to the parked rover touches it
PARKED_MOVED, PARKED_TURNED = 0.003, 1.0   # m, degrees: the parked rover disturbed (struck by a rock or an arm)
WAIT_MOVED, WAIT_TURNED = 0.010, 5.0       # m, degrees
REST_SPEED, REST_SPIN, REST_STEPS = 0.01, 0.5, 20
RELEASE = 0.05

REC = tk.Recorder()


def footprint(spec: tk.Body, st: tk.BodyState) -> np.ndarray:
    return tk.body_points(spec, st.pos, st.rotation())


def dense_points(spec: tk.Body, st: tk.BodyState, n: int = 6) -> np.ndarray:
    """Points on every face of the rock's boxes (an n x n grid per face), for contact tests."""
    R, out = st.rotation(), []
    g = np.linspace(-0.5, 0.5, n)
    for s in spec.shapes:
        size, c, r = np.asarray(s.size), np.asarray(s.centre), s.rotation()
        for ax in range(3):
            u, v = [i for i in range(3) if i != ax]
            for sign in (-0.5, 0.5):
                for a in g:
                    for b in g:
                        p = np.zeros(3)
                        p[ax], p[u], p[v] = sign * size[ax], a * size[u], b * size[v]
                        out.append(c + r @ p)
    return st.pos + np.asarray(out) @ R.T


PATH = (STRIPS[0][0], STRIPS[1][1])   # the rover's swept path: between the outer edges of the two tape strips


def cleared(spec: tk.Body, st: tk.BodyState) -> bool:
    """A rock is clear when it lies wholly in the ditch (every part nearer the robot than the road's near edge, and
    below the road top), or when it is still on the road with no part over the rover's path (the band between the
    outer edges of the two tape strips). Measured on the rock's axis-aligned outline seen from above."""
    pts = footprint(spec, st)
    x0, x1 = pts[:, 0].min(), pts[:, 0].max()
    if x1 < NEAR_X and pts[:, 2].min() < DECK_TOP - 0.010 and ROAD_Y[0] <= st.pos[1] <= ROAD_Y[1]:
        return True                                          # in the ditch beside the road, not off one of its ends
    on_road = pts[:, 2].min() > DECK_TOP - 0.010 and ROAD_Y[0] <= st.pos[1] <= ROAD_Y[1]
    return bool(on_road and (x1 < PATH[0] or x0 > PATH[1]))


def on_a_strip(spec: tk.Body, st: tk.BodyState) -> bool:
    return not cleared(spec, st)


@tk.on_reset
def _reset(environment=None) -> None:
    REC.reset()
    if environment is not None:
        REC.values["wait0"] = tk.body_state(environment, "waiting_rover").pos.copy()


@tk.on_step
def _record(environment) -> None:
    REC.steps += 1
    wait = tk.body_state(environment, "waiting_rover")
    parked = tk.body_state(environment, "parked_rover")
    w0 = REC.values.setdefault("wait0", wait.pos.copy())
    REC.latch("waiting_moved", float(np.linalg.norm(wait.pos[:2] - w0[:2])) > WAIT_MOVED, "the waiting rover moved")
    h0 = REC.values.setdefault("wait_h0", tk.heading(wait.rotation()))
    REC.latch("waiting_moved", abs(math.degrees(math.remainder(tk.heading(wait.rotation()) - h0, 2 * math.pi)))
              > WAIT_TURNED, "the waiting rover was turned")
    p0 = REC.values.setdefault("parked0", (parked.pos.copy(), tk.heading(parked.rotation())))
    if (float(np.linalg.norm(parked.pos - p0[0])) > PARKED_MOVED
            or abs(math.degrees(math.remainder(tk.heading(parked.rotation()) - p0[1], 2 * math.pi))) > PARKED_TURNED
            or tk.tilt_deg(parked.rotation()) > PARKED_TURNED):
        REC.latch("parked_touched", True, "the parked rover was struck (moved or turned)")
    grips = tk.grippers(environment)
    released, rest = True, True
    for name, (spec, _) in ROCKS.items():
        st = tk.body_state(environment, name)
        REC.latch("over_edge", st.pos[0] > FAR_X, f"{name}'s centre passed the far edge")
        pts = dense_points(spec, st)
        if tk.distance_to_body(PARKED, parked, pts) < TOUCH:
            REC.latch("parked_touched", True, f"{name} touched the parked rover")
        released &= tk.released(spec, st, grips, RELEASE)
        rest &= st.speed() < REST_SPEED and st.spin() < REST_SPIN
    REC.window("rest", rest, REST_STEPS)
    REC.values["released"] = released


def evaluate(states: dict, rec: tk.Recorder = REC) -> tuple[float, str]:
    for foul in ("over_edge", "parked_touched", "waiting_moved"):
        if rec.flag(foul):
            return 0.0, rec.why.get(foul, foul)
    if not (rec.full("rest", REST_STEPS) and all(rec.windows["rest"])):
        return 0.0, "rocks not at rest"
    if not rec.values.get("released", False):
        return 0.0, "a gripper is within 5 cm of a rock"
    total, done = 0.0, []
    for name, (spec, worth) in ROCKS.items():
        if not on_a_strip(spec, states[name]):
            total += worth
            done.append(name)
    return round(total, 3), f"cleared {done}"


_LAST: dict = {}


def score(objects) -> float:
    """0 if at any time a rock's centre passes the far edge, a rock comes within 3 mm of the parked rover or the parked
    rover is moved more than 3 mm or turned more than 1 degree, or the waiting rover is moved more than 1 cm
    horizontally or turned more than 5 degrees. Otherwise, at the end, with the rocks at rest and
    released: 0.7 for the big rock and 0.15 for each small one when it is clear (wholly in the ditch, or on the road
    wholly outside the rover's path)."""
    s, why = evaluate(objects)
    _LAST.update(score=s, why=why)
    return s


def success(objects) -> bool:
    return evaluate(objects)[0] == 1.0


def report() -> dict:
    return {"last": dict(_LAST), "fouls": {k: REC.why[k] for k in REC.flags if REC.flags[k]}}
