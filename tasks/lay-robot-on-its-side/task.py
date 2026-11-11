"""On the station table: a packed-earth pad, with a strip of turf and two grey rocks along its far side. A small
grey humanoid robot (a passive 1:3 model, 44 cm tall, 1.24 kg, 14 limp joints with hard end stops) lies on its back
on the pad, along the table's width, head to the robot's right, both arms straight along its trunk (20 degrees out
from its sides), legs straight. The robot must lay it on its left side so that it stays there on its own.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import dummy as D  # noqa: E402
import trailkit as tk  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 3.0      # a full score ends the trial once it has held 3 s more (5 s still in all)

INSTRUCTION = "Lay the humanoid robot on the pad on its left side so that it stays there on its own."

TOP = tk.TOP
TEX = HERE / "textures"

# -- layout (world frame: x forward from the arm bases, y to the robot's left, metres) ---------------------------
PAD_H = 0.010                        # packed-earth pad: a 1 cm board on the table
GROUND = TOP + PAD_H
PAD_X = (0.03, 0.46)
PAD_Y = (-0.25, 0.25)                # between the arms' base plates
TURF_X = 0.40                        # turf strip from here to the pad's far edge (away from the scored contacts)
DUMMY_X = 0.28                       # the dummy's hip midpoint, lying along y
DUMMY_Y = 0.0
ROCK = (0.46, 0.45, 0.43)
ROCKS = [((0.43, 0.20, 0.0), (0.05, 0.06, 0.045)), ((0.43, -0.21, 0.0), (0.06, 0.05, 0.035))]

PAD = tk.body("pad", pos=((PAD_X[0] + PAD_X[1]) / 2, 0.0, TOP), shapes=[
    tk.Shape.span((-(PAD_X[1] - PAD_X[0]) / 2, PAD_Y[0], 0.0), ((PAD_X[1] - PAD_X[0]) / 2, PAD_Y[1], PAD_H), ROCK,
                  friction=(0.6, 0.5)),
    tk.Shape.tile((-(PAD_X[1] - TURF_X) / 2, 0.0, PAD_H + 0.0004), (TURF_X - PAD_X[0], PAD_Y[1] - PAD_Y[0]),
                  str(TEX / "earth.png"), texture_m=0.12),
    tk.Shape.tile(((TURF_X - PAD_X[0]) / 2, 0.0, PAD_H + 0.0006), (PAD_X[1] - TURF_X, PAD_Y[1] - PAD_Y[0]),
                  str(TEX / "turf.png"), texture_m=0.08),
] + [tk.Shape.box((x - (PAD_X[0] + PAD_X[1]) / 2, y, PAD_H + s[2] / 2), s, ROCK, friction=(0.7, 0.6), tag="rock")
     for (x, y, _), s in ROCKS])

_RZ180 = np.diag([-1.0, -1.0, 1.0])  # half a turn about vertical: the dummy's head toward -y, its left toward -x


def head_minus_y(rot_xyzw) -> tuple:
    return tk.quat_from_rot(_RZ180 @ tk.rot_from_xyzw(rot_xyzw))


START_ROT = head_minus_y(D.on_back())   # chest up, head toward -y, its left side toward the robot
ARMS_OUT = 0.35                      # rad: both arms straight along the trunk, 20 degrees out from its sides
START_POSE = {"l_sh_roll": ARMS_OUT, "r_sh_roll": -ARMS_OUT}
DUMMY = D.dummy("dummy", pos=(DUMMY_X, DUMMY_Y, GROUND + D.rest_height(START_ROT, START_POSE) + 0.002),
                rot_xyzw=START_ROT, pose=START_POSE)
OBJECTS = [PAD, DUMMY]

# -- scoring -------------------------------------------------------------------------------------------------------
SIDE_DEG = 25.0           # on its left side: the line from the right shoulder to the left within this of straight down
SIDE_STEPS = 20           # 0.25 needs it on its left side for this many control steps in a row (1 s)
LIFT_FOUL = 0.040         # m: the trunk's lowest point above the ground, at any time
SPINE_FOUL_DEG = 30.0     # the spine (hips to head) more than this from level, at any time: stood up, not rolled
REST_SPEED, REST_SPIN, REST_JOINT, REST_STEPS = 0.01, 0.5, 0.5, 40   # m/s, rad/s, rad/s; 2 s at 20 Hz
RELEASE = 0.05            # m: no gripper point and no point along either arm within this of the dummy
ROCK_CLEAR = 0.01         # m: no point of a rock within this of the dummy (it rests on the pad and its own limbs)
PAD_BELOW = 0.002         # m: every point of the dummy over the pad and no more than this below its top

REC = tk.Recorder()


def trunk_lift(st: tk.BodyState) -> float:
    """Height (m) of the lowest point of the trunk, pelvis and head above the ground."""
    return D.lowest(DUMMY, st, which={"base"}) - GROUND


def spine_tilt(st: tk.BodyState) -> float:
    """Degrees between the spine (hips to head) and level."""
    return abs(90.0 - D.angle_deg(D.spine(st), (0.0, 0.0, 1.0)))


def side_angle(st: tk.BodyState) -> float:
    """Degrees between the line from the right shoulder to the left and straight down (0: on its left side)."""
    return D.angle_deg(D.shoulder_line(st), (0.0, 0.0, -1.0))


def on_left_side(st: tk.BodyState) -> bool:
    return side_angle(st) <= SIDE_DEG


def _rock_points(step: float = 0.005) -> np.ndarray:
    """Points over the surface of both rocks (world), every ``step`` metres."""
    out = []
    for (x, y, _), s in ROCKS:
        lo = np.array([x - s[0] / 2, y - s[1] / 2, GROUND])
        hi = np.array([x + s[0] / 2, y + s[1] / 2, GROUND + s[2]])
        axes = [np.linspace(lo[i], hi[i], max(2, int(round((hi[i] - lo[i]) / step)) + 1)) for i in range(3)]
        g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        on_face = np.any(np.isclose(g, lo) | np.isclose(g, hi), axis=1)
        out.append(g[on_face])
    return np.concatenate(out)


ROCK_POINTS = _rock_points()


def rock_gap(st: tk.BodyState) -> float:
    """Smallest distance (m) between a rock and the dummy."""
    return D.distance(DUMMY, st, ROCK_POINTS)


def on_pad(st: tk.BodyState) -> bool:
    """Every point of the dummy's colliders inside the pad's outline and no more than 2 mm below its top."""
    p = D.points(DUMMY, st)
    return bool((p[:, 0] >= PAD_X[0]).all() and (p[:, 0] <= PAD_X[1]).all() and (p[:, 1] >= PAD_Y[0]).all()
                and (p[:, 1] <= PAD_Y[1]).all() and (p[:, 2] >= GROUND - PAD_BELOW).all())


def joint_speed(environment) -> float:
    asset = environment.unwrapped.scene["dummy"]
    return float(np.abs(tk._np(asset.data.joint_vel)[0]).max())


@tk.on_reset
def _reset(environment=None) -> None:
    REC.reset()


@tk.on_step
def _record(environment) -> None:
    st = tk.body_state(environment, "dummy")
    REC.steps += 1
    lift = trunk_lift(st)
    REC.extreme("max_lift", lift)
    REC.latch("lifted", lift > LIFT_FOUL, "the trunk was lifted more than 4 cm off the ground")
    REC.extreme("max_spine_tilt", spine_tilt(st))
    REC.latch("stood", spine_tilt(st) > SPINE_FOUL_DEG, "the spine was tilted more than 30 degrees from level")
    REC.window("rest", st.speed() < REST_SPEED and st.spin() < REST_SPIN and joint_speed(environment) < REST_JOINT,
               REST_STEPS)
    REC.window("released", D.released(DUMMY, st, tk.grippers(environment), RELEASE), REST_STEPS)
    side = REC.window("side", on_left_side(st), SIDE_STEPS)
    REC.latch("was_on_side", len(side) == SIDE_STEPS and all(side), "on its left side for 1 s")


def evaluate(st: tk.BodyState, rec: tk.Recorder = REC) -> tuple[float, str]:
    """(score, reason) from the recorder and the dummy's final state."""
    if rec.flag("lifted"):
        return 0.0, "the trunk was lifted more than 4 cm off the ground"
    if rec.flag("stood"):
        return 0.0, "the body was stood up (spine more than 30 degrees from level)"
    a = side_angle(st)
    rest = rec.full("rest", REST_STEPS) and all(rec.windows["rest"])
    released = rec.full("released", REST_STEPS) and all(rec.windows["released"])
    clear = rock_gap(st) >= ROCK_CLEAR
    pad = on_pad(st)
    if a <= SIDE_DEG and rest and released and clear and pad:
        return 1.0, f"on its left side ({a:.0f} degrees), released and at rest on its own"
    if rec.flag("was_on_side"):
        why = (f"did not stay on its left side (now {a:.0f} degrees from it)" if a > SIDE_DEG else
               "still held or touched in the last 2 s" if not released else
               "not at rest" if not rest else
               "leaning on a rock" if not clear else
               "not wholly on the pad")
        return 0.25, f"was on its left side for 1 s, but {why}"
    return 0.0, f"never on its left side for 1 s (now {a:.0f} degrees from it)"


_LAST: dict = {}


def score(objects) -> float:
    """0 if at any time the lowest point of the trunk, pelvis and head was more than 4 cm above the ground, or the
    spine (the line from the hips to the head) was more than 30 degrees from level (the body was stood up). The dummy
    is on its left side when the line from its right shoulder to its left is within 25 degrees of straight down.
    1.0 if at the end it is on its left side, at rest (base under 1 cm/s and 0.5 rad/s, every joint under 0.5 rad/s)
    and released (no gripper point and no point along either arm within 5 cm of it), both for each of the last 40
    control steps (2 s at 20 Hz), every point of it is over the pad and no more than 2 mm below the pad's top, and no
    rock is within 1 cm of it; 0.25 if it was on its left side for 20 control
    steps in a row (1 s) at some time but does not meet all of that at the end; otherwise 0."""
    s, why = evaluate(objects["dummy"])
    _LAST.update(score=s, why=why)
    return s


def success(objects) -> bool:
    return evaluate(objects["dummy"])[0] == 1.0


def report() -> dict:
    return {"last": dict(_LAST), "fouls": {k: REC.why[k] for k in REC.flags if REC.flags[k] and k in ("lifted", "stood")},
            "was_on_side": REC.why.get("was_on_side", ""),
            "max_lift_cm": round(100 * REC.extremes.get("max_lift", 0.0), 2),
            "max_spine_tilt_deg": round(REC.extremes.get("max_spine_tilt", 0.0), 1)}
