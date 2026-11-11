"""On the station table: a packed-earth pad, with a strip of turf and two grey rocks along its far side. A small
grey humanoid robot (a passive 1:3 model, 44 cm tall, 1.24 kg, 14 limp joints with hard end stops) lies face down on
the pad, along the table's width, head to the robot's left, both arms straight along its trunk (20 degrees out from
its sides), legs straight. The robot must turn it onto its back.

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
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "The humanoid robot on the pad has fallen face down. Turn it onto its back."

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

START_ROT = D.face_down()
ARMS_OUT = 0.35                      # rad: both arms straight along the trunk, 20 degrees out from its sides
START_POSE = {"l_sh_roll": ARMS_OUT, "r_sh_roll": -ARMS_OUT}
DUMMY = D.dummy("dummy", pos=(DUMMY_X, DUMMY_Y, GROUND + D.rest_height(START_ROT, START_POSE) + 0.002),
                rot_xyzw=START_ROT, pose=START_POSE)
OBJECTS = [PAD, DUMMY]

# -- scoring -------------------------------------------------------------------------------------------------------
CHEST_UP_DEG = 12.0       # full score: the chest's normal within this of straight up
CHEST_PART_DEG = 45.0     # 0.25: within this
LIFT_FOUL = 0.040         # m: the trunk's lowest point above the ground, at any time
REST_SPEED, REST_SPIN, REST_JOINT, REST_STEPS = 0.01, 0.5, 0.5, 20   # m/s, rad/s, rad/s; 1 s at 20 Hz
RELEASE = 0.05            # m: no gripper point and no point along either arm within this of the dummy
SPINE_FOUL_DEG = 30.0     # the spine (hips to head) more than this from level, at any time: stood up, not rolled
TRAP_FROM_DEG = 135.0     # an arm under the body counts as trapped once the chest is this close to up (turned 45)

REC = tk.Recorder()


def trunk_lift(st: tk.BodyState) -> float:
    """Height (m) of the lowest point of the trunk, pelvis and head above the ground."""
    return D.lowest(DUMMY, st, which={"base"}) - GROUND


def spine_tilt(st: tk.BodyState) -> float:
    """Degrees between the spine (hips to head) and level."""
    return abs(90.0 - D.angle_deg(D.spine(st), (0.0, 0.0, 1.0)))


def chest_angle(st: tk.BodyState) -> float:
    """Degrees between the direction the chest faces and straight up."""
    return D.angle_deg(D.chest_normal(st), (0.0, 0.0, 1.0))


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
    REC.window("rest", st.speed() < REST_SPEED and st.spin() < REST_SPIN and joint_speed(environment) < REST_JOINT,
               REST_STEPS)
    REC.extreme("max_spine_tilt", spine_tilt(st))
    REC.latch("stood", spine_tilt(st) > SPINE_FOUL_DEG, "the spine was tilted more than 30 degrees from level")
    REC.window("released", D.released(DUMMY, st, tk.grippers(environment), RELEASE), REST_STEPS)
    REC.latch("turned", chest_angle(st) <= TRAP_FROM_DEG, "turned 45 degrees from face down")
    if REC.flag("turned"):
        arms = [k for k in D.limb_under_body(DUMMY, st) if k.endswith(D.ARM_LINKS)]
        REC.latch("trapped", bool(arms), f"{', '.join(arms)} under the body while it was turned")


def evaluate(st: tk.BodyState, rec: tk.Recorder = REC) -> tuple[float, str]:
    """(score, reason) from the recorder and the dummy's final state."""
    if rec.flag("lifted"):
        return 0.0, "the trunk was lifted more than 4 cm off the ground"
    if rec.flag("stood"):
        return 0.0, "the body was stood up (spine more than 30 degrees from level)"
    if not (rec.full("rest", REST_STEPS) and all(rec.windows["rest"])):
        return 0.0, "not at rest"
    if not (rec.full("released", REST_STEPS) and all(rec.windows["released"])):
        return 0.0, "a gripper or arm came within 5 cm of the dummy in the last second"
    a = chest_angle(st)
    under = D.limb_under_body(DUMMY, st)
    if a <= CHEST_UP_DEG and not under:
        if rec.flag("trapped"):
            return 0.25, "on its back, but an arm was trapped under it during the turn"
        return 1.0, "on its back, chest up, no limb under the body"
    if a <= CHEST_PART_DEG:
        return 0.25, (f"chest up within 45 degrees ({a:.0f}), limb under the body: {under}" if under
                      else f"chest up within 45 degrees but tilted ({a:.0f})")
    return 0.0, f"not on its back (chest {a:.0f} degrees from up)"


_LAST: dict = {}


def score(objects) -> float:
    """0 if at any time the lowest point of the trunk, pelvis and head was more than 4 cm above the ground, or the
    spine (the line from the hips to the head) was more than 30 degrees from level (the body was stood up). Otherwise
    the end state counts, and is 0 unless the dummy is at rest (base under 1 cm/s and 0.5 rad/s, every joint under
    0.5 rad/s) and released (no gripper point and no point along either arm within 5 cm of it), both for each of the
    last 20 control steps (1 s at 20 Hz). A limb is under the body when a point of an upper arm, forearm, thigh, shank
    or foot (sampled every 5 mm along it) has the trunk or pelvis straight above it, leaving out the first 3 cm from
    the shoulder or hip joint; the head is not judged. 1.0 if the chest faces up (within 12 degrees of vertical), no
    limb is under the body, and at no step from the first time the body had turned 45 degrees from face down on was
    an arm (upper arm or forearm) under it; 0.25 if the chest faces up within 45 degrees but a limb is under the body, or an arm was
    under it during the turn, or the trunk is tilted more than 12 degrees; otherwise 0. (A leg under the pelvis is
    not judged during the turn: a body on its side has its lower leg there.)"""
    s, why = evaluate(objects["dummy"])
    _LAST.update(score=s, why=why)
    return s


def success(objects) -> bool:
    return evaluate(objects["dummy"])[0] == 1.0


def report() -> dict:
    return {"last": dict(_LAST), "fouls": {k: REC.why[k] for k in REC.flags if REC.flags[k]},
            "max_lift_cm": round(100 * REC.extremes.get("max_lift", 0.0), 2),
            "max_spine_tilt_deg": round(REC.extremes.get("max_spine_tilt", 0.0), 1)}
