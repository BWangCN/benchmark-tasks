"""On the station table: a packed-earth pad with a turf strip along its far edge, and a dark charging pad near its
front. A grey four-legged robot (a passive 1:2 model of a mid-size robot dog, 1.9 kg, 12 limp joints with hard end
stops, a handle over each hip) has fallen on its right side and lies propped on its two right legs, belly down and
tilted. Its flat battery sits in a bay in its belly, held by a magnetic catch. The robot must take the battery out
and put it on the charging pad. The bay opens only with the belly up.

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

import quadruped as Q  # noqa: E402
import quadruped_spec as S  # noqa: E402
import trailkit as tk  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "The quadruped robot has fallen and its battery is flat. Take the battery out and put it on the charging pad."

TOP = tk.TOP
TEX = HERE / "textures"

# -- layout (world frame: x forward from the arm bases, y to the robot's left, metres) ---------------------------
PAD_H = 0.010
GROUND = TOP + PAD_H
PAD_X = (0.03, 0.48)
PAD_Y = (-0.30, 0.30)
TURF_X = 0.43
EARTH_C = (0.42, 0.31, 0.21)
ROBOT_X = 0.33                           # trunk centre, lying along y, head toward +y
CHARGER = ((0.08, 0.22), (-0.27, -0.13))  # charging pad's outline: x, y (14 x 14 cm)
CHARGER_H = 0.006
CHARGER_C = (0.12, 0.13, 0.15)

PAD = tk.body("pad", pos=((PAD_X[0] + PAD_X[1]) / 2, 0.0, TOP), shapes=[
    tk.Shape.span((-(PAD_X[1] - PAD_X[0]) / 2, PAD_Y[0], 0.0), ((PAD_X[1] - PAD_X[0]) / 2, PAD_Y[1], PAD_H), EARTH_C,
                  friction=(0.6, 0.5)),
    tk.Shape.tile((-(PAD_X[1] - TURF_X) / 2, 0.0, PAD_H + 0.0004), (TURF_X - PAD_X[0], PAD_Y[1] - PAD_Y[0]),
                  str(TEX / "earth.png"), texture_m=0.12),
    tk.Shape.tile(((TURF_X - PAD_X[0]) / 2, 0.0, PAD_H + 0.0006), (PAD_X[1] - TURF_X, PAD_Y[1] - PAD_Y[0]),
                  str(TEX / "turf.png"), texture_m=0.08),
    tk.Shape.span((CHARGER[0][0] - (PAD_X[0] + PAD_X[1]) / 2, CHARGER[1][0], PAD_H),
                  (CHARGER[0][1] - (PAD_X[0] + PAD_X[1]) / 2, CHARGER[1][1], PAD_H + CHARGER_H), CHARGER_C,
                  friction=(0.6, 0.5), tag="charger"),
])
# The robot is spawned standing (trailkit authors every link at joint value 0, and on its side the legs would start
# inside the ground) and is laid on its side by the reset hook below, before the first control step.
# The start: how this robot lies when it has fallen on its right side, recorded from the simulator (laid at 35
# degrees onto its right side with straight legs, it settles in 0.4 s into this pose and stays): propped on its two
# right legs, belly 140 degrees from up (its back up and tilted away from the arms), right legs splayed out.
START_ROT = (0.22922, 0.25732, 0.70286, 0.62228)
START_POSE = {
    "fl_hip": -0.3228,
    "fl_thigh": 0.2406,
    "fl_calf": -0.1023,
    "fr_hip": 0.9706,
    "fr_thigh": 0.4121,
    "fr_calf": -0.7211,
    "rl_hip": -0.2216,
    "rl_thigh": 0.5120,
    "rl_calf": -0.1490,
    "rr_hip": 1.0472,
    "rr_thigh": 0.3733,
    "rr_calf": -0.0077,
}
START_POS = (ROBOT_X, 0.0, GROUND + 0.0688 + 0.0015)
DOG = Q.quadruped("dog", pos=(ROBOT_X, 0.0, GROUND + Q.rest_height(Q.upright(), Q.stand_pose()) + 0.003),
                  rot_xyzw=Q.upright(), pose=Q.stand_pose())
BATTERY = Q.battery("battery", DOG)
OBJECTS = [PAD, DOG, BATTERY]

# -- scoring -------------------------------------------------------------------------------------------------------
LIFT_FOUL = 0.060         # m: the trunk's lowest point more than this above the ground: lifted, at any time (the legs
#                           hang 20 cm and can trail on the ground while the trunk is carried)
OUT_OF_BAY = 0.030        # m: the battery's centre this far from its seat: out of the bay
ON_CHARGER = 0.005        # m: the battery's lowest point within this of the charging pad's top
HELD_NEAR = 0.020         # m: a gripper point this close to the battery: it is being handled
LEFT_CATCH = 0.004        # m: the battery's centre this far from its seat: it has left the catch
REST_MOVE = 0.002         # m over the last REST_STEPS, robot trunk and battery
REST_STEPS = 20
RELEASE = 0.05

REC = tk.Recorder()


def robot_lowest(st: tk.BodyState) -> float:
    """Height (m) of the lowest point of the trunk (with its head and handles) above the ground (the pad's top)."""
    return Q.lowest(DOG, st, which={"base"}) - GROUND


def battery_points(st: tk.BodyState) -> np.ndarray:
    return tk.body_points(BATTERY, st.pos, st.rotation())


def on_charger(bat: tk.BodyState) -> bool:
    """The battery's whole outline (tab included) over the charging pad, its lowest point on the pad's top."""
    p = battery_points(bat)
    inside = bool(np.all((p[:, 0] >= CHARGER[0][0]) & (p[:, 0] <= CHARGER[0][1]) & (p[:, 1] >= CHARGER[1][0])
                         & (p[:, 1] <= CHARGER[1][1])))
    return inside and abs(float(p[:, 2].min()) - (GROUND + CHARGER_H)) <= ON_CHARGER


def out_of_bay(dog: tk.BodyState, bat: tk.BodyState) -> bool:
    return float(np.linalg.norm(dog.to_local(bat.pos) - Q.SEAT)) >= OUT_OF_BAY


def lay_down(environment) -> None:
    """Lay the robot on its right side (lying pose, at rest) and seat its battery: the start of every trial."""
    import torch

    base = environment.unwrapped
    tk.write_body_pose(environment, "dog", START_POS, START_ROT)
    asset = base.scene["dog"]
    names = list(asset.joint_names)
    q = torch.zeros((1, len(names)), dtype=torch.float32, device=base.device)
    for k, v in START_POSE.items():
        q[0, names.index(f"{k}_joint")] = float(v)
    asset.write_joint_position_to_sim_index(position=q)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
    tk.write_body_pose(environment, "battery", (ROBOT_X, PAD_Y[1] - 0.05, GROUND + 0.02), (0.0, 0.0, 0.0, 1.0))
    for _ in range(2):                    # let the robot's links take the written pose, then seat the battery against it
        base.sim.step(render=False)
        base.scene.update(base.physics_dt)
    Q.seat_battery(environment, "battery")
    R = tk.rot_from_xyzw(tk.body_state(environment, "dog").quat_xyzw)
    # the catch acts from the first physics step (its own hook first runs after the first control step)
    f = -S.CATCH_FORCE * (R @ Q.OUT_AXIS)
    dev = base.device
    zero = torch.zeros((1, 1, 3), device=dev)
    for name, force in (("battery", f), ("dog", -f)):
        a = base.scene[name]
        i = list(a.body_names).index("base") if name == "dog" else 0
        a.permanent_wrench_composer.reset()
        a.permanent_wrench_composer.add_forces_and_torques(
            torch.tensor([[force.tolist()]], dtype=torch.float32, device=dev), zero, body_ids=[i], is_global=True)


def _refresh_first_observation() -> None:
    """The embodiment takes its first observation before the reset hooks run, so it would show the robot at its spawn
    pose (standing); take it again once the robot has been laid down. Wraps the embodiment's reset once, from here."""
    module = sys.modules.get("yam_arena.inspect_embodiment")
    if module is None or getattr(module.YamArenaEmbodiment, "_laid_down_obs", False):
        return
    cls = module.YamArenaEmbodiment
    inner = cls.reset

    def reset(self, *a, **k):
        inner(self, *a, **k)
        for _ in range(3):
            self._env.unwrapped.sim.render()
        return self._observation()

    cls.reset = reset
    cls._laid_down_obs = True


_refresh_first_observation()


@tk.on_reset
def _reset(environment=None) -> None:
    REC.reset()
    if environment is not None:
        lay_down(environment)


@tk.on_step
def _record(environment) -> None:
    REC.steps += 1
    dog = tk.body_state(environment, "dog")
    bat = tk.body_state(environment, "battery")
    REC.extreme("max_robot_lowest", robot_lowest(dog))
    REC.latch("lifted", robot_lowest(dog) > LIFT_FOUL, f"the trunk was {100 * robot_lowest(dog):.1f} cm off the ground")
    left_catch = not Q.battery_in_bay(dog, bat, tol=LEFT_CATCH)
    grips = tk.grippers(environment)
    if left_catch and not REC.flag("drawn") and not REC.flag("fell_out"):
        held = tk.distance_to_body(BATTERY, bat, np.vstack([g["points"] for g in grips.values()])) < HELD_NEAR
        REC.latch("drawn", held, "the battery was drawn out of its catch by a gripper")
        REC.latch("fell_out", not held, "the battery left its catch with no gripper on it (it fell or was knocked out)")
    hist = REC.values.setdefault("poses", [])
    hist.append(np.array([dog.pos, bat.pos]))
    del hist[:-REST_STEPS]
    REC.window("rest", len(hist) == REST_STEPS and float(np.linalg.norm(hist[-1] - hist[0], axis=1).max()) < REST_MOVE,
               REST_STEPS)
    REC.window("released", Q.released(DOG, dog, grips, RELEASE) and tk.released(BATTERY, bat, grips, RELEASE),
               REST_STEPS)


def evaluate(dog: tk.BodyState, bat: tk.BodyState, rec: tk.Recorder = REC) -> tuple[float, str]:
    if rec.flag("lifted"):
        return 0.0, "the trunk was lifted more than 6 cm off the ground"
    if rec.flag("fell_out"):
        return 0.0, "the battery left its catch with no gripper on it"
    if not (rec.full("rest", REST_STEPS) and all(rec.windows["rest"])):
        return 0.0, "the robot or the battery not at rest for the last second"
    if not (rec.full("released", REST_STEPS) and all(rec.windows["released"])):
        return 0.0, "a gripper or an arm within 5 cm of the robot or the battery in the last second"
    if not out_of_bay(dog, bat):
        return 0.0, "the battery is still in its bay"
    if on_charger(bat):
        return 1.0, "the battery is out and lies on the charging pad"
    return 0.5, "the battery is out of the bay but not on the charging pad"


_LAST: dict = {}


def score(objects) -> float:
    """0 if at any time the lowest point of the robot's trunk (with its head and handles) is more than 6 cm above the
    ground (the pad's top): it was lifted; or if, the first time the battery leaves its catch (its centre more than 4 mm
    from its seat), no gripper point is within 2 cm of it: it fell or was knocked out. Otherwise the end state counts, and is 0 unless the robot's trunk and the
    battery are at rest (neither centre moved more than 2 mm over the last second) and released (no point of either
    gripper, and no point along either arm from its shoulder joint to its palm, within 5 cm of the robot or the
    battery), both for each of the last 20 control steps (1 s at 20 Hz), and unless the battery is out of its bay (its
    centre at least 3 cm from its seat). Then: 1.0 if the battery's whole outline, its grip tab included, lies over the
    charging pad and its lowest point is within 5 mm of the pad's top; 0.5 otherwise."""
    s, why = evaluate(objects["dog"], objects["battery"])
    _LAST.update(score=s, why=why)
    return s


def success(objects) -> bool:
    return evaluate(objects["dog"], objects["battery"])[0] == 1.0


def report() -> dict:
    return {"last": dict(_LAST), "fouls": {k: REC.why[k] for k in REC.flags if REC.flags[k]},
            "max_robot_lowest_cm": round(100 * REC.extremes.get("max_robot_lowest", 0.0), 2)}
