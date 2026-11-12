"""On the station table: a packed-earth pad. A small grey humanoid robot (a passive 1:3 model, 44 cm tall, 1.24 kg,
14 limp joints with hard end stops) lies on its back on the pad, along the table's width, between two loose concrete rocks
that run beside its trunk and thighs, head to the robot's left, arms along its trunk, legs straight. An orange hatch is on its back. The
robot must leave it so that the hatch can be opened: the body cannot be rolled over between the rocks, and raised to
sitting it falls back; carried past its balance point, it folds forward onto its own thighs with its back up.

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

import dummy as D  # noqa: E402
import trailkit as tk  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("The humanoid robot on the pad lies on its back between two rocks. Leave it with its back hatch "
               "facing up so that the hatch can be opened.")

TOP = tk.TOP
TEX = HERE / "textures"

# -- layout (world frame: x forward from the arm bases, y to the robot's left, metres) ---------------------------
PAD_H = 0.010
GROUND = TOP + PAD_H
PAD_X = (0.03, 0.46)
PAD_Y = (-0.25, 0.25)
TURF_X = 0.41                            # turf strip from here to the pad's far edge, away from the work
DUMMY_X = 0.28                           # the point midway between the hip joints, lying along y
ROCK = (0.04, 0.26, 0.06)                # across, along y, tall
ROCK_Y = (-0.12, 0.14)                   # beside the thighs and the trunk
ROCK_GAP = 0.15                          # between the rocks' inner faces
ROCK_MASS = 2300 * ROCK[0] * ROCK[1] * ROCK[2]          # concrete: 1.44 kg
ROCK_C = (0.55, 0.54, 0.51)
EARTH_C = (0.42, 0.31, 0.21)

PAD = tk.body("pad", pos=((PAD_X[0] + PAD_X[1]) / 2, 0.0, TOP), shapes=[
    tk.Shape.span((-(PAD_X[1] - PAD_X[0]) / 2, PAD_Y[0], 0.0), ((PAD_X[1] - PAD_X[0]) / 2, PAD_Y[1], PAD_H), EARTH_C,
                  friction=(0.6, 0.5)),
    tk.Shape.tile((-(PAD_X[1] - TURF_X) / 2, 0.0, PAD_H + 0.0004), (TURF_X - PAD_X[0], PAD_Y[1] - PAD_Y[0]),
                  str(TEX / "earth.png"), texture_m=0.12),
    tk.Shape.tile(((TURF_X - PAD_X[0]) / 2, 0.0, PAD_H + 0.0006), (PAD_X[1] - TURF_X, PAD_Y[1] - PAD_Y[0]),
                  str(TEX / "turf.png"), texture_m=0.08),
])
ROCKS = [tk.body(f"rock_{i}", pos=(DUMMY_X + s * (ROCK_GAP / 2 + ROCK[0] / 2), (ROCK_Y[0] + ROCK_Y[1]) / 2,
                                   GROUND + ROCK[2] / 2 + 0.0005),
                 shapes=[tk.Shape.box((0, 0, 0), ROCK, ROCK_C, friction=(0.7, 0.6), tag="rock")], mass=ROCK_MASS)
         for i, s in enumerate((-1, 1))]

# on its back, along y, head toward +y (the robot's left): chest up, the dummy's left toward +x
START_ROT = D.on_back()
START_SPINE = np.array([0.0, 1.0, 0.0])
START_POSE = dict(D.ARMS_ALONG)
DUMMY = D.dummy("dummy", pos=(DUMMY_X, 0.0, GROUND + D.rest_height(START_ROT, START_POSE) + 0.002), rot_xyzw=START_ROT,
                pose=START_POSE)
OBJECTS = [PAD, *ROCKS, DUMMY]

# -- scoring -------------------------------------------------------------------------------------------------------
HATCH_UP_DEG = 70.0       # full score: the hatch's normal within this of straight up
RAISED_DEG = 60.0         # 0.25: the spine rose more than this above level at some time
PELVIS_MOVE = 0.04        # m: the pelvis (the point between the hip joints) moved less than this, seen from above
LIFT_FOUL = 0.03          # m: the pelvis's lowest point more than this above the pad, at any time
ROCK_FOUL = 0.01          # m: a rock moved more than this, at any time
ON_HATCH = 0.03           # m: a limb point within this outside the hatch's face, over the hatch: it lies on it
REST_MOVE, REST_TURN, REST_JOINT = 0.002, 1.0, 2.0      # m, degrees, degrees over the last REST_STEPS
REST_STEPS = 40           # 2 s at 20 Hz
RELEASE = 0.05

REC = tk.Recorder()


def hatch_angle(st: tk.BodyState) -> float:
    """Degrees between the direction the back hatch faces and straight up."""
    return D.angle_deg(D.hatch_normal(st), (0.0, 0.0, 1.0))


def raise_angle(st: tk.BodyState) -> float:
    """Degrees the trunk is raised: the elevation of the spine (the line from the hips to the head) above level, in 3-D:
    0 lying, 90 sitting upright; folded forward onto the thighs it reads about 50 again."""
    s = D.spine(st)
    return math.degrees(math.asin(float(np.clip(s[2] / np.linalg.norm(s), -1.0, 1.0))))


def pelvis_lift(st: tk.BodyState) -> float:
    pts = D.points(DUMMY, st, which={"base"}, tags=("pelvis",))
    return float(pts[:, 2].min()) - GROUND


def limbs_on_hatch(st: tk.BodyState) -> list[str]:
    """Arm and leg links with a point lying on the hatch: over its 4 x 5 cm face (with 5 mm around it) and within 3 cm
    outside it."""
    R, p = st.rotation(), np.asarray(st.pos, float)
    hw, hh = D.S.HATCH_SIZE
    out = []
    for ln in DUMMY.links:
        if not any(sh.tag in D.SEGMENT_TAGS for sh in ln.shapes):
            continue
        loc = (D._segment_points(DUMMY, st, ln.name) - p) @ R
        if not len(loc):
            continue
        outward = -loc[:, 0] - D.S.TRUNK_SIZE[0] / 2
        on = ((outward >= -0.002) & (outward <= ON_HATCH) & (np.abs(loc[:, 1]) <= hw / 2 + 0.005)
              & (np.abs(loc[:, 2] - D.S.HATCH_Z) <= hh / 2 + 0.005))
        if on.any():
            out.append(ln.name)
    return out


def between_rocks(st: tk.BodyState, rocks: list) -> bool:
    """The pelvis (the point between the hip joints) lies between the rocks' inner faces, seen from above."""
    inner = []
    for r in rocks:
        c = np.array([r.to_world((sx * ROCK[0] / 2, sy * ROCK[1] / 2, 0.0)) for sx in (-1, 1) for sy in (-1, 1)])
        inner.append(c[:, 0].max() if r.pos[0] < st.pos[0] else c[:, 0].min())
    return min(inner) < float(st.pos[0]) < max(inner)


def rock_corners(r: tk.BodyState) -> np.ndarray:
    return np.array([r.to_world((sx * ROCK[0] / 2, sy * ROCK[1] / 2, sz * ROCK[2] / 2))
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


@tk.on_reset
def _reset(environment=None) -> None:
    REC.reset()


@tk.on_step
def _record(environment) -> None:
    REC.steps += 1
    st = tk.body_state(environment, "dummy")
    rocks = [tk.body_state(environment, r.name) for r in ROCKS]
    REC.values.setdefault("pelvis0", np.asarray(st.pos[:2], float))
    REC.values.setdefault("rocks0", [rock_corners(r) for r in rocks])
    REC.extreme("max_raise", raise_angle(st))
    REC.extreme("max_pelvis_lift", pelvis_lift(st))
    REC.latch("pelvis_lifted", pelvis_lift(st) > LIFT_FOUL, f"the pelvis was lifted {100 * pelvis_lift(st):.1f} cm")
    moved = max(float(np.linalg.norm(rock_corners(r) - r0, axis=1).max()) for r, r0 in zip(rocks, REC.values["rocks0"]))
    REC.extreme("max_rock_move", moved)
    REC.latch("rock_moved", moved > ROCK_FOUL, f"a rock moved {100 * moved:.1f} cm")
    hist = REC.values.setdefault("poses", [])
    hist.append((np.asarray(st.pos, float), st.rotation(), np.array(list(st.joints.values()), float)))
    del hist[:-REST_STEPS]
    if len(hist) == REST_STEPS:
        (p0, r0, q0), (p1, r1, q1) = hist[0], hist[-1]
        turn = math.degrees(math.acos(float(np.clip((np.trace(r0.T @ r1) - 1) / 2, -1, 1))))
        REC.window("rest", float(np.linalg.norm(p1 - p0)) < REST_MOVE and turn < REST_TURN
                   and float(np.degrees(np.abs(q1 - q0)).max()) < REST_JOINT, REST_STEPS)
    REC.window("released", D.released(DUMMY, st, tk.grippers(environment), RELEASE), REST_STEPS)


def evaluate(st: tk.BodyState, rocks: list, rec: tk.Recorder = REC) -> tuple[float, str]:
    """(score, reason) from the recorder and the final state."""
    if rec.flag("pelvis_lifted"):
        return 0.0, "the pelvis was lifted more than 3 cm off the pad"
    if rec.flag("rock_moved"):
        return 0.0, "a rock was moved more than 1 cm"
    if not (rec.full("rest", REST_STEPS) and all(rec.windows["rest"])):
        return 0.0, "not at rest for the last 2 s"
    if not (rec.full("released", REST_STEPS) and all(rec.windows["released"])):
        return 0.0, "a gripper or an arm within 5 cm of the robot in the last 2 s"
    a = hatch_angle(st)
    p0 = rec.values.get("pelvis0", np.asarray(DUMMY.pos[:2], float))
    moved = float(np.linalg.norm(np.asarray(st.pos[:2], float) - p0))
    in_place = between_rocks(st, rocks) and moved < PELVIS_MOVE
    if not in_place:
        return 0.0, f"the pelvis left its place between the rocks (moved {100 * moved:.1f} cm)"
    on = limbs_on_hatch(st)
    if a <= HATCH_UP_DEG and not on:
        return 1.0, f"folded forward: hatch {a:.0f} degrees from up, nothing on it, pelvis moved {100 * moved:.1f} cm"
    if rec.extremes.get("max_raise", -90.0) > RAISED_DEG:
        why = f"limb on the hatch: {on}" if on else f"hatch {a:.0f} degrees from up"
        return 0.25, f"the trunk was raised past 60 degrees but the end state fails ({why})"
    return 0.0, f"hatch {a:.0f} degrees from up; the trunk never rose past 60 degrees"


_LAST: dict = {}


def _states(objects):
    return objects["dummy"], [objects[r.name] for r in ROCKS]


def score(objects) -> float:
    """0 if at any time the lowest point of the pelvis is more than 3 cm above the pad, or any corner of a rock has
    moved more than 1 cm. Otherwise the end state counts, and is 0 unless the robot is at rest (its pelvis moved under
    2 mm, it turned under 1 degree and no joint changed more than 2 degrees over the last 2 s) and released (none of
    six points on each arm's gripper and wrist, and none of the points sampled along each arm from its shoulder joint
    to its palm, within 5 cm of the robot), both for each of the last 40 control steps (2 s at 20 Hz), and unless the
    pelvis (the point between the hip joints) is still between the rocks' inner faces and has moved less than 4 cm,
    seen from above. Then: 1.0 if the back hatch faces up (its normal within 70 degrees of straight up) and no arm or
    leg segment lies on the hatch (a point of it over the hatch's face, or within 5 mm around it, and less than 3 cm
    outside it); 0.25 if not, but the spine (the line from the hips to the head) rose more than 60 degrees above
    level at some time; 0 otherwise."""
    s, why = evaluate(*_states(objects))
    _LAST.update(score=s, why=why)
    return s


def success(objects) -> bool:
    return evaluate(*_states(objects))[0] == 1.0


def report() -> dict:
    return {"last": dict(_LAST), "fouls": {k: REC.why[k] for k in REC.flags if REC.flags[k]},
            "max_raise_deg": round(REC.extremes.get("max_raise", 0.0), 1),
            "max_pelvis_lift_cm": round(100 * REC.extremes.get("max_pelvis_lift", 0.0), 2),
            "max_rock_move_cm": round(100 * REC.extremes.get("max_rock_move", 0.0), 2)}
