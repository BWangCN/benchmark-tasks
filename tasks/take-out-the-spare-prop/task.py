"""On the station table: a concrete slab rests on four props, one under each corner, each sticking out past the
slab's end. One prop is 2 mm short and carries nothing: the slab stands on the other three, its weight (a lead
plate cast into it) lying inside their triangle. Which prop is short changes from trial to trial, and the slab is
turned to match, so nothing in view shows it. The robot must take out the prop that carries nothing and put it on the pad, leaving the slab where it is. A prop that carries the slab
is held by the slab's weight and drags the slab when pulled; the spare one moves at a touch.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import rubble as rb  # noqa: E402
import trailkit as tk  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Take out the prop that carries nothing and put it on the pad. Leave the slab as it is."

TOP = tk.TOP
TEX = HERE / "textures"

# -- layout (world frame: x forward from the arm bases, y to the robot's left, metres) ---------------------------
SHEET_T = 0.0006
GROUND = TOP + SHEET_T
X0, Y0 = 0.28, 0.0                      # the slab's middle
SLAB = (0.14, 0.24, 0.02)               # x, y, thickness
PROP = (0.04, 0.10, 0.04)               # x, y, height: concrete, half under a corner of the slab, half past its end
PROP_MASS = 2400 * PROP[0] * PROP[1] * PROP[2]
SHORT_BY = 0.002                        # the spare prop is this much short
CORNERS = ((-1, -1), (1, -1), (1, 1), (-1, 1))     # (sx, sy) of corner k: prop k's place
PX, PY = 0.045, SLAB[1] / 2             # props' centres from the slab's middle
SLAB_MASS = 3.1                         # with a lead plate cast into it: its weight lies off its middle
COM_OFF = (0.012, 0.022)                # the centre of mass from the slab's middle, away from the spare corner (it
#                                         puts the weight well inside the triangle of the three carrying props); the
#                                         slab looks the same either way up and either way round, so nothing shows it
PAD = ((0.13, 0.21), (-0.07, 0.07))     # x, y: where the spare prop goes, in front of the slab
PAD_T = 0.002
PAD_TOP = GROUND + PAD_T
EARTH = (0.42, 0.31, 0.21)
PAD_GREY = (0.62, 0.62, 0.60)
SPARE_SPEC = 3                          # the body that is short: prop_3
SPAWN_CORNER = 2                        # where the scene file puts it; every reset puts it at a random corner


def prop_pos(k_body: int, corner: int) -> tuple:
    sx, sy = CORNERS[corner]
    h = PROP[2] - (SHORT_BY if k_body == SPARE_SPEC else 0.0)
    return (X0 + sx * PX, Y0 + sy * PY, GROUND + h / 2 + 0.0002)


def slab_quat(spare_corner: int) -> tuple:
    """The slab's turn that puts its centre of mass away from ``spare_corner``: the scene builds it for SPAWN_CORNER,
    and a half turn about x, y or z moves the offset to each of the other diagonals."""
    sx, sy = CORNERS[spare_corner]
    s0x, s0y = CORNERS[SPAWN_CORNER]
    if (sx, sy) == (s0x, s0y):
        return (0.0, 0.0, 0.0, 1.0)
    if sy == s0y:
        return (0.0, 1.0, 0.0, 0.0)             # half turn about y: x flips
    if sx == s0x:
        return (1.0, 0.0, 0.0, 0.0)             # half turn about x: y flips
    return (0.0, 0.0, 1.0, 0.0)                 # half turn about z: both flip


def layout(spare_corner: int) -> dict:
    """Corner of each prop body: the spare body at ``spare_corner``, the others in order at the rest."""
    rest = [c for c in range(4) if c != spare_corner]
    out, i = {}, 0
    for k in range(4):
        if k == SPARE_SPEC:
            out[k] = spare_corner
        else:
            out[k] = rest[i]
            i += 1
    return out


SHEET = tk.body("sheet", pos=(0.27, 0.0, TOP), shapes=[
    tk.Shape.span((-0.17, -0.45, 0.0), (0.17, 0.45, SHEET_T), EARTH, friction=rb.CONCRETE_FR),
    tk.Shape.tile((0, 0, SHEET_T + 0.0002), (0.34, 0.90), str(TEX / "earth.png"), texture_m=0.12),
    tk.Shape.span((PAD[0][0] - 0.27, PAD[1][0], SHEET_T), (PAD[0][1] - 0.27, PAD[1][1], SHEET_T + PAD_T), PAD_GREY,
                  friction=rb.CONCRETE_FR, tag="pad")])
_L0 = layout(SPAWN_CORNER)
PROPS = [rb.block(f"prop_{k}", prop_pos(k, _L0[k]), (PROP[0], PROP[1], PROP[2] - (SHORT_BY if k == SPARE_SPEC else 0)),
                  mass=PROP_MASS) for k in range(4)]
_S0X, _S0Y = CORNERS[SPAWN_CORNER]
SLAB_BODY = tk.body("slab", pos=(X0, Y0, GROUND + PROP[2] + SLAB[2] / 2 + 0.0006),
                    shapes=[tk.Shape.box((0, 0, 0), SLAB, rb.CONCRETE, friction=rb.CONCRETE_FR)], mass=SLAB_MASS,
                    com=(-_S0X * COM_OFF[0], -_S0Y * COM_OFF[1], 0.0))
OBJECTS = [SHEET, *PROPS, SLAB_BODY]

# -- scoring -------------------------------------------------------------------------------------------------------
CORNER_TOL = 0.001        # m: a slab corner this far from its height on the props (up or down): the slab was moved
SLAB_SHIFT = 0.003        # m: the slab's middle this far from its place, along the ground: moved
ON_PAD = 0.005            # m: the prop's bottom within this of the pad's top: on it
REST_MOVE = 0.002         # m: the spare prop's and the slab's middles moved under this over the last REST_STEPS
REST_STEPS = 40
RELEASE = 0.05
SLAB_Z = GROUND + PROP[2]                 # the slab's underside, on the three carrying props

REC = tk.Recorder()
STATE = {"spare_corner": SPAWN_CORNER}


def slab_corners(st: tk.BodyState) -> np.ndarray:
    """The slab's four bottom corners (world): at each corner, the lower of its two faces' points (the slab may lie
    either way up)."""
    hx, hy, hz = SLAB[0] / 2, SLAB[1] / 2, SLAB[2] / 2
    out = []
    for sx, sy in CORNERS:
        a, b = np.asarray(st.to_world((sx * hx, sy * hy, -hz))), np.asarray(st.to_world((sx * hx, sy * hy, hz)))
        out.append(a if a[2] <= b[2] else b)
    return np.array(out)


def slab_moved(st: tk.BodyState) -> tuple[bool, str]:
    dz = float(np.max(np.abs(slab_corners(st)[:, 2] - SLAB_Z)))
    shift = float(np.hypot(st.pos[0] - X0, st.pos[1] - Y0))
    if dz > CORNER_TOL:
        return True, f"a slab corner moved {1000 * dz:.1f} mm up or down"
    if shift > SLAB_SHIFT:
        return True, f"the slab moved {1000 * shift:.1f} mm"
    return False, ""


def prop_corners(st: tk.BodyState) -> np.ndarray:
    """The spare prop's eight corners (world), at its own height."""
    hx, hy, hz = PROP[0] / 2, PROP[1] / 2, (PROP[2] - SHORT_BY) / 2
    return np.array([st.to_world((a * hx, b * hy, c * hz)) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])


def on_pad(st: tk.BodyState) -> bool:
    """Every corner of the prop over the pad, its lowest corner within 5 mm of the pad's top (lying on it)."""
    c = prop_corners(st)
    inside = np.all((c[:, 0] >= PAD[0][0]) & (c[:, 0] <= PAD[0][1]) & (c[:, 1] >= PAD[1][0]) & (c[:, 1] <= PAD[1][1]))
    return bool(inside and abs(float(c[:, 2].min()) - PAD_TOP) <= ON_PAD)


def clear_of_slab(st: tk.BodyState) -> bool:
    """No corner of the prop within the slab's outline seen from above."""
    c = prop_corners(st)
    hx, hy = SLAB[0] / 2, SLAB[1] / 2
    under = (np.abs(c[:, 0] - X0) < hx) & (np.abs(c[:, 1] - Y0) < hy)
    # a prop longer than the slab is wide could straddle it with its corners outside: test its middle too
    mid = np.asarray(st.pos)
    return bool(not np.any(under) and not (abs(mid[0] - X0) < hx and abs(mid[1] - Y0) < hy))


def track_step(rec: tk.Recorder, slab: tk.BodyState) -> None:
    moved, why = slab_moved(slab)
    rec.extreme("corner_dev", float(np.max(np.abs(slab_corners(slab)[:, 2] - SLAB_Z))))
    rec.latch("slab_moved", moved, why)


def evaluate(spare: tk.BodyState, rec: tk.Recorder = REC) -> tuple[float, str]:
    if rec.flag("slab_moved"):
        return 0.0, rec.why["slab_moved"]
    if not (rec.full("rest", REST_STEPS) and all(rec.windows["rest"])):
        return 0.0, "the spare prop or the slab not at rest for 2 s"
    if not (rec.full("released", REST_STEPS) and all(rec.windows["released"])):
        return 0.0, "a gripper or an arm within 5 cm of the spare prop or the slab in the last 2 s"
    if on_pad(spare):
        return 1.0, "the spare prop on the pad, the slab where it was"
    if clear_of_slab(spare):
        return 0.5, "the spare prop out from under the slab, not on the pad"
    return 0.0, "the spare prop still under the slab"


def spare_name() -> str:
    return f"prop_{SPARE_SPEC}"


_RNG = np.random.default_rng(int(os.environ["SPARE_SEED"]) if os.environ.get("SPARE_SEED") else None)


def pick_corner() -> int:
    """The short prop's corner for the next trial: SPARE_CORNER fixes it; else drawn from a generator seeded by
    SPARE_SEED (unseeded if that is unset)."""
    forced = os.environ.get("SPARE_CORNER")
    if forced is not None and forced != "":
        return int(forced) % 4
    return int(_RNG.integers(4))


@tk.on_reset
def _reset(environment=None) -> None:
    REC.reset()
    c = pick_corner()
    STATE["spare_corner"] = c
    if environment is None:
        return
    lay = layout(c)
    for k in range(4):
        tk.write_body_pose(environment, f"prop_{k}", prop_pos(k, lay[k]), (0.0, 0.0, 0.0, 1.0))
    tk.write_body_pose(environment, "slab", (X0, Y0, GROUND + PROP[2] + SLAB[2] / 2 + 0.0003), slab_quat(c))
    base = environment.unwrapped
    for _ in range(int(0.5 / base.physics_dt)):          # settle the slab on the props before the first record
        base.sim.step(render=False)
        base.scene.update(base.physics_dt)


@tk.on_step
def _record(environment) -> None:
    REC.steps += 1
    slab = tk.body_state(environment, "slab")
    spare = tk.body_state(environment, spare_name())
    track_step(REC, slab)
    hist = REC.values.setdefault("centres", [])
    hist.append(np.array([spare.pos, slab.pos]))
    del hist[:-REST_STEPS]
    REC.window("rest", len(hist) == REST_STEPS and float(np.max(np.linalg.norm(hist[-1] - hist[0], axis=1))) < REST_MOVE,
               REST_STEPS)
    grips = tk.grippers(environment)
    REC.window("released", tk.released(PROPS[SPARE_SPEC], spare, grips, RELEASE)
               and tk.released(SLAB_BODY, slab, grips, RELEASE), REST_STEPS)


_LAST: dict = {}


def score(objects) -> float:
    """0 if at any time a corner of the slab's underside moves more than 1 mm up or down from its height on the
    props, or the slab's middle moves more than 3 mm along the ground. Otherwise, at the end, after 2 s with the
    short prop and the slab at rest (their middles moved under 2 mm over the last 2 s) and released (no point of
    either gripper, and no point along either arm from its shoulder joint to its palm, within 5 cm of either): 1.0
    the short prop on the pad (every corner over the pad, its lowest corner within 5 mm of the pad's top); 0.5 the
    short prop out from under the slab (no corner of it, and not its middle, inside the slab's outline seen from
    above); 0 otherwise. A full score ends the trial once it has held for 1 s."""
    s, why = evaluate(objects[spare_name()])
    _LAST.update(score=s, why=why)
    return s


def success(objects) -> bool:
    return evaluate(objects[spare_name()])[0] == 1.0


def report() -> dict:
    return {"last": dict(_LAST), "steps": REC.steps, "spare_corner": STATE["spare_corner"],
            "corner_dev_mm": round(1000 * REC.extremes.get("corner_dev", 0.0), 2), "fouls": dict(REC.why)}
