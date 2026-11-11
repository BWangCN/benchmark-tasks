"""On the station table: an earth track runs away from the robot. Its near half is a grey rock deck 3.8 cm high, its far
half a lower earth track at table level. A small yellow-and-white trail crawler (a 1:18 rover, motor disconnected so
its wheels roll freely) stands on the deck facing the drop, a few centimetres short of it, with a tow tab at the
middle of each bumper. The robot must bring the rover down onto the lower track without letting it drop and without
lifting it clear of the ground.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 --scorer task_score ...
         (the descent test counts in 20 Hz control steps)
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

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

INSTRUCTION = ("Get the rover down off the ledge onto the lower track without letting it drop or lifting it "
               "clear.")

TOP = tk.TOP
TEX = HERE / "textures"

# -- layout (world frame: x forward from the arm bases, y to the robot's left, metres) ---------------------------
DECK_H = 0.038                        # the step: taller than the tyres' radius (2.8 cm)
DECK_X = (0.10, 0.40)                 # the deck, from near the robot to its edge
LOW_X = (DECK_X[1], 0.74)             # the lower track beyond the edge
TRACK_Y = (-0.17, 0.17)
DECK_TOP = TOP + DECK_H
ROVER_X = 0.255                       # rover origin on the deck, facing the edge: front tyres 4.0 cm short of it
ROCK_GREY = (0.52, 0.50, 0.47)

# tow blocks at the middle of each bumper, the one part of the rover a gripper can close on
TAB_SIZE = (0.048, 0.016, 0.020)    # a tow bar stub: 4.6 cm out, 16 mm thick, 20 mm tall
TAB_X = rv.S.LENGTH / 2 + TAB_SIZE[0] / 2 - 0.002
TAB_Z = rv.S.BUMPER_HEIGHT
TAB_COLOR = (0.20, 0.20, 0.22)

_dx = DECK_X[1] - DECK_X[0]
_ly = TRACK_Y[1] - TRACK_Y[0]
_lx = LOW_X[1] - LOW_X[0]
DECK = tk.body("deck", pos=((DECK_X[0] + DECK_X[1]) / 2, 0.0, TOP), shapes=[
    tk.Shape.span((-_dx / 2, -_ly / 2, 0.0), (_dx / 2, _ly / 2, DECK_H), ROCK_GREY, friction=(0.8, 0.7)),
])                                    # grey rock all over: the step shows against the earth track below it
LOWER = tk.body("lower_track", pos=((LOW_X[0] + LOW_X[1]) / 2, 0.0, TOP), shapes=[
    tk.Shape.span((-_lx / 2, -_ly / 2, 0.0), (_lx / 2, _ly / 2, 0.0006), (0.46, 0.34, 0.23), friction=(0.8, 0.7)),
    tk.Shape.tile((0, 0, 0.0008), (_lx, _ly), str(TEX / "earth.png"), texture_m=0.12),
])
TABS = (tk.Shape.box((TAB_X, 0.0, TAB_Z), TAB_SIZE, TAB_COLOR, friction=(1.0, 0.9), tag="tow_f"),
        tk.Shape.box((-TAB_X, 0.0, TAB_Z), TAB_SIZE, TAB_COLOR, friction=(1.0, 0.9), tag="tow_r"))
ROVER = rv.rover_l3("rover", pos=(ROVER_X, 0.0, DECK_TOP + 0.0012), rot_xyzw=rv.upright(0.0), extra_shapes=TABS)
OBJECTS = [DECK, LOWER, ROVER]

# -- scoring -------------------------------------------------------------------------------------------------------
DESCENT_MIN_STEPS = 12   # control steps (0.6 s at 20 Hz): each tyre must take at least this long to come down the
#                          step, from its centre past the edge and 3 mm below its height on the deck until it rests on
#                          the lower track (within 1 cm of it): about 2.5 cm, under 4.2 cm/s on average. Rolled off the edge, a tyre swings and falls down
#                          in a few steps; lowered by a hand at 1-3 cm/s it takes 1.7 to 3 s
DROP_STEP = 0.010        # m: on the deck, a tyre's lowest point falling this far in one control step: let fall
DROP_SPAN, DROP_WINDOW = 0.015, 3   # ... or this far over 3 control steps
DESCENT_TOP = DECK_H + rv.TYRE_R - 0.003    # the axle centre's height above the table that starts the descent
REARM = 0.015            # m: an axle lifted this far off the lower track again, for 3 control steps, is no longer
#                          down (its drops count again); a bounce is shorter
HOP_ALL, HOP_STEPS = 0.015, 3   # m: all four tyres this far clear for 3 control steps (0.15 s): lifted clear
STAGGER_STEPS = 10       # the second axle comes down at least this many control steps (0.5 s) after the first
LIFT_CLEAR = 0.008       # m: a tyre is off the ground when its lowest point is this far above the surface under it
LIFT_STEPS = 10          # ... all four at once for this many control steps (0.5 s): the rover lifted clear
TIPPED_DEG = 30.0
UPRIGHT_DEG = 10.0
DOWN_TOL = 0.010         # m: a tyre is down on the lower track when its lowest point is within this of it, inside it
REST_SPEED, REST_SPIN, REST_STEPS = 0.01, 0.5, 20
RELEASE = 0.05
AXLES = {"front": ("wheel_fl", "wheel_fr"), "rear": ("wheel_rl", "wheel_rr")}
OFF_DECK = 0.010         # m: a tyre's lowest point this far below the deck's top while short of its edge: it went off
#                          the deck's side or near end (the only way down is over the edge)
RATE_DROP, RATE_WINDOW = 0.012, 3   # m: a tyre's centre, past the deck's edge, falling more than this over 3 control
#                          steps (0.15 s, 8 cm/s on average): dropped. Lowered by a hand at 1-3 cm/s it falls 0.2 to 0.5 cm

REC = tk.Recorder()


def tyre_lows(st: tk.BodyState) -> dict[str, np.ndarray]:
    """Each tyre's lowest point (world)."""
    return {n: pts[np.argmin(pts[:, 2])] for n, pts in rv.l3_tyre_points(ROVER, st).items()}


def support_height(p) -> float:
    """Height of the surface under a point: the deck top over the deck, the table elsewhere."""
    on_deck = DECK_X[0] <= p[0] <= DECK_X[1] and TRACK_Y[0] <= p[1] <= TRACK_Y[1]
    return DECK_TOP if on_deck else TOP


def lifted_clear(lows: dict) -> bool:
    return all(p[2] - support_height(p) > LIFT_CLEAR for p in lows.values())


def down(p) -> bool:
    """A tyre's lowest point resting on the lower track."""
    return (LOW_X[0] <= p[0] <= LOW_X[1] and TRACK_Y[0] <= p[1] <= TRACK_Y[1] and abs(p[2] - TOP) <= DOWN_TOL)


WHEEL_AXLE = {w: a for a, ws in AXLES.items() for w in ws}


def _drop(rec: tk.Recorder, w: str, why: str) -> None:
    axle = WHEEL_AXLE[w]
    why = f"the {axle} axle dropped ({w[-2:]} tyre {why})"
    rec.latch(f"drop_{w}", True, why)
    rec.latch(f"drop_{axle}", True, why)
    rec.latch("early_drop", rec.values.get("first_down") is None, why + ", before either axle was down")


def _track_tyre(rec: tk.Recorder, w: str, low, centre) -> None:
    """One tyre past the deck's edge: its own descent time (from its centre 3 mm below its height on the deck, past the
    edge, until it rests on the lower track) and its rate of fall; timed again when lifted off the track again."""
    cz, past = float(centre[2]) - TOP, float(centre[0]) > DECK_X[1]
    zs = rec.values.setdefault(f"cz_{w}", [])
    zs.append(cz)
    del zs[:-(RATE_WINDOW + 1)]
    if past and len(zs) == RATE_WINDOW + 1 and zs[0] - zs[-1] > RATE_DROP:
        _drop(rec, w, f"fell {100 * (zs[0] - zs[-1]):.1f} cm in {RATE_WINDOW} control steps")
    up = rec.values.get(f"up_{w}", 0) + 1 if low[2] - TOP > REARM else 0
    rec.values[f"up_{w}"] = up
    if rec.values.get(f"down_{w}") and up >= HOP_STEPS:      # lifted off the track again (not a bounce): timed again
        rec.values[f"down_{w}"] = False
        rec.values.pop(f"start_{w}", None)
        rec.values.pop(f"took_{w}", None)
    if rec.values.get(f"down_{w}"):
        return
    if cz > DESCENT_TOP or not past:
        rec.values.pop(f"start_{w}", None)                  # on the deck or back up at its height: starts afresh
    elif f"start_{w}" not in rec.values:
        rec.values[f"start_{w}"] = rec.steps
    if down(low):
        took = rec.steps - rec.values.get(f"start_{w}", rec.steps)
        rec.values[f"took_{w}"], rec.values[f"down_{w}"] = took, True
        if took < DESCENT_MIN_STEPS:
            _drop(rec, w, f"down the step in {took} control steps, {took / 20:.2f} s")


def track_step(rec: tk.Recorder, lows: dict, tilt: float, held: bool = True, centres: dict | None = None) -> None:
    """One control step of the whole-trial record: each axle's descent time, the order the axles came down, fouls.
    ``centres``: each wheel's centre (world)."""
    hist = rec.values.setdefault("hist", [])                 # each tyre's lowest point, the last DROP_WINDOW + 1 steps
    hist.append({w: float(p[2]) for w, p in lows.items()})
    del hist[:-(DROP_WINDOW + 1)]
    if len(hist) > 1:                                        # lifted on the deck and let fall back onto it
        on_deck = [w for w, p in lows.items() if support_height(p) == DECK_TOP and p[2] - DECK_TOP < 0.004]
        step = max([hist[-2][w] - hist[-1][w] for w in on_deck], default=0.0)
        span = max([hist[0][w] - hist[-1][w] for w in on_deck], default=0.0)
        if step > DROP_STEP or span > DROP_SPAN:
            rec.latch("early_drop", True, f"the rover was let fall onto the deck ({100 * max(step, span):.1f} cm)")
    if centres is not None:
        for w in WHEEL_AXLE:
            _track_tyre(rec, w, lows[w], centres[w])
    for axle, wheels in AXLES.items():
        if all(rec.values.get(f"down_{w}") for w in wheels) and not rec.values.get(f"down_{axle}"):
            rec.values[f"down_{axle}"] = True
            rec.values[f"took_{axle}"] = max(rec.values.get(f"took_{w}") or 0 for w in wheels)
            if "first_down" not in rec.values:
                rec.values["first_down"], rec.values["first_step"] = axle, rec.steps
            elif rec.values["first_down"] != axle and "second_step" not in rec.values:
                rec.values["second_step"] = rec.steps
                rec.latch("together", rec.steps - rec.values["first_step"] < STAGGER_STEPS,
                          "both axles came down together (the rover carried down)")
        elif not all(rec.values.get(f"down_{w}") for w in wheels):
            rec.values[f"down_{axle}"] = False
    off = [w for w, p in lows.items() if p[0] < DECK_X[1] and p[2] < DECK_TOP - OFF_DECK]
    rec.latch("off_deck", bool(off), f"the {off[0][-2:] if off else ''} tyre went off the deck's side or near end")
    run = rec.values.get("clear_run", 0) + 1 if lifted_clear(lows) else 0
    rec.values["clear_run"] = run
    high = all(p[2] - support_height(p) > HOP_ALL for p in lows.values())
    hop = rec.values.get("hop_run", 0) + 1 if high else 0          # carried for 3 steps, not a bounce of 1 or 2
    rec.values["hop_run"] = hop
    rec.latch("lifted", run >= LIFT_STEPS or hop >= HOP_STEPS,
              "all four tyres off the ground (the rover lifted clear)")
    rec.latch("tipped", tilt > TIPPED_DEG, "the rover tipped past 30 degrees")


@tk.on_reset
def _reset(environment=None) -> None:
    REC.reset()


@tk.on_step
def _record(environment) -> None:
    REC.steps += 1
    st = tk.body_state(environment, "rover")
    released = tk.released(ROVER, st, tk.grippers(environment), RELEASE)
    centres = {w: np.asarray(st.links[w][0]) for ws in AXLES.values() for w in ws}
    track_step(REC, tyre_lows(st), tk.tilt_deg(st.rotation()), held=not released, centres=centres)
    REC.window("rest", st.speed() < REST_SPEED and st.spin() < REST_SPIN, REST_STEPS)
    REC.values["released"] = released


def evaluate(st: tk.BodyState, rec: tk.Recorder = REC) -> tuple[float, str]:
    for foul in ("lifted", "tipped", "together", "early_drop", "off_deck"):
        if rec.flag(foul):
            return 0.0, rec.why.get(foul, foul)
    first = rec.values.get("first_down")
    dropped = [a for a in AXLES if rec.flag(f"drop_{a}")]
    # the axle that goes over the edge first must be lowered; one that dropped before any axle came down is that one
    if dropped and (first is None or first in dropped):
        a = first or dropped[0]
        return 0.0, rec.why.get(f"drop_{a}", f"the {a} axle dropped")
    if not (rec.full("rest", REST_STEPS) and all(rec.windows["rest"])):
        return 0.0, "not at rest"
    if not rec.values.get("released", False):
        return 0.0, "a gripper or an arm is within 5 cm of the rover"
    lows = tyre_lows(st)
    if not (tk.tilt_deg(st.rotation()) <= UPRIGHT_DEG and all(down(p) for p in lows.values())):
        return 0.0, "the rover is not down on the lower track"
    if dropped:
        return 0.5, f"down; the first axle lowered, but {rec.why.get(f'drop_{dropped[0]}', 'the second dropped')}"
    return 1.0, "down on the lower track, both axles lowered"


_LAST: dict = {}


def score(objects) -> float:
    """0 if at any time the rover is lifted clear (all four tyres more than 8 mm above the surfaces under them for
    0.5 s, or more than 1.5 cm above them for 3 control steps), is let fall onto the deck, a tyre goes off the deck's side or near end, tips past 30 degrees, comes down with both axles within 0.5 s of each other, has an axle drop before either axle rests on the
    lower track, or the first axle to come down drops (a tyre of it, past the deck's edge, falls more than 1.2 cm over 3 control steps, or takes under 0.6 s,
    12 control steps, to come down the step: from its centre past the edge and 3 mm below its height on the deck until
    it rests on the lower track; timed per tyre, afresh after a lift). Otherwise, at the end, at rest and released,
    with all four tyres on the lower track and upright: 1.0 if neither axle dropped, 0.5 if the second axle dropped."""
    s, why = evaluate(objects["rover"])
    _LAST.update(score=s, why=why)
    return s


def success(objects) -> bool:
    return evaluate(objects["rover"])[0] == 1.0


def report() -> dict:
    return {"last": dict(_LAST), "fouls": {k: REC.why[k] for k in REC.flags if REC.flags[k]},
            "first_down": REC.values.get("first_down"), "steps": REC.steps,
            "descent_steps": {a: REC.values.get(f"took_{a}") for a in AXLES},
            "rest_window": sum(REC.windows.get("rest", []))}
