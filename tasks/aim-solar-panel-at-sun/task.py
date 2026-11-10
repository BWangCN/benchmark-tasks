"""On the station table: a small yellow-and-white trail crawler (a 1:18 rover, 26 x 12 x 12 cm, 0.5 kg) is parked on
a turf pad in front of the robot, nose away from it. On its rear bed stands a solar panel (10 x 7 cm, dark blue, with a
white sighting disc printed with two rings, and a dark peg standing square at its centre) on a two-axis friction mount: it turns about the
vertical and tilts about a horizontal hinge through its middle; an orange aiming handle sticks out of its near edge. Ahead and to the left a small warm lamp
on a 42 cm post stands in for the sun. The robot must point the panel straight at the lamp.

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
import rover_spec as RS  # noqa: E402
import trailkit as tk  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Point the rover's solar panel straight at the sun lamp."

TOP = tk.TOP
TEX = HERE / "textures"

# -- layout (world frame: x forward from the arm bases, y to the robot's left, metres) ---------------------------
ROVER_XY = (0.36, 0.02)               # rover origin (midway between its axles)
ROVER_HEADING = math.radians(35.0)    # nose away from the robot, turned 35 degrees to the left
LAMP = np.array([0.446, 0.187, TOP + 0.425])   # centre of the lamp's bulb: 39 cm from the panel, 43 degrees up, and
#                                               inside the top camera's view
LAMP_R = 0.012                        # a small bulb: 3.6 degrees across from the panel, so the shadow stays sharp
LAMP_INTENSITY = 7.0e4                # the brightest light at the panel: the peg casts one clear shadow

# -- the mount, in the rover frame ----------------------------------------------------------------------------------
MOUNT = (-0.100, 0.0, 0.125)          # top of the mast on the rear bed (the panel clears the cab at any angle)
HINGE_H = 0.035                       # tilt hinge above the turntable's joint
PANEL = (0.070, 0.100, 0.004)         # x (tilts up), y (along the hinge), thickness
SIGHT_R = 0.030                       # white sighting disc
PEG_R, PEG_L = 0.003, 0.060           # dark peg standing square on the panel: its shadow reads in the top camera
RING_R = PEG_R + PEG_L * math.tan(math.radians(8.0))    # 11.4 mm: the shadow's far edge at the 8-degree limit
RING2_R = PEG_R + PEG_L * math.tan(math.radians(20.0))  # 24.8 mm: ... at the 20-degree limit
HANDLE_L, HANDLE_W, HANDLE_T = 0.045, 0.020, 0.006   # aiming handle: a flat bar out of the middle of the near edge
TURN0, TILT0 = math.radians(-15.0), 0.0   # start: flat, turned 15 degrees to the right
TURN_LIM = (math.radians(-170), math.radians(170))
TILT_LIM = (math.radians(-5), math.radians(50))   # a hard stop: past about 52 the panel's low edge meets the
#                                                    turntable (the aim needs 46.7)

GREY = (0.30, 0.30, 0.32)
CELL_BLUE = (0.08, 0.12, 0.35)
WHITE = (0.95, 0.95, 0.93)

PEDESTAL = tk.Shape.span((MOUNT[0] - 0.010, -0.010, 0.080), (MOUNT[0] + 0.010, 0.010, MOUNT[2]), GREY, tag="mast")
TURNTABLE = tk.Link(
    "turntable", "base", "revolute", "Z", MOUNT, mass=0.020, limits=TURN_LIM, damping=0.3, friction=0.0,
    q0=TURN0, armature=0.0005,
    shapes=(tk.Shape.cylinder((0, 0, 0.004), 0.018, 0.008, color=GREY, tag="turntable"),
            tk.Shape.span((-0.006, -0.059, 0.0), (0.006, -0.053, HINGE_H + 0.006), GREY, tag="yoke"),
            tk.Shape.span((-0.006, 0.053, 0.0), (0.006, 0.059, HINGE_H + 0.006), GREY, tag="yoke"),
            tk.Shape.span((-0.006, -0.059, 0.0), (0.006, 0.059, 0.004), GREY, tag="yoke")))
PANEL_LINK = tk.Link(
    "panel", "turntable", "revolute", "Y", (0.0, 0.0, HINGE_H), mass=0.030, limits=TILT_LIM, damping=0.3,
    friction=0.0, q0=TILT0, armature=0.0002, com=(0.0, 0.0, 0.0),   # balanced on its hinge, handle included
    shapes=(tk.Shape.box((0, 0, 0), PANEL, CELL_BLUE, tag="panel"),
            tk.Shape.cylinder((0, 0, PANEL[2] / 2 + 0.0004), SIGHT_R, 0.0008, color=WHITE, tag="sight"),
            # rings printed round the peg's foot, inner edges at the shadow's reach for 20 and 8 degrees (0.8 mm lines)
            tk.Shape.cylinder((0, 0, PANEL[2] / 2 + 0.0010), RING2_R + 0.0008, 0.0002, color=(0.15, 0.15, 0.16), tag="ring"),
            tk.Shape.cylinder((0, 0, PANEL[2] / 2 + 0.0012), RING2_R, 0.0002, color=WHITE, tag="ring"),
            tk.Shape.cylinder((0, 0, PANEL[2] / 2 + 0.0014), RING_R + 0.0008, 0.0002, color=(0.15, 0.15, 0.16), tag="ring"),
            tk.Shape.cylinder((0, 0, PANEL[2] / 2 + 0.0016), RING_R, 0.0002, color=WHITE, tag="ring"),
            tk.Shape.cylinder((0, 0, PANEL[2] / 2 + PEG_L / 2), PEG_R, PEG_L, color=(0.15, 0.15, 0.16), tag="peg"),
            tk.Shape.box((-PANEL[0] / 2 - HANDLE_L / 2, 0, 0), (HANDLE_L, HANDLE_W, HANDLE_T),
                         color=(0.85, 0.35, 0.05), tag="handle"),
            tk.Shape.span((-0.002, -0.0525, -0.003), (0.002, -0.0505, 0.003), GREY, tag="pin"),
            tk.Shape.span((-0.002, 0.0505, -0.003), (0.002, 0.0525, 0.003), GREY, tag="pin")))

rover_shapes = rv.l1_shapes() + [PEDESTAL]
ROVER = tk.body("rover", pos=(ROVER_XY[0], ROVER_XY[1], TOP + 0.0005), shapes=rover_shapes, mass=RS.MASS,
                rot_xyzw=rv.upright(ROVER_HEADING), com=(0.0, 0.0, RS.COM_HEIGHT), color=rv.BODY_COLOR,
                friction=(RS.BODY_FRICTION, RS.BODY_FRICTION * 0.9), links=(TURNTABLE, PANEL_LINK))

LAMP_POST = tk.body("sun_lamp", pos=(LAMP[0], LAMP[1], TOP), shapes=[
    tk.Shape.cylinder((0, 0, 0.004), 0.030, 0.008, color=(0.15, 0.15, 0.16)),
    tk.Shape.cylinder((0, 0, (LAMP[2] - TOP) / 2), 0.006, LAMP[2] - TOP - 0.02, color=(0.15, 0.15, 0.16)),
    tk.Shape.sphere((0, 0, LAMP[2] - TOP), LAMP_R, color=(1.0, 0.92, 0.70)),
], lights=({"centre": (0.0, 0.0, float(LAMP[2] - TOP)), "radius": LAMP_R * 0.8, "intensity": LAMP_INTENSITY,
            "color": (1.0, 0.86, 0.62)},))

CHOCK = (0.022, 0.026, 0.016)          # wheel chock: x (along the rover), y, height; taped to the pad


def _chocks() -> list:
    """Four chocks, one in front of each front tyre and one behind each rear tyre, in the pad's frame (the pad is
    square to the world; the rover is turned by ROVER_HEADING on it)."""
    c, s_ = math.cos(ROVER_HEADING), math.sin(ROVER_HEADING)
    q = tk.quat_xyzw_from_axis_angle((0, 0, 1), ROVER_HEADING)
    out = []
    for sx in (1, -1):
        for sy in (1, -1):
            lx = sx * (rv.AXLE_X + rv.TYRE_R + 0.002 + CHOCK[0] / 2)
            ly = sy * rv.WHEEL_Y
            out.append(tk.Shape.box((c * lx - s_ * ly, s_ * lx + c * ly, CHOCK[2] / 2), CHOCK, (0.85, 0.75, 0.10),
                                    rot_xyzw=q, friction=(0.9, 0.8), tag="chock"))
    return out


PAD = tk.body("parking_pad", pos=(ROVER_XY[0], ROVER_XY[1], TOP), shapes=[
    tk.Shape.span((-0.17, -0.11, 0.0), (0.17, 0.11, 0.0005), (0.25, 0.45, 0.20)),
    tk.Shape.tile((0, 0, 0.0007), (0.34, 0.22), str(TEX / "turf.png"), texture_m=0.08),
    *_chocks(),
])
OBJECTS = [PAD, LAMP_POST, ROVER]

# -- scoring -------------------------------------------------------------------------------------------------------
FULL_DEG, HALF_DEG = 8.0, 20.0        # panel normal to the line from the panel's centre to the lamp
MOVED, TURNED, TIPPED = 0.02, 10.0, 10.0   # rover fouls (m in 3D, degrees of heading, degrees of tilt), any time
REST_STEPS = 20                       # at rest: the last 20 control steps (1 s at 20 Hz)
REST_SPEED, REST_SPIN, PANEL_STILL_DEG = 0.01, 0.5, 0.05   # panel: under 0.05 deg per step, 1 deg/s
RELEASE = 0.05

REC = tk.Recorder()
START = {"pos": np.array([ROVER_XY[0], ROVER_XY[1], TOP]), "heading": ROVER_HEADING}


def panel_frame(st: tk.BodyState) -> tuple[np.ndarray, np.ndarray]:
    """World (centre, rotation) of the panel."""
    return st.links["panel"]


def aim_error_deg(st: tk.BodyState) -> float:
    """Angle (deg) between the panel's normal and the line from its centre to the lamp."""
    c, R = panel_frame(st)
    to_lamp = LAMP - c
    to_lamp /= np.linalg.norm(to_lamp)
    return math.degrees(math.acos(float(np.clip(R[:, 2] @ to_lamp, -1.0, 1.0))))


def rover_moved(st: tk.BodyState) -> tuple[float, float]:
    """(distance moved in m, in 3D, of the point midway between the axles; turn in degrees) since the start."""
    d = float(np.linalg.norm(st.pos - START["pos"]))
    turn = abs(math.degrees(math.remainder(tk.heading(st.rotation()) - START["heading"], 2 * math.pi)))
    return d, turn


@tk.on_reset
def _reset(environment=None) -> None:
    REC.reset()
    if environment is not None:
        st = tk.body_state(environment, "rover")
        START.update(pos=st.pos.copy(), heading=tk.heading(st.rotation()))


@tk.on_step
def _record(environment) -> None:
    st = tk.body_state(environment, "rover")
    REC.steps += 1
    d, turn = rover_moved(st)
    REC.latch("moved", d > MOVED, f"rover moved {100 * d:.1f} cm")
    REC.latch("turned", turn > TURNED, f"rover turned {turn:.1f} deg")
    REC.latch("tipped", tk.tilt_deg(st.rotation()) > TIPPED, "rover tilted past 10 degrees")
    normal = panel_frame(st)[1][:, 2]
    prev = REC.values.get("normal")
    swing = 0.0 if prev is None else math.degrees(math.acos(float(np.clip(prev @ normal, -1, 1))))
    REC.values["normal"] = normal
    REC.window("rest", st.speed() < REST_SPEED and st.spin() < REST_SPIN and swing < PANEL_STILL_DEG, REST_STEPS)
    grips = tk.grippers(environment)
    REC.values["released"] = all(tk.distance_to_body(ROVER, st, g["points"]) >= RELEASE
                                 and _panel_distance(st, g["points"]) >= RELEASE for g in grips.values())


def _panel_distance(st: tk.BodyState, points) -> float:
    """Distance from points to the mount's moving parts (turntable, yoke and panel), as boxes."""
    best = np.inf
    for ln in (TURNTABLE, PANEL_LINK):
        p, R = st.links[ln.name]
        for s in ln.shapes:
            c = p + R @ np.asarray(s.centre)
            half = s.half_extent()
            best = min(best, float(tk.point_box_distance(points, c, R, half).min()))
    return best


def evaluate(st: tk.BodyState, rec: tk.Recorder = REC) -> tuple[float, str]:
    if rec.flag("moved") or rec.flag("turned") or rec.flag("tipped"):
        return 0.0, "the rover was moved, turned or tipped"
    if not (rec.full("rest", REST_STEPS) and all(rec.windows["rest"])):
        return 0.0, "not at rest"
    if not rec.values.get("released", False):
        return 0.0, "a gripper is within 5 cm of the rover or the panel"
    err = aim_error_deg(st)
    if err <= FULL_DEG:
        return 1.0, f"aimed within {err:.1f} deg"
    if err <= HALF_DEG:
        return 0.5, f"aimed within {err:.1f} deg"
    return 0.0, f"off by {err:.1f} deg"


_LAST: dict = {}


def score(objects) -> float:
    """0 if at any time the rover is moved more than 2 cm, turned more than 10 degrees or tilted more than 10 degrees.
    Otherwise, at the end, with
    the rover and panel at rest and released: 1.0 if the panel's normal is within 8 degrees of the line from the
    panel's centre to the lamp; 0.5 within 20 degrees."""
    s, why = evaluate(objects["rover"])
    _LAST.update(score=s, why=why)
    return s


def success(objects) -> bool:
    return evaluate(objects["rover"])[0] == 1.0


def report() -> dict:
    return {"last": dict(_LAST), "fouls": {k: REC.why[k] for k in REC.flags if REC.flags[k]}}
