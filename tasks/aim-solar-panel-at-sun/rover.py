"""The rover as a trailkit body, from ``rover_spec``.

Level L1 (``rover_l1``): one rigid body. Chassis rails, the body shell, the cab and roof, two bumper bars and four
tyre cylinders (axles along the rover's y). The wheels do not turn: the real crawler with its drivetrain engaged
skids on its tyres when pushed.

Rover frame: x forward, y left, z up; origin on the ground midway between the axles and the wheel tracks.
Shape tags: ``tyre_fl``, ``tyre_fr``, ``tyre_rl``, ``tyre_rr``, ``shell``, ``cab``, ``bumper_f``, ``bumper_r``,
``chassis``.

License: MIT (see LICENSE next to this file).
"""

from __future__ import annotations

import math

import numpy as np

import rover_spec as S
import trailkit as tk

BODY_COLOR = (0.95, 0.72, 0.05)     # signal yellow shell: stands out on brown earth and green turf
CAB_COLOR = (0.92, 0.92, 0.90)      # white roof
TYRE_COLOR = (0.08, 0.08, 0.08)
CHASSIS_COLOR = (0.20, 0.20, 0.22)
BUMPER_COLOR = (0.12, 0.12, 0.13)

TYRE_R = S.TYRE_DIAMETER / 2
AXLE_X = S.WHEELBASE / 2
WHEEL_Y = S.TRACK / 2
TYRE_TAGS = ("tyre_fl", "tyre_fr", "tyre_rl", "tyre_rr")
TYRE_CENTRES = {
    "tyre_fl": (AXLE_X, WHEEL_Y, TYRE_R), "tyre_fr": (AXLE_X, -WHEEL_Y, TYRE_R),
    "tyre_rl": (-AXLE_X, WHEEL_Y, TYRE_R), "tyre_rr": (-AXLE_X, -WHEEL_Y, TYRE_R),
}


def l1_shapes(body_color=BODY_COLOR) -> list[tk.Shape]:
    tyre_fr = (S.TYRE_FRICTION, S.TYRE_FRICTION * 0.9)
    body_fr = (S.BODY_FRICTION, S.BODY_FRICTION * 0.9)
    half_l, half_w = S.BODY_LENGTH / 2, S.BODY_WIDTH / 2
    bx, by, bz = S.BUMPER_SIZE
    shapes = [
        # chassis rails and skid plate, between the tyres
        tk.Shape.span((-0.105, -0.028, S.GROUND_CLEARANCE), (0.105, 0.028, S.BODY_BOTTOM + 0.004), CHASSIS_COLOR,
                      body_fr, "chassis"),
        # body shell over the wheels, from its lower edge to the belt line
        tk.Shape.span((-half_l, -half_w, S.BODY_BOTTOM), (half_l, half_w, 0.085), body_color, body_fr, "shell"),
        # cab and roof, set back from the bonnet
        tk.Shape.span((-0.070, -0.046, 0.085), (S.ROOF_LENGTH - 0.070, 0.046, S.HEIGHT), CAB_COLOR, body_fr, "cab"),
        # bumper bars at the ends
        tk.Shape.box((S.LENGTH / 2 - bx / 2, 0.0, S.BUMPER_HEIGHT), S.BUMPER_SIZE, BUMPER_COLOR, friction=body_fr,
                     tag="bumper_f"),
        tk.Shape.box((-S.LENGTH / 2 + bx / 2, 0.0, S.BUMPER_HEIGHT), S.BUMPER_SIZE, BUMPER_COLOR, friction=body_fr,
                     tag="bumper_r"),
    ]
    for tag, c in TYRE_CENTRES.items():
        shapes.append(tk.Shape.cylinder(c, TYRE_R, S.TYRE_WIDTH, axis="Y", color=TYRE_COLOR, friction=tyre_fr, tag=tag))
    return shapes


def rover_l1(name: str, pos, rot_xyzw=(0.0, 0.0, 0.0, 1.0), body_color=BODY_COLOR) -> tk.Body:
    return tk.body(name, pos=pos, shapes=l1_shapes(body_color), mass=S.MASS, rot_xyzw=rot_xyzw,
                   com=(0.0, 0.0, S.COM_HEIGHT), color=body_color, friction=(S.BODY_FRICTION, S.BODY_FRICTION * 0.9),
                   angular_damping=0.05)


# -- poses ----------------------------------------------------------------------------------------------------

def upright(heading: float) -> tuple[float, float, float, float]:
    return tk.quat_xyzw_from_axis_angle((0, 0, 1), heading)


def on_left_side(heading: float) -> tuple[float, float, float, float]:
    """Lying on its left side (its left tyres down), nose along ``heading``: rolled -90 deg about its own x axis,
    so its roof faces to the left of the heading and its belly (the tyre treads) to the right."""
    roll = tk.quat_xyzw_from_axis_angle((1, 0, 0), -math.pi / 2)
    return tk.quat_mul_xyzw(upright(heading), roll)


def side_origin_height() -> float:
    """Height of the rover origin above the ground when it lies on a side (on its tyres' outer faces)."""
    return S.TRACK / 2 + S.TYRE_WIDTH / 2


# -- state helpers ----------------------------------------------------------------------------------------------

def tyre_contacts(st: tk.BodyState) -> dict[str, np.ndarray]:
    """World point under each tyre's centre at the bottom of the tyre (rover frame z = 0)."""
    return {tag: st.to_world((c[0], c[1], 0.0)) for tag, c in TYRE_CENTRES.items()}


def tyre_centres(st: tk.BodyState) -> dict[str, np.ndarray]:
    return {tag: st.to_world(c) for tag, c in TYRE_CENTRES.items()}


def tyre_box_ys(st: tk.BodyState) -> dict[str, tuple[float, float]]:
    """World y range of each tyre (its whole cylinder), for edge-distance tests."""
    out = {}
    R = st.rotation()
    for tag, c in TYRE_CENTRES.items():
        s = tk.Shape.cylinder(c, TYRE_R, S.TYRE_WIDTH, axis="Y")
        pts = st.pos + s.surface_points(16) @ R.T
        out[tag] = (float(pts[:, 1].min()), float(pts[:, 1].max()))
    return out


# -- level L3: rolling wheels ------------------------------------------------------------------------------------

WHEEL_LINK_MASS = 0.030     # approx: one wheel and tyre (moved from the base's mass)
WHEEL_DRAG = 0.0005         # N m s/rad: bearing and drivetrain drag with the motor pinion removed (approx)


def l3_wheel_links(tyre_friction=None) -> tuple[tk.Link, ...]:
    """Four wheel links on revolute joints about the rover's y axis, at the wheel centres. Each tyre is a cylinder
    in its link's frame; the joints have a little drag and no spring."""
    fr = tyre_friction or (S.TYRE_FRICTION, S.TYRE_FRICTION * 0.9)
    links = []
    for tag, c in TYRE_CENTRES.items():
        links.append(tk.Link(tag.replace("tyre", "wheel"), "base", "revolute", "Y", c, mass=WHEEL_LINK_MASS,
                             damping=WHEEL_DRAG, effort=5.0,
                             shapes=(tk.Shape.cylinder((0, 0, 0), TYRE_R, S.TYRE_WIDTH, axis="Y", color=TYRE_COLOR,
                                                       friction=fr, tag=tag),
                                     tk.Shape.cylinder((0, 0, 0), TYRE_R * 0.55, S.TYRE_WIDTH + 0.002, axis="Y",
                                                       color=(0.75, 0.75, 0.78), tag=tag + "_rim"))))
    return tuple(links)


def rover_l3(name: str, pos, rot_xyzw=(0.0, 0.0, 0.0, 1.0), body_color=BODY_COLOR, extra_shapes=(),
             extra_links=()) -> tk.Body:
    """The rover with free-rolling wheels (motor pinion removed): the base carries everything but the tyres."""
    shapes = [s for s in l1_shapes(body_color) if not s.tag.startswith("tyre")] + list(extra_shapes)
    return tk.body(name, pos=pos, shapes=shapes, mass=S.MASS - 4 * WHEEL_LINK_MASS, rot_xyzw=rot_xyzw,
                   com=(0.0, 0.0, S.COM_HEIGHT), color=body_color,
                   friction=(S.BODY_FRICTION, S.BODY_FRICTION * 0.9),
                   links=l3_wheel_links() + tuple(extra_links), angular_damping=0.05)


def l3_tyre_points(spec: tk.Body, st: tk.BodyState) -> dict[str, np.ndarray]:
    """World points on each tyre's rim (16 per tyre) from the wheel links' poses."""
    out = {}
    for ln in spec.links:
        if not ln.name.startswith("wheel_"):
            continue
        p, R = st.links[ln.name]
        tyre = ln.shapes[0]
        out[ln.name] = p + tyre.surface_points(16) @ R.T
    return out
