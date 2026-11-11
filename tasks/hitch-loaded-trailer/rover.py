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
WHEEL_DRAG = 0.002          # N m s/rad: bearing and drivetrain drag with the motor pinion removed (approx)


def l3_wheel_links(tyre_friction=None, drag: float = WHEEL_DRAG) -> tuple[tk.Link, ...]:
    """Four wheel links on revolute joints about the rover's y axis, at the wheel centres. Each tyre is a cylinder
    in its link's frame; the joints have a little drag and no spring."""
    fr = tyre_friction or (S.TYRE_FRICTION, S.TYRE_FRICTION * 0.9)
    links = []
    for tag, c in TYRE_CENTRES.items():
        links.append(tk.Link(tag.replace("tyre", "wheel"), "base", "revolute", "Y", c, mass=WHEEL_LINK_MASS,
                             damping=drag, effort=5.0,
                             shapes=(tk.Shape.cylinder((0, 0, 0), TYRE_R, S.TYRE_WIDTH, axis="Y", color=TYRE_COLOR,
                                                       friction=fr, tag=tag),
                                     tk.Shape.cylinder((0, 0, 0), TYRE_R * 0.55, S.TYRE_WIDTH + 0.002, axis="Y",
                                                       color=(0.75, 0.75, 0.78), tag=tag + "_rim"))))
    return tuple(links)


def rover_l3(name: str, pos, rot_xyzw=(0.0, 0.0, 0.0, 1.0), body_color=BODY_COLOR, extra_shapes=(),
             extra_links=(), wheel_drag: float = WHEEL_DRAG) -> tk.Body:
    """The rover with free-rolling wheels (motor pinion removed): the base carries everything but the tyres."""
    shapes = [s for s in l1_shapes(body_color) if not s.tag.startswith("tyre")] + list(extra_shapes)
    return tk.body(name, pos=pos, shapes=shapes, mass=S.MASS - 4 * WHEEL_LINK_MASS, rot_xyzw=rot_xyzw,
                   com=(0.0, 0.0, S.COM_HEIGHT), color=body_color,
                   friction=(S.BODY_FRICTION, S.BODY_FRICTION * 0.9),
                   links=l3_wheel_links(drag=wheel_drag) + tuple(extra_links), angular_damping=0.05)


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


# -- level L4: loaded, steerable (vehicle line) ----------------------------------------------------------------------
#
# The rolling rover (L3 wheels) built heavy (about 13 kg of lead filling the body, centre of mass 5.5 cm up), so it behaves towards the arms as a
# real vehicle behaves towards a person: it can be rolled and steered, not lifted or shoved sideways. The front wheels
# hang on one steering link that turns about the middle of the front axle (both wheels turn together; it stands for
# the real kingpins joined by the tie rod) within +-30 degrees. Joint friction and damping stand for the unpowered
# steering servo's gear train: the wheels stay where they are put. A steering tiller (a pin on the servo saver, 4.5 cm
# ahead of the pivot, poking up in front of the bonnet) lets a gripper steer by moving it sideways; tow tabs with a
# post at each end take a gripper or a hook. The front (steered) tyres touch the ground at a point (crowned tyres): a
# flat tread's line contact resists steering at a standstill more than an arm can push; the rear tyres
# keep the flat tread, whose line contact stops the loaded rover creeping at rest.

L4_STEER_FRICTION = 0.0       # PhysX joint friction coefficient: does not hold the steering, unused
L4_STEER_HOLD = 0.15          # N m: the steering holds its angle up to this torque, then slips (trailkit hold)
L4_STEER_DAMPING = 0.05       # N m s/rad
L4_STEER_MASS = 0.020
L4_WHEEL_HOLD = 0.004         # N m per wheel: static (breakaway) friction of the bearings and tread, as a trailkit
L4_WHEEL_HOLD_K = 0.1         # hold of stiffness 0.1 N m/rad (0.4 added 6 N of drag at walking pace): stops the creep at rest
L4_WHEEL_DRAG = 0.02          # N m s/rad per wheel: rolling resistance and bearing drag of the loaded crawler (about
#                               6 N to roll it at 6 cm/s; it coasts about 1.5 cm from 10 cm/s)
TILLER_R = 0.005                       # the steering tiller: a pin on the servo saver, ahead of the steering pivot
TILLER_ARM = 0.045                     # its distance ahead of the pivot (steering 20 degrees moves it 15 mm sideways)
TILLER_TOP = 0.112                     # its top above the ground (the bonnet is at 0.085; the bumper top at 0.053)
TILLER_LEN = 0.040
KEY_COLOR = (0.78, 0.12, 0.10)
KEY_TOP = TILLER_TOP                   # kept for older callers
TOW_TAB = (0.050, 0.016, 0.016)        # tab beyond each bumper: length (x), width (y), height (z)
TOW_POST_R, TOW_POST_H = 0.005, 0.030  # a vertical post at the tab's tip (a hook or a loop of line goes over it)
TOW_COLOR = (0.20, 0.20, 0.22)
TOW_X = S.LENGTH / 2 + TOW_TAB[0] / 2 - 0.002          # tab centre along the rover's x
TOW_POST_X = S.LENGTH / 2 + TOW_TAB[0] - 0.002 - 0.007  # post centre
ROOF_LOAD_SIZE = (0.070, 0.060, 0.030)
ROOF_LOAD_COLOR = (0.25, 0.30, 0.55)


def l4_tow_shapes() -> list[tk.Shape]:
    out = []
    for s, tag in ((1, "tow_f"), (-1, "tow_r")):
        out.append(tk.Shape.box((s * TOW_X, 0.0, S.BUMPER_HEIGHT), TOW_TAB, TOW_COLOR, friction=(1.0, 0.9), tag=tag))
        out.append(tk.Shape.cylinder((s * TOW_POST_X, 0.0, S.BUMPER_HEIGHT + TOW_TAB[2] / 2 + TOW_POST_H / 2),
                                     TOW_POST_R, TOW_POST_H, axis="Z", color=TOW_COLOR, friction=(0.6, 0.5),
                                     tag=tag + "_post"))
    return out


def l4_steer_link(steer0: float = 0.0, friction: float = L4_STEER_FRICTION, damping: float = L4_STEER_DAMPING,
                  limit_deg: float = S.L4_STEER_LIMIT_DEG, hold: float = L4_STEER_HOLD) -> tk.Link:
    lim = math.radians(limit_deg)
    zc = TILLER_TOP - TILLER_LEN / 2 - TYRE_R             # tiller centre above the joint (at the axle height)
    return tk.Link("steer", "base", "revolute", "Z", (AXLE_X, 0.0, TYRE_R), mass=L4_STEER_MASS, damping=damping,
                   friction=friction, limits=(-lim, lim), q0=steer0, effort=5.0, hold=hold, target=steer0,
                   shapes=(tk.Shape.box((0, 0, 0), (0.010, 0.070, 0.008), CHASSIS_COLOR, tag="front_axle"),
                           tk.Shape.cylinder((TILLER_ARM, 0, zc), TILLER_R, TILLER_LEN, axis="Z", color=KEY_COLOR,
                                             friction=(1.0, 0.9), tag="tiller")))


def rover_l4(name: str, pos, rot_xyzw=(0.0, 0.0, 0.0, 1.0), steer0: float = 0.0, roof_load: float = 0.0,
             mass: float = S.L4_MASS, body_color=BODY_COLOR, extra_shapes=(), extra_links=(),
             wheel_drag: float = L4_WHEEL_DRAG, steer_friction: float = L4_STEER_FRICTION,
             steer_damping: float = L4_STEER_DAMPING, wheel_friction: float = 0.0,
             steer_hold: float = L4_STEER_HOLD, tyre_contact: str = "sphere",
             wheel_hold: float = L4_WHEEL_HOLD, **body_kw) -> tk.Body:
    """The loaded rover: free-rolling wheels, the front pair steered together (joint ``steer``, radians, + to the
    left), ballast low between the axles. ``roof_load`` (kg, e.g. ``S.L4_ROOF_LOAD``) adds a case on the roof and
    raises the centre of mass. Links: ``steer``, ``wheel_fl``, ``wheel_fr`` (on ``steer``), ``wheel_rl``,
    ``wheel_rr``."""
    steer = l4_steer_link(steer0, steer_friction, steer_damping, hold=steer_hold)
    links = []
    for ln in l3_wheel_links(drag=wheel_drag):
        front = ln.name.startswith("wheel_f")
        shapes = ln.shapes
        if tyre_contact == "sphere" and front:
            # the tread touches the ground at a point (a crowned tyre): the cylinder is kept for its look only
            tyre = ln.shapes[0]
            shapes = (tk.Shape(tyre.kind, tyre.centre, radius=tyre.radius, length=tyre.length, axis=tyre.axis,
                               color=tyre.color, collide=False, tag=tyre.tag + "_look"),
                      tk.Shape.sphere((0, 0, 0), TYRE_R, color=TYRE_COLOR, friction=tyre.friction, tag=tyre.tag),
                      *ln.shapes[1:])
        ln = tk.Link(ln.name, "steer" if front else "base", ln.joint, ln.axis,
                     (0.0, ln.anchor[1], 0.0) if front else ln.anchor, shapes, ln.mass, damping=ln.damping,
                     effort=ln.effort, friction=wheel_friction, hold=wheel_hold, hold_stiffness=L4_WHEEL_HOLD_K)
        links.append(ln)
    shapes = [s for s in l1_shapes(body_color) if not s.tag.startswith("tyre")] + l4_tow_shapes() + list(extra_shapes)
    base_mass = mass - 4 * WHEEL_LINK_MASS - L4_STEER_MASS
    com_z = S.L4_COM_HEIGHT
    if roof_load > 0:
        shapes.append(tk.Shape.box((-0.015, 0.0, S.HEIGHT + ROOF_LOAD_SIZE[2] / 2), ROOF_LOAD_SIZE, ROOF_LOAD_COLOR,
                                   friction=(0.6, 0.5), tag="roof_load"))
        com_z = (base_mass * com_z + roof_load * S.L4_ROOF_LOAD_HEIGHT) / (base_mass + roof_load)
        base_mass += roof_load
    return tk.body(name, pos=pos, shapes=shapes, mass=base_mass, rot_xyzw=rot_xyzw, com=(0.0, 0.0, com_z),
                   color=body_color, friction=(S.BODY_FRICTION, S.BODY_FRICTION * 0.9),
                   links=(steer, *links, *extra_links), angular_damping=0.05, **body_kw)


def steer_angle(st: tk.BodyState) -> float:
    """The steering angle (radians, + to the left) of an L4 rover's state."""
    return float(st.joints.get("steer", 0.0))


def l4_link_points(spec: tk.Body, st: tk.BodyState) -> np.ndarray:
    """World points on every link collider (the wheels, the front axle, the steering key)."""
    pts = []
    for ln in spec.links:
        if ln.name not in st.links:
            continue
        p, R = st.links[ln.name]
        for s in ln.shapes:
            if s.collide:
                pts.append(p + s.surface_points(8) @ R.T)
    return np.vstack(pts) if pts else np.zeros((0, 3))


def l4_released(spec: tk.Body, st: tk.BodyState, grips: dict, clearance: float = 0.05) -> bool:
    """``tk.released`` over the base and every link (a gripper on the steering key or a wheel counts as holding)."""
    if not tk.released(spec, st, grips, clearance):
        return False
    pts = l4_link_points(spec, st)
    if len(pts) == 0:
        return True
    for g in grips.values():
        for key in ("points", "arm"):
            if key in g and len(g[key]):
                d = np.linalg.norm(pts[None, :, :] - np.asarray(g[key])[:, None, :], axis=2).min()
                if d < clearance:
                    return False
    return True


def l4_wheel_centres(st: tk.BodyState) -> dict[str, np.ndarray]:
    return {n: np.asarray(st.links[n][0], float) for n in ("wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr")
            if n in st.links}


def tiller_point(st: tk.BodyState, steer: float | None = None, h: float | None = None) -> np.ndarray:
    """World point on the tiller's axis at height ``h`` above the ground (rover frame; default 1.5 cm below its top)
    for the steering angle ``steer`` (default: the current one)."""
    q = steer_angle(st) if steer is None else steer
    z = (TILLER_TOP - 0.015) if h is None else h
    return st.to_world((AXLE_X + TILLER_ARM * math.cos(q), TILLER_ARM * math.sin(q), z))


def steer_with_tiller(rover_state, A, target: float, tcp_for, h: float, rate: float, timeout_s: float = 12.0,
                      reach: float = 0.05, tol: float = 0.008):
    """Generator: with the tiller gripped, steer to ``target`` (rad) closed loop on the measured angle. The commanded
    tiller point runs ahead of the measured one along the arc (by up to ``reach`` metres past the target point), so
    the arm pushes hard enough to overcome the tyres' scrub at a standstill. ``rover_state()`` returns the current
    rover state, ``tcp_for(point)`` the tool point that puts the pads on ``point``."""
    import demokit as dk
    q = steer_angle(rover_state())
    qc = q
    for _ in range(int(timeout_s * dk.CONTROL_HZ)):
        st = rover_state()
        q = steer_angle(st)
        err = target - q
        if abs(err) < tol:
            break
        qc += math.copysign(min(rate / dk.CONTROL_HZ, abs(target - qc) + reach / TILLER_ARM), err)
        lead = max(min(qc - q, reach / TILLER_ARM), -reach / TILLER_ARM)
        A.aim(tcp_for(tiller_point(st, steer=q + lead, h=h)), A.rot)
        yield
    # ease off: the tool back onto the tiller where it now is
    for _ in range(6):
        A.aim(tcp_for(tiller_point(rover_state(), h=h)), A.rot)
        yield


# -- the line rig's fittings on the rover (vehicle line, kit part B) --------------------------------------------------

SHEAVE_R = 0.025             # the pulley's sheave: radius at the bottom of its groove
SHEAVE_T = 0.010             # groove width
SHEAVE_FLANGE_R = 0.033
SHEAVE_COLOR = (0.70, 0.72, 0.75)


def l4_sheave_link(end: str = "front", height: float = 0.075, drag: float = 0.0005) -> tk.Link:
    """A pulley block on the rover's tow post at ``end``: a free-spinning sheave (revolute about the vertical) whose
    groove is ``height`` above the ground, with flanges that keep the line on. Pass it in ``rover_l4(extra_links=...)``."""
    s = 1 if end == "front" else -1
    x = s * TOW_POST_X
    flange = (0.0, 0.0, SHEAVE_T / 2 + 0.002)
    shapes = (tk.Shape.cylinder((0, 0, 0), SHEAVE_R, SHEAVE_T, axis="Z", color=SHEAVE_COLOR, friction=(0.3, 0.2),
                                tag="sheave"),
              tk.Shape.cylinder(flange, SHEAVE_FLANGE_R, 0.003, axis="Z", color=SHEAVE_COLOR, tag="sheave_flange"),
              tk.Shape.cylinder(tuple(-v for v in flange), SHEAVE_FLANGE_R, 0.003, axis="Z", color=SHEAVE_COLOR,
                                tag="sheave_flange"))
    # the block hangs on a short strap from the post: the sheave clear of the tab, its groove 5 mm ahead of the tab tip
    off = (S.LENGTH / 2 + TOW_TAB[0] - 0.002) - TOW_POST_X + SHEAVE_R + 0.008
    return tk.Link(f"sheave_{end}", "base", "revolute", "Z", (x + s * off, 0.0, height), shapes=shapes,
                   mass=0.04, damping=drag, effort=2.0)
