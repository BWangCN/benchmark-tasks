"""The passive 1:2 quadruped (a powered-off robot dog) as a trailkit articulated body, from ``quadruped_spec``.

One rigid base (the trunk with its head, four hip handles and the battery bay) and 12 pin hinges with hard end stops:
per leg an abduction hinge (the leg swings sideways), a thigh hinge (fore and aft) and a knee. The joints are limp
(no spring, light damping), as on a powered-off robot. The battery is a separate free body seated in the belly bay,
held by a magnetic catch (a force of ``CATCH_FORCE`` toward its seat while it is within ``CATCH_RANGE`` of it, with
the reaction on the trunk), so it stays in whichever way the robot lies and comes out at a pull of about 3 N.

The robot's own parts collide with each other (a leg can be trapped under the trunk or rest on another leg); hinged
neighbours do not (PhysX filters jointed pairs).

Robot frame: origin at the trunk's centre; x forward (head), y left, z up (back). Leg prefixes ``fl``, ``fr``,
``rl``, ``rr``; links ``{leg}_hip`` (abduction, with the hip housing), ``{leg}_thigh``, ``{leg}_calf`` (with the foot).

License: MIT (see LICENSE next to this file).
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

import quadruped_spec as S
import trailkit as tk

# The knee's range (-156 to -48 degrees) does not contain 0, and trailkit authors every link at joint value 0, which
# PhysX cannot start from. So the knee hinge's zero is put at the standing knee angle: hinge value = knee - KNEE_ZERO.
KNEE_ZERO = math.radians(S.STAND["knee"])
CALF_DIR = np.array([-math.sin(KNEE_ZERO), 0.0, -math.cos(KNEE_ZERO)])   # knee to foot, in the calf link frame


def knee_hinge(knee_rad: float) -> float:
    """Hinge value of a knee angle (the model's convention, radians)."""
    return knee_rad - KNEE_ZERO


def knee_angle(hinge: float) -> float:
    """Knee angle (the model's convention, radians) of a hinge value."""
    return hinge + KNEE_ZERO


LEGS = {"fl": (1.0, 1.0), "fr": (1.0, -1.0), "rl": (-1.0, 1.0), "rr": (-1.0, -1.0)}   # (fore/aft, left/right)
_QUADRUPEDS: set[str] = set()
_BATTERIES: dict[str, str] = {}      # battery body name -> robot body name


def _rad(lim):
    return (math.radians(lim[0]), math.radians(lim[1]))


_CALF_ROT = tk.quat_xyzw_from_axis_angle((0.0, 1.0, 0.0), KNEE_ZERO)


def links(pose: dict | None = None, knee_overrange: float = 0.0) -> tuple[tk.Link, ...]:
    """The 12 hinged links. ``pose`` maps link names to start joint values (radians). ``knee_overrange`` (degrees, default
    0) lets each knee straighten that far past its real stop in the simulator, so that a task can score a knee forced
    past its range by position (a hinge with a wider limit standing in for a joint that would break)."""
    pose = dict(pose or {})
    d, e = S.JOINT_DAMPING, S.JOINT_EFFORT
    out = []
    for leg, (fx, sy) in LEGS.items():
        def L(name, parent, axis, anchor, shapes, mass, lim):
            return tk.Link(f"{leg}_{name}", parent if parent == "base" else f"{leg}_{parent}", "revolute", axis,
                           tuple(map(float, anchor)), tuple(shapes), float(mass), limits=_rad(lim), damping=d,
                           effort=e, armature=S.JOINT_ARMATURE, q0=float(pose.get(f"{leg}_{name}", 0.0)))

        thigh_lim = S.FRONT_THIGH if fx > 0 else S.REAR_THIGH
        out += [
            L("hip", "base", "X", (fx * S.HIP_X, sy * S.HIP_Y, 0.0), (
                tk.Shape.cylinder((0.0, sy * S.HOUSING_Y, 0.0), S.HOUSING_R, S.HOUSING_LEN, "Y", S.JOINT_COLOR,
                                  S.SHELL_FRICTION, "housing"),
            ), S.HIP_MASS, S.ABDUCTION),
            L("thigh", "hip", "Y", (0.0, sy * S.THIGH_Y, 0.0), (
                tk.Shape.box((0.0, 0.0, -S.THIGH_LEN / 2), (S.THIGH_SECTION[0], S.THIGH_SECTION[1], S.THIGH_LEN),
                             S.SHELL, friction=S.SHELL_FRICTION, tag="thigh"),
            ), S.THIGH_MASS, thigh_lim),
            L("calf", "thigh", "Y", (0.0, 0.0, -S.THIGH_LEN), (
                tk.Shape("capsule", tuple((CALF_DIR * S.CALF_LEN / 2).tolist()), radius=S.CALF_R,
                         length=S.CALF_LEN - 2 * S.CALF_R, axis="Z", rot_xyzw=_CALF_ROT, color=S.SHELL,
                         friction=S.SHELL_FRICTION, tag="calf"),
                tk.Shape.sphere(tuple((CALF_DIR * S.CALF_LEN).tolist()), S.FOOT_R, S.JOINT_COLOR, S.FOOT_FRICTION,
                                "foot"),
            ), S.CALF_MASS, (math.degrees(math.radians(S.KNEE[0]) - KNEE_ZERO),
                             math.degrees(math.radians(S.KNEE[1]) - KNEE_ZERO) + knee_overrange)),
        ]
    return tuple(out)


def base_shapes() -> list[tk.Shape]:
    """The trunk built round the belly bay, the head, and the four hip handles."""
    lx, ly, lz = S.TRUNK_SIZE[0] / 2, S.TRUNK_SIZE[1] / 2, S.TRUNK_SIZE[2] / 2
    (bx0, bx1), (by0, by1), (bz0, bz1) = S.BAY
    c, f = S.SHELL, S.SHELL_FRICTION
    out = [
        tk.Shape.span((-lx, -ly, bz1), (lx, ly, lz), c, f, "trunk"),               # above the bay
        tk.Shape.span((bx1, -ly, -lz), (lx, ly, bz1), c, f, "trunk"),              # ahead of the bay
        tk.Shape.span((-lx, -ly, -lz), (bx0, ly, bz1), c, f, "trunk"),             # behind it
        tk.Shape.span((bx0, by1, -lz), (bx1, ly, bz1), c, f, "trunk"),             # left wall
        tk.Shape.span((bx0, -ly, -lz), (bx1, by0, bz1), c, f, "trunk"),            # right wall
        tk.Shape.cylinder(S.HEAD_CYL[0], S.HEAD_CYL[1], S.HEAD_CYL[2], "Z", c, f, "head"),
        tk.Shape.sphere(S.HEAD_SPHERE[0], S.HEAD_SPHERE[1], c, f, "head"),
    ]
    h = S.HANDLE_COLOR
    for leg, (fx, sy) in LEGS.items():
        x, y, z = fx * S.HANDLE_X, sy * S.HANDLE_Y, S.HANDLE_Z
        out.append(tk.Shape.capsule((x, y, z), S.HANDLE_R, S.HANDLE_LEN, "X", h, (0.8, 0.7), f"handle_{leg}"))
        for end in (-1.0, 1.0):                   # brackets: from the trunk's top edge out to the rung's ends
            xe = x + end * (S.HANDLE_LEN / 2 + S.BRACKET / 2)
            y0, y1 = sy * (ly - 0.002), y + sy * S.HANDLE_R
            out.append(tk.Shape.span((xe - S.BRACKET / 2, min(y0, y1), lz - 0.002),
                                     (xe + S.BRACKET / 2, max(y0, y1), z + S.HANDLE_R), h, (0.8, 0.7), "bracket"))
    return out


def battery_shapes() -> list[tk.Shape]:
    bx, by, bz = S.BATTERY_SIZE
    tx, ty, tz = S.TAB_SIZE
    return [tk.Shape.box((0, 0, 0), S.BATTERY_SIZE, S.BATTERY_COLOR, friction=S.BATTERY_FRICTION, tag="battery"),
            tk.Shape.box((0, 0, -bz / 2 - tz / 2), S.TAB_SIZE, S.TAB_COLOR, friction=(0.8, 0.7), tag="tab")]


# the battery's seat (its centre) in the robot frame: its top 0.5 mm under the bay's roof, outer face flush with the belly
SEAT = np.array([0.0, 0.0, S.BAY[2][1] - 0.0005 - S.BATTERY_SIZE[2] / 2])
OUT_AXIS = np.array([0.0, 0.0, -1.0])           # the battery comes out of the belly along the robot's -z

_BASE_MASS = S.BASE_MASS_TOTAL - S.BATTERY_MASS
_COM = (0.0, 0.0, 0.004)                         # a little above centre: the bay is empty below


def quadruped(name: str, pos, rot_xyzw, pose: dict | None = None, knee_overrange: float = 0.0) -> tk.Body:
    """The robot with its trunk centre at ``pos`` turned by ``rot_xyzw``, joints at ``pose`` (radians);
    ``knee_overrange`` as in ``links``."""
    _QUADRUPEDS.add(name)
    return tk.body(name, pos=pos, shapes=base_shapes(), mass=_BASE_MASS, rot_xyzw=rot_xyzw, com=_COM,
                   friction=S.SHELL_FRICTION, links=links(pose, knee_overrange), angular_damping=0.05,
                   solver_iterations=32)


def battery(name: str, robot: tk.Body, seated: bool = True, pos=None, rot_xyzw=None) -> tk.Body:
    """The battery, seated in ``robot``'s bay at its spawn pose (or at ``pos``, ``rot_xyzw``)."""
    _BATTERIES[name] = robot.name
    if seated:
        _SEATED.add(name)
        R = tk.rot_from_xyzw(robot.spawn_xyzw)
        pos = np.asarray(robot.pos, float) + R @ SEAT
        rot_xyzw = robot.spawn_xyzw
    return tk.body(name, pos=pos, shapes=battery_shapes(), mass=S.BATTERY_MASS, rot_xyzw=rot_xyzw,
                   friction=S.BATTERY_FRICTION, solver_iterations=32)


# -- lying orientations: the robot along world y, head toward +y (the robot's left) ---------------------------------

def _along_y(R_local) -> tuple:
    Rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])     # robot x -> world y
    return tk.quat_from_rot(Rz @ np.asarray(R_local, float))


def _rx(deg):
    t = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])


def upright() -> tuple:
    """Back up, belly down; its left toward -x (the arms)."""
    return _along_y(np.eye(3))


def on_back() -> tuple:
    """Belly up (the bay facing up); its left toward +x."""
    return _along_y(_rx(180.0))


def on_left_side() -> tuple:
    """Left side down; the back faces -x (toward the arms), the belly +x."""
    return _along_y(_rx(-90.0))


def on_right_side() -> tuple:
    """Right side down; the back faces +x, the belly -x (toward the arms)."""
    return _along_y(_rx(90.0))


def stand_pose() -> dict:
    """Joint values of the standing pose (the model's home keyframe)."""
    t, k = math.radians(S.STAND["thigh"]), knee_hinge(math.radians(S.STAND["knee"]))
    return {f"{leg}_{j}": v for leg in LEGS for j, v in (("thigh", t), ("calf", k))}


def lying_pose(knee_deg: float = -50.0, thigh_deg: float = 0.0) -> dict:
    """Legs out from the body, knees nearly as straight as they go (50 degrees; the stop is 48): a robot that fell and went limp."""
    t, k = math.radians(thigh_deg), knee_hinge(math.radians(knee_deg))
    return {f"{leg}_{j}": v for leg in LEGS for j, v in (("thigh", t), ("calf", k))}


def folded_pose(thigh_deg: float = 69.0) -> dict:
    """Knees folded to their stop (-156 degrees), thighs swung back: how the robot ends when it collapses from standing
    (kit self-test 1)."""
    t, k = math.radians(thigh_deg), knee_hinge(math.radians(S.KNEE[0]))
    return {f"{leg}_{j}": v for leg in LEGS for j, v in (("thigh", t), ("calf", k))}


# -- geometry on the whole robot (base and links) --------------------------------------------------------------------

def frames(spec: tk.Body, st: tk.BodyState) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """World (position, rotation) of the base and every link."""
    return {"base": (np.asarray(st.pos, float), st.rotation())} | {k: (np.asarray(p), np.asarray(R))
                                                                   for k, (p, R) in st.links.items()}


def placed_shapes(spec: tk.Body, st: tk.BodyState, which=None, tags=None):
    """(owner, shape, world centre, world rotation) of every collider; ``which`` limits the owners."""
    fr = frames(spec, st)
    owners = [("base", spec.shapes)] + [(ln.name, ln.shapes) for ln in spec.links]
    out = []
    for owner, shapes in owners:
        if which is not None and owner not in which:
            continue
        p, R = fr[owner]
        for s in shapes:
            if s.collide and (tags is None or s.tag in tags):
                out.append((owner, s, p + R @ np.asarray(s.centre), R @ s.rotation()))
    return out


def points(spec: tk.Body, st: tk.BodyState, which=None, tags=None) -> np.ndarray:
    """World surface points of the colliders (optionally of some owners, or of shapes with some tags)."""
    pts = [_shape_world_points(s, c, r) for _o, s, c, r in placed_shapes(spec, st, which, tags)]
    return np.vstack(pts) if pts else np.zeros((0, 3))


def _shape_world_points(s: tk.Shape, centre, rot) -> np.ndarray:
    """Surface points of one shape placed at a world centre and rotation."""
    local = replace(s, centre=(0.0, 0.0, 0.0), rot_xyzw=(0.0, 0.0, 0.0, 1.0)).surface_points()
    return np.asarray(centre) + local @ np.asarray(rot).T


def lowest(spec: tk.Body, st: tk.BodyState, which=None) -> float:
    return float(points(spec, st, which)[:, 2].min())


def distance(spec: tk.Body, st: tk.BodyState, pts, which=None) -> float:
    """Smallest distance from any of ``pts`` (world) to the robot's colliders."""
    pts = np.atleast_2d(np.asarray(pts, float))
    best = np.inf
    for _o, s, c, r in placed_shapes(spec, st, which):
        if s.kind == "box":
            d = tk.point_box_distance(pts, c, r, np.asarray(s.size) / 2)
        elif s.kind == "sphere":
            d = np.maximum(np.linalg.norm(pts - c, axis=1) - s.radius, 0.0)
        else:
            a = r @ tk._AXES[s.axis]
            d = np.maximum(tk._point_segment_distance(pts, c - a * s.length / 2, c + a * s.length / 2) - s.radius, 0.0)
        best = min(best, float(d.min()))
    return best


def released(spec: tk.Body, st: tk.BodyState, grips: dict, clearance: float = 0.05) -> bool:
    """No gripper point and no point along either arm within ``clearance`` of any part of the robot."""
    return all(distance(spec, st, g["points"]) >= clearance
               and ("arm" not in g or distance(spec, st, g["arm"]) >= clearance) for g in grips.values())


def back_normal(st: tk.BodyState) -> np.ndarray:
    """World direction the back faces (the robot's +z); the belly faces the opposite way."""
    return st.rotation()[:, 2]


def angle_deg(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return math.degrees(math.acos(float(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


def belly_up_deg(st: tk.BodyState) -> float:
    """Degrees between the belly's outward normal and straight up (0: on its back, 180: upright)."""
    return angle_deg(-back_normal(st), (0.0, 0.0, 1.0))


def foot_points(spec: tk.Body, st: tk.BodyState) -> dict[str, np.ndarray]:
    """World centre of each foot."""
    fr = frames(spec, st)
    return {leg: fr[f"{leg}_calf"][0] + fr[f"{leg}_calf"][1] @ (CALF_DIR * S.CALF_LEN) for leg in LEGS}


def knee_points(spec: tk.Body, st: tk.BodyState) -> dict[str, np.ndarray]:
    fr = frames(spec, st)
    return {leg: fr[f"{leg}_calf"][0] for leg in LEGS}


def hip_points(spec: tk.Body, st: tk.BodyState) -> dict[str, np.ndarray]:
    """World position of each thigh joint."""
    fr = frames(spec, st)
    return {leg: fr[f"{leg}_thigh"][0] for leg in LEGS}


def handle_points(spec: tk.Body, st: tk.BodyState) -> dict[str, np.ndarray]:
    """World centre of each hip handle's rung."""
    return {leg: st.to_world(next(s for s in spec.shapes if s.tag == f"handle_{leg}").centre) for leg in LEGS}


def seat_world(st: tk.BodyState) -> np.ndarray:
    return st.to_world(SEAT)


def battery_out(robot_st: tk.BodyState, bat_st: tk.BodyState) -> float:
    """How far (m) the battery's centre is drawn out of its seat along the bay's axis (0 seated)."""
    rel = robot_st.to_local(bat_st.pos) - SEAT
    return float(rel @ OUT_AXIS)


def battery_in_bay(robot_st: tk.BodyState, bat_st: tk.BodyState, tol: float = 0.004) -> bool:
    """The battery's centre within ``tol`` of its seat (seated in the bay)."""
    return float(np.linalg.norm(robot_st.to_local(bat_st.pos) - SEAT)) <= tol


def rest_height(rot_xyzw, pose: dict | None = None) -> float:
    """Height of the trunk centre above the ground when the robot lies in this orientation and pose."""
    spec = quadruped("_probe", (0, 0, 0), rot_xyzw, pose)
    _QUADRUPEDS.discard("_probe")
    st = tk.state_at(spec, (0, 0, 0), spec.spawn_xyzw)
    low = lowest(spec, st)
    R = tk.rot_from_xyzw(spec.spawn_xyzw)
    bat = _shape_world_points(battery_shapes()[1], R @ (SEAT + np.array([0, 0, -S.BATTERY_SIZE[2] / 2 - S.TAB_SIZE[2] / 2])), R)
    return -min(low, float(bat[:, 2].min()))


# -- in the simulator: self-collision for the robot, and the battery's magnetic catch --------------------------------

def _install_self_collision() -> None:
    if getattr(tk, "_quadruped_patch", False):
        return
    original = tk._author_body

    def author(prim_path, cfg, *a, **k):
        prim = original(prim_path, cfg, *a, **k)
        if cfg.body_name in _QUADRUPEDS:
            from isaaclab.sim.utils.stage import get_current_stage
            from pxr import PhysxSchema

            base = get_current_stage().GetPrimAtPath(f"{prim_path}/base")
            PhysxSchema.PhysxArticulationAPI(base).GetEnabledSelfCollisionsAttr().Set(True)
        return prim

    tk._author_body = author
    tk._quadruped_patch = True


CATCH_STATE: dict[str, bool] = {}     # battery name -> the catch is holding it (for logs and tests)


def catch_force(robot_st: tk.BodyState, bat_st: tk.BodyState) -> np.ndarray:
    """World force (N) the catch puts on the battery: ``CATCH_FORCE`` along the robot's +z while the battery is within
    ``CATCH_RANGE`` of its seat along the bay's axis and within 3 mm of it across; 0 otherwise."""
    rel = robot_st.to_local(bat_st.pos) - SEAT
    out = float(rel @ OUT_AXIS)
    across = float(np.linalg.norm(rel - out * OUT_AXIS))
    if -0.002 <= out <= S.CATCH_RANGE and across <= 0.003:
        return -S.CATCH_FORCE * (robot_st.rotation() @ OUT_AXIS)
    return np.zeros(3)


def _apply_catch(environment) -> None:
    import torch

    base = environment.unwrapped
    for bat, robot in _BATTERIES.items():
        if bat not in tk._BODIES or robot not in tk._BODIES:
            continue
        rs, bs = tk.body_state(environment, robot), tk.body_state(environment, bat)
        f = catch_force(rs, bs)
        CATCH_STATE[bat] = bool(np.any(f))
        dev = base.device
        b_asset, r_asset = base.scene[bat], base.scene[robot]
        zero = torch.zeros((1, 1, 3), device=dev)
        b_asset.permanent_wrench_composer.reset()
        b_asset.permanent_wrench_composer.add_forces_and_torques(
            torch.tensor([[f.tolist()]], dtype=torch.float32, device=dev), zero, body_ids=[0], is_global=True)
        i = list(r_asset.body_names).index("base")
        r_asset.permanent_wrench_composer.reset()
        r_asset.permanent_wrench_composer.add_forces_and_torques(          # the reaction, at the trunk
            torch.tensor([[(-f).tolist()]], dtype=torch.float32, device=dev), zero, body_ids=[i], is_global=True)


_SEATED: set[str] = set()      # batteries that start seated: re-seated after each reset


def seat_battery(environment, bat: str) -> None:
    """Put the battery in its seat in the robot's bay as the robot now lies, at rest."""
    rs = tk.body_state(environment, _BATTERIES[bat])
    tk.write_body_pose(environment, bat, seat_world(rs), rs.quat_xyzw, rs.lin_vel, rs.ang_vel)


def _catch_on_reset(environment) -> None:
    """The reset writes the robot's start joints and lets it move; the battery is then put back in its seat."""
    if environment is None or not _BATTERIES:
        return
    for bat in _SEATED:
        if bat in tk._BODIES:
            seat_battery(environment, bat)
    _apply_catch(environment)


_install_self_collision()
tk.on_step(_apply_catch)
tk.on_reset(_catch_on_reset)
