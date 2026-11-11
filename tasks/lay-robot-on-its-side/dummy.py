"""The passive 1:3 humanoid dummy as a trailkit articulated body, from ``dummy_spec``.

One rigid base (pelvis, trunk and head: the robot has no waist joint) and 14 pin hinges with hard end stops:
per side shoulder pitch and roll, elbow, hip pitch and roll, knee and ankle pitch. A two-axis joint is two hinges
with a light joint block between them. The joints are limp (no spring, light damping), as on a powered-off robot.

Unlike trailkit's other articulated bodies (rover, lines), the dummy's own parts collide with each other: an arm
can be trapped under the trunk, a leg can rest on the other. Hinged neighbours do not collide (PhysX filters
jointed pairs), and the layout keeps every other pair apart at the zero pose.

Dummy frame: origin midway between the hip joints; x forward (chest), y left, z up (head). Link names, per side
``l``/``r``: ``{s}_sh_pitch``, ``{s}_sh_roll`` (housing and upper arm), ``{s}_elbow`` (forearm with hand),
``{s}_hip_pitch``, ``{s}_hip_roll`` (housing and thigh), ``{s}_knee`` (shank), ``{s}_ankle`` (foot).
Joint values: as in ``dummy_spec`` (degrees there, radians here), except the elbow, whose hinge value is minus
its flexion (the hinge turns about +y, and flexion brings the forearm forward).

License: MIT (see LICENSE next to this file).
"""

from __future__ import annotations

import math

import numpy as np

import dummy_spec as S
import trailkit as tk

SIDES = {"l": 1.0, "r": -1.0}
LIMBS = ("sh_roll", "elbow", "hip_roll", "knee", "ankle")        # links that carry a segment
ARM_LINKS = ("sh_roll", "elbow")
LEG_LINKS = ("hip_roll", "knee", "ankle")
_DUMMIES: set[str] = set()


def _rad(lim):
    return (math.radians(lim[0]), math.radians(lim[1]))


def _mirror(lim):
    return (-lim[1], -lim[0])


def links(pose: dict | None = None) -> tuple[tk.Link, ...]:
    """The 14 hinged links (plus 4 joint blocks). ``pose`` maps link names to start joint values (radians)."""
    pose = dict(pose or {})
    d, e, c, m = S.JOINT_DAMPING, S.JOINT_EFFORT, S.JOINT_COLOR, S.CONNECTOR_MASS
    out = []
    for s, sign in SIDES.items():
        def L(name, parent, axis, anchor, shapes, mass, lim):
            return tk.Link(f"{s}_{name}", parent if parent == "base" else f"{s}_{parent}", "revolute", axis,
                           tuple(map(float, anchor)), tuple(shapes), float(mass), limits=_rad(lim), damping=d,
                           effort=e, q0=float(pose.get(f"{s}_{name}", 0.0)))

        roll_arm = S.SHOULDER_ROLL if sign > 0 else _mirror(S.SHOULDER_ROLL)
        roll_leg = S.HIP_ROLL if sign > 0 else _mirror(S.HIP_ROLL)
        elbow = _mirror(S.ELBOW)                       # hinge value = -flexion
        ua, fa = S.UPPER_ARM_LEN, S.FOREARM_LEN
        th, sh = S.THIGH_LEN, S.SHANK_LEN
        out += [
            L("sh_pitch", "base", "Y", (0, sign * S.SHOULDER_Y, S.SHOULDER_Z), (), m, S.SHOULDER_PITCH),
            L("sh_roll", "sh_pitch", "X", (0, 0, 0), (
                tk.Shape.sphere((0, 0, 0), S.SHOULDER_HOUSING_R, c, S.SHELL_FRICTION, "housing"),
                tk.Shape.capsule((0, 0, -ua / 2), S.UPPER_ARM_R, ua - 2 * S.UPPER_ARM_R, "Z", S.SHELL,
                                 S.SHELL_FRICTION, "upper_arm"),
            ), S.UPPER_ARM_MASS - m, roll_arm),
            L("elbow", "sh_roll", "Y", (0, 0, -ua), (
                tk.Shape.capsule((0, 0, -fa / 2), S.FOREARM_R, fa - 2 * S.FOREARM_R, "Z", S.SHELL,
                                 S.SHELL_FRICTION, "forearm"),
            ), S.FOREARM_MASS, elbow),
            L("hip_pitch", "base", "Y", (0, sign * S.HIP_Y, 0), (), m, S.HIP_PITCH),
            L("hip_roll", "hip_pitch", "X", (0, 0, 0), (
                tk.Shape.sphere((0, 0, 0), S.HIP_HOUSING_R, c, S.SHELL_FRICTION, "housing"),
                tk.Shape.capsule((0, 0, -th / 2), S.THIGH_R, th - 2 * S.THIGH_R, "Z", S.SHELL, S.SHELL_FRICTION,
                                 "thigh"),
            ), S.THIGH_MASS - m, roll_leg),
            L("knee", "hip_roll", "Y", (0, 0, -th), (
                tk.Shape.capsule((0, 0, -sh / 2), S.SHANK_R, sh - 2 * S.SHANK_R, "Z", S.SHELL, S.SHELL_FRICTION,
                                 "shank"),
            ), S.SHANK_MASS, S.KNEE),
            L("ankle", "knee", "Y", (0, 0, -sh), (
                tk.Shape.box((S.FOOT_FORWARD, 0, -S.ANKLE_HEIGHT + S.FOOT_SIZE[2] / 2), S.FOOT_SIZE, c,
                             friction=S.SOLE_FRICTION, tag="foot"),
            ), S.FOOT_MASS, S.ANKLE_PITCH),
        ]
    return tuple(out)


def base_shapes() -> list[tk.Shape]:
    px, py, _ = S.PELVIS_SIZE
    tx, ty, _ = S.TRUNK_SIZE
    hx, hy, _ = S.HEAD_SIZE
    hw, hh = S.HATCH_SIZE
    return [
        tk.Shape.span((-px / 2, -py / 2, S.PELVIS_Z[0]), (px / 2, py / 2, S.PELVIS_Z[1]), S.SHELL, S.SHELL_FRICTION,
                      "pelvis"),
        tk.Shape.span((-tx / 2, -ty / 2, S.TRUNK_Z[0]), (tx / 2, ty / 2, S.TRUNK_Z[1]), S.SHELL, S.SHELL_FRICTION,
                      "trunk"),
        tk.Shape.span((-hx / 2, -hy / 2, S.HEAD_Z[0]), (hx / 2, hy / 2, S.HEAD_Z[1]), S.SHELL, S.SHELL_FRICTION,
                      "head"),
        # the back hatch: a visual panel on the trunk's back (-x face), 0.6 mm proud
        tk.Shape("box", (-tx / 2 - 0.0006, 0.0, S.HATCH_Z), size=(0.001, hw, hh), color=S.HATCH_COLOR,
                 collide=False, tag="hatch"),
    ]


# base mass centre: the pelvis at its box centre, the trunk with the head at 2/5 up the trunk-and-head stack
_BASE_MASS = S.TRUNK_HEAD_MASS + S.PELVIS_MASS
_COM_Z = (S.PELVIS_MASS * sum(S.PELVIS_Z) / 2
          + S.TRUNK_HEAD_MASS * (S.TRUNK_Z[0] + 0.4 * (S.HEAD_Z[1] - S.TRUNK_Z[0]))) / _BASE_MASS


def dummy(name: str, pos, rot_xyzw, pose: dict | None = None) -> tk.Body:
    """The dummy with its base at ``pos`` turned by ``rot_xyzw``, joints at ``pose`` (radians)."""
    _DUMMIES.add(name)
    return tk.body(name, pos=pos, shapes=base_shapes(), mass=_BASE_MASS, rot_xyzw=rot_xyzw, com=(0, 0, _COM_Z),
                   friction=S.SHELL_FRICTION, links=links(pose), angular_damping=0.05, solver_iterations=32)


# -- lying orientations: the dummy along world y, head toward +y (the robot's left) -------------------------------

def _quat(R) -> tuple[float, float, float, float]:
    return tk.quat_from_rot(np.asarray(R, float))


def on_back() -> tuple:
    """Chest up; the dummy's left toward +x (away from the robot)."""
    return _quat(np.column_stack([(0, 0, 1), (1, 0, 0), (0, 1, 0)]))


def face_down() -> tuple:
    """Chest down; the dummy's left toward -x (toward the robot)."""
    return _quat(np.column_stack([(0, 0, -1), (-1, 0, 0), (0, 1, 0)]))


def on_left_side() -> tuple:
    """Left side down; the chest faces +x (away from the robot)."""
    return _quat(np.column_stack([(1, 0, 0), (0, 0, -1), (0, 1, 0)]))


def on_right_side() -> tuple:
    """Right side down; the chest faces -x (toward the robot)."""
    return _quat(np.column_stack([(-1, 0, 0), (0, 0, 1), (0, 1, 0)]))


ARMS_ALONG = {"l_sh_roll": 0.12, "r_sh_roll": -0.12}   # arms along the trunk, hands just clear of the thighs


# -- geometry on the whole dummy (base and links) ----------------------------------------------------------------

def frames(spec: tk.Body, st: tk.BodyState) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """World (position, rotation) of the base and every link."""
    return {"base": (np.asarray(st.pos, float), st.rotation())} | {k: (np.asarray(p), np.asarray(R))
                                                                   for k, (p, R) in st.links.items()}


def placed_shapes(spec: tk.Body, st: tk.BodyState, which=None):
    """(owner, shape, world centre, world rotation) of every collider; ``which`` limits the owners."""
    fr = frames(spec, st)
    owners = [("base", spec.shapes)] + [(ln.name, ln.shapes) for ln in spec.links]
    out = []
    for owner, shapes in owners:
        if which is not None and owner not in which:
            continue
        p, R = fr[owner]
        for s in shapes:
            if s.collide:
                out.append((owner, s, p + R @ np.asarray(s.centre), R @ s.rotation()))
    return out


def points(spec: tk.Body, st: tk.BodyState, which=None, tags=None) -> np.ndarray:
    """World surface points of the colliders (optionally of some owners, or of shapes with some tags)."""
    fr = frames(spec, st)
    owners = [("base", spec.shapes)] + [(ln.name, ln.shapes) for ln in spec.links]
    pts = []
    for owner, shapes in owners:
        if which is not None and owner not in which:
            continue
        p, R = fr[owner]
        for s in shapes:
            if s.collide and (tags is None or s.tag in tags):
                pts.append(p + s.surface_points() @ R.T)
    return np.vstack(pts) if pts else np.zeros((0, 3))


def lowest(spec: tk.Body, st: tk.BodyState, which=None) -> float:
    return float(points(spec, st, which)[:, 2].min())


def distance(spec: tk.Body, st: tk.BodyState, pts, which=None) -> float:
    """Smallest distance from any of ``pts`` (world) to the dummy's colliders."""
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
    """No gripper point and no point along either arm within ``clearance`` of any part of the dummy."""
    return all(distance(spec, st, g["points"]) >= clearance
               and ("arm" not in g or distance(spec, st, g["arm"]) >= clearance) for g in grips.values())


def chest_normal(st: tk.BodyState) -> np.ndarray:
    """World direction the chest faces (the dummy's +x)."""
    return st.rotation()[:, 0]


def hatch_normal(st: tk.BodyState) -> np.ndarray:
    """World direction the back hatch faces (the dummy's -x)."""
    return -st.rotation()[:, 0]


def shoulder_line(st: tk.BodyState) -> np.ndarray:
    """World direction from the right shoulder to the left (the dummy's +y)."""
    return st.rotation()[:, 1]


def spine(st: tk.BodyState) -> np.ndarray:
    """World direction from the hips to the head (the dummy's +z)."""
    return st.rotation()[:, 2]


def angle_deg(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return math.degrees(math.acos(float(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


def rest_height(rot_xyzw, pose: dict | None = None) -> float:
    """Height of the base origin above the ground when the dummy lies in this orientation and pose."""
    spec = dummy("_probe", (0, 0, 0), rot_xyzw, pose)
    _DUMMIES.discard("_probe")
    st = tk.state_at(spec, (0, 0, 0), spec.spawn_xyzw)
    return -lowest(spec, st)


def _ray_up_hits_box(pts, centre, rot, half) -> np.ndarray:
    """For each point, whether the vertical ray up from it passes through the box (slab test)."""
    o = (np.atleast_2d(pts) - centre) @ rot          # box frame
    d = np.array([0.0, 0.0, 1.0]) @ rot               # world up in the box frame
    t0 = np.full(len(o), -np.inf)
    t1 = np.full(len(o), np.inf)
    for k in range(3):
        if abs(d[k]) < 1e-9:
            inside = np.abs(o[:, k]) <= half[k]
            t0 = np.where(inside, t0, np.inf)
            continue
        a, b = (-half[k] - o[:, k]) / d[k], (half[k] - o[:, k]) / d[k]
        t0 = np.maximum(t0, np.minimum(a, b))
        t1 = np.minimum(t1, np.maximum(a, b))
    return (t0 <= t1) & (t1 > 0.0)


SEGMENT_TAGS = ("upper_arm", "forearm", "thigh", "shank", "foot")
NEAR_JOINT = 0.03     # limb points this close to the shoulder or hip joint are not judged (they flank the base)


SAMPLE_STEP = 0.005   # m: spacing of the points sampled along each limb segment and over the foot


def _segment_points(spec: tk.Body, st: tk.BodyState, link: str) -> np.ndarray:
    """World points on the surface of a link's limb segments, every 5 mm along each capsule's axis (a ring of 8 at
    each step, the end caps included) and on a 5 mm grid over each box's faces."""
    p, R = frames(spec, st)[link]
    ln = next(x for x in spec.links if x.name == link)
    out = []
    for s in ln.shapes:
        if not (s.collide and s.tag in SEGMENT_TAGS):
            continue
        c, Rs = p + R @ np.asarray(s.centre), R @ s.rotation()
        if s.kind == "capsule":
            a = Rs @ tk._AXES[s.axis]
            u = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
            u /= np.linalg.norm(u)
            v = np.cross(a, u)
            half, r = s.length / 2, s.radius
            for t in np.arange(-half - r, half + r + 1e-9, SAMPLE_STEP):
                rr = r if abs(t) <= half else float(np.sqrt(max(r * r - (abs(t) - half) ** 2, 0.0)))
                for k in range(8):
                    ang = k * np.pi / 4
                    out.append(c + a * t + rr * (np.cos(ang) * u + np.sin(ang) * v))
        else:
            h = np.asarray(s.size) / 2
            g = [np.arange(-h[i], h[i] + 1e-9, SAMPLE_STEP) if h[i] > 0 else np.zeros(1) for i in range(3)]
            for x in g[0]:
                for y in g[1]:
                    for z in g[2]:
                        q = np.array([x, y, z])
                        if np.any(np.isclose(np.abs(q), h, atol=SAMPLE_STEP / 2)):
                            out.append(c + Rs @ q)
    return np.array(out) if out else np.zeros((0, 3))


def limb_under_body(spec: tk.Body, st: tk.BodyState) -> list[str]:
    """Arm and leg links with a point straight below the trunk or pelvis (a vertical line up from it passes through
    either box). Each segment is sampled every 5 mm along its length (``_segment_points``). Points within
    ``NEAR_JOINT`` of the shoulder or hip joint are left out (they flank the body)."""
    fr = frames(spec, st)
    p0, R0 = fr["base"]
    boxes = [(p0 + R0 @ np.asarray(s.centre), R0 @ s.rotation(), np.asarray(s.size) / 2)
             for s in spec.shapes if s.tag in ("trunk", "pelvis")]
    out = []
    for ln in spec.links:
        if not any(sh.tag in SEGMENT_TAGS for sh in ln.shapes):
            continue
        pts = _segment_points(spec, st, ln.name)
        if ln.name.endswith(("sh_roll", "hip_roll")):
            pts = pts[np.linalg.norm(pts - fr[ln.name][0], axis=1) > NEAR_JOINT]
        if len(pts) and any(_ray_up_hits_box(pts, c, r, h).any() for c, r, h in boxes):
            out.append(ln.name)
    return out


# -- self-collision: enabled for the dummy only, without editing trailkit ------------------------------------------

def _install_self_collision() -> None:
    if getattr(tk, "_dummy_patch", False):
        return
    original = tk._author_body

    def author(prim_path, cfg, *a, **k):
        prim = original(prim_path, cfg, *a, **k)
        if cfg.body_name in _DUMMIES:
            from isaaclab.sim.utils.stage import get_current_stage
            from pxr import PhysxSchema

            base = get_current_stage().GetPrimAtPath(f"{prim_path}/base")
            PhysxSchema.PhysxArticulationAPI(base).GetEnabledSelfCollisionsAttr().Set(True)
        return prim

    tk._author_body = author
    tk._dummy_patch = True


_install_self_collision()
