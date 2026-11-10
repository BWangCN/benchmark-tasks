"""trailkit: compound bodies, terrain, a per-step recorder and scoring helpers for yam_arena task files.

Keep this file next to the task file and import it before declaring ``OBJECTS``:

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import trailkit as tk

    DECK = tk.body("deck", pos=(0.3, 0.0, tk.TOP + 0.025), shapes=[tk.Shape.box((0, 0, 0), (0.4, 0.3, 0.05))])
    ROVER = tk.body("rover", pos=..., shapes=[...], mass=0.5, rot_xyzw=...)
    OBJECTS = [DECK, ROVER]

What it adds to yam_arena's single-primitive objects:

- ``Body``: one rigid body (``mass`` given) or one fixed collider group (``mass=None``) made of boxes, cylinders,
  spheres and capsules, each with its own placement, rotation, colour and friction, and textured visual tiles.
  The spawn orientation is a full quaternion (``rot_xyzw``), so a body can start on its side.
- ``BodyState``: the body origin's pose (not a bounding-box centre), velocities, world bounds and helpers.
- ``on_step`` / ``on_reset``: callbacks that run after every control step (before the task is scored) and after
  every reset (once the scene has settled), with the Isaac Lab environment.
- ``grippers(env)``: points on each gripper (palm, tool point, both fingertips) and how far its fingers are open.
- ``Recorder``: latched flags, extremes and short windows, so ``score()`` and ``success()`` are pure functions of
  the recorder and the final state.
- Geometry helpers: tilt, heading, the lowest point of a body, distances from points to a body's shapes.

yam_arena itself is not modified: importing this module installs an import hook that wraps ``yam_arena.scene``
(object creation and state) and the embodiment's ``reset`` and per-step scoring when they are imported.
Pure Python at import time; everything that needs Isaac Sim is imported lazily.

License: MIT (see LICENSE next to this file).
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import math
import sys
from dataclasses import dataclass, field, replace

import numpy as np
from yam_arena.task_file import TABLE_TOP, Box, ObjectState

__all__ = [
    "TOP", "Shape", "Body", "BodyState", "body", "on_step", "on_reset", "current_env", "grippers", "Recorder",
    "quat_xyzw_from_axis_angle", "quat_mul_xyzw", "rot_from_xyzw", "tilt_deg", "heading", "point_box_distance",
    "body_points", "lowest_z", "distance_to_body", "install",
]

TOP = TABLE_TOP["station"]
Color = tuple[float, float, float]


# ---------------------------------------------------------------------------------------------------------------
# rotations (quaternions are x, y, z, w, as yam_arena uses)
# ---------------------------------------------------------------------------------------------------------------

def quat_xyzw_from_axis_angle(axis, angle: float) -> tuple[float, float, float, float]:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    s = math.sin(angle / 2)
    return (float(a[0] * s), float(a[1] * s), float(a[2] * s), math.cos(angle / 2))


def quat_mul_xyzw(q1, q2) -> tuple[float, float, float, float]:
    """q1 * q2 (apply q2 first)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)


def rot_from_xyzw(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def tilt_deg(rot: np.ndarray) -> float:
    """Angle (deg) between the body's z axis and the world's vertical."""
    return math.degrees(math.acos(float(np.clip(rot[2, 2], -1.0, 1.0))))


def heading(rot: np.ndarray) -> float:
    """Yaw (rad) of the body's x axis projected on the ground."""
    return math.atan2(rot[1, 0], rot[0, 0])


# ---------------------------------------------------------------------------------------------------------------
# shapes and bodies
# ---------------------------------------------------------------------------------------------------------------

_AXES = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}


@dataclass(frozen=True)
class Shape:
    """One collider (or visual-only tile) of a body, in the body frame.

    ``kind``: ``box`` (``size`` = edge lengths), ``cylinder`` and ``capsule`` (``radius``, ``length`` of the
    straight part, along ``axis``), ``sphere`` (``radius``) or ``tile`` (a visual-only flat rectangle, ``size``
    x and y, facing +z, with ``texture`` repeated every ``texture_m`` metres).
    ``rot_xyzw`` turns the shape in the body frame. ``friction`` (static, dynamic) overrides the body's.
    """

    kind: str
    centre: tuple[float, float, float]
    size: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.0
    length: float = 0.0
    axis: str = "Z"
    rot_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    color: Color = (0.5, 0.5, 0.5)
    friction: tuple[float, float] | None = None
    collide: bool = True
    texture: str | None = None
    texture_m: float = 0.1
    tag: str = ""

    @staticmethod
    def box(centre, size, color=(0.5, 0.5, 0.5), rot_xyzw=(0.0, 0.0, 0.0, 1.0), friction=None, tag="") -> Shape:
        return Shape("box", tuple(map(float, centre)), size=tuple(map(float, size)), color=color,
                     rot_xyzw=tuple(rot_xyzw), friction=friction, tag=tag)

    @staticmethod
    def span(lo, hi, color=(0.5, 0.5, 0.5), friction=None, tag="") -> Shape:
        """An axis-aligned box from corner ``lo`` to corner ``hi``."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        return Shape.box((lo + hi) / 2, hi - lo, color=color, friction=friction, tag=tag)

    @staticmethod
    def cylinder(centre, radius, length, axis="Z", color=(0.5, 0.5, 0.5), friction=None, tag="") -> Shape:
        return Shape("cylinder", tuple(map(float, centre)), radius=float(radius), length=float(length), axis=axis,
                     color=color, friction=friction, tag=tag)

    @staticmethod
    def capsule(centre, radius, length, axis="Z", color=(0.5, 0.5, 0.5), friction=None, tag="") -> Shape:
        return Shape("capsule", tuple(map(float, centre)), radius=float(radius), length=float(length), axis=axis,
                     color=color, friction=friction, tag=tag)

    @staticmethod
    def sphere(centre, radius, color=(0.5, 0.5, 0.5), friction=None, tag="") -> Shape:
        return Shape("sphere", tuple(map(float, centre)), radius=float(radius), color=color, friction=friction,
                     tag=tag)

    @staticmethod
    def tile(centre, size_xy, texture: str, texture_m=0.1, color=(1.0, 1.0, 1.0), tag="") -> Shape:
        return Shape("tile", tuple(map(float, centre)), size=(float(size_xy[0]), float(size_xy[1]), 0.0),
                     color=color, collide=False, texture=texture, texture_m=float(texture_m), tag=tag)

    # -- geometry in the body frame -------------------------------------------------------------------------

    def rotation(self) -> np.ndarray:
        return rot_from_xyzw(self.rot_xyzw)

    def half_extent(self) -> np.ndarray:
        """Half size of the shape's axis-aligned bounding box in the body frame."""
        r = self.rotation()
        if self.kind in ("box", "tile"):
            return np.abs(r) @ (np.asarray(self.size) / 2)
        if self.kind == "sphere":
            return np.full(3, self.radius)
        a = r @ _AXES[self.axis]
        half_len = self.length / 2 + (self.radius if self.kind == "capsule" else 0.0)
        # a cylinder's extent: along its axis half_len, across it the radius
        return np.abs(a) * half_len + self.radius * np.sqrt(np.clip(1 - a * a, 0, 1))

    def surface_points(self, n: int = 12) -> np.ndarray:
        """Points on the shape's surface (corners and rims), body frame: enough for lowest-point tests."""
        c, r = np.asarray(self.centre), self.rotation()
        if self.kind in ("box", "tile"):
            h = np.asarray(self.size) / 2
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h
            return c + corners @ r.T
        if self.kind == "sphere":
            d = np.vstack([np.eye(3), -np.eye(3)])
            return c + self.radius * d
        a = r @ _AXES[self.axis]
        u = np.cross(a, [0.0, 0.0, 1.0] if abs(a[2]) < 0.9 else [1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(a, u)
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        ring = self.radius * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
        ends = (-self.length / 2, self.length / 2)
        pts = [c + e * a + ring for e in ends]
        if self.kind == "capsule":
            pts.append(np.array([c + (ends[0] - self.radius) * a, c + (ends[1] + self.radius) * a]))
        return np.vstack(pts)


@dataclass(frozen=True)
class Link:
    """A rigid link on a joint, for articulated bodies. ``parent`` is ``"base"`` (the body) or another link's
    name. The joint sits at ``anchor`` in the parent's frame, its axes parallel to the parent's; ``axis`` is the
    joint axis in that frame. The link frame coincides with the joint frame at ``q = 0``; ``shapes`` are in it.
    Revolute limits in radians, prismatic in metres. ``damping`` (N m s/rad or N s/m), ``stiffness`` toward
    ``target`` (a spring; 0 for none) and ``friction`` (PhysX joint friction coefficient) act on the joint."""

    name: str
    parent: str
    joint: str
    axis: str
    anchor: tuple[float, float, float]
    shapes: tuple[Shape, ...]
    mass: float
    limits: tuple[float, float] | None = None
    damping: float = 0.0
    stiffness: float = 0.0
    target: float = 0.0
    friction: float = 0.0
    q0: float = 0.0
    com: tuple[float, float, float] | None = None
    armature: float = 0.0
    effort: float = 50.0            # N m or N: the most the joint's drive (spring and damping) can exert


@dataclass(frozen=True)
class Body(Box):
    """A body made of ``shapes``. Subclasses ``Box`` only so yam_arena's task-file checks accept it: ``pos`` is
    the body origin, ``size`` its bounding box in the body frame, ``mass`` None for a fixed body."""

    shapes: tuple[Shape, ...] = ()
    rot_xyzw: tuple[float, float, float, float] | None = None
    friction: tuple[float, float] = (0.8, 0.7)
    com: tuple[float, float, float] | None = None
    angular_damping: float = 0.05
    linear_damping: float = 0.0
    contact_offset: float = 0.002
    solver_iterations: int = 16
    bounds_lo: tuple[float, float, float] = (0.0, 0.0, 0.0)
    links: tuple[Link, ...] = ()
    lights: tuple[dict, ...] = ()     # each: centre (body frame), radius, intensity, color (linear RGB)
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def _check(self) -> None:
        super()._check()
        if not any(s.collide for s in self.shapes):
            raise ValueError(f"{self.name}: a body needs at least one collider")

    @property
    def spawn_xyzw(self) -> tuple[float, float, float, float]:
        if self.rot_xyzw is not None:
            return tuple(self.rot_xyzw)
        return (0.0, 0.0, math.sin(self.yaw / 2), math.cos(self.yaw / 2))

    def shapes_tagged(self, tag: str) -> list[Shape]:
        return [s for s in self.shapes if s.tag == tag]


def body(name: str, pos, shapes, mass: float | None = None, rot_xyzw=None, yaw: float = 0.0,
         color: Color = (0.5, 0.5, 0.5), friction=(0.8, 0.7), com=None, links=(), lights=(), **kw) -> Body:
    """Build a ``Body``; its bounding box is computed from the shapes."""
    shapes = tuple(shapes)
    lo = np.min([np.asarray(s.centre) - s.half_extent() for s in shapes], axis=0)
    hi = np.max([np.asarray(s.centre) + s.half_extent() for s in shapes], axis=0)
    size = tuple(np.maximum(hi - lo, 1e-4).tolist())
    return Body(name=name, pos=tuple(map(float, pos)), size=size, mass=mass, color=color, yaw=yaw, shapes=shapes,
                rot_xyzw=None if rot_xyzw is None else tuple(map(float, rot_xyzw)), friction=tuple(friction),
                com=None if com is None else tuple(map(float, com)), bounds_lo=tuple(lo.tolist()),
                links=tuple(links), lights=tuple(lights), **kw)


def moved(spec: Body, pos=None, rot_xyzw=None) -> Body:
    """The same body spawned at another pose."""
    return replace(spec, pos=tuple(pos) if pos is not None else spec.pos,
                   rot_xyzw=tuple(rot_xyzw) if rot_xyzw is not None else spec.rot_xyzw)


def _axis_rot(axis: str, angle: float) -> np.ndarray:
    return rot_from_xyzw(quat_xyzw_from_axis_angle(_AXES[axis], angle))


def link_poses(spec: Body, base_pos, base_rot, joints: dict[str, float]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """World (position, rotation) of every link for given joint positions (forward kinematics)."""
    out = {"base": (np.asarray(base_pos, float), np.asarray(base_rot, float))}
    pending = list(spec.links)
    while pending:
        progressed = False
        for ln in list(pending):
            if ln.parent not in out:
                continue
            p, R = out[ln.parent]
            q = float(joints.get(ln.name, ln.q0))
            jp = p + R @ np.asarray(ln.anchor)
            if ln.joint == "revolute":
                out[ln.name] = (jp, R @ _axis_rot(ln.axis, q))
            else:
                out[ln.name] = (jp + R @ (_AXES[ln.axis] * q), R)
            pending.remove(ln)
            progressed = True
        if not progressed:
            raise ValueError(f"{spec.name}: links with unknown parents {[ln.parent for ln in pending]}")
    return out


@dataclass(frozen=True)
class BodyState(ObjectState):
    """``pos`` and ``quat_xyzw`` are the body origin's pose (world, relative to the env origin); ``joints`` the
    joint positions of an articulated body, ``links`` each link's world (position, rotation)."""

    joints: dict = field(default_factory=dict)
    links: dict = field(default_factory=dict)

    def rotation(self) -> np.ndarray:
        return rot_from_xyzw(self.quat_xyzw)

    def to_world(self, local) -> np.ndarray:
        return self.pos + np.asarray(local, float) @ self.rotation().T

    def to_local(self, world) -> np.ndarray:
        return (np.asarray(world, float) - self.pos) @ self.rotation()

    def speed(self) -> float:
        return float(np.linalg.norm(self.lin_vel))

    def spin(self) -> float:
        return float(np.linalg.norm(self.ang_vel))


def state_at(spec: Body, pos, quat_xyzw, lin_vel=(0, 0, 0), ang_vel=(0, 0, 0), joints=None) -> BodyState:
    """A BodyState for a body at a given pose (for offline tests and for fixed bodies)."""
    pos, quat = np.asarray(pos, float), np.asarray(quat_xyzw, float)
    pts = body_points(spec, pos, rot_from_xyzw(quat))
    joints = {ln.name: ln.q0 for ln in spec.links} | dict(joints or {})
    links = link_poses(spec, pos, rot_from_xyzw(quat), joints) if spec.links else {}
    return BodyState(pos=pos, quat_xyzw=quat, lin_vel=np.asarray(lin_vel, float), ang_vel=np.asarray(ang_vel, float),
                     low=pts.min(0), high=pts.max(0), joints=joints, links=links)


# -- geometry on bodies -----------------------------------------------------------------------------------------

def body_points(spec: Body, pos, rot, tags: tuple[str, ...] | None = None) -> np.ndarray:
    """World points on the surface of the body's colliders (optionally only shapes with these tags)."""
    pts = [s.surface_points() for s in spec.shapes if s.collide and (tags is None or s.tag in tags)]
    return np.asarray(pos, float) + np.vstack(pts) @ np.asarray(rot).T


def lowest_z(spec: Body, st: BodyState) -> float:
    return float(body_points(spec, st.pos, st.rotation())[:, 2].min())


def point_box_distance(points, centre, rot, half) -> np.ndarray:
    """Distance from each point (rows) to a solid box; 0 inside."""
    local = (np.atleast_2d(points) - centre) @ rot
    return np.linalg.norm(np.maximum(np.abs(local) - half, 0.0), axis=1)


def _point_segment_distance(points, a, b) -> np.ndarray:
    p = np.atleast_2d(points)
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + np.outer(t, ab)), axis=1)


def distance_to_body(spec: Body, st: BodyState, points) -> float:
    """Smallest distance from any of ``points`` (world) to the body's colliders. Cylinders are measured as capsules
    (their end faces bulge by the radius), which errs on the side of reporting them closer."""
    R = st.rotation()
    best = np.inf
    pts = np.atleast_2d(points)
    for s in spec.shapes:
        if not s.collide:
            continue
        c = st.pos + R @ np.asarray(s.centre)
        r = R @ s.rotation()
        if s.kind == "box":
            d = point_box_distance(pts, c, r, np.asarray(s.size) / 2)
        elif s.kind == "sphere":
            d = np.maximum(np.linalg.norm(pts - c, axis=1) - s.radius, 0.0)
        else:
            a = r @ _AXES[s.axis]
            seg = _point_segment_distance(pts, c - a * s.length / 2, c + a * s.length / 2)
            d = np.maximum(seg - s.radius, 0.0)
        best = min(best, float(d.min()))
    return best


# ---------------------------------------------------------------------------------------------------------------
# spawning (Isaac Sim only)
# ---------------------------------------------------------------------------------------------------------------

_BODIES: dict[str, Body] = {}


def _author_body(prim_path: str, cfg, translation=None, orientation=None, **_):
    """Write the body into the USD stage under ``prim_path``.

    Without links: one Xform carrying the rigid body (when it has a mass) with one child prim per shape.
    With links: the Xform is a plain container; ``base`` (the articulation root) and every link are rigid bodies
    side by side under it, joined by revolute or prismatic joints."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

    from yam_arena.station import linear_rgb

    spec = _BODIES[cfg.body_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    t = translation if translation is not None else (0.0, 0.0, 0.0)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, t)))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(w), float(x), float(y), float(z)))

    materials: dict[tuple[float, float], str] = {}

    def physics_material(fr: tuple[float, float]) -> str:
        if fr not in materials:
            path = f"{prim_path}/Looks/phys{len(materials)}"
            m = sim_utils.RigidBodyMaterialCfg(static_friction=fr[0], dynamic_friction=fr[1], restitution=0.0,
                                               friction_combine_mode="average")
            m.func(path, m)
            materials[fr] = path
        return materials[fr]

    looks: dict[Color, str] = {}

    def look(color: Color) -> str:
        if color not in looks:
            path = f"{prim_path}/Looks/look{len(looks)}"
            v = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.7)
            v.func(path, v)
            looks[color] = path
        return looks[color]

    def textured(name: str, texture: str) -> UsdShade.Material:
        mpath = Sdf.Path(f"{prim_path}/Looks/{name}")
        mat = UsdShade.Material.Define(stage, mpath)
        pbr = UsdShade.Shader.Define(stage, mpath.AppendChild("pbr"))
        pbr.CreateIdAttr("UsdPreviewSurface")
        pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
        reader = UsdShade.Shader.Define(stage, mpath.AppendChild("st"))
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, mpath.AppendChild("tex"))
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        rgb = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb)
        mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
        return mat

    def place(xf, s: Shape, scale=None):
        xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*s.centre))
        qx, qy, qz, qw = s.rot_xyzw
        xf.AddOrientOp().Set(Gf.Quatf(float(qw), float(qx), float(qy), float(qz)))
        if scale is not None:
            xf.AddScaleOp().Set(Gf.Vec3f(*scale))

    def make_rigid(prim, mass: float, com=None, iterations: int = 16):
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(float(mass))
        if com is not None:
            mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*com))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(0.5)
        rb.CreateSolverPositionIterationCountAttr(int(iterations))
        rb.CreateSolverVelocityIterationCountAttr(2)
        rb.CreateAngularDampingAttr(float(spec.angular_damping))
        rb.CreateLinearDampingAttr(float(spec.linear_damping))

    def add_shapes(parent: str, shapes, tag: str):
        for i, s in enumerate(shapes):
            path = f"{parent}/{s.kind}_{tag}{i}"
            if s.kind == "box":
                g = UsdGeom.Cube.Define(stage, path)
                g.CreateSizeAttr(1.0)
                place(g, s, s.size)
            elif s.kind == "cylinder":
                g = UsdGeom.Cylinder.Define(stage, path)
                g.CreateAxisAttr(s.axis)
                g.CreateRadiusAttr(s.radius)
                g.CreateHeightAttr(s.length)
                place(g, s)
            elif s.kind == "capsule":
                g = UsdGeom.Capsule.Define(stage, path)
                g.CreateAxisAttr(s.axis)
                g.CreateRadiusAttr(s.radius)
                g.CreateHeightAttr(s.length)
                place(g, s)
            elif s.kind == "sphere":
                g = UsdGeom.Sphere.Define(stage, path)
                g.CreateRadiusAttr(s.radius)
                place(g, s)
            elif s.kind == "tile":
                g = UsdGeom.Mesh.Define(stage, path)
                hx, hy = s.size[0] / 2, s.size[1] / 2
                g.CreatePointsAttr([Gf.Vec3f(-hx, -hy, 0), Gf.Vec3f(hx, -hy, 0), Gf.Vec3f(hx, hy, 0),
                                    Gf.Vec3f(-hx, hy, 0)])
                g.CreateFaceVertexCountsAttr([4])
                g.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
                g.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
                g.SetNormalsInterpolation("vertex")
                g.CreateDoubleSidedAttr(False)
                u, v = s.size[0] / s.texture_m, s.size[1] / s.texture_m
                st = UsdGeom.PrimvarsAPI(g).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                          UsdGeom.Tokens.vertex)
                st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(u, 0), Gf.Vec2f(u, v), Gf.Vec2f(0, v)])
                place(g, s)
                UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(textured(f"tex_{tag}{i}", s.texture))
                continue
            else:
                raise ValueError(f"{spec.name}: unknown shape kind {s.kind!r}")
            gp = g.GetPrim()
            bind_visual_material(str(gp.GetPath()), look(s.color), stage=stage)
            if s.collide:
                UsdPhysics.CollisionAPI.Apply(gp)
                pc = PhysxSchema.PhysxCollisionAPI.Apply(gp)
                pc.CreateContactOffsetAttr(float(spec.contact_offset))
                pc.CreateRestOffsetAttr(0.0)
                bind_physics_material(str(gp.GetPath()), physics_material(tuple(s.friction or spec.friction)),
                                      stage=stage)

    def add_lights(parent: str):
        for i, lt in enumerate(spec.lights):
            light = UsdLux.SphereLight.Define(stage, f"{parent}/light{i}")
            light.CreateRadiusAttr(float(lt.get("radius", 0.025)))
            light.CreateIntensityAttr(float(lt["intensity"]))
            light.CreateColorAttr(Gf.Vec3f(*lt.get("color", (1.0, 0.85, 0.65))))
            light.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*lt["centre"]))

    if not spec.links:
        if spec.mass is not None:
            make_rigid(root.GetPrim(), spec.mass, spec.com, spec.solver_iterations)
        add_shapes(prim_path, spec.shapes, "")
        add_lights(prim_path)
        return root.GetPrim()

    # articulated: base and links as sibling rigid bodies under the container, placed at q = 0 (the joints read 0
    # there); the reset hook below writes each joint's start position q0
    poses = link_poses(spec, np.zeros(3), np.eye(3), {ln.name: 0.0 for ln in spec.links})
    base_path = f"{prim_path}/base"
    base = UsdGeom.Xform.Define(stage, base_path)
    base.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0))
    base.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
    make_rigid(base.GetPrim(), spec.mass, spec.com, spec.solver_iterations)
    add_shapes(base_path, spec.shapes, "")
    add_lights(base_path)
    for ln in spec.links:
        lp = f"{prim_path}/{ln.name}"
        p, R = poses[ln.name]
        q = quat_from_rot(R)
        xf = UsdGeom.Xform.Define(stage, lp)
        xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, p)))
        xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q[3], q[0], q[1], q[2]))
        make_rigid(xf.GetPrim(), ln.mass, ln.com, spec.solver_iterations)
        add_shapes(lp, ln.shapes, "")
        parent = base_path if ln.parent == "base" else f"{prim_path}/{ln.parent}"
        jp = f"{lp}/{ln.name}_joint"
        j = (UsdPhysics.RevoluteJoint if ln.joint == "revolute" else UsdPhysics.PrismaticJoint).Define(stage, jp)
        j.CreateAxisAttr(ln.axis)
        j.CreateBody0Rel().SetTargets([Sdf.Path(parent)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(lp)])
        # joint frame 0: at the anchor, parallel to the parent; frame 1: the link frame (parallel to the parent
        # at q = 0), so the joint reads q0 at spawn
        j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, ln.anchor)))
        j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))     # link origin: joint position 0 at spawn
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        if ln.limits is not None:
            lo, hi = ln.limits
            if ln.joint == "revolute":
                lo, hi = math.degrees(lo), math.degrees(hi)
            j.CreateLowerLimitAttr(float(lo))
            j.CreateUpperLimitAttr(float(hi))
        # a drive on every joint: its damping (and spring, if any) act through it; PhysX applies no gains to a
        # joint without one
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular" if ln.joint == "revolute" else "linear")
        drive.CreateTypeAttr("force")
        to_deg = math.pi / 180 if ln.joint == "revolute" else 1.0       # USD angular gains are per degree
        drive.CreateStiffnessAttr(float(ln.stiffness) * to_deg)
        drive.CreateDampingAttr(float(ln.damping) * to_deg)
        drive.CreateMaxForceAttr(float(ln.effort))
        drive.CreateTargetPositionAttr(float(math.degrees(ln.target) if ln.joint == "revolute" else ln.target))
        pj = PhysxSchema.PhysxJointAPI.Apply(j.GetPrim())
        pj.CreateJointFrictionAttr(float(ln.friction))
        if ln.armature:
            pj.CreateArmatureAttr(float(ln.armature))
    UsdPhysics.ArticulationRootAPI.Apply(base.GetPrim())
    art = PhysxSchema.PhysxArticulationAPI.Apply(base.GetPrim())
    art.CreateEnabledSelfCollisionsAttr(False)
    art.CreateSolverPositionIterationCountAttr(int(spec.solver_iterations))
    art.CreateSolverVelocityIterationCountAttr(2)
    return root.GetPrim()


def quat_from_rot(R: np.ndarray) -> tuple[float, float, float, float]:
    """(x, y, z, w) of a rotation matrix."""
    R = np.asarray(R, float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return (x, y, z, w)


def _arena_object(spec: Body):
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class TrailBodyCfg(SpawnerCfg):
        func: object = clone(_author_body)
        body_name: str = ""

    class TrailObject(Object):
        def _generate_articulation_cfg(self):
            # an explicit effort limit: without one the drive (and so the joint's damping and spring) gets none
            actuators = {ln.name: ImplicitActuatorCfg(joint_names_expr=[f"{ln.name}_joint"], stiffness=ln.stiffness,
                                                      damping=ln.damping, friction=ln.friction,
                                                      joint_effort_limit=ln.effort)
                         for ln in spec.links}
            cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators=actuators,
                                  init_state=ArticulationCfg.InitialStateCfg(
                                      joint_pos={f"{ln.name}_joint": ln.q0 for ln in spec.links},
                                      joint_vel={f"{ln.name}_joint": 0.0 for ln in spec.links}))
            return self._add_initial_pose_to_cfg(cfg)

    if spec.links:
        kind = ObjectType.ARTICULATION
    else:
        kind = ObjectType.RIGID if spec.mass is not None else ObjectType.BASE
    return TrailObject(name=spec.name, object_type=kind, spawner_cfg=TrailBodyCfg(body_name=spec.name),
                       initial_pose=Pose(position_xyz=tuple(spec.pos), rotation_xyzw=spec.spawn_xyzw))


def _np(x) -> np.ndarray:
    x = getattr(x, "torch", x)
    return x.detach().cpu().numpy().astype(np.float64)


def _read_state(environment, spec: Body) -> BodyState:
    if spec.mass is None and not spec.links:
        return state_at(spec, spec.pos, spec.spawn_xyzw)
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins[0])
    pose = _np(asset.data.root_link_pose_w)[0]
    vel = _np(asset.data.root_link_vel_w)[0] if hasattr(asset.data, "root_link_vel_w") else _np(asset.data.root_vel_w)[0]
    joints = {}
    if spec.links:
        names = list(asset.joint_names)
        q = _np(asset.data.joint_pos)[0]
        joints = {ln.name: float(q[names.index(f"{ln.name}_joint")]) for ln in spec.links}
    return state_at(spec, pose[:3] - origin, pose[3:7], vel[:3], vel[3:6], joints=joints)


def body_state(environment, name: str) -> BodyState:
    return _read_state(environment, _BODIES[name])


def write_body_pose(environment, name: str, pos, quat_xyzw, lin_vel=(0, 0, 0), ang_vel=(0, 0, 0)) -> None:
    """Teleport a free body (env 0); for resets of start poses the task-file format cannot express."""
    import torch

    base = environment.unwrapped
    asset = base.scene[name]
    origin = getattr(base.scene.env_origins, "torch", base.scene.env_origins)
    pose = torch.tensor([[*pos, *quat_xyzw]], dtype=origin.dtype, device=origin.device)
    pose[:, :3] += origin[0]
    asset.write_root_pose_to_sim_index(root_pose=pose)
    v = torch.tensor([[*lin_vel, *ang_vel]], dtype=origin.dtype, device=origin.device)
    asset.write_root_velocity_to_sim_index(root_velocity=v)


def reset_joints(environment) -> None:
    """Write every articulated body's start joint positions (``Link.q0``), at rest. Runs after each reset."""
    import torch

    base = environment.unwrapped
    for spec in _BODIES.values():
        if not spec.links:
            continue
        asset = base.scene[spec.name]
        names = list(asset.joint_names)
        q = torch.zeros((1, len(names)), dtype=torch.float32, device=base.device)
        for ln in spec.links:
            q[0, names.index(f"{ln.name}_joint")] = float(ln.q0)
        if hasattr(asset, "write_joint_position_to_sim_index"):
            asset.write_joint_position_to_sim_index(position=q)
            asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
        else:
            asset.write_joint_state_to_sim(q, torch.zeros_like(q))


# ---------------------------------------------------------------------------------------------------------------
# the robot as a policy-independent observer sees it
# ---------------------------------------------------------------------------------------------------------------

GRIPPER_LINKS = ("link5", "gripper", "tip_left", "tip_right")   # wrist joint, palm, both fingertips
ARM_CHAIN = ("link2", "link3", "link4", "link5")                 # the arm from the shoulder to the wrist, then the palm
ARM_SAMPLES = 8                                                  # points per segment, along the line between joints


def grippers(environment) -> dict[str, dict]:
    """Per arm: ``points`` (wrist joint, palm, both fingertip links, the tool point and the point between the pad
    tips; world, m) and ``opening`` (finger gap, m, both fingers summed)."""
    from yam_arena.constants import FINGER_JOINTS, PAD_TIP_PAST_TCP, TCP_OFFSET

    base = environment.unwrapped
    origin = _np(base.scene.env_origins[0])
    out = {}
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        poses = _np(robot.data.body_link_pose_w)[0]
        names = list(robot.body_names)
        pts = []
        missing = [link for link in GRIPPER_LINKS if link not in names]
        if missing:
            raise RuntimeError(f"{side} arm has no links {missing}: the release test would be incomplete")
        for link in GRIPPER_LINKS:
            pts.append(poses[names.index(link), :3] - origin)
        g = poses[names.index("gripper")]
        R = rot_from_xyzw(g[3:7])
        tcp = g[:3] - origin + R @ np.asarray(TCP_OFFSET)
        pts += [tcp, tcp + R[:, 2] * PAD_TIP_PAST_TCP]
        ids = [names_j for names_j in robot.joint_names]
        q = _np(robot.data.joint_pos)[0]
        opening = float(sum(abs(q[ids.index(j)]) for j in FINGER_JOINTS if j in ids))
        chain = [poses[names.index(link), :3] - origin for link in ARM_CHAIN if link in names] + [g[:3] - origin]
        arm = [a + (b - a) * t for a, b in zip(chain[:-1], chain[1:]) for t in np.linspace(0.0, 1.0, ARM_SAMPLES)]
        out[side] = {"points": np.array(pts), "arm": np.array(arm), "opening": opening}
    return out


def released(spec: Body, st: BodyState, grips: dict, clearance: float = 0.05) -> bool:
    """No gripper point, and no point along either arm (the line from the shoulder joint through each joint to the
    palm), within ``clearance`` of the body: a forearm resting on an object cannot hold it up."""
    return all(distance_to_body(spec, st, g["points"]) >= clearance
               and ("arm" not in g or distance_to_body(spec, st, g["arm"]) >= clearance) for g in grips.values())


# ---------------------------------------------------------------------------------------------------------------
# recorder
# ---------------------------------------------------------------------------------------------------------------

class Recorder:
    """Whole-trial memory, updated once per control step by the task's ``on_step`` callback.

    ``latch(key, cond)``: a flag that, once set, stays set until the next reset (fouls, milestones).
    ``extreme(key, value)``: the largest value seen. ``window(key, value, n)``: the last n values.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.flags: dict[str, bool] = {}
        self.why: dict[str, str] = {}
        self.extremes: dict[str, float] = {}
        self.windows: dict[str, list] = {}
        self.values: dict = {}
        self.steps = 0

    def latch(self, key: str, cond: bool, why: str = "") -> bool:
        if cond and not self.flags.get(key, False):
            self.flags[key] = True
            self.why[key] = f"step {self.steps}: {why}" if why else f"step {self.steps}"
        self.flags.setdefault(key, False)
        return self.flags[key]

    def flag(self, key: str) -> bool:
        return bool(self.flags.get(key, False))

    def extreme(self, key: str, value: float) -> float:
        self.extremes[key] = max(self.extremes.get(key, -np.inf), float(value))
        return self.extremes[key]

    def window(self, key: str, value, n: int) -> list:
        w = self.windows.setdefault(key, [])
        w.append(value)
        del w[:-n]
        return w

    def full(self, key: str, n: int) -> bool:
        return len(self.windows.get(key, [])) >= n


# ---------------------------------------------------------------------------------------------------------------
# hooks into yam_arena
# ---------------------------------------------------------------------------------------------------------------

_STEP_FNS: list = []
_RESET_FNS: list = []
_CURRENT: dict = {"env": None}


def on_step(fn):
    """Run ``fn(env)`` after every control step, before the task file is scored."""
    _STEP_FNS.append(fn)
    return fn


def on_reset(fn):
    """Run ``fn(env)`` after every reset, once the scene has settled. ``fn(None)`` is also fine to call offline."""
    _RESET_FNS.append(fn)
    return fn


def current_env():
    return _CURRENT["env"]


def _wrap_scene(module) -> None:
    if getattr(module, "_trailkit", False):
        return
    make_object, read_states = module.task_object, module.object_states

    def task_object(spec):
        if isinstance(spec, Body):
            _BODIES[spec.name] = spec
            return _arena_object(spec)
        return make_object(spec)

    def object_states(environment, names):
        ours = [n for n in names if n in _BODIES]
        theirs = [n for n in names if n not in _BODIES]
        out = read_states(environment, theirs) if theirs else {}
        out.update({n: _read_state(environment, _BODIES[n]) for n in ours})
        return {n: out[n] for n in names}

    module.task_object, module.object_states = task_object, object_states
    module._trailkit = True


def _wrap_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_trailkit", False):
        return
    score_step, do_reset = cls._task_step, cls.reset

    def _task_step(self, *a, **k):
        _CURRENT["env"] = self._env
        for fn in _STEP_FNS:
            fn(self._env)
        return score_step(self, *a, **k)

    def reset(self, *a, **k):
        obs = do_reset(self, *a, **k)
        _CURRENT["env"] = self._env
        if any(b.links for b in _BODIES.values()):
            reset_joints(self._env)
            sim = self._env.unwrapped.sim
            sim.step(render=False)
            self._env.unwrapped.scene.update(self._env.unwrapped.physics_dt)
            for _ in range(3):
                sim.render()
            obs = self._observation()
        for fn in _RESET_FNS:
            fn(self._env)
        return obs

    cls._task_step, cls.reset = _task_step, reset
    cls._trailkit = True


_WRAPPERS = {"yam_arena.scene": _wrap_scene, "yam_arena.inspect_embodiment": _wrap_embodiment}


class _PostImport(importlib.abc.MetaPathFinder):
    """Finds nothing itself: for the modules in ``_WRAPPERS`` it asks the other finders, then runs the wrapper
    right after the module has executed."""

    def find_spec(self, fullname, path, target=None):
        wrapper = _WRAPPERS.get(fullname)
        if wrapper is None:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                inner = spec.loader

                class _Loader(importlib.abc.Loader):
                    def create_module(self, s):
                        return inner.create_module(s)

                    def exec_module(self, module):
                        inner.exec_module(module)
                        wrapper(module)

                spec.loader = _Loader()
                return spec
        return None


def install() -> None:
    """Wrap yam_arena now where it is imported, and on import otherwise. Safe to call twice."""
    for name, wrapper in _WRAPPERS.items():
        if name in sys.modules:
            wrapper(sys.modules[name])
    if not any(isinstance(f, _PostImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _PostImport())


install()
