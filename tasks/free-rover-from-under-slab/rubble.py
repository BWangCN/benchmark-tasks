"""Rubble props: a concrete slab, square concrete blocks and cribs, a short steel pipe
section lying on its side (a drum), a debris face with a steel plate, an L-shaped lump of broken block, and a point
sampler for "does any piece come near the rover" tests. Rigid bodies built with trailkit; nothing here touches
yam_arena directly.

Densities: concrete 2.4 g/cm3; broken cinder block 1.5 g/cm3; the drum a thin-walled steel pipe section (its mass
given, not computed).
"""
from __future__ import annotations

import math

import numpy as np

import trailkit as tk

CONCRETE = (0.62, 0.61, 0.58)
BLOCK_GREY = (0.55, 0.54, 0.51)
ROCK = (0.52, 0.50, 0.47)
STEEL = (0.45, 0.47, 0.50)
RUST = (0.42, 0.30, 0.22)
CONCRETE_FR = (0.8, 0.7)
STEEL_FR = (0.25, 0.25)          # smooth steel on steel: the drum and its plate


def slab(name: str, centre, size=(0.30, 0.16, 0.012), mass: float | None = None, friction=CONCRETE_FR) -> tk.Body:
    """A flat concrete slab, centred at ``centre`` (world), axis-aligned. Mass from concrete if not given."""
    m = mass if mass is not None else 2400 * size[0] * size[1] * size[2]
    return tk.body(name, pos=tuple(centre), shapes=[tk.Shape.box((0, 0, 0), size, CONCRETE, friction=friction)],
                   mass=m)


def block(name: str, centre, size, color=BLOCK_GREY, mass: float | None = None, friction=CONCRETE_FR) -> tk.Body:
    """A square concrete block or crib standing at ``centre`` (world)."""
    m = mass if mass is not None else 2400 * size[0] * size[1] * size[2]
    return tk.body(name, pos=tuple(centre), shapes=[tk.Shape.box((0, 0, 0), size, color, friction=friction)], mass=m)


def drum(name: str, centre, radius=0.05, length=0.07, mass=0.6, friction=STEEL_FR) -> tk.Body:
    """A short steel pipe section lying on its side, its axis along y, centred at ``centre`` (world)."""
    q = tk.quat_xyzw_from_axis_angle((1, 0, 0), math.pi / 2)        # cylinder axis Z -> along y
    return tk.body(name, pos=tuple(centre), rot_xyzw=q,
                   shapes=[tk.Shape.cylinder((0, 0, 0), radius, length, axis="Z", color=STEEL, friction=friction)],
                   mass=mass)


def debris_face_shapes(contact, slope_deg: float, length=0.06, width=0.08, thickness=0.02,
                       plate_friction=STEEL_FR) -> list:
    """Shapes (in the frame whose origin is the ground under the pile) of a steel plate on a debris face falling
    toward +x at ``slope_deg``, its top face through ``contact`` (that frame), with a fill under its high end."""
    phi = math.radians(slope_deg)
    n = np.array([math.sin(phi), 0.0, math.cos(phi)])
    c = np.asarray(contact, float) - n * thickness / 2
    q = tk.quat_xyzw_from_axis_angle((0, 1, 0), phi)
    return [tk.Shape.box(tuple(c), (length, width, thickness), RUST, rot_xyzw=q, friction=plate_friction, tag="plate"),
            tk.Shape.span((c[0] - 0.025, -width / 2, 0.0), (c[0] + 0.03, width / 2, max(0.002, contact[2] - 0.012)),
                          ROCK, friction=CONCRETE_FR, tag="debris")]


def l_rock(name: str, origin, a_centre, a_size, b_centre, b_size, mass: float | None = None,
           friction=CONCRETE_FR) -> tk.Body:
    """An L-shaped lump of broken cinder block: two boxes (``a`` and ``b``) given in the frame at ``origin``."""
    m = mass if mass is not None else 1500 * (np.prod(a_size) + np.prod(b_size))
    return tk.body(name, pos=tuple(origin), shapes=[tk.Shape.box(tuple(a_centre), tuple(a_size), ROCK, friction=friction),
                                                    tk.Shape.box(tuple(b_centre), tuple(b_size), ROCK, friction=friction)],
                   mass=m)


def sample_points(spec: tk.Body, st: tk.BodyState, step: float = 0.015) -> np.ndarray:
    """Points over the faces of a body's boxes and over its cylinders (world), about ``step`` apart: for distance
    tests against another body."""
    pts = []
    for s in spec.shapes:
        if not s.collide:
            continue
        c, r = np.asarray(s.centre, float), s.rotation()
        if s.kind == "box":
            h = np.asarray(s.size) / 2
            g = [np.linspace(-h[i], h[i], max(2, int(2 * h[i] / step) + 1)) for i in range(3)]
            for ax in range(3):
                for sign in (-1, 1):
                    o = [i for i in range(3) if i != ax]
                    u, v = np.meshgrid(g[o[0]], g[o[1]])
                    p = np.zeros((u.size, 3))
                    p[:, ax] = sign * h[ax]
                    p[:, o[0]], p[:, o[1]] = u.ravel(), v.ravel()
                    pts.append(c + p @ r.T)
        else:
            pts.append(s.surface_points(16))
    local = np.vstack(pts)
    R = st.rotation()
    return st.pos + local @ R.T


def pipe(name: str, centre, radius=0.02, length=0.18, axis: str = "x", mass=0.7, friction=(0.5, 0.5),
         color=STEEL) -> tk.Body:
    """A pipe section (a solid cylinder standing in for it; its mass given) lying with its axis along ``axis``
    ("x" or "y"), centred at ``centre`` (world)."""
    q = (tk.quat_xyzw_from_axis_angle((0, 1, 0), math.pi / 2) if axis == "x"
         else tk.quat_xyzw_from_axis_angle((1, 0, 0), math.pi / 2))
    return tk.body(name, pos=tuple(centre), rot_xyzw=q,
                   shapes=[tk.Shape.cylinder((0, 0, 0), radius, length, axis="Z", color=color, friction=friction)],
                   mass=mass)
