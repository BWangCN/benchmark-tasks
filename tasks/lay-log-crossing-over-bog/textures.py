"""Write the tiling ground textures (turf, earth, sand, rock) as PNG files.

    python textures.py OUT_DIR        # writes turf.png, earth.png, sand.png, rock.png (256 x 256, tileable)

Procedural: value noise on a periodic lattice (so the images tile), mixed into each ground's palette, plus fine
speckle (grit, grass blades). Deterministic (fixed seeds). License: MIT (see LICENSE next to this file).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SIZE = 256

PALETTES = {
    # (dark, light, speckle) in sRGB 0..255
    "turf": ((44, 92, 34), (96, 150, 58), (130, 176, 80)),
    "earth": ((92, 66, 44), (140, 104, 70), (70, 50, 34)),
    "sand": ((196, 170, 122), (226, 204, 156), (170, 146, 104)),
    "rock": ((104, 102, 98), (150, 147, 140), (80, 78, 76)),
}


def periodic_noise(rng: np.random.Generator, cells: int, size: int = SIZE) -> np.ndarray:
    """Smooth value noise in 0..1 that wraps at the image edges."""
    lattice = rng.random((cells, cells))
    t = np.arange(size) * cells / size
    i0 = np.floor(t).astype(int)
    f = t - i0
    f = f * f * (3 - 2 * f)
    i1 = (i0 + 1) % cells
    a = lattice[np.ix_(i0, i0)]
    b = lattice[np.ix_(i0, i1)]
    c = lattice[np.ix_(i1, i0)]
    d = lattice[np.ix_(i1, i1)]
    fx, fy = f[None, :], f[:, None]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def texture(kind: str, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed + list(PALETTES).index(kind))
    dark, light, speck = (np.array(c, float) for c in PALETTES[kind])
    n = 0.55 * periodic_noise(rng, 4) + 0.30 * periodic_noise(rng, 16) + 0.15 * periodic_noise(rng, 64)
    img = dark + (light - dark) * n[..., None]
    density = {"turf": 0.12, "earth": 0.04, "sand": 0.06, "rock": 0.03}[kind]
    mask = rng.random((SIZE, SIZE)) < density
    if kind == "turf":
        # short blades: stretch the speckle a pixel or two along a random direction
        mask |= np.roll(mask, 1, axis=0) & (rng.random((SIZE, SIZE)) < 0.6)
    img[mask] = 0.5 * img[mask] + 0.5 * speck
    return np.clip(img, 0, 255).astype(np.uint8)


def write_all(out_dir) -> list[Path]:
    from PIL import Image

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for kind in PALETTES:
        p = out / f"{kind}.png"
        Image.fromarray(texture(kind)).save(p)
        paths.append(p)
    return paths


if __name__ == "__main__":
    for p in write_all(sys.argv[1] if len(sys.argv) > 1 else "textures"):
        print(p)
