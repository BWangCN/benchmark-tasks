"""demokit: closed-loop scripted demos for yam_arena task files.

    python demo.py task.py OUT_DIR [--variant NAME] [--max-steps N] [--no-record] [--debug]

A task's ``demo.py`` defines ``script(task, ctx, variant)``: a generator that sets arm targets and yields once per
control step. It may read only what a policy could observe on a real rig: object poses (as a camera-based pose
estimate would give them) through ``ctx.states()``, and the robot's own joint state. ``run()`` plays it in the
standard ``yam_arena`` embodiment with the task's own scorer, records the three cameras to MP4, saves the first
top-camera frame as ``setup.png`` and writes ``result.json``.

License: MIT (see LICENSE next to this file).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

CONTROL_HZ = 20
TABLE = 0.7475                 # station table top (yam_arena.task_file.TABLE_TOP["station"])
TRAVEL_CEILING = TABLE + 0.18  # highest tool point used for travel moves; the arms reach poorly above it


def tool_rotation(approach, close) -> np.ndarray:
    """Tool (TCP) rotation whose z axis points along ``approach`` and whose fingers close along ``close``
    (the tool's x axis). Both are world directions; ``close`` is made orthogonal to ``approach``."""
    z = np.asarray(approach, float)
    z /= np.linalg.norm(z)
    x = np.asarray(close, float) - z * float(np.dot(close, z))
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


DOWN_Y = tool_rotation((0, 0, -1), (0, 1, 0))   # pointing down, fingers closing along world y
DOWN_X = tool_rotation((0, 0, -1), (1, 0, 0))   # pointing down, fingers closing along world x


class Arm:
    """One arm's command: Cartesian moves solved by IK, warm-started from the last command."""

    def __init__(self, side: str):
        from yam_arena import ik, station
        from yam_arena.constants import READY

        self.side = side
        self.base = np.asarray(station.BASES[side], float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.home = (self.pos.copy(), self.rot.copy())
        self.worst_mm = 0.0
        self.worst_deg = 0.0
        self.misses: list = []   # (mm, deg, target) of IK solutions that missed by > 5 mm or 6 deg

    def solve(self, pos, rot, iters: int = 60):
        from yam_arena import ik

        q, pe, re = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base,
                             iters=iters)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), np.asarray(rot, float)
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.worst_deg = max(self.worst_deg, math.degrees(re))
        if pe > 0.005 or re > 0.1:
            self.misses.append((round(pe * 1000, 1), round(math.degrees(re), 1), tuple(np.round(pos, 3))))
        return pe, re

    @property
    def command(self) -> np.ndarray:
        return np.r_[self.q, self.grip]

    def move(self, pos, rot=None, seconds: float = 1.0, grip: float | None = None):
        """Generator: straight-line move of the tool point with smoothstep timing and slerped rotation."""
        from yam_arena import ik

        rot = self.rot if rot is None else np.asarray(rot, float)
        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else float(grip)
        n = max(1, round(seconds * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self.solve(p0 + s * (np.asarray(pos, float) - p0), ik.slerp_matrix(r0, rot, s))
            yield

    def via(self, pos, rot=None, seconds: float = 2.0, lift: float = 0.06, grip: float | None = None,
            ceiling: float = TRAVEL_CEILING):
        """Generator: up by ``lift`` (no higher than ``ceiling``), across above the goal, then down to it."""
        rot = self.rot if rot is None else rot
        up = min(max(self.pos[2], float(pos[2])) + lift, ceiling)
        up = max(up, float(pos[2]), min(self.pos[2], ceiling))
        if up > self.pos[2] + 1e-3:
            yield from self.move((*self.pos[:2], up), None, seconds * 0.3, grip)
        else:
            yield from self.set_grip(self.grip if grip is None else grip, 0.05)
        yield from self.move((*np.asarray(pos)[:2], up), rot, seconds * 0.4)
        yield from self.move(pos, rot, seconds * 0.3)

    def grip_rotation(self, pos, approach, close) -> np.ndarray:
        """The tool rotation pointing along ``approach`` that closes along ``close`` or ``-close`` (the same grip for
        symmetric fingers), whichever the arm reaches at ``pos`` with its wrist's last joint further from its limits."""
        from yam_arena import ik

        best = None
        for c in (np.asarray(close, float), -np.asarray(close, float)):
            rot = tool_rotation(approach, c)
            q, pe, re = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base,
                                 iters=120)
            margin = min(q[5] - ik.LOWER[5], ik.UPPER[5] - q[5])
            key = (pe > 0.005 or re > 0.1, -margin)
            if best is None or key < best[0]:
                best = (key, rot)
        return best[1]

    def transit(self, pos, rot=None, seconds: float = 3.0, height: float = 0.12, grip: float | None = None):
        """Generator: down (or up) to ``height`` above the table where the arm is, across at that height while turning
        to ``rot``, then straight to the goal. Unlike ``via`` it never travels at the home height, where pointing the
        tool down near the robot is out of reach."""
        rot = self.rot if rot is None else np.asarray(rot, float)
        h = TABLE + height
        h = max(h, float(pos[2]))
        yield from self.move((*self.pos[:2], h), None, seconds * 0.3, grip)
        yield from self.move((*np.asarray(pos)[:2], h), rot, seconds * 0.45)
        yield from self.move(pos, rot, seconds * 0.25)

    def set_grip(self, grip: float, seconds: float = 0.6):
        yield from self.move(self.pos, self.rot, seconds, grip)

    def aim(self, pos, rot=None):
        """Track a target this step (closed-loop phases)."""
        self.solve(pos, self.rot if rot is None else rot)

    def go_home(self, seconds: float = 2.0):
        """Generator: a little up, in toward the home point at a middle height, then up to home."""
        mid = (*self.home[0][:2], min(max(self.pos[2] + 0.03, TABLE + 0.15), TRAVEL_CEILING))
        yield from self.move(self.pos + (0, 0, min(0.03, max(0.0, TRAVEL_CEILING - self.pos[2]))), None, seconds * 0.2)
        yield from self.move(mid, self.home[1], seconds * 0.5)
        yield from self.move(self.home[0], self.home[1], seconds * 0.3)


def smooth(s: float) -> float:
    """Smoothstep from 0 to 1."""
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def together(*gens):
    """Run generators in lockstep until all are done (two arms moving at once)."""
    gens = list(gens)
    while gens:
        alive = []
        for g in gens:
            try:
                next(g)
                alive.append(g)
            except StopIteration:
                pass
        gens = alive
        if gens:
            yield


class Ctx:
    def __init__(self, emb):
        self.emb = emb
        self.left = Arm("left")
        self.right = Arm("right")
        self.notes: list[str] = []
        self.step = 0

    def states(self):
        from yam_arena.scene import object_states

        return object_states(self.emb._env, self.emb.task.names)

    def note(self, text: str) -> None:
        self.notes.append(f"step {self.step}: {text}")
        print(f"[demo] step {self.step}: {text}", flush=True)

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.command, self.right.command))


def _mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264", quality=8,
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, script, variant: str = "", max_steps: int = 1200, record: bool = True,
        cam_width: int = 320, cam_height: int = 176, hold_s: float = 3.0, debug=None, debug_every: int = 20) -> dict:
    """Play ``script(task, ctx, variant)`` and score it with the task file's own scorer. The episode runs until the
    task reports success, the run limit, or ``hold_s`` after the script ends (so the final state is scored at
    rest)."""
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    t0 = time.time()
    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    obs = emb.reset(Scene(id="demo", instruction="demo"))
    task = emb.task.module
    ctx = Ctx(emb)
    gen = script(task, ctx, variant)
    frames: dict[str, list] = {}

    def keep(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    keep(obs)
    steps, ended, success, info = 0, None, False, {}
    while steps < max_steps:
        if ended is None:
            try:
                next(gen)
            except StopIteration:
                ended = steps
                ctx.note("script done")
        res = emb.step(ctx.action())
        steps += 1
        ctx.step = steps
        info = res.info or {}
        keep(res.observation)
        if debug is not None and (steps % debug_every == 0):
            print(f"[demo] step {steps} score {info.get('score')} | {debug(task, ctx)}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "variant": variant,
              "script_steps": ended, "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2),
                                                     "right": round(ctx.right.worst_mm, 2)},
              "ik_misses": {"left": len(ctx.left.misses), "right": len(ctx.right.misses),
                            "first": (ctx.left.misses + ctx.right.misses)[:6]},
              "notes": ctx.notes, "wall_s": round(time.time() - t0, 1)}
    if hasattr(task, "report"):
        result["report"] = task.report()
    if record:
        import imageio

        for cam, fs in frames.items():
            _mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), None)
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    name = f"result_{variant}.json" if variant else "result.json"
    (out / name).write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, default=str), flush=True)
    return result


def main(script, debug=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--variant", default="")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--cam-width", type=int, default=320)
    ap.add_argument("--cam-height", type=int, default=176)
    a = ap.parse_args()
    run(a.task, a.out, script, a.variant, a.max_steps, record=not a.no_record, cam_width=a.cam_width,
        cam_height=a.cam_height, debug=debug if a.debug else None)
