# Find by touch and take out the one prop under a slab that carries nothing

A concrete slab with a lead plate cast into it (3.1 kg, its weight off its middle) stands on four props, one under each corner, each half under the slab
and half sticking out past its end. One prop is 2 mm short: the slab rests on the other three. The gap is under the
slab's edge, and the slab is turned at every reset so that its heavy end lies away from the short prop: it looks the
same any way round, so nothing in view tells which prop is short. The robot is told to take out the prop that carries nothing, put it on the pad, and leave the slab as it is.

A prop that carries the slab is pinned by its share of the weight (about 10 N, so about 19 N of friction above and
below it) and drags the slab when it is pulled. The spare one weighs 0.38 kg and slides at a touch. The short corner
is drawn at random every trial, so it has to be found by nudging each prop gently. Pressing the slab's corners to feel
for a rock, the way a short foot under a rocking base is found, does not work here: the slab stands steady on three,
and a press hard enough to rock it moves a corner more than the 1 mm the scoring allows. The scoring looks only at where the slab and the props
are.

## Files

| File | What it is |
|---|---|
| `task.py` | The yam_arena task file: the sheet and pad, the slab and props; the random corner at each reset; the whole-trial recorder; `score()` and `success()` |
| `demo.py`, `demokit.py` | The scripted full-score demo and its wrong-method variants |
| `trailkit.py`, `rubble.py`, `textures.py`, `textures/` | Scene building, state reading and the props |

## Run

```bash
inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 \
  --instruction "Take out the prop that carries nothing and put it on the pad. Leave the slab as it is." \
  --scorer task_score --max-steps 2400
```

```bash
python demo.py task.py demo_out --max-steps 2400                    # full score
SPARE_CORNER=1 python demo.py task.py demo_out --max-steps 2400     # the short prop at corner 1 (0 to 3)
python demo.py task.py out --variant take_nearest --no-record       # take the right arm's nearest prop untested
python demo.py task.py out --variant press_corners --no-record      # press each slab corner, then take the nearest
python demo.py task.py out --variant idle --no-record               # do nothing
```

The short prop's corner is drawn at random at every reset (`SPARE_SEED` seeds the draw; unset, it is unseeded);
`SPARE_CORNER` (0 to 3) fixes it for a run.
Corners: 0 near-right, 1 far-right, 2 far-left, 3 near-left ("near" is toward the robot).

## Demo and variants

| Variant | What it does | Score | Steps | IK misses (left / right) |
|---|---|---|---|---|
| (none), 4 runs (the short prop at corners 0, 1, 3 and 2) | nudge each prop in turn until one slides; take it out along y, carry it to the pad | 1.0 each (slab corners within 0.03 mm) | 589, 695, 798, 953 | 0 / 0 |
| `take_nearest` (short prop at corner 2) | take the right arm's nearest prop without testing | 0.0 (it carries the slab: the slab dragged 4.1 mm at step 128, then came down) | 510 | 0 / 0 |
| `press_corners` (short prop at corner 2) | press each slab corner 4 mm to feel for a rock, then take the nearest prop | 0.0 (the presses moved no corner 1 mm, though the left arm missed some of its far-left targets; the prop taken carries the slab: dragged 3.5 mm at step 592) | 974 | 69 / 0 |
| `idle` | nothing | 0.0 | 670 | 0 / 0 |

Dream-1, one run of 2,400 steps: 0.0.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- **Foul, at any time** (the trial then scores 0): a corner of the slab's underside more than 1 mm up or down from
  its height on the props, or the slab's middle more than 3 mm from its place along the ground.
- **At the end**, after 2 s with the short prop and the slab at rest (their middles moved under 2 mm over the last
  2 s) and released (no point of either gripper, and no point along either arm from its shoulder joint to its palm,
  within 5 cm of either):
  - 1.0: the short prop on the pad (every corner over the pad, its lowest corner within 5 mm of the pad's top);
  - 0.5: the short prop out from under the slab (no corner of it, and not its middle, inside the slab's outline
    seen from above);
  - 0 otherwise.
- A full score ends the trial once it has held for 1 s.

## Premise probe

Run before the build, gravity and one arm (the slab then carried a separate 1.5 kg block in place of the cast-in plate, the
same weight in the same place):

- Nudged with the same command (closed fingers aimed 5 mm into the prop for 1.5 s), three layouts: the short prop
  moved 3.7, 4.3 and 4.1 mm; the carrying props 0.15 to 0.35 mm; the slab 0.04 mm at most.
- The short prop drawn out along y and lifted, at three corners: 68 to 71 mm out, every slab corner within 0.01 mm.
- A carrying prop left out of the scene: two slab corners 2.1 mm down. A carrying prop pulled (props laid along x in
  an earlier layout): the slab dragged 12 mm.

## Fidelity

Rigid bodies, 240 Hz physics, 20 Hz control. Concrete friction 0.8 static, 0.7 dynamic on every face. The 2 mm gap
is a bench-scale stand-in for a loose crib; a real one may be loose by a fraction of a millimetre and is found the
same way. The demo's nudge is a fixed hand offset held 1.5 s, not a measured force. The slab's off-centre weight is a set centre
of mass.

On a real table: the slab's corners against a dial gauge or a ruler at a mark, the end state from a photo.

## Sources

- U.S. Army Corps of Engineers, Urban Search and Rescue Program, *Shoring Operations Guide*, Edition 5.0
  (October 2021).
- Fire Engineering, "Timber cribbing use": check that each crib stays tight as the load comes on.
