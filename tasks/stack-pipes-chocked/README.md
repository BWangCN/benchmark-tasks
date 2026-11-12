# Stack three smooth pipes on a narrow pad with the bottom pair chocked first

Three short steel pipe sections, greased, lie loose beside a grey pad 12 cm wide (three side by side, 15 cm, would overhang it; the
scorer tests the pipes' axes against the pad's outline, so a pipe whose axis is on the edge counts as on the pad).
Two rubber-soled steel chocks lie on the other side. The robot is told to stack the three pipes on the pad.

Three equal pipes in a pyramid spread on their own whenever the friction between them is below 2 - sqrt(3) = 0.27,
as between greased steel pipes. Two pipes laid side by side stand alone, so nothing looks wrong until the third is set
on them: its weight wedges them apart and the stack collapses. The bottom pair has to be blocked before the top pipe goes on (a chock along
each outer side, or one laid across both beyond the top pipe's end). The scoring looks only at where the pipes end.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the sheet and pad, the pipes and chocks; the whole-trial recorder; `score()` and `success()` |
| `rubble.py` | The pipe sections and blocks as rigid bodies |
| `trailkit.py` | Compound rigid bodies, textured tiles, the per-step recorder and scoring helpers; it hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena |
| `demo.py`, `demokit.py` | The scripted full-score demo and its near-miss variants |
| `textures/`, `textures.py` | The earth look, and the code that generated it |
| `LICENSE` | MIT |

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source <ARENA_DIR>/yam.env`), from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Stack the three pipes on the pad, lengthwise along it." \
  --scorer task_score --max-steps 2600 --store-frames --log-dir ~/yam-eval/pipes-dream
```

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 2600                  # full score
python demo.py task.py out --variant no_chocks --no-record       # stack them with no chocks
python demo.py task.py out --variant one_chock --no-record       # chock one side only
python demo.py task.py out --variant chocks_after --no-record    # set the top pipe, then both chocks
python demo.py task.py out --variant idle --no-record            # do nothing
```

1. The left arm lays the two chocks along the pad, their inner faces about 6.4 cm either side of its middle line.
2. The right arm lays a pipe between them and rolls it out against one chock (closed fingers on its side), then lays
   the second and rolls it against the other.
3. The right arm sets the third pipe into the groove between them and lets go.

Every piece is taken at its middle (the hand steered onto it by its own measured fingertip), carried, steered on its
measured position while held 6 mm above its place, lowered and let go. The script reads only the poses of the pipes
and chocks and the arms' own joint state.

| Variant | What it does | Score | Steps | IK misses (left / right), worst |
|---|---|---|---|---|
| (none), 3 runs | chocks first, the bottom pair rolled against them, the third pipe set on top | 1.0, 1.0, 1.0 (top pipe's axis 5.8 cm above the pad) | 1,790 | 22 / 0, 11.7 / 0.1 mm |
| `no_chocks` | the same with no chocks | 0.0 (the bottom pair rolls apart, 37 cm on one side; the top pipe falls off the sheet) | 1,087 | 0 / 220 (the right arm reaching after a pipe that rolled away) |
| `one_chock` | one chock only | 0.25 (two pipes on the pad, the third not on them) | 1,429 | 15 / 0, 23.7 mm |
| `chocks_after` | the top pipe set first, then both chocks | 0.0 (the stack spreads before the chocks are laid; the pipes still rolling at the end) | 1,680 | 182 / 220 (the arms reaching after pipes that rolled away) |
| `idle` | nothing | 0.0 | 670 | 0 / 0 |

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- **End state**, after 2 s with the three pipes at rest (no pipe's centre moved more than 2 mm over the last 2 s) and
  released (no point of either gripper, and no point along either arm from its shoulder joint to its palm, within
  5 cm of any pipe):
  - 1.0: the three pipes' axes wholly inside the pad's outline, all three lying along x (within about 18 degrees),
    two lying on the pad (axis 2.5 cm above it, within 4 mm) and the third resting on both (its axis between theirs
    along y, at least 4 cm above the pad, within 5.3 cm of each of theirs so that it touches both, its middle
    within 6 cm of theirs along x, and no chock within 3 mm under it along its length: it rests on the pipes, not on
    a chock);
  - 0.25: two pipes lying on the pad (along x within about 18 degrees, axes inside the pad's outline and 2.5 cm
    above it within 4 mm) and no full pyramid; a pipe or a column of pipes standing on end does not lie on the pad;
  - 0 otherwise.
- A full score ends the trial once it has held for 1 s.

On a real table: the end state by eye or from a photo.

## Runtime and seeds

A full 2,600-step Dream-1 trial takes about 8 minutes on an L40S, Isaac Sim's start included. The layout is fixed
(`-E object_jitter=0`); there are no seeds to choose. The demo reaches full score in 1,790 steps (90 s).

## Fidelity

Rigid bodies, 240 Hz physics, 20 Hz control. The pipes are solid cylinders with a thin-walled pipe's mass; the simulator has no rolling resistance, so a stacked pipe can keep turning in place and the rest test reads positions. The pyramid's spreading threshold (0.27) is the statics of equal cylinders; in the simulator a pyramid set down also spreads at 0.35, so rougher pipes are not tested.

Probe (gravity only, the pyramid spawned resting, three runs each): with no chock the bottom pair rolls 36 cm apart and
the top pipe ends on the pad; with one chock the unchocked pipe rolls away and the top pipe falls to the pad; with both chocks the stack settles and stands.

## Real-world build

- **Sheet and pad:** an earth-brown sheet of painted PVC foil 0.6 mm thick, x 10 to 44 cm, y -45 to 45; a grey pad
  2 mm thick, 22 x 12 cm, from x 14 to 36, y -6 to 6.
- **Pipes:** three steel tube sections 5 cm across, 12 cm long, about 0.25 kg each, wiped with grease.
- **Chocks:** two steel bars 12 x 4 x 4 cm (1.5 kg) with rubber soles.

## Sources

- OSHA 29 CFR 1926.250(b)(6), General requirements for storage: "Structural steel, poles, pipe, bar stock, and other
  cylindrical materials, unless racked, shall be stacked and blocked so as to prevent spreading or tilting."
- Plastics Pipe Institute, "Storing PE Pipe": block each row so that pipe cannot roll.
- Friction as run (the simulator averages the two faces' values): pipe on pipe 0.15, pipe on the sheet 0.33, pipe
  on a chock 0.53 (the chocks' 0.9 is their rubber soles' value, applied to all their faces), chock on the sheet 0.7.
- Static friction of greased steel on steel: 0.16 (MISUMI's table) and 0.11 (lubricated, ingdemurtas.it's table).

## License

MIT (`LICENSE`), the task authors. No third-party assets: all shapes and textures are generated by code in this
folder.
