# Bring a free-rolling rover down a ledge one axle at a time

An earth track runs away from the robot. Its near half is a grey rock deck 3.8 cm high, its far half a lower track at
table level. A small trail crawler (a real 1:18 crawler, 26 x 12 x 12 cm, 0.5 kg, driveshafts removed so the wheels
roll freely) stands on the deck facing the edge. A short tow bar sticks out from the middle of each bumper.
The robot is told to get the rover down onto the lower track without letting it drop or lifting it clear.

A vehicle is brought down a step one axle at a time: it is held at a fixed point and each axle is lowered in turn,
never rolled off the edge. Attendants take a wheelchair down a high kerb this way (the assisted curb descent of the
Wheelchair Skills Program; O*NET 31-1122.00, Personal Care Aides), and trail crews bring a vehicle down a ledge the same way on a held
line. The step here is taller than the wheels' radius: rolled off, an axle pivots over the edge and then falls,
reaching the lower track a tenth of a second after it leaves the deck's height. Held by the front
bar with its nose down, the rover sits on free wheels and is easily pushed or pulled off, so the other hand takes the
rear before the first lets go.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: deck, lower track, the rolling rover with its tow bars; the whole-trial recorder; `score()` and `success()` |
| `trailkit.py` | Compound rigid bodies, articulated links (the rover's wheels), textured tiles, the per-step recorder and scoring helpers; it hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena |
| `rover.py`, `rover_spec.py` | The rover with free-rolling wheels, and every number of it with its source |
| `demo.py`, `demokit.py` | The scripted full-score demo and its near-miss variants |
| `textures/`, `textures.py` | The earth look, and the code that generated it |
| `LICENSE` | MIT |

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source <ARENA_DIR>/yam.env`, as set up by the
walkthrough), from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Get the rover down off the ledge onto the lower track without letting it drop or lifting it clear." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/ledge-descent-dream
```

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 2400                               # full score
python demo.py task.py out --variant push_off --no-record                      # push it off the edge
python demo.py task.py out --variant lift_down --no-record                     # carry it down by both bars
python demo.py task.py out --variant front_then_push --no-record --max-steps 2400   # lower the front, push the rear off
python demo.py task.py out --variant hold_then_drop --no-record   # hold the front 6 mm under deck height, let go
python demo.py task.py out --variant idle --no-record                          # do nothing
```

1. The right arm reaches across the front tow bar from the right, tilted 17 degrees down, and closes on its top and
   bottom. It rolls the rover forward on its wheels until the front axle is just past the edge, and lowers the nose
   onto the lower track, turning the hand with the rover's pitch (closed loop on the rover's measured pose).
2. The left arm takes the rear bar from above before the right lets go.
3. The left arm rolls the rover on until the rear axle is past the edge, lowers it, opens and backs away.

The demo reads only the rover's pose and the arms' own joint state.

| Variant | What it does | Score |
|---|---|---|
| (none), 3 runs | both axles lowered in turn, with the hand-over | 1.0, 1.0, 1.0 |
| `hold_then_drop` | carry the front axle past the edge, hold it 6 mm under deck height for 1 s, let go | 0.0 (a front tyre falls 1.4 cm in 3 control steps) |
| `push_off` | push the rover off the edge by its rear bumper | 0.0 (a front tyre falls 2.9 cm in 3 control steps) |
| `lift_down` | carry it down by both bars | 0.0 (lifted clear) |
| `front_then_push` | lower the front properly, then push the rear off | 0.5 (a rear tyre falls 1.8 cm in 3 control steps) |
| `idle` | nothing | 0.0 |

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- A tyre has **dropped** when, past the deck's edge, its centre falls more than 1.2 cm over 3 control steps (0.15 s,
  8 cm/s on average; lowered by a hand it falls 0.2 to 0.5 cm), or when it comes down the step in under 0.6 s
  (12 control steps at 20 Hz), timed for each tyre on its own from its centre being past the deck's edge and 3 mm below
  its height rolling on the deck until it rests on the lower track (lowest point within 1 cm of it), about 2.5 cm.
  Rolled or pushed off the edge, a tyre swings and falls down in 2 control steps; lowered by a hand it takes 1.7 to
  3 s. A tyre counts as resting on the track until it is lifted 1.5 cm off it again for 3 control steps (a bounce is
  shorter); lifted again, it is timed afresh. An axle has **dropped** when either of its tyres has.
- **Fouls, at any time**, the trial then scoring 0: the rover lifted on the deck and let fall back onto it (a tyre's
  lowest point falling more than 1 cm in one control step, or 1.5 cm over three, onto the deck); a tyre going off the
  deck's side or near end (its lowest point 1 cm below the deck's top while short of the deck's edge); all four tyres more than 8 mm above the surfaces under them (deck
  top or table) for 0.5 s, or more than 1.5 cm above them for 3 control steps in a row (0.15 s; a bounce is shorter) (lifted clear); the rover tilting more
  than 30 degrees; an axle dropping before either axle rests on the lower track; the first axle to come down there
  having dropped; the second axle coming down less than 0.5 s after the first (carried down whole).
- **End state**, with the rover at rest (under 1 cm/s and 0.5 rad/s for the last 20 control steps, 1 s at 20 Hz) and
  released (none of six points on each arm, its wrist joint, palm, both fingertips, tool point and the point between
  the pad tips, and no point along the arm from its shoulder joint to its palm, within 5 cm of the rover), all four tyres on the lower track (lowest point within 1 cm of it, inside
  its outline) and the rover within 10 degrees of level; otherwise 0:
  - 1.0: neither axle dropped;
  - 0.5: the second axle dropped.
- A full score ends the trial once it has held for 1 s.

On a real table: each axle's descent is timed from a side video at 20 frames per second; the end state with a photo.

## Runtime and seeds

A full 2,400-step Dream-1 trial takes about 7 minutes on an L40S, Isaac Sim's start included. The layout is fixed
(`-E object_jitter=0`); there are no seeds to choose. The demo reaches full score in 575 steps (29 s).

## Fidelity

Rigid bodies at 240 Hz physics, 20 Hz control. The rover's base is one rigid body at the crawler's published size and
mass (tyre size and centre of mass estimated from photographs, see `rover_spec.py`); its four wheels turn freely on
revolute joints with a little drag (0.002 N m s/rad), as with the driveshafts removed; its suspension is rigid, as with the shocks locked.
Friction: tyres 1.0, tow bars 1.0, deck and track 0.8. A drop is judged from the tyres' fall per control step; no
impact forces or damage are simulated.

## Real-world build

- **Deck:** a 30 x 34 cm plywood deck on battens, 3.8 cm tall, clamped to the table, painted rock grey all over. **Lower track:** an earth-brown sheet of painted PVC foil 0.6 mm thick, taped to the table.
- **Rover:** a stock 1:18 trail crawler (Traxxas TRX-4M, Bronco body), decals removed, painted yellow with a white
  roof; dummy battery; driveshafts removed (each wheel rolls freely on its own) and shocks locked. No electrics are live.
- **Tow bars:** two dark grey 3D-printed PLA bars 4.8 x 1.6 x 2.0 cm, bolted to the middle of each bumper's face.

## License

MIT (`LICENSE`), the task authors. No third-party assets: all shapes and textures are generated by code in this
folder.
