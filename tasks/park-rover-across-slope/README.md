# Park a free-rolling rover across a slope before taking its stop block away

A small trail crawler (a real 1:18 crawler, 26 x 12 x 12 cm, 0.5 kg, its driveshafts removed so its wheels roll
freely) stands nose uphill on a board tilted 10 degrees, held only by a red stop block against the downhill side of
its left front tyre. A short tow bar sticks out from its rear bumper. There are no chocks. The robot is told to take
the stop block away and leave the rover parked on the slope.

A vehicle with free wheels left on a slope rolls away along its wheels unless it stands across the fall line, wheels
square to the slope. A disabled vehicle or a trailer is turned across the grade before it is released. Taken away as
the rover stands, the stop block lets it roll off the board. One hand has to hold the rover by its tail while the
other takes the block away. Then the rover is swung about its front axle until it stands square across the slope:
with free wheels, a few degrees off the contour it still creeps away (about 4 mm/s at 3 degrees, 9 mm/s at 6). The
scoring counts the rover parked only if it creeps less than 5 mm in 3 s, which needs its heading within about 1.1
degrees of the contour. Holding it some other way while the block is taken away (for instance lifting it whole) is
equally accepted; the scoring asks only for the end state and that it never rolled off.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the slope, the rolling rover with its tow bar, the stop block; the whole-trial recorder; `score()` and `success()` |
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
  --instruction "Take the stop block away from the rover's wheel and leave the rover parked on the slope." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/park-across-dream
```

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200                     # full score
python demo.py task.py out --variant remove_first --no-record         # take the stop block away as the rover stands
python demo.py task.py out --variant off_contour --no-record          # park it 6 degrees off the contour
python demo.py task.py out --variant idle --no-record                 # do nothing
```

1. The right arm takes the rear tow bar from above and lifts the tail 1.5 cm off the slope: it now holds the rover.
2. The left arm takes the stop block off the slope by its outer end and sets it on the table.
3. The right arm swings the rover 90 degrees about its front axle (the front wheels roll as it turns) until it stands
   across the slope, trims its heading to within 0.15 degrees (closed loop on its measured pose), sets the tail
   down and lets go.

The demo reads only object poses and the arms' own joint state.

| Variant | What it does | Score |
|---|---|---|
| (none) | hold the tail, take the block away, swing the rover across | 1.0 (heading 0.3 degrees off the contour; 2.5 mm in the scored 3 s) |
| `remove_first` | take the stop block away as the rover stands | 0.0 (the rover rolls off the board, step 117) |
| `off_contour` | swing it across but set it down 6 degrees off the contour | 0.0 (it creeps away and rolls off the board, step 515) |
| `idle` | nothing | 0.0 |

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- **Foul, at any time**, the trial then scoring 0: the lowest point of a tyre leaving the board's outline (the rover
  rolled off).
- **End state**, with the rover still (its origin moving less than 5 mm over the last 3 s, 60 control steps at
  20 Hz, a creep under 1.7 mm/s; the rover and the stop block turning under 0.5 rad/s, the block under 1 cm/s) and
  released over those 3 s (none of six points on each arm, its wrist joint, palm, both fingertips, tool point and the
  point between the pad tips, and no point along the arm from its shoulder joint to its palm, within 5 cm of the rover
  or the stop block):
  - 1.0: all four tyres rest on the board (lowest point within 8 mm of its surface, inside its outline) and every
    point of the stop block's surface is at least 3 cm from the rover (tyres, body and tow bar), wherever it is put;
  - 0 otherwise.
- A full score ends the trial once it has held for 1 s.

On a real table: rolling off is plain to see; the end state with a photo.

## Runtime and seeds

A full 1,200-step Dream-1 trial takes about 4 minutes on an L40S, Isaac Sim's start included. The layout is fixed
(`-E object_jitter=0`); there are no seeds to choose. The demo reaches full score in 538 steps (27 s).

## Fidelity

Rigid bodies at 240 Hz physics, 20 Hz control. The rover's base is one rigid body at the crawler's published size and
mass (tyre size and centre of mass estimated from photographs, see `rover_spec.py`); its four wheels turn freely on
revolute joints with a little drag (0.002 N m s/rad) and no static friction, as with the driveshafts removed. So even a
few degrees off the contour the simulated rover rolls slowly away; a real crawler's bearings hold a little, which makes
the real task slightly easier. A simulated rover parked within the 1.1-degree limit still creeps (under 1.7 mm/s) and
would reach the board's side edge in about half a minute; the trial ends 1 s after the full score has held. Its
suspension is rigid. The stop block is rubber-soled aluminium (friction 0.9). Rolling off
is judged from the tyres' positions. No forces or damage are simulated.

## Real-world build

- **Slope:** a plywood board 40 x 40 x 2 cm on a pine wedge frame at 10 degrees, clamped to the table, top painted earth
  brown.
- **Rover:** a stock 1:18 trail crawler (Traxxas TRX-4M), body painted yellow with a white roof; dummy battery;
  driveshafts removed. No electrics are live.
- **Tow bar:** a dark grey 3D-printed PLA bar 4.8 x 1.6 x 2.0 cm bolted to the middle of the rear bumper's face.
- **Stop block:** a red aluminium block 3 x 8 x 3 cm with a rubber sole, 0.19 kg.

## License

MIT (`LICENSE`), the task authors. No third-party assets: all shapes and textures are generated by code in this
folder.
