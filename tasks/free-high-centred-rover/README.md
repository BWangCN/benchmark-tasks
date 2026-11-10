# Free a rover high-centred on a rock without dragging or dropping it

A small trail crawler (a real 1:18 crawler, 26 x 12 x 12 cm, 0.5 kg, its driveshafts removed so each wheel rolls
freely) is high-centred on a flat grey rock on an earth track: its chassis rests on the rock and all four tyres hang
2.5 cm off the ground. The rock sticks out 4 cm beyond the rover's right side; a short tow bar sticks out from the
middle of the rear bumper. The robot is told to get the rock out and set the rover back on its wheels.

A high-centred vehicle is freed by taking its weight off the obstacle first: recovery crews jack or lift the vehicle,
remove the rock, and lower the vehicle onto its wheels. Pulled from under the full weight, the rock drags the rover
along with it, and once the rock is out the rover drops 2.5 cm onto its wheels. One arm must hold the rover up while
the other slides the rock out; then the rover is lowered gently. Lifting the rover whole and setting it down gently
is also allowed.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: track, rock, the rolling rover with its tow bar; the whole-trial recorder; `score()` and `success()` |
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
  --instruction "The rover is stuck on the rock. Get the rock out and set the rover back on its wheels where it stands." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/high-centred-dream
```

The drop test counts in 20 Hz control steps; run at `control_hz=20`.

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 2400                 # full score
python demo.py task.py out --variant pull_rock --no-record        # pull the rock out from under the rover
python demo.py task.py out --variant fast_pull --no-record        # the same pull, three times as fast
python demo.py task.py out --variant hold_and_pull --no-record    # hold the rear tow bar where it is, pull the rock
python demo.py task.py out --variant idle --no-record             # do nothing
```

1. The left arm takes the rear tow bar from above and lifts the rear about 9 cm, turning the hand with the rover's
   pitch (closed loop on its measured pose): the rover tips onto its front tyres and its chassis rises off the rock.
2. The right arm grips the end of the rock that sticks out past the rover's right side and slides it 12.5 cm out
   sideways, clear of the rover, and lets go.
3. The left arm lowers the rear, closed loop, until the rear tyres touch the track, then opens and backs away.

The demo reads only the rover's and the rock's poses and the arms' own joint state.

| Variant | What it does | Score |
|---|---|---|
| (none) | lift the rear, slide the rock out, lower | 1.0 |
| `pull_rock` | slide the rock out from under the rover | 0.0 (the rock carries the rover 11.1 cm sideways: dragged) |
| `fast_pull` | the same pull, three times as fast | 0.0 (carried 11.3 cm: dragged) |
| `hold_and_pull` | hold the rear tow bar where it is and pull the rock out | 0.0 (the rover drops 1.2 cm in one control step) |
| `idle` | nothing | 0.0 |

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- A **drop**: the lowest point of one of the rover's tyres falling more than 1 cm in one control step (0.2 m/s at
  20 Hz), or more than 1.5 cm over three control steps (0.1 m/s sustained). The test counts 20 Hz control steps.
- **Fouls, at any time**, the trial then scoring 0: a drop; the rover dragged with the rock (carried more than 5 cm
  sideways, across its heading, net, while its body rests on the rock: some point of the rock's top face within 4 mm
  of it); the rover rolling more than 30 degrees or pitching more than 45.
- Why both rules: with its wheels hanging, nothing but its inertia holds the rover sideways, so a rock pulled from
  under the full weight carries it along (about 11 cm in simulation) and stays under it; the drag rule makes that
  0. Held still by a hand while the rock is pulled, the rover drops off the rock's edge, which the drop rule makes
  0 (1.2 cm in one control step). With the tail lifted first, the rock only rubs the body as it slides (1.7 cm in the
  demo). The drag is the net sideways excursion: yaw or movement along the rover on the rock are not counted.
- **End state**, with the rover and the rock at rest (each under 1 cm/s and 0.5 rad/s for the last 20 control steps,
  1 s at 20 Hz) and released over those 20 steps (none of six points on each arm, its wrist joint, palm, both fingertips, tool point and
  the point between the pad tips, and no point along the arm from its shoulder joint to its palm, within 5 cm of the
  rover or the rock), the rock wholly outside the rover's outline
  seen from above (bumper to bumper, tyre to tyre), the rover within 10 degrees of level with all four tyres on the
  track (lowest point within 8 mm of it):
  - 1.0: the rover within 2.5 cm and 5 degrees of where it stood;
  - 0.5: otherwise;
  - 0 if the end state is not reached.
- A full score ends the trial once it has held for 1 s.

On a real table: drops are read from two side videos (one per side) at 20 frames per second; the end state with a
photo and a ruler.

## Runtime and seeds

A full 2,400-step Dream-1 trial takes about 7 minutes on an L40S, Isaac Sim's start included. The layout is fixed
(`-E object_jitter=0`); there are no seeds to choose. The demo reaches full score in 505 steps (25 s); its peak pitch is about 30 degrees.

## Fidelity

Rigid bodies at 240 Hz physics, 20 Hz control. The rover's base is one rigid body at the crawler's published size and
mass (tyre size and centre of mass estimated from photographs, see `rover_spec.py`); its four wheels turn freely on
revolute joints with a little drag (0.002 N m s/rad), as with the driveshafts removed; its suspension is rigid, as with
the shocks locked. Friction: tyres 1.0, tow bar 1.0, rock 0.8, track 0.8, rover body 0.5. A drop is judged from the tyres' fall per control
step; no impact forces or damage are simulated.

## Real-world build

- **Track:** an earth-brown sheet of painted PVC foil 0.6 mm thick, taped to the table.
- **Rock:** a block of pine 5 x 14 x 5 cm painted rock grey, ballasted with steel shot to 0.30 kg.
- **Rover:** a stock 1:18 trail crawler (Traxxas TRX-4M, Bronco body), decals removed, painted yellow with a white
  roof; dummy battery; driveshafts removed (each wheel rolls freely) and shocks replaced with solid links. No
  electrics are live.
- **Tow bar:** a dark grey 3D-printed PLA bar 4.8 x 1.6 x 2.0 cm bolted to the middle of the rear bumper's face.

## License

MIT (`LICENSE`), the task authors. No third-party assets: all shapes and textures are generated by code in this
folder.
