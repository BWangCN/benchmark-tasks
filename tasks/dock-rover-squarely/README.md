# Square a fixed-wheel rover up before pushing it onto its charging dock

A small trail crawler (a real 1:18 crawler, 26 x 12 x 12 cm, 0.5 kg, its driveshafts removed so its wheels roll freely
and its steering fixed straight) stands on an earth track in front of its charging dock: a block with two contact
pads and no guides. Its front axle is on the dock's centreline, but the rover is turned 12 degrees off the dock's
axis. A short tow bar sticks out from its rear bumper. The robot is told to put the rover on its dock.

A vehicle whose wheels cannot steer goes only where its wheels point. Pushed as it stands, the rover runs wide and
reaches the dock about 4 cm off centre and still turned; the dock has no guides to correct it, and that push scores 0.
The rover must be brought square before it reaches the dock. The scoring checks only the end state, so it does not
prescribe how: the demo lifts the tail by the tow bar, swings it about the front axle until the rover is square, sets
it down and pushes it straight in; shoving the tail across on its tyres, or lifting the rover whole and setting it in
square, also scores 1.0.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: track, dock, the rolling rover with its tow bar; the whole-trial recorder; `score()` and `success()` |
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
  --instruction "Put the rover on its charging dock." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/dock-dream
```

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200                    # full score
python demo.py task.py out --variant push_straight --no-record       # push it in as it stands
python demo.py task.py out --variant idle --no-record                # do nothing
```

1. The left arm takes the rear tow bar from above, lifts the tail 1.5 cm (the rear wheels clear, the front ones stay
   down), swings it about the front axle until the rover is square to the dock, and sets it down.
2. The right arm, fingers closed and pointing forward, pushes the rover by the end of its tow bar straight into the
   dock (closed loop on its measured pose) until both bumper corners touch the pads, presses 2 mm on for 0.5 s, and
   withdraws slowly.

The demo reads only the rover's pose and the arms' own joint state.

| Variant | What it does | Score |
|---|---|---|
| (none) | square it up, push it in | 1.0 (both bumper corners on the pads, centre 3.9 mm off, heading 0.2 degrees) |
| `push_straight` | push it in as it stands | 0.0 (it arrives turned 12 degrees and off centre) |
| `idle` | nothing | 0.0 |

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- **Foul, at any time**, the trial then scoring 0: the rover tilting more than 30 degrees.
- **End state**, with the rover at rest (under 5 mm/s and 0.5 rad/s for the last 20 control steps, 1 s at 20 Hz),
  released (none of six points on each arm, its wrist joint, palm, both fingertips, tool point and the point between
  the pad tips, and no point along the arm from its shoulder joint to its palm, within 5 cm of the rover's body) and all four
  tyres on the track (lowest point within 8 mm of it):
  - 1.0: docked: both front corners of the front bumper within 3 mm of the contact pads' face, the rover's heading
    within 2 degrees of the dock's axis, and the bumper's centre within 8 mm of the dock's centreline;
  - 0 otherwise.
- A full score ends the trial once it has held for 1 s.

On a real table: the end state with a photo and a ruler (or the dock's charge light).

## Runtime and seeds

A full 1,200-step Dream-1 trial takes about 4 minutes on an L40S, Isaac Sim's start included. The layout is fixed
(`-E object_jitter=0`); there are no seeds to choose. The demo reaches full score in 401 steps (20 s).

## Fidelity

Rigid bodies at 240 Hz physics, 20 Hz control. The rover's base is one rigid body at the crawler's published size and
mass (tyre size and centre of mass estimated from photographs, see `rover_spec.py`); its four wheels turn freely on
revolute joints with a little drag (0.002 N m s/rad), as with the driveshafts removed, and they do not steer, as with
the steering fixed straight; its suspension is rigid. Friction: tyres 1.0, track 0.8, dock plate 0.6, pads 0.8, tow bar 1.0. Contact with the pads
is judged from the bumper's position; no electrical contact is simulated.

## Real-world build

- **Track:** an earth-brown sheet of painted PVC foil 0.6 mm thick, taped to the table.
- **Dock:** a grey ABS block 3 x 16 x 7 cm screwed to the table, with two orange contact pads 1.4 x 1.4 cm, 6 mm
  proud, 6 cm apart, 4.5 cm up.
- **Rover:** a stock 1:18 trail crawler (Traxxas TRX-4M), body painted yellow with a white roof; dummy battery;
  driveshafts removed; steering servo replaced by a fixed link holding the wheels straight. No electrics are live.
- **Tow bar:** a dark grey 3D-printed PLA bar 4.8 x 1.6 x 2.0 cm bolted to the middle of the rear bumper's face.

## License

MIT (`LICENSE`), the task authors. No third-party assets: all shapes and textures are generated by code in this
folder.
