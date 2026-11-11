# Push a rover that cannot be lifted across a washout by taking it at an angle

An earth track 57 cm wide runs across the station table from the robot's right to its left. Across it runs a washout,
a gap 5 cm wide and 3 cm deep; beyond it, a marked pad. A loaded rover (built heavy at the size of a 1:18 trail
crawler, 14 kg; free-rolling wheels; steering servo unpowered, so the front wheels stay where they are put) stands on
the near side, wheels straight, square to the gap. The robot is told to push it across the washout to the pad.

Taken square, both front tyres drop into the gap together and wedge: the push cannot lift the axle out, and lifting the
rover is a foul. Taken at about 30 degrees, one wheel is over the gap at a time and the other three carry the rover
level. The angle has to be set first: the steering is turned at a standstill and the rover walked round on it, then
straightened and walked across, and handed from one arm to the other to reach the pad. The scoring looks only at the
result and at the lifting foul.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: track with the washout and the pad, the loaded rover with its steering; the whole-trial recorder; `score()` and `success()` |
| `trailkit.py` | Compound rigid bodies, articulated links with stick-slip joint friction (the steering servo and the wheel bearings), textured tiles, the per-step recorder and scoring helpers; it hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena |
| `rover.py`, `rover_spec.py` | The loaded, steerable rover, and every number of it with its source |
| `demo.py`, `demokit.py` | The scripted full-score demo and its near-miss variants |
| `textures/`, `textures.py` | The earth look, and the code that generated it |
| `LICENSE` | MIT |

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source <ARENA_DIR>/yam.env`), from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The rover has lost power. Push it across the washout to the pad without lifting it." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/washout-dream
```

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 2400                       # full score
python demo.py task.py out --variant square_on --no-record            # walk it straight at the gap
python demo.py task.py out --variant lift_the_nose --no-record        # lift the nose over the gap
python demo.py task.py out --variant idle --no-record                 # do nothing
```

1. The right arm grips the rear tow tab from above. The left arm grips the steering tiller and turns the front wheels
   to full lock to the right (closed loop on the measured angle).
2. The right arm walks the rover along its heading while the left arm keeps the tiller where it is: the rover turns
   right until it is 40 degrees off square to the gap.
3. The left arm straightens the steering and lets go; the right arm walks the rover across the gap at that angle, one
   wheel over the gap at a time, until the tab reaches y 3 cm (or x 33 cm), and lets go.
4. The left arm, fingers closed and pointing down, pushes the same tab on by its end face until the rear tyres are
   past y 23.5 cm, then backs off.

The script reads only the rover's pose and steering angle and the arms' own joint state.

| Variant | What it does | Score | Steps | IK misses (left / right), worst |
|---|---|---|---|---|
| (none), 3 runs | steer at a standstill, walk round 40 degrees, straighten (34.3 degrees off square), walk across, hand over, push on | 1.0, 1.0, 1.0 (deepest dip 2.0 mm; rear tyres end at y 24.2 cm) | 1,823 | 44 / 17, 20.4 / 24.0 mm (the left arm at the tiller, the right at the end of its walk) |
| `square_on` | walk it straight at the gap | 0.0 (both front tyres over the washout at once at step 193; they drop 15.4 mm into it) | 1,060 | 0 / 0 |
| `lift_the_nose` | lift the nose by the front tab (10 cm lift command) and walk it at the gap | 0.0 (the 14 kg nose does not rise; both front tyres over the washout at once at step 534) | 1,188 | 17 / 0, 41.4 mm |
| `idle` | nothing | 0.0 | 870 | 0 / 0 |

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- **Fouls, at any time**, the trial then scoring 0: both tyres of an axle more than 1 cm above the ground under them
  at once (the rover lifted; over the washout the ground is taken at the track's surface); both tyres of an axle over
  the washout at once (the middle of each tread's lowest line between its edges: the rover taken square, or an axle
  carried across level, perhaps under the lifting threshold by two arms).
- **End state**, after 2 s with the rover at rest (under 5 mm/s and 0.05 rad/s) and released (no point of either
  gripper, and no point along either arm from its shoulder joint to its palm, within 5 cm of the rover, its wheels and
  steering):
  - 1.0: all four tyres beyond the washout (every point of each tread past its far edge) and on the pad (past y 22 cm,
    the near edge of the pad's tape line, inside the track, the lowest point within 5 mm of the surface);
  - 0.25: at least one tyre beyond the washout;
  - 0 otherwise.
- A full score ends the trial once it has held for 1 s.

On a real table: the end state by eye or from a photo (the tyres past the pad's line); the lift from a side video.

## Runtime and seeds

A full 2,400-step Dream-1 trial takes about 7 minutes on an L40S, Isaac Sim's start included. The layout is fixed
(`-E object_jitter=0`); there are no seeds to choose. The demo reaches full score in 1,823 steps (91 s).

## Fidelity

Rigid bodies at 240 Hz physics, 20 Hz control; suspension rigid. The rover's base is one rigid body at the crawler's
published size (`rover_spec.py`), 14 kg with its centre of mass 5.5 cm up: at that mass neither arm can lift an axle
more than 5 mm even with a 10 cm lift command, and a 12 cm sideways push moves it under 1 mm. Its wheels turn freely on
revolute joints with drag (0.02 N m s/rad) and a breakaway of 0.004 N m. The front axle is a turntable: one steering
link turning about the axle's middle, within 30 degrees each way, held by a stick-slip friction of 0.15 N m for the
unpowered servo. The front tyres touch the ground at a point (a crowned tread, as built): with a flat tread's line
contact the arm cannot turn the steering of a standing 14 kg rover in the simulator. Turning a front tyre by its
leading edge moved the steering 0.5 degrees for a 25 mm push, so the tiller pin is fitted for a hand on the steering.
Friction: tyres 1.0, track 0.9, tow tabs and tiller 1.0. The washout's edges are square.

## Real-world build

- **Track:** a slab of painted plywood 3.2 cm thick, earth brown, x 13 to 70 cm, y -45 to 56, with a slot 5 cm wide
  and 3 cm deep across it from y 15 to 20 (the washout, its floor painted dark brown) and a white tape line (6 mm)
  across it at y 22 (the pad beyond).
- **Rover:** built for the rig at the size of a 1:18 trail crawler: an aluminium plate chassis with steel axles, the
  body shell of a Traxxas TRX-4M (polycarbonate, painted yellow, white roof), about 13 kg of lead (1.15 L) cast to fill
  the body between and over the axles, 14 kg in all. Hard rubber wheels 5.6 cm across, the front pair with a rounded
  (crowned) tread. The front axle is a turntable on a central pivot bolt with a friction washer (about 0.15 N m), so
  both front wheels turn together; driveshafts and motor absent, so all four wheels roll freely.
- **Tow tabs:** dark grey 3D-printed PLA tabs 5.0 x 1.6 x 1.6 cm on the middle of each bumper, each with a post 1 cm
  across and 3 cm tall at its tip.
- **Tiller:** a red 3D-printed PLA pin 1 cm across and 4 cm tall on an arm of the turntable axle that reaches forward
  under the bonnet's lip, 4.5 cm ahead of the pivot.

## License

MIT (`LICENSE`), the task authors. No third-party assets: all shapes and textures are generated by code in this
folder.
