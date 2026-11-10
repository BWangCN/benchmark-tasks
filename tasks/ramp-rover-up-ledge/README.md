# Build ramp rocks under a rover's wheels to roll it up a ledge

A small trail crawler (a real 1:18 crawler, 26 x 12 x 12 cm, 0.5 kg) stands on a lower earth track facing a rock
ledge 3.8 cm high. Its motor pinion is out, so its wheels roll freely, but its tyres are only 5.6 cm across: a free
wheel cannot roll up a step taller than its radius, and pushed straight at the ledge the rover only jams its front
tyres against the face. Four stepped ramp rocks lie beside the track. The robot is told to get the rover up onto the
ledge without lifting it.

The way up is the recovery crew's: build a ramp under each front wheel, tight to the step and on the wheel's line,
then roll the vehicle up. Two ramps at the vehicle's track spacing, each pressed against the ledge's face, carry the
front wheels and then the rear wheels up as the rover is pushed. A ramp off the wheel line, or standing off the face,
lets the wheel drop into the gap.

Off-road recovery and trail crews build rock ramps ("stacking" a ledge) this way every day; field robots that keep
other robots moving will need to read a step against a wheel and build what the wheel needs (O*NET 17-3024.01,
Robotics Technicians; 45-4011.00, Forest and Conservation Workers).

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the lower track, the ledge, the rolling rover and the four ramp rocks; the whole-trial recorder; `score()` and `success()` |
| `trailkit.py` | Compound rigid bodies, articulated links (the rover's wheels), the per-step recorder and scoring helpers; it hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena |
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
  --instruction "The rover can't climb the ledge on its own. Get it up onto the ledge without lifting it." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/ledge-dream
```

## Scripted demo

```bash
python demo.py task.py demo_out                                 # full score: demo_*.mp4, setup.png, result.json
python demo.py task.py out --variant push_only --no-record      # push the rover straight at the ledge, no ramp
python demo.py task.py out --variant ramp_only --no-record      # set the two ramp rocks and stop
python demo.py task.py out --variant idle --no-record           # do nothing
```

1. The left arm takes a ramp rock by its high step and sets it on the rover's left wheel line, high end toward the
   ledge, then taps it tight against the ledge's face with its closed fingers; the right arm does the same on the right
   wheel line.
2. The right arm, fingers closed and pointing forward, puts its fingertips to the middle of the rear bumper and
   pushes the rover slowly forward (closed loop on the rover's measured pose): the front wheels roll up the steps
   onto the ledge, then the rear wheels.
3. It withdraws.

The demo reads only object poses (the rover and the rocks) and the arms' own joint state.

| Variant | What it does | Score |
|---|---|---|
| (none) | build both ramps, push the rover up | 1.0 (success at step 881) |
| `push_only` | push the rover straight at the ledge | 0.0 (the front wheels jam against the step) |
| `ramp_only` | set the ramps and stop | 0.25 |
| `wheelie` | press the tail down so the nose rises, as for a wheelchair at a kerb | 0.0 (lift foul) |
| `pulse_wheelie` | the same in a short 0.4 s press | 0.0 (lift foul) |
| `idle` | nothing | 0.0 |

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- **Fouls, at any time:** any tyre is more than 1 cm above the surface under it for 0.5 s, or more than 1.8 cm above it even for one control step (the rover lifted, hopped, or tipped up by one end; surfaces are the track, the ledge top and the steps of ramp rocks lying flat on the table; a tyre on a ramp step never rises more than one step, 1.27 cm, while a bare tyre must rise 2.8 cm to cross the face); a front tyre reaches the ledge without a ramp rock set under its own line at that moment (as in the 0.25 clause); or the rover tilts more than 30 degrees. The trial then scores 0 whatever happens afterwards.
- **End state**, with the rover and the ramp rocks at rest (each under 1 cm/s and 0.5 rad/s for the last 20 control
  steps, 1 s at 20 Hz) and released (none of six points on each arm, its wrist joint, palm, both fingertips, tool
  point and the point between the pad tips, and no point along the arm from its shoulder joint to its palm, within 5 cm of the rover or a ramp rock); otherwise 0:
  - 1.0: all four tyres rest on the ledge (each tyre's lowest point within 1 cm of the ledge top and inside its
    outline), the rover within 10 degrees of level;
  - 0.5: both front tyres rest on the ledge;
  - 0.25: under each front wheel's line (the y of that front tyre's centre, the rover facing the ledge within 15 degrees), a ramp rock lies flat on the table (tilted under 10 degrees) with its high step facing the ledge (within 20 degrees), the end of that step from 5 mm short of the ledge's face to 6 mm past it (pressed on the face), and its centre within 1.5 cm of the wheel's line;
  - 0 otherwise.
- A full score ends the trial once it has held for 1 s.

On a real table: lifting and tipping are seen in a side video; the end state with a photo and a ruler.

## Runtime and seeds

A full 2,400-step Dream-1 trial takes about 7 minutes on an L40S, Isaac Sim's start included. The layout is fixed
(`-E object_jitter=0`); there are no seeds to choose. The demo reaches full score in 881 steps (44 s).

## Fidelity

Rigid bodies at 240 Hz physics, 20 Hz control. The rover's base is one rigid body at the crawler's published size
and mass (tyre size, body outline and centre of mass estimated from photographs, see `rover_spec.py`); its four
wheels turn freely on revolute joints with a little drag (0.002 N m s/rad), as with the motor pinion removed; its
suspension is rigid, so the body pitches with the wheels where a real crawler's would articulate. Friction: tyres
1.0, rocks 1.0, ledge 0.8. The tyre diameter (5.6 cm) is an estimate from photographs; the ledge is 3.8 cm so that a bare wheel jams
even if the real tyre is a few millimetres larger and soft. The ramp rocks are stepped blocks rather than rough stones. Lifting is judged from the
tyres' height above the surfaces under them, as from a side video. No forces or damage are simulated.

## Real-world build

- **Lower track:** an earth-brown sheet of painted PVC foil 0.6 mm thick, taped to the table. **Ledge:** a 30 x 34 cm plywood deck on battens, 3.8 cm
  tall, clamped to the table, top painted earth brown, sides rock grey.
- **Rover:** a stock 1:18 trail crawler (Traxxas TRX-4M, Bronco body), decals removed, painted yellow with a white
  roof; dummy battery; motor pinion removed so the wheels roll freely. No electrics are live.
- **Ramp rocks:** four three-step blocks (6 x 4 cm, steps 1.27, 2.53 and 3.8 cm), 3D-printed PLA or sawn hardwood, painted
  grey, filled with steel shot to 150 g each.

## License

MIT (`LICENSE`), the task authors. No third-party assets: all shapes and textures are generated by code in this
folder.
