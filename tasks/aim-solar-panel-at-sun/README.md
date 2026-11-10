# Aim a field rover's solar panel at the sun by its shadow sight

A small trail crawler (a real 1:18 crawler, 26 x 12 x 12 cm, 0.5 kg) is parked on a turf pad in front of the robot, against four wheel chocks.
On its rear bed a solar panel (10 x 7 cm) stands on a two-axis friction mount: it turns about the vertical and tilts
about a hinge through its middle. A dark peg stands square on a white sighting disc at the panel's centre, inside two printed rings, and an
orange flat handle sticks out of the panel's near edge. Ahead and to the left a warm lamp on a 42 cm post,
in the top camera's view, stands in for the sun. The robot is told only to point the panel straight at the sun lamp.

The panel faces the sun when the peg's shadow shrinks to its foot: the sun sight that installers use to aim solar
panels and trackers. Getting there means turning and tilting together on two coupled axes, toward a target that is a
light in space, and reading the result from a shadow. The mount sits on a light vehicle, so the robot must also keep
the rover still: shoved or turned, it fouls the trial, and turning the whole rover toward the lamp is not allowed.

Solar-powered field robots and remote stations are kept running by technicians who clean, aim and service their
panels (O*NET 17-3024.01, Robotics Technicians; 47-2231.00, Solar Photovoltaic Installers). Aiming by a shadow is
how the panel's real alignment is checked in the field, without instruments.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the pad, the lamp and the rover with its articulated mount; the whole-trial recorder; `score()` and `success()` |
| `trailkit.py` | Compound rigid bodies, articulated links on damped joints, lights, the per-step recorder and scoring helpers; it hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena |
| `rover.py`, `rover_spec.py` | The rover as one rigid body, and every number of it with its source |
| `demo.py`, `demokit.py` | The scripted full-score demo and its near-miss variants |
| `textures/`, `textures.py` | The turf look, and the code that generated it |
| `LICENSE` | MIT |

The task runs with the standard `yam_arena` embodiment and the `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source <ARENA_DIR>/yam.env`, as set up by the
walkthrough), from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Point the rover's solar panel straight at the sun lamp." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/aim-panel-dream
```

## Scripted demo

```bash
python demo.py task.py demo_out                                  # full score: demo_*.mp4, setup.png, result.json
python demo.py task.py out --variant half_way --no-record        # stop the swing a third short
python demo.py task.py out --variant turn_rover --no-record      # aim by turning the whole rover by its bumper
python demo.py task.py out --variant idle --no-record            # do nothing
```

1. The left arm steadies the rover: closed fingers rest on its bonnet, well clear of the panel's sweep.
2. The right arm comes in low behind the orange handle, along its axis, and clamps its flat faces.
3. It swings the panel along a straight path in turn and tilt from where it is to where it faces the lamp; the
   direction to the lamp comes from the lamp's position on the table and the panel's measured pose. A short closed-
   loop trim on the panel's measured orientation removes what is left.
4. It opens, slides back off the handle and withdraws; the left arm lifts off and withdraws.

The demo reads only object poses (the rover, the panel, the lamp) and the arms' own joint state.

| Variant | What it does | Score |
|---|---|---|
| (none) | steady the rover, swing the panel to face the lamp | 1.0 (0.1 degrees off; success at step 438) |
| `half_way` | the same, but stopping a third short | 0.5 (16.8 degrees off) |
| `turn_rover` | aim by pushing the rover's tail round | 0.0 (the rover is turned) |
| `idle` | nothing | 0.0 (46.6 degrees off) |

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- **Fouls, at any time:** the rover (the point midway between its axles) moves more than 2 cm, turns more than 10
  degrees from where it was parked, or tilts more than 10 degrees. The trial then scores 0 whatever happens afterwards.
- **End state**, with the rover and panel at rest (the rover under 1 cm/s and 0.5 rad/s, the panel turning less than
  0.05 degrees per control step, for the last 20 control steps, 1 s at 20 Hz) and released (none of six points on
  each arm, its wrist joint, palm, both fingertips, tool point and the point between the pad tips, and no point along the arm from its shoulder joint to its palm, within 5 cm of the
  rover or the mount's moving parts); otherwise 0:
  - 1.0: the angle between the panel's normal and the line from the panel's centre to the lamp's centre is at most
    8 degrees (on a real panel: the peg's shadow ends inside the inner printed ring);
  - 0.5: at most 20 degrees (the shadow ends inside the outer ring);
  - 0 otherwise.
- A full score ends the trial once it has held for 1 s.

On a real table the angle is read with the shadow itself: the shadow's far edge reaches 3 mm (the peg's radius)
plus 60 mm x tan(angle) from the peg's centre, so the rings' inner edges sit at 11.4 mm (8 degrees) and 24.8 mm (20
degrees). The bulb's light spreads the tip over about 3 degrees (plus or minus 1.5); judge the middle of the blur. A foul is seen in the video or with a tape mark at the rover's wheels.

## Runtime and seeds

A full 1,200-step trial takes about 5 minutes on an L40S, Isaac Sim's start included. The layout is fixed
(`-E object_jitter=0`); there are no seeds to choose. The demo reaches full score in 438 steps (22 s).

## Fidelity

Rigid bodies at 240 Hz physics, 20 Hz control. The rover is one rigid body at the crawler's published length, width,
height, wheelbase and mass; its tyre size, body outline and centre of mass are estimates from photographs (see
`rover_spec.py`). The two-axis friction mount is two damped joints (0.3 N m s/rad): they hold the panel wherever it is
let go, as a friction mount does, but have no breakaway torque: in simulation the panel moves at less force than on
a real mount, which makes keeping the rover still somewhat easier. The sun is a sphere light 1.9 cm across inside the 2.4 cm bulb (2.9 degrees seen from the
panel), the brightest light at the panel, so the peg casts one clear shadow (see the demo's top-camera frames); the score is computed from the
panel's normal and the lamp's position, which is exactly what the shadow shows. At the camera size Dream-1 uses
(320 x 176) the bulb is about 4 pixels across, in the top camera's view with its whole post; the sighting disc is
about 12 pixels across and the peg's shadow at the start about 7 pixels long. At the 20-degree ring the shadow is
about 4 pixels, at the 8-degree ring under 2, so the last few degrees are judged from the panel's pose against the
lamp as much as from the shadow; the wrist cameras see the disc closely. No forces or
damage are simulated.

## Real-world build

- **Rover:** a stock 1:18 trail crawler (Traxxas TRX-4M, Bronco body), decals removed and the shell painted yellow
  with a white roof; dummy battery; motor off, drivetrain engaged. No electrics are live.
- **Mount:** 3D-printed PLA, fixed to the rear bed with double-sided tape: mast 2 x 2 x 4.5 cm, a turntable on an M4
  bolt with a nylon friction washer, a yoke with the tilt hinge on an M3 bolt and a friction washer. Panel: 2 mm
  plywood or PLA 10 x 7 cm, painted dark blue, a white 6 cm disc glued at its centre, a 6 mm dowel 6 cm long glued
  square at the disc's centre, a 4.5 x 2 x 0.6 cm handle glued to the middle of the near edge. Balance the panel on its
  hinge with a small weight under the far edge if needed.
- **Sun lamp:** a small warm LED bulb 2.4 cm across (about 250 lm) in a plain socket on a 42 cm lamp post at (44.6, 18.7), the brightest light near the
  table; dim the room lights.
- **Pad:** a 34 x 22 cm piece of artificial turf, with four 3D-printed wheel chocks taped to it.
- **Lamp post:** clamp its base to the table.

## License

MIT (`LICENSE`), the task authors. No third-party assets: all shapes and textures are generated by code in this
folder.
