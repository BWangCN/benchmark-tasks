# Straighten a loaded rover's turned front wheels before walking it into its bay

A loaded rover (a rover built heavy at the size of a 1:18 trail crawler, 14 kg in all; its wheels roll freely; its
steering holds where it is put) stands in a painted
lane 16 cm wide with its front wheels turned 25 degrees to the left. The robot is told to move it along the lane into
the bay at the far end, keeping all four wheels in the lane.

A vehicle moved with its steered wheels turned runs off its line: walked down the lane as it stands, this rover turns
left and leaves the lane within a few centimetres. It is too heavy to drag or lift straight (14 kg; its tyres hold it
sideways), forcing the nose across scrubs the tyres, and steering it on the move (like a cart by its drawbar) or towing
it by the steering means running with the wheels turned (in the simulator, towing by the tiller barely moved it). The steering has to be centred at a standstill, then the rover moved. The demo holds the
body by its rear tab while it straightens the steering; in the simulator the body turns about as much without the hold
(3.5 against 3.1 degrees), so the hold is the demo's choice, not a requirement. The scoring looks only at the result.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: lane, bay, the loaded rover with its steering; the whole-trial recorder; `score()` and `success()` |
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
  --instruction "Move the rover along the lane into the bay. Keep all four wheels in the lane." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/straighten-dream
```

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 2400                       # full score
python demo.py task.py out --variant push_only --no-record              # walk it down the lane as it stands
python demo.py task.py out --variant shove_nose --no-record             # shove the nose straight, then walk it
python demo.py task.py out --variant tiller_only --no-record            # straighten without holding the body
python demo.py task.py out --variant tow_by_tiller --no-record          # one arm tows it by the tiller
python demo.py task.py out --variant lift_front --no-record             # lift the nose and walk it
python demo.py task.py out --variant drive_with_tiller --no-record      # steer it while walking it
python demo.py task.py out --variant straighten_only --no-record        # straighten the wheels, do not move it
python demo.py task.py out --variant idle --no-record                   # do nothing
```

1. The right arm grips the rear tow tab from above and holds the body.
2. The left arm grips the steering tiller (the red pin ahead of the bonnet) and carries it along its arc until the
   wheels are straight (closed loop on the measured steering angle), then lets go and goes home.
3. Still holding the rear tow tab, the right arm walks the rover down the lane along its heading (it rolls on its
   wheels; closed loop on its measured pose; the hand turns with the rover), lets go, and pushes it the rest of the way
   into the bay with closed fingers against the tab's end, then withdraws.

The demo reads only the rover's pose and steering angle and the arms' own joint state.

| Variant | What it does | Score | Steps | IK misses (left / right), worst |
|---|---|---|---|---|
| (none), 3 runs | hold, straighten, walk and push into the bay | 1.0, 1.0, 1.0 (steering 0.5 degrees, body turned 3.1 degrees while straightening; 3.5 cm from the start at most while turned; tyres slid 0.5 cm in all) | 1,543 | 69 / 85, 16.3 / 19.7 mm (the left arm's approach to the tiller and the right arm's push, at the edge of their reach) |
| `push_only` | walk it down the lane as it stands | 0.0 (the front-left tyre leaves the lane at step 145) | 1,250 | 0 / 845, 26.5 mm |
| `drive_with_tiller` | walk it while steering it with the tiller, like a cart by its drawbar | 0.0 (the front-left tyre leaves the lane; it was 7.9 cm from its start with the wheels turned) | 1,998 | 628 / 0, 31.3 mm |
| `shove_nose` | shove the nose straight, then walk it | 0.0 (the front-right tyre leaves the lane at step 65) | 1,428 | 0 / 168, 497 mm |
| `tiller_only` | straighten without holding the body, then walk and push it as the demo does | 0.0 (straightened to 0.4 degrees with the body turned 3.5 degrees, against 3.1 with the hold; a front tyre left the lane during the final push, at step 1557) | 1,700 | 121 / 87, 16.4 / 19.6 mm |
| `tow_by_tiller` | one arm tows it by the tiller, like a wagon's handle | 0.0 (it hardly moves; the steering stays at 23 degrees) | 1,240 | 727 / 0, 16.7 mm |
| `lift_front` | lift the nose by its tab (10 cm lift command) and walk it | 0.0 (the nose does not rise; the rover hardly moves; front wheels at 28 degrees) | 1,428 | 60 / 0, 16.7 mm |
| `straighten_only` | straighten the wheels, do not move it | 0.25 | 511 | 75 / 0, 16.7 mm |
| `idle` | nothing | 0.0 (front wheels turned 25 degrees) | 670 | 0 / 0 |

The demo reads the steering angle from the joint; on a real rig it is the tiller's (or the wheels') angle to the body,
seen by the top camera.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- **Fouls, at any time**, the trial then scoring 0:
  - a tyre leaves the lane: the outer edge of its tread (its full width) more than 5 mm past the inner
    edge of a painted side line (y = +-8 cm), or any part of a tyre past either end of the lane;
  - the tyres slide sideways more than 2 cm in all: each wheel centre's motion along its own axle, summed over the
    trial for all four wheels, whether or not the tyre touches the lane (a tyre dragged just clear counts too);
  - both tyres of an axle more than 3 mm above the lane's surface for 0.5 s (lifted);
  - the front wheels turned more than 5 degrees while the rover is more than 5 cm from its start: the steering must
    be straightened before the rover is moved (steering it on the move, like a cart by its drawbar, does not count).
- **End state**, with the rover upright (tilted under 15 degrees), at rest (under 1 cm/s and 0.5 rad/s for the last 20
  control steps, 1 s at 20 Hz), released (no point of either gripper, and no point along either arm from its shoulder
  joint to its palm, within 5 cm of the rover's body, wheels, steering link or tiller) and the front wheels within 5
  degrees of straight:
  - 1.0: all four tyres wholly inside the bay (x 36 to 64, between its two painted lines) and the heading within 8
    degrees of the lane;
  - 0.5: the rover moved at least 10 cm along the lane;
  - 0.25: not moved that far;
  - 0 otherwise (tilted, not at rest, not released, or the front wheels turned more than 5 degrees).
- A full score ends the trial once it has held for 1 s.

On a real table: the end state with a photo and a ruler; the lane, lift and turned-wheels fouls from an overhead video;
the scrub only approximately (sideways tyre marks on the painted lane).

## Runtime and seeds

A full 2,400-step Dream-1 trial takes about 7 minutes on an L40S, Isaac Sim's start included. The layout is fixed
(`-E object_jitter=0`); there are no seeds to choose. The demo reaches full score in 1,543 steps (77 s).

## Fidelity

Rigid bodies at 240 Hz physics, 20 Hz control; suspension rigid. The rover's base is one rigid body at the crawler's
published size (`rover_spec.py`), 14 kg with its centre of mass 5.5 cm up: at that mass neither arm can lift an axle
more than 5 mm even with a 10 cm lift command, and a 12 cm sideways push moves it under 1 mm. Its wheels turn freely on
revolute joints with drag (0.02 N m s/rad) and a breakaway of 0.004 N m. The front axle is a turntable: one steering
link turning about the axle's middle, within 30 degrees each way, held by a stick-slip friction of 0.15 N m for the
unpowered servo. The front tyres touch the ground at a point (a crowned tread, as built): with a flat tread's line
contact the arm cannot turn the steering of a standing 14 kg rover in the simulator. Turning a front tyre by its
leading edge moved the steering 0.5 degrees for a 25 mm push, so the tiller pin is fitted for a hand on the steering.
Friction: tyres 1.0, lane 0.8, tow tabs and tiller 1.0.

## Real-world build

- **Lane:** an earth-brown sheet of painted PVC foil 0.6 mm thick, taped to the table, x 10 to 72 cm, y -14 to 14,
  with white tape lines (6 mm) along y = +-8 cm and across at x = 36 and 64 cm (the bay).
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
