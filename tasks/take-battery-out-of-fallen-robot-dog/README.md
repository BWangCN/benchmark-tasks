# Roll a fallen robot dog off its propping legs to take its battery out of the belly

A powered-off four-legged robot (a passive 1:2 model of a mid-size robot dog, 1.9 kg, 12 limp joints with hard end
stops, a handle over each hip) has fallen on its right side on an earth pad. Its flat battery sits in a bay in its
belly, held by a light magnetic catch. The robot is told to take the battery out and put it on the charging pad.

A robot dog that has gone limp does not lie on its side: it rolls onto its loose legs and props itself there, belly
down and tilted, with the battery bay facing down toward the ground. The battery cannot be drawn there, and pushed at the
trunk the body slides on its legs instead of turning over. It has to be rolled over its own propping legs onto its
back, by the two handles of its high side, front and rear together; then the battery comes out along the bay's axis,
past the legs that now stick up beside it. Neither the roll nor the legs is mentioned in the instruction.

Legged robots are sent where people are kept out, and when one falls and goes flat it is recovered where it lies.
Operators of such robots roll them over to change the battery in the belly: a university laboratory's protocol for a
commercial robot dog says "After roll over, press and hold Power Button to shut down and remove battery from Spot
belly" (https://uwrobotlearning.github.io/Protocols/spot.html). Keeping field robots running is the work of robotics
technicians (O*NET 17-3024.01).

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the pad, the charging pad, the robot and its battery, the start pose, the whole-trial recorder, `score()` and `success()` |
| `quadruped.py`, `quadruped_spec.py` | The passive robot dog as one articulated body of boxes, capsules, cylinders and spheres, its battery and magnetic catch, and every number with its source |
| `trailkit.py` | Compound rigid and articulated bodies, textured tiles, the per-step recorder and scoring helpers; it hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena |
| `demo.py`, `demokit.py` | The scripted full-score demo (closed loop on the robot's and the battery's measured poses) and its near-miss variants |
| `textures/`, `textures.py` | Earth and turf looks, and the code that generated them (not needed at run time) |
| `test_scoring.py` | Offline tests of the scorer on synthetic states (`python -m pytest test_scoring.py`, no simulator) |
| `LICENSE` | MIT |

The task runs with the standard `yam_arena` embodiment and the `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source <ARENA_DIR>/yam.env`), from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The quadruped robot has fallen and its battery is flat. Take the battery out and put it on the charging pad." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/robot-dog-battery-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out                                  # full score: demo_*.mp4, setup.png, result.json
python demo.py task.py out --variant push_trunk --no-record      # push the trunk over toward its back
python demo.py task.py out --variant pull_propped --no-record    # draw the battery out as it lies, propped
python demo.py task.py out --variant pull_propped_level --no-record   # the same, the hand level from the arm side
python demo.py task.py out --variant lift_clear --no-record      # lift it by two handles and set it down
python demo.py task.py out --variant jolt --no-record            # lift the high side 5 cm and drop it, twice
python demo.py task.py out --variant roll_only --no-record       # turn it onto its back, leave the battery
python demo.py task.py out --variant idle --no-record            # do nothing
```

1. Both hands take the two hip handles of the robot's high (left) side, pointing down, fingers closing across each
   rung, so that the rung turns in the grip like a pin: the left hand the front one, the right hand the rear one.
2. Together they carry the handles up and over a pivot at the robot's low outer edge: the robot rolls across its
   propping legs onto its back. Once the belly is within 70 degrees of straight up they open and back off; the
   robot rests there, belly toward the arms, its legs in the air.
3. The right hand comes level with the battery's grip tab, out along the bay's axis (from above it would meet the
   raised legs), takes the tab with its fingers across it, draws the battery straight out of the bay against the
   catch, turns to point down so the battery hangs below, carries it to the charging pad and sets it down there.
   (Had a leg lain within 5 cm of the tab's line, the left hand would first hold it aside.)

The demo reads only the poses of the robot and the battery and the robot's joint angles (what a camera-based pose
estimate gives), and the arms' own joint state.

| Variant | What it does | Score |
|---|---|---|
| Dream-1 | the model, with the task's instruction, 1,600 steps | 0.0: the right arm went to the charging pad and the arms hovered near the robot without touching it; it stayed propped on its legs |
| (none) | roll it over by its high side's handles, then draw the battery out and set it on the charging pad | 1.0, three runs of three (success at step 616) |
| `push_trunk` | push the trunk's top edge over toward its back for 10 s | 0.0: the robot slides on its legs; it never reaches its back and the battery stays in |
| `pull_propped` | go for the tab as the robot lies, propped, along the bay's axis | 0.0: the bay faces down (its outward direction 50 degrees below level, the tab 3.9 cm above the pad): the hand cannot come in along it; nothing is drawn |
| `pull_propped_level` | the same, the hand level and pointing at the robot from the arm side, pulled back along the bay's axis | 0.0: the hand does not get under the belly; the battery stays in |
| `lift_clear` | lift the robot 10 cm by its two high-side handles and set it down | 0.0: the trunk rises 11.5 cm off the pad (foul at step 151) |
| `jolt` | lift the high side 5 cm by its handles and let it drop, twice, to shake the battery out | 0.0: the trunk rises 5.8 cm (under the foul); the battery stays in its bay |
| `roll_only` | roll it onto its back, leave the battery | 0.0: the battery is still in its bay |
| `idle` | nothing | 0.0 |

| Demo run | Score | Steps | IK misses (left, right) | Worst IK error |
|---|---|---|---|---|
| 1, 2, 3 | 1.0 each | 616 each | 69, 71 | 31 mm (right arm, during the roll, where the handles pass their highest) |

Premise, checked before the build (`push_trunk`'s motion, three runs at this scale): the push ends with the belly 155,
150 and 110 degrees from up, never on its back, the robot slid 6 to 42 cm.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- **Foul, at any time:** the lowest point of the robot's trunk (with its head and handles) more than 6 cm above the
  ground, the pad's top (it was lifted: it stands for a full-size machine that is rolled, not carried; the demo's roll raises it 3.6 cm; the legs are
  not judged, they hang 20 cm and can trail on the ground); or, the first time the battery leaves its catch (its centre more
  than 4 mm from its seat), no gripper point is within 2 cm of it (it fell or was knocked out: in the start pose the
  bay faces 50 degrees below level, and a battery jolted 3 mm out slides out by its own weight). The trial then
  scores 0.
- **End state.** It counts only if the robot's trunk and the battery are at rest (neither centre moved more than 2 mm
  over the last second: positions, not speeds) and released (no point of either gripper, and no point along either
  arm from its shoulder joint to its palm, within 5 cm of the robot or the battery), for each of the last 20 control
  steps (1 s at 20 Hz); and 0 unless the battery is out of its bay (its centre at least 3 cm from its seat).
  - 1.0: the battery's whole outline, its grip tab included, lies over the charging pad and its lowest point is
    within 5 mm of the pad's top;
  - 0.5: the battery is out but not on the charging pad.
- A full score ends the trial once it has held for 1 s (`SUCCESS_HOLD_S`).

On a real table: the charging pad's outline is taped; a photo from above shows the battery inside it; the foul is
judged from the side video against a millimetre scale.

## Runtime and seeds

The layout is fixed (`-E object_jitter=0`); there are no seeds to choose. The demo reaches full score in 616 steps
(31 s) and takes about 5 minutes on an NVIDIA L40S, Isaac Sim's start included.

## Fidelity

Rigid links at 240 Hz physics, 20 Hz control. The robot's sizes, mass shares and joint ranges are a 1:2 scale of a
public mid-size quadruped model (the Unitree Go2 in MuJoCo Menagerie, BSD-3-Clause: lengths x 1/2, masses x 1/8);
only numbers are taken, and it is built from boxes, capsules, cylinders and spheres, grey, with no logo. Its 12
hinges are limp (light viscous damping, no friction or spring) with hard end stops, and its own parts collide with
each other. The hip handles and the belly battery are ours, sized for the gripper. The magnetic catch is a 3 N force
toward the seat while the battery is within 3 mm of it (a real magnet's pull also falls off within millimetres). The
start pose is the one this body settles into by itself in under a second when laid on its right side; it is written
at reset so that every trial starts the same. The turf strip is drawn on the rigid pad. None of this changes what the
task measures: a limp legged body props itself on its legs, and the bay is reached by rolling it over them.

## Real-world build

- **Robot:** 3D-printed PLA links (grey) on steel pin hinges with printed end stops, no springs or electronics, to
  the sizes and masses in `quadruped_spec.py` (masses brought up with steel shot); four printed hip handles; a printed
  battery block (slate blue, 140 g, an orange tab) in a belly bay, held by a 3 N magnetic catch (a cabinet magnet and
  a steel plate).
- **Pad:** a 45 x 60 x 1 cm plywood board clamped to the table, painted earth brown, a 5 cm strip of flat artificial
  grass along its far edge; a 14 x 14 x 0.6 cm dark grey plate as the charging pad.

## Licence

MIT (`LICENSE`). The robot's numbers are derived from the Unitree Go2 model in MuJoCo Menagerie (BSD-3-Clause); no
mesh, texture or other file of it is used. The ground textures are generated by `textures.py` (ours, MIT).
