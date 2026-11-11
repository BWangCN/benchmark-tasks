# Turn a fallen humanoid robot onto its back without rolling it onto its own arm

A small humanoid robot, powered off, lies face down on a packed-earth pad, along the table's width, head to the
robot's left, arms straight along its trunk and legs straight. It is a passive 1:3 model of a mid-size humanoid
(44 cm, 1.24 kg, 14 limp joints with hard end stops). The robot is told only to turn it onto its back.

A powered-off humanoid is not a rigid body: every limb falls where its joints let it. Rolled about its long axis, the
body pivots on the edge it rolls toward, and the arm lying beside that edge stays on the ground: the body rolls onto
it and the arm ends up trapped under the back, bearing the body's weight; in this model the trapped arm props the body
on its side. The right way is
the one taught for a casualty log roll and for recovering a fallen robot: first move that arm clear (here, swept round
its shoulder up beside the head), then roll. The instruction mentions neither the arm nor the hazard; the test is
foreseeing where a free limb ends up when the body it is attached to is turned.

Humanoid and legged robots fall, and are recovered where they lie, by technicians today and more and more by other
robots. Turning one over without trapping or forcing a limb is part of the job of the technicians who keep field and
factory robots running (O*NET 17-3024.01, Robotics Technicians).

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the pad and the dummy, the whole-trial recorder, `score()` and `success()` |
| `dummy.py`, `dummy_spec.py` | The passive humanoid as one articulated body of boxes, capsules and spheres, and every number of it with its source |
| `trailkit.py` | Compound rigid and articulated bodies, textured tiles, the per-step recorder and scoring helpers; it hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena |
| `demo.py`, `demokit.py` | The scripted full-score demo (closed loop on the dummy's measured pose) and its near-miss variants |
| `textures/`, `textures.py` | Earth and turf looks, and the code that generated them (not needed at run time) |
| `test_scoring.py` | Offline tests of the scorer on synthetic states (`python -m pytest test_scoring.py`, no simulator) |
| `LICENSE` | MIT |

The task runs with the standard `yam_arena` embodiment and the `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source <ARENA_DIR>/yam.env`), from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The humanoid robot on the pad has fallen face down. Turn it onto its back." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/turn-onto-back-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out                                    # full score: demo_*.mp4, setup.png, result.json
python demo.py task.py out --variant roll_only --no-record         # roll it over without moving its arm first
python demo.py task.py out --variant lift_and_flip --no-record     # pick the body up and turn it in the air
python demo.py task.py out --variant idle --no-record              # do nothing
```

The body is rolled toward the robot, about its near edge, so its left arm (on the robot's side) is the one at risk.

1. **Clear the near arm.** The left hand takes the near forearm 10 cm from the shoulder, fingers about 4 cm apart and
   closing across it, and carries it round the shoulder, just off the ground, from 20 to 80 degrees out from the side;
   the right hand takes it there and carries it on to 120 degrees, beside the head. Each carry runs on an arc about
   the shoulder, just inside full stretch, as a limp arm moves (pulled straight, it would drag the body).
2. **Roll.** The left hand grips the head across its width (the head is rigid with the trunk) and carries it 68
   degrees round the body's near edge, turning with it. From 30 degrees on, the right hand comes onto the trunk's
   highest far edge, fingers closed, and pushes; once it is pressing, the left hand lets go and backs out along its
   fingers, and the right hand pushes the body past its side onto its back.
3. Both hands withdraw.

A push alone does not roll it: face down, the body slides or swivels at about 7 N, while a side push at its 5 cm
height would need 15 to 20 N to turn it, so the far side has to be lifted, as a rescuer does.

The demo reads only the dummy's pose and joint angles (what a camera-based pose estimate gives) and the arms' own
joint state.

| Variant | What it does | Score |
|---|---|---|
| Dream-1 | the model, with the task's instruction, 1,200 steps | 0.0: the right arm reached over the body's back and touched it near the shoulder, then drew back; the body stayed face down |
| (none) | clear the near arm up beside the head, then roll | 1.0, three runs of three (success at step 651) |
| `roll_only` | the naive way: the same roll without moving the arm first | 0.0, three runs of three: the arm is trapped under the body (upper arm and forearm) and props it on its side, chest 97 degrees from up |
| `lift_and_flip` | pick it up (head and far thigh) and turn it in the air | 0.0 (the trunk is lifted 5.4 cm at step 150: foul) |
| `idle` | nothing | 0.0 |

The scoring gives 0.25 to a body brought chest up within 45 degrees with a limb under it, or with an arm trapped
during the turn and pulled out afterwards (`test_scoring.py` in this folder checks both offline). In this model the trapped
arm props the body on its side instead, so the naive roll scores 0.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- **Fouls, at any time:** the lowest point of the trunk, pelvis and head more than 4 cm above the pad; or the spine
  (the line from the hips to the head) more than 30 degrees from level. The body stands for a full-size machine
  (about 35 kg): it is rolled, not picked up or stood on end. A roll about the long axis keeps the spine level; the
  demo's stays within 15 degrees of it. The trial then scores 0 whatever happens afterwards.
- **End state.** It counts only if the dummy is at rest (trunk under 1 cm/s and 0.5 rad/s, every joint under 0.5
  rad/s) and released (none of six points on each arm, its wrist joint, palm, both fingertips, tool point and the
  point between the pad tips, and none of the points sampled along the arm from its shoulder joint to its palm, within 5 cm of the
  dummy), both for each of the last 20 control steps (1 s at 20 Hz); otherwise 0.
- **A limb under the body:** any point of an upper arm, forearm, thigh, shank or foot, sampled every 5 mm along it,
  with the trunk or pelvis straight above it, leaving out the first 3 cm from the shoulder or hip joint (they flank the
  body). The head is not judged: the cleared arm lies beside it, and its weight on a limb is small.
- **Whole-trial rule:** from the first time the body has turned 45 degrees from face down on, an arm (upper arm or
  forearm) under it at any step marks it trapped, even if the body has turned back; pulling it out afterwards does not undo that. A leg under the pelvis is not judged
  during the turn, because a body on its side has its lower leg there.
  - 1.0: chest up (the trunk's front within 12 degrees of straight up), no limb under the body, and no arm trapped
    during the turn;
  - 0.25: chest up within 45 degrees, but a limb lies under the body, or an arm was trapped during the turn, or the
    trunk is tilted more than 12 degrees;
  - 0 otherwise, including lying on its side or still face down.
- A full score ends the trial once it has held for 1 s (`SUCCESS_HOLD_S`).

On a real table: a millimetre scale card stands beside the pad, and the side camera reads the trunk's lowest point (4
cm) and the spine's angle against it; the end state is checked with a photo from above (no limb hidden under the
body) and one from the side (the chest's tilt).

## Runtime and seeds

A full 1,200-step Dream-1 trial took about 3 minutes on an RTX 4070 Ti SUPER, Isaac Sim's start included.

The layout is fixed (`-E object_jitter=0`); there are no seeds to choose. The demo reaches full score in about 650
steps (33 s) and takes about 2 minutes on an RTX 4070 Ti SUPER, Isaac Sim's start included (add about 4 minutes on a
fresh machine while it compiles its shaders).

## Fidelity

Rigid links at 240 Hz physics, 20 Hz control. The dummy's sizes, mass shares and joint ranges are a 1:3 scale of a
public mid-size humanoid model (the Unitree G1 in MuJoCo Menagerie, BSD-3-Clause: lengths x 1/3, masses x 1/27); only
numbers are taken, and it is built from boxes, capsules and spheres. It has 14 hinges (shoulder pitch and roll, elbow,
hip pitch and roll, knee, ankle pitch per side): no waist, wrist, hip-yaw or neck joints. Its hinges are limp (damping
0.002 N m s/rad, no friction or spring; under a tenth of the gravity torque of the forearm held out level), with hard end
stops; its own parts collide with each other, so a limb can be trapped under the trunk. Friction: shell 0.6, soles
0.9, pad 0.6. The turf strip is drawn on the rigid pad, away from the work. No forces, motors or damage are simulated.
None of these simplifications changes what the task measures: whether a limb ends up under a rolled body depends on
where it lies relative to the edge the body turns about.

## Real-world build

- **Dummy:** 3D-printed PLA links (grey shell, darker joints) on steel pin hinges with hard printed end stops and no
  springs, to the sizes and masses in `dummy_spec.py` (masses brought up with steel shot glued inside the links); an
  orange 4 x 5 cm printed panel on the back. No electronics.
- **Pad:** a 43 x 50 x 1 cm plywood board clamped to the table, top painted earth brown; a 6 x 50 cm strip of flat
  artificial-grass mat (pile 2 mm or less) glued along its far edge.
- **Rocks:** two softwood blocks, 5 x 6 x 4.5 cm and 6 x 5 x 3.5 cm, painted rock grey, glued at the far corners.

## Licence

MIT (`LICENSE`). The dummy's numbers are derived from the Unitree G1 model in MuJoCo Menagerie (BSD-3-Clause); no
mesh, texture or other file of it is used. The ground textures are generated by `textures.py` (ours, MIT).
