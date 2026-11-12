# Fold a limp humanoid robot forward past its balance point so its back hatch faces up

A small humanoid robot, powered off, lies on its back on an earth pad between two loose concrete rocks that run
beside its trunk and thighs. It is a passive 1:3 model of a mid-size humanoid (44 cm, 1.24 kg, 14 limp joints with
hard end stops) with an orange service hatch on its back. The robot is told to leave it with the hatch facing up so
that it can be opened.

Between the rocks the body cannot be rolled over onto its front, which is how a fallen humanoid is usually turned
(TK-4J8HRQ). The only way to bring its back up is to fold it at the hips. Raised part of the way and let go, the trunk falls back onto its back; carried past its
balance point (about 95 degrees from lying) it folds forward onto its own thighs and rests there with its back up.
Unlike TK-T93HNS (a limp humanoid propped on its side by its own limbs), nothing props it here: how far the trunk is
carried decides it. The instruction mentions neither the fold nor the balance point.

Humanoid and legged robots fall and are recovered where they lie, more and more by other robots in cells and sites
that people are kept out of while a fallen machine is assessed. Reaching a service hatch without forcing a joint is
part of the job of the technicians who keep such robots running (O*NET 17-3024.01, Robotics Technicians).

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the pad, the two rocks and the robot, the whole-trial recorder, `score()` and `success()` |
| `dummy.py`, `dummy_spec.py` | The passive humanoid as one articulated body of boxes, capsules and spheres, and every number of it with its source |
| `trailkit.py` | Compound rigid and articulated bodies, textured tiles, the per-step recorder and scoring helpers; it hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena |
| `demo.py`, `demokit.py` | The scripted full-score demo (closed loop on the robot's measured pose) and its near-miss variants |
| `textures/`, `textures.py` | Earth and turf looks, and the code that generated them (not needed at run time) |
| `test_scoring.py` | Offline tests of the scorer on synthetic states (`python -m pytest test_scoring.py`, no simulator) |
| `LICENSE` | MIT |

The task runs with the standard `yam_arena` embodiment and the `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source <ARENA_DIR>/yam.env`), from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The humanoid robot on the pad lies on its back between two rocks. Leave it with its back hatch facing up so that the hatch can be opened." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/fold-forward-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out                                       # full score: demo_*.mp4, setup.png, result.json
python demo.py task.py out --variant sit_and_release --no-record      # raise it to 75 degrees and let go
python demo.py task.py out --variant roll_between_rocks --no-record   # try to roll it over by its far shoulder
python demo.py task.py out --variant roll_in_place --no-record        # roll it over about its own spine
python demo.py task.py out --variant legs_up --no-record              # fold the legs up instead of the trunk
python demo.py task.py out --variant idle --no-record                 # do nothing
```

1. The left hand takes the head across its width, just above its base, fingers closing along x; the head turns in
   the grip like a pin. (The wrists lie against the rocks: there is no room for fingers beside them.)
2. It carries the head round the hip axis (the hand's arc is set for 112 degrees; the trunk follows it to about
   107): the trunk rises through sitting and on past its balance point.
3. It lets go and backs off: the trunk folds on over the thighs and rests there, the hatch 51 degrees from straight up.

The demo reads only the robot's pose and joint angles (what a camera-based pose estimate gives) and the arm's own
joint state.

| Variant | What it does | Score |
|---|---|---|
| Dream-1 | the model, with the task's instruction, 1,200 steps | 0.0: the right arm reached in over the robot's legs and the left arm came near, but neither moved it; it stayed on its back |
| (none) | carry the trunk past its balance point by the head, let go | 1.0, three runs of three (success at step 369) |
| `sit_and_release` | the same, but let go at 75 degrees | 0.25: raised to 66 degrees, it falls back onto its back |
| `roll_between_rocks` | lift the far shoulder and push it toward the near rock, as for rolling a body onto its front | 0.0: the rocks stop the roll; it stays on its back |
| `roll_in_place` | carry the far upper arm round the spine's axis, up and over, 170 degrees | 0.0: the limp arm swings over alone; the body stays on its back |
| `legs_up` | fold both legs up over the trunk instead | 0.0: the trunk never rises; the swinging leg also moved a rock 1.07 cm (foul) |
| `idle` | nothing | 0.0 |

| Demo run | Score | Steps | IK misses (left, right) | Worst IK error |
|---|---|---|---|---|
| 1, 2, 3 | 1.0 each | 369 each | 79, 0 | 34 mm (left arm, at the top of the arc, 18 cm above the table) |

A harder roll, by a scripted force on the trunk itself (a separate check, not part of the task: a push at the far
shoulder housing, up and toward the near rock, rising 1 N/s to 6 N for 6 s), turned the body until the hatch was 76
degrees from up; it then rode up the rock and fell back onto its back, lifting the pelvis 5.5 cm and moving a rock 5.9
cm (both fouls). Rolled over, the 12.4 cm shoulders have to turn through their 13.4 cm diagonal inside the 15 cm gap
and against 6 cm walls.

The IK misses are the left hand's lag at the top of the head's arc, where it is near the arm's reach; the head turns in
the grip, so the lag only slows the carry. With the arc 1 cm lower (two runs) the trunk stopped short of its balance
point and fell back to sitting against the rocks (0.25), so the full arc is kept.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- **Fouls, at any time:** the lowest point of the pelvis more than 3 cm above the pad (the body stands for a
  full-size machine: it is folded where it lies, not lifted), or any corner of a rock moved more than 1 cm. The trial then scores 0
  whatever happens afterwards.
- **End state.** It counts only if the robot is at rest (its pelvis moved under 2 mm, it turned under 1 degree and no
  joint changed more than 2 degrees over the last 2 s: positions, not speeds) and released (none of six points on each
  arm's gripper and wrist, and none of the points sampled along each arm from its shoulder joint to its palm, within
  5 cm of the robot), for each of the last 40 control steps (2 s at 20 Hz); otherwise 0.
  - first, the pelvis (the point between the hip joints) must still be between the rocks' inner faces and have moved
    less than 4 cm, seen from above (the machine is folded where it lies, not dragged out);
  - 1.0: the back hatch faces up (its normal within 70 degrees of straight up) and no arm or leg segment lies on it
    (a point over the hatch's face, or within 5 mm around it, and less than 3 cm outside it);
  - 0.25: the hatch is not up or a limb lies on it, but the spine (the line from the hips to the head) rose more
    than 60 degrees above level at some time;
  - 0 otherwise, and always 0 if the pelvis has left its place between the rocks or moved 4 cm or more.
- Rest poses measured between the rocks, by the hatch's angle from straight up: on its back (180), folded forward
  (51) and sitting propped against the rocks (about 90, seen when the carry stopped short). By the trunk's turn
  about the hips from lying: let go before about 70 degrees it falls back; carried past about 95 it folds forward.
  The 70-degree hatch line separates the folded pose from the others. A hatch facing sideways on a propped, sitting
  body does not count: the instruction asks for it facing up.
- A full score ends the trial once it has held for 1 s (`SUCCESS_HOLD_S`).

On a real table: a protractor card stands beside the pad and a side photo reads the hatch's angle; a photo from above
shows the pelvis between the rocks against tape marks; the fouls are judged from the video against a millimetre scale.

## Runtime and seeds

The layout is fixed (`-E object_jitter=0`); there are no seeds to choose. The demo reaches full score in 369 steps
(18 s) and takes about 4 minutes on an NVIDIA L40S, Isaac Sim's start included.

## Fidelity

Rigid links at 240 Hz physics, 20 Hz control. The robot's sizes, mass shares and joint ranges are a 1:3 scale of a
public mid-size humanoid model (the Unitree G1 in MuJoCo Menagerie, BSD-3-Clause: lengths x 1/3, masses x 1/27); only
numbers are taken, and it is built from boxes, capsules and spheres. It has 14 hinges (shoulder pitch and roll,
elbow, hip pitch and roll, knee, ankle pitch per side) and no waist joint; its hinges are limp (damping 0.002 N m
s/rad, no friction or spring) with hard end stops, and its own parts collide with each other. Friction: shell 0.6,
soles 0.9, pad 0.6, rocks 0.7. The turf strip is drawn on the rigid pad. No forces, motors or damage are simulated.
None of this changes what the task measures: where a limp body comes to rest depends on whether it was carried past
its balance point.

## Real-world build

- **Robot:** 3D-printed PLA links (grey shell, darker joints) on steel pin hinges with hard printed end stops and no
  springs, to the sizes and masses in `dummy_spec.py` (masses brought up with steel shot glued inside the links); an
  orange 4 x 5 cm printed hatch panel on the back. No electronics.
- **Pad:** a 43 x 50 x 1 cm plywood board clamped to the table, top painted earth brown; a 5 x 50 cm strip of flat
  artificial-grass mat (pile 2 mm or less) glued along its far edge.
- **Rocks:** two cast-concrete blocks 4 x 26 x 6 cm (about 1.4 kg each), laid loose on the pad.

## Licence

MIT (`LICENSE`). The robot's numbers are derived from the Unitree G1 model in MuJoCo Menagerie (BSD-3-Clause); no
mesh, texture or other file of it is used. The ground textures are generated by `textures.py` (ours, MIT).
