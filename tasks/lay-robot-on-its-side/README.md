# Lay a limp humanoid robot on its side so that it stays there

A small humanoid robot, powered off, lies on its back on a packed-earth pad, along the table's width, head to the
robot's right, arms straight along its trunk and legs straight. It is a passive 1:3 model of a mid-size humanoid
(44 cm, 1.24 kg, 14 limp joints with hard end stops). The robot is told to lay it on its left side so that it stays
there on its own.

On its side the body balances on a narrow edge. Laid there with its limbs straight and let go, it rolls back off its
side within 2 s (measured: the `roll_and_release` variant below, let go at step 239, flat on its back by step 280).
It stays when its own upper limbs prop it,
as in a casualty's recovery position: the upper arm brought forward in front of the chest, or the upper knee forward
onto the ground. Each prop is a limb on free joints, placed through those joints. In this model either prop alone is
enough (measured: `arm_only` 1.0; in `knee_only`, carried by the head with only the knee brought forward, it
stays on its side, still at 4 degrees to the end of the run); the demo places both, as first aiders are taught.
The instruction names no prop; the test is foreseeing that a limp body will not stay where it is put, and using its
own limbs to hold it.

Humanoid and legged robots fall, and are recovered where they lie, by technicians today and more and more by other
robots. Laying a powered-off one on its side, where neither its back panel nor its face is against the ground, and so
that it stays there while it waits, is part of the job of the technicians who keep field and factory robots running (O*NET 17-3024.01, Robotics
Technicians).

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
  --instruction "Lay the humanoid robot on the pad on its left side so that it stays there on its own." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/lay-on-side-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out                                       # full score: demo_*.mp4, setup.png, result.json
python demo.py task.py out --variant roll_and_release --no-record     # carry it onto its side by the head and let go
python demo.py task.py out --variant arm_only --no-record             # bring only the far arm forward
python demo.py task.py out --variant knee_only --no-record            # carry it by the head, bring only the far knee forward
python demo.py task.py out --variant hold_to_the_end --no-record      # the full method, but never let go
python demo.py task.py out --variant idle --no-record                 # do nothing
```

The body is rolled toward the robot, about its near edge, onto its left side; its right (upper) arm and leg become the
props. The demo is the first-aid recovery roll, with the far limbs as handles:

1. **Take the far limbs.** The right hand takes the far forearm 4.5 cm past the elbow, fingers closing across it; the
   left hand takes the far thigh 2 cm short of the knee, pointing down.
2. **Roll.** Together, over 8 s, the two hands turn the body 85 degrees round its near edge. The right hand moves as if
   fixed to the forearm, raising the arm as the body turns, so that when the body is on its side the arm points
   forward out of the chest. The left hand carries the knee up, across the near leg and down in front. The dummy's
   shoulder has no twist joint: a hand that kept pointing down would twist the arm and with it the whole body.
3. **Let go.** Both hands open and withdraw. The body settles 5 degrees forward onto its knee and forearm and stays
   (watched for 43 s after release, unchanged).

The demo reads only the dummy's pose and joint angles (what a camera-based pose estimate gives) and the arms' own
joint state.

| Variant | What it does | Score |
|---|---|---|
| Dream-1 | the model, with the task's instruction, 1,200 steps | 0.0: both arms came in over the trunk and head and pressed on it; the body stayed on its back |
| (none) | the recovery roll: far arm and knee brought forward as the body turns, then let go | 1.0, three runs of three (success at step 435; on its side at 5 degrees) |
| `roll_and_release` | the naive way: grip the head, carry the body onto its side (8 degrees from it), hold 1.5 s, let go | 0.25: on its side for 1 s, then it rolls back onto its back |
| `arm_only` | the recovery roll with the far leg kept straight: only the arm is brought forward | 1.0: on its side at 10 degrees, stays |
| `knee_only` | grip the head, carry the body onto its side while the left hand brings the far knee forward; let go | 0.25: it stays on its side (4 degrees), but part of it ends off the pad |
| `hold_to_the_end` | the full method, but the hands never let go | 0.25: on its side, but still held |
| `idle` | nothing | 0.0 |

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- **Fouls, at any time:** the lowest point of the trunk, pelvis and head more than 4 cm above the pad; or the spine
  (the line from the hips to the head) more than 30 degrees from level. The body stands for a full-size machine
  (about 35 kg): it is rolled, not picked up or stood on end. The demo lifts the trunk at most 2.2 cm and tilts the
  spine at most 9 degrees; every variant run stays under 2.6 cm (the lower arm folds in under the side as the body
  turns). The trial then scores 0 whatever happens afterwards.
- **On its left side:** the line from the right shoulder joint to the left within 25 degrees of straight down.
- **End state.** 1.0 if the dummy is on its left side, at rest (trunk under 1 cm/s and 0.5 rad/s, every joint under
  0.5 rad/s) and released (none of six points on each arm, its wrist joint, palm, both fingertips, tool point and the
  point between the pad tips, and none of the points sampled along the arm from its shoulder joint to its palm, within
  5 cm of the dummy), both for each of the last 40 control steps (2 s at 20 Hz); every point of it over the pad and no
  more than 2 mm below the pad's top; and no part of it within 1 cm of a rock (it rests on the pad and its own limbs
  only, not on the table or against the pad's edge or a rock).
- **0.25** if it was on its left side for 20 control steps in a row (1 s) at some time, but at the end it rolled off,
  is still held, is still moving, leans on a rock or is not wholly on the pad.
- **0** otherwise, including lying on its right side, front or back throughout.
- A full score ends the trial once it has held for 3 s more (`SUCCESS_HOLD_S`; 5 s still in all). If the run reaches
  its limit first, the end state is judged on the last 2 s.

On a real table: the end state is read from a photo from the side (the shoulder line against vertical, and a gap under
each gripper) and one from above (the rocks' clearance); a millimetre scale card beside the pad gives the trunk's
lowest point (4 cm).

## Runtime and seeds

A full 1,200-step Dream-1 trial took about 3 minutes on an RTX 4070 Ti SUPER, Isaac Sim's start included.

The layout is fixed (`-E object_jitter=0`); there are no seeds to choose. The demo reaches full score in about 435
steps (22 s) and takes about 2 minutes on an RTX 4070 Ti SUPER, Isaac Sim's start included (add about 4 minutes on a
fresh machine while it compiles its shaders).

## Fidelity

Rigid links at 240 Hz physics, 20 Hz control. The dummy's sizes, mass shares and joint ranges are a 1:3 scale of a
public mid-size humanoid model (the Unitree G1 in MuJoCo Menagerie, BSD-3-Clause: lengths x 1/3, masses x 1/27); only
numbers are taken, and it is built from boxes, capsules and spheres. It has 14 hinges (shoulder pitch and roll, elbow,
hip pitch and roll, knee, ankle pitch per side; zero with the limbs straight along the body; only roll mirrors between the sides): no waist, wrist, hip-yaw, shoulder-yaw or neck joints. Its hinges are
limp (damping 0.002 N m s/rad, no friction or spring; under a tenth of the gravity torque of the forearm held
out level; a real powered-off robot's geared joints have some back-drive friction), with hard end stops; its own parts collide with each other. Friction: shell 0.6, soles 0.9, pad 0.6. The turf strip is
drawn on the rigid pad, away from the work. No forces, motors or damage are simulated. None of these simplifications
changes what the task measures: whether a limp body stays on its side depends on where its free limbs lie relative to
the edge it rests on.

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
