# Right a rover that has rolled onto its side at the edge of a drop-off

A raised earth track runs away from the robot. Its right-hand long edge is a 5 cm drop to the table, with a bare rock
face. A small trail crawler (a real 1:18 crawler, 26 x 12 x 12 cm, 0.5 kg) has rolled onto its left side on the
track, nose away from the robot, its tyres toward the drop-off and 4 cm from the edge. The robot is told only to get
it back on its wheels.

Tipping it where it lies fails. A vehicle on its side is righted by tipping it over its lower tyres, and that pivot carries
it a full body-width sideways: tipped where it lies, its far tyres land past the edge and it goes over. The right
way is the one a recovery marshal uses: first move it back from the edge (slide or drag it along the track), then
tip it onto its wheels; or turn it about its middle, held by both bumpers, so that it does not travel toward the edge.
Lifting it bodily is also allowed (a 0.5 kg crawler hangs heavily from one 9 x 1.6 cm bumper bar) and scores by where
it is set down: the scoring rewards the outcome, a rover on its wheels clear of the edge, however it is reached. The instruction does not mention the edge or the drop, and nothing in the task is hard to
grasp: the test is foreseeing where a body lands
when it is toppled.

Field robots roll over on slopes, rock and verges, and are righted by whoever reaches them: a person today, more and
more another robot (a base station's arm, a second vehicle). Recovering a vehicle without causing a second accident
is the core of the job for off-road marshals and vehicle recovery crews, and of the technicians who keep field
robots running (O*NET 17-3024.01, Robotics Technicians).

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the track and rover, the whole-trial recorder, `score()` and `success()` |
| `trailkit.py` | Compound rigid bodies (boxes, cylinders, textured tiles), the per-step recorder and scoring helpers; it hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena |
| `rover.py`, `rover_spec.py` | The rover as one rigid body, and every number of it with its source |
| `demo.py`, `demokit.py` | The scripted full-score demo (closed loop on the rover's measured pose) and its near-miss variants |
| `textures/`, `textures.py` | Earth and turf looks, and the code that generated them (not needed at run time) |
| `LICENSE` | MIT |

The task runs with the standard `yam_arena` embodiment and the `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source <ARENA_DIR>/yam.env`, as set up by the walkthrough), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The rover has rolled onto its side on the track. Get it back on its wheels." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/right-rover-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out                                   # full score: demo_*.mp4, setup.png, result.json
python demo.py task.py out --variant tip_in_place --no-record     # tip it over where it lies
python demo.py task.py out --variant drag_only --no-record        # slide it back from the edge, never right it
python demo.py task.py out --variant idle --no-record             # do nothing
```

1. The right arm, fingers closed and pointing down, sets them against the rover's chassis on the drop-off side and
   pushes it along the track, away from the edge, until its treads are 18 cm from the edge. The push is aimed
   through the rover's centre of mass and shifted along it to cancel any turn (closed loop on the measured pose).
2. The right arm withdraws. The left arm, fingers closed, presses on the upper part of the roof, square to it, and
   follows it round as the rover turns over its lower tyres, until it passes its balance point and drops onto its
   wheels.
3. Both arms withdraw.

The demo reads only the rover's pose (what a camera-based pose estimate gives) and the arms' own joint state.

| Variant | What it does | Score |
|---|---|---|
| (none) | push back from the edge, then tip onto the wheels | 1.0 (success at step 384) |
| `tip_in_place` | the naive way: tip it over where it lies | 0.0 (the rover goes over the edge at step 120) |
| `drag_only` | push it back from the edge (treads 18 cm from it) and leave it on its side | 0.25 |
| `idle` | nothing | 0.0 |

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- **Foul, at any time:** the rover leaves the deck, that is its lowest point is more than 1 cm below the deck top
  (over any of the deck's edges). The trial then scores 0 whatever happens afterwards.
- **End state**, with the rover at rest (under 1 cm/s and 0.5 rad/s for the last 20 control steps, 1 s at 20 Hz) and
  released (none of six points on each arm, its wrist joint, palm, both fingertips, tool point and the point between
  the pad tips, and no point along the arm from its shoulder joint to its palm, within 5 cm of the rover); otherwise 0:
  - 1.0: upright (roof within 10 degrees of level), all four tyres resting on the deck (each tyre's lowest point
    within 1 cm of the deck top and inside the deck's outline), every tyre at least 2 cm from the drop-off edge;
  - 0.5: upright with all four tyres on the deck, at least one closer than 2 cm to the edge;
  - 0.25: still lying on a side (its axles within 30 degrees of vertical), but its nearest point at least 8 cm farther from the
    drop-off edge than at the start (moved back but not righted);
  - 0 otherwise, including lying where it started or ending on its roof.
- A full score ends the trial once it has held for 1 s (`SUCCESS_HOLD_S`).

On a real table: the foul is seen in the video (the rover drops off the deck); the end state is checked with a
photo and a tape measure from the edge.

## Runtime and seeds

A full 1,200-step Dream-1 trial took 4.7 minutes on an L40S, Isaac Sim's start included (add 5 minutes on a fresh
instance while it compiles its shaders); the demo takes about 1.5 minutes. The layout is fixed (`-E object_jitter=0`); there are no seeds to choose. The demo reaches full score in
about 400 steps (20 s).

## Fidelity

Rigid bodies at 240 Hz physics, 20 Hz control. The rover is one rigid body of boxes and cylinders at the crawler's
published size (26.2 cm long, 12.4 cm wide over the tyres, 11.7 cm tall, 15.5 cm wheelbase) and mass (0.468 kg
plus a 40 g dummy battery), with its centre of mass 4.8 cm above the ground; numbers we could not find published
(tyre size, body-shell outline, centre of mass) are estimates, marked so in `rover_spec.py`. Its suspension is rigid
and its wheels do not turn, as with the real crawler's drivetrain engaged and the motor off. Friction: tyres 1.0,
body 0.5, deck 0.6. On its side the rover rests on its tyre sidewalls and leans onto its cab when pushed, as the real
one does. The grass strip is drawn on the deck surface; the real mat's pile (2 mm or less) is within the scoring's 1 cm tyre
tolerance. Leaving the deck is judged from the rover's geometry (its lowest point), as it would be from a video. No forces
or damage are simulated. None of these simplifications changes what the task measures: where a toppled body lands
depends only on its width and where it pivots.

## Real-world build

- **Rover:** a stock 1:18 trail crawler (Traxxas TRX-4M, Bronco body), decals removed and the shell painted yellow
  with a white roof; its battery replaced by a dummy of the same mass; motor off, drivetrain engaged. No electrics
  are live during a trial.
- **Track:** a 38 x 36 cm plywood deck on 3.2 cm battens (5 cm in all), top painted earth brown, the right-hand
  sides painted rock grey; a 6 x 38 cm strip of flat artificial-grass mat (pile 2 mm or less) glued along the
  left-hand edge. Clamp the
  deck to the table at the positions in `task.json`.
- The drop is 5 cm onto the table: the rover is not damaged by the foul it scores.

## License

MIT (`LICENSE`), the task authors. No third-party assets: all shapes and textures are generated by code in this
folder.
