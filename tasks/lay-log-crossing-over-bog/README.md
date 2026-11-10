# Lay a corduroy log mat across a bog and push the rover over it

An earth track runs away from the robot on two low decks, 2 cm high, with an 8.3 cm bog between them (dark wet mud at
table level). A small trail crawler (a real 1:18 crawler, 26 x 12 x 12 cm, 0.5 kg, its motor pinion removed so the
wheels roll freely) stands on the near deck facing the bog. Four logs lie on the turf verges beside the track. The
robot is told to lay the logs, then push the rover across onto the far bank.

Trail crews cross soft ground with a corduroy mat: logs laid crosswise and tight between the banks, so that none can
roll or drop away under a wheel. Here the four logs fill the bog exactly when laid that way, side by side, each seated on
the mud; a log missing, askew or riding up leaves a wheel in the mud. The mat is then proved by pushing the free-rolling rover across
onto the far deck; a wheel that reaches the bog fouls the trial.

Trail crews (O*NET 45-4011.00, Forest and Conservation Workers) build corduroy and puncheon crossings on soft ground,
and field robots that keep their own routes open will need to build the footing a wheel needs.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: decks, bog, verges, the rolling rover and the logs; the whole-trial recorder; `score()` and `success()` |
| `trailkit.py` | Compound rigid bodies, articulated links (the rover's wheels), textured tiles, the per-step recorder and scoring helpers; it hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena |
| `rover.py`, `rover_spec.py` | The rover with free-rolling wheels, and every number of it with its source |
| `demo.py`, `demokit.py` | The scripted full-score demo and its near-miss variants |
| `textures/`, `textures.py` | The earth and turf looks, and the code that generated them |
| `LICENSE` | MIT |

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source <ARENA_DIR>/yam.env`, as set up by the
walkthrough), from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The track ahead is a bog. Lay the logs, then push the rover across onto the far bank." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/corduroy-dream
```

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 2400                          # full score
python demo.py task.py out --variant logs_only --no-record --max-steps 2400   # lay the mat and stop
python demo.py task.py out --variant push_only --no-record                # push the rover straight into the bog
python demo.py task.py out --variant idle --no-record                     # do nothing
```

1. The arms take the logs one by one from the verges (the left arm those on the left, the right arm those on the
   right), each gripped across its middle, and drop each into its slot across the bog from just above the decks, far
   side first: the mat is exactly four logs wide, so there is no room for the fingers beside a log being set down.
2. The right arm tamps the row tight toward the far bank with its closed fingers on the nearest log. A log left
   riding on its neighbours is pressed down into its slot with the closed fingertips, and the row tamped again.
3. The right arm, fingers closed and pointing forward, pushes the rover by the middle of its rear bumper (closed loop
   on its measured pose) across the mat onto the far deck, then withdraws.

The demo reads only object poses (the rover and the logs) and the arms' own joint state.

| Variant | What it does | Score |
|---|---|---|
| (none) | lay and tamp the mat, push the rover across | 1.0 |
| `logs_only` | lay and tamp the mat, leave the rover | 0.5 |
| `push_only` | push the rover without a mat | 0.0 (a tyre touches the bog) |
| `idle` | nothing | 0.0 |

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- **Fouls, at any time:** a tyre touches the bog (its lowest point within 4 mm of the bog sheet's top, over the bog); a tyre is over the bog with no log under it (no log lying crosswise in the bog within 1.6 cm of the tyre's lowest point along the track, under it across the track, with the tyre within 4 mm of the log's top, the log lying in the bog and not held up); a tyre is over the bog before all four logs are laid as for the 0.5 credit below; a tyre leaves the track (its lowest point outside the decks and the bog seen from above: the rover may not go round); the rover moves faster than 0.3 m/s (it could jump the bog between control steps); or all four tyres are more than 8 mm above the surfaces under them (deck tops, the top of a log lying in the bog over the bog, the table elsewhere): the rover lifted. The trial then scores 0 whatever happens afterwards.
- **End state**, with the rover and the logs at rest (each under 1 cm/s and 0.5 rad/s for the last 20 control steps,
  1 s at 20 Hz) and released (none of six points on each arm, its wrist joint, palm, both fingertips, tool point and
  the point between the pad tips, and no point along the arm from its shoulder joint to its palm, within 5 cm of the rover or a log); otherwise 0:
  - 0.5: all four logs lie in the bog crosswise (within 10 degrees of the track's width and of level), resting on the bog (centre no more than 1.4 cm above the table), each spanning both wheel paths, with no gap wider than 5 mm between neighbouring logs or at the decks (taken corner to corner, 2.0 cm per log; the bog leaves 3 mm of play for four logs, so the gap test catches a log missing, askew or riding up);
  - 1.0: in addition, all four tyres rest on the far deck;
  - 0 otherwise.
- A full score ends the trial once it has held for 1 s.

On a real table: a wheel in the bog is seen in a side video; the mat is checked with a ruler and a photo.

## Runtime and seeds

A full 2,400-step Dream-1 trial takes about 7 minutes on an L40S, Isaac Sim's start included. The layout is fixed
(`-E object_jitter=0`); there are no seeds to choose. The demo reaches full score in 1,596 steps (80 s).

## Fidelity

Rigid bodies at 240 Hz physics, 20 Hz control. The bog is a rigid sheet: a real bog would let a wheel sink; here a
wheel that reaches it fouls the trial, judged from the tyres' height. The rover's base is one rigid body at the
crawler's published size and mass (tyre size and centre of mass estimated); its wheels turn freely on revolute joints
with some drag (0.006 N m s/rad), as with the motor pinion removed; its suspension is rigid. The logs have a six-sided
section (2.0 cm across the corners) and angular damping of 4.0 (PhysX units, heavy), standing in for a real log's uneven bark: a perfect
cylinder rolls away at the slightest touch in a rigid-body simulator, which real logs on turf do not. No forces or
damage are simulated.

## Real-world build

- **Decks:** two plywood decks 2.0 cm tall and 24 cm wide, clamped to the table 8.3 cm apart, tops painted earth
  brown. **Bog:** a sheet of dark brown rubber between them. **Verges:** two pieces of artificial turf.
- **Rover:** a stock 1:18 trail crawler (Traxxas TRX-4M, Bronco body), decals removed, painted yellow with a white
  roof; dummy battery; motor pinion removed. No electrics are live.
- **Logs:** four hexagonal dowels of a dense hardwood (jatoba, about 900 kg/m3), 2.0 cm across the corners, 18 cm
  long, 42 g each, natural brown.

## License

MIT (`LICENSE`), the task authors. No third-party assets: all shapes and textures are generated by code in this
folder.
