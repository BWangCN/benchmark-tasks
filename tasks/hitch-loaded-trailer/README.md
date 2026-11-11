# Hold a fallen trailer tongue up while rolling the heavy rover back under it, then lower it onto the ball

A loaded two-wheeled trailer (9 kg) stands chocked on an earth track, its tongue resting on the ground with a socket
at its tip. A loaded rover (built heavy at the size of a 1:18 trail crawler, 14 kg; free-rolling wheels) stands 12 cm
ahead of the socket, its back to the trailer, with a tow ball on a post at its rear, 8.4 cm up. The robot is told to hitch
the trailer to the rover.

Neither body can be carried to the other: the trailer is chocked and its tongue falls when let go, and the rover can
be rolled but not lifted. One hand has to carry the tongue's tip (about 5 N) at the height of the ball while the other
rolls the rover back until the ball is under the socket, and the tongue is then lowered onto the ball. Backed up with
the tongue on the ground, the rover passes over the socket; dropping the tongue onto the ball is a foul.
The scoring looks only at the result and at those fouls.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: track, chocked trailer with its socket, the loaded rover with its tow ball; the whole-trial recorder; `score()` and `success()` |
| `trailkit.py` | Compound rigid bodies, articulated links with stick-slip joint friction, textured tiles, the per-step recorder and scoring helpers; it hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena |
| `rover.py`, `rover_spec.py` | The loaded rover, and every number of it with its source |
| `demo.py`, `demokit.py` | The scripted full-score demo and its near-miss variants |
| `textures/`, `textures.py` | The earth look, and the code that generated it |
| `LICENSE` | MIT |

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source <ARENA_DIR>/yam.env`), from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Hitch the trailer to the rover." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/hitch-dream
```

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1600                    # full score
python demo.py task.py out --variant back_up_only --no-record        # back the rover up, the tongue left on the ground
python demo.py task.py out --variant lift_and_drop --no-record       # hold the tongue high, back up, let it fall
python demo.py task.py out --variant small_drop --no-record          # let it fall from the ball's top (about 1 cm)
python demo.py task.py out --variant tiny_drop --no-record           # let it fall from 6 mm above the ball's top
python demo.py task.py out --variant overshoot_drop --no-record      # back the rover 5 cm too far, drop the tongue on its bar
python demo.py task.py out --variant idle --no-record                # do nothing
```

1. The left arm grips the trailer's tongue 6 cm behind the socket and lifts it until the socket's rim is 1.5 cm above
   the rover's tow ball.
2. The right arm, fingers closed against the end of the rover's front tow tab, pushes the rover backward until the
   ball is 9 mm short of the socket (closed loop on the measured poses): the socket swings that far toward the rover as
   the tongue comes down. Then it withdraws.
3. The left arm lowers the tongue at 1 cm/s until the socket has stopped on the ball (its rim less than 2 mm above the
   ball's centre, and still), opens and withdraws.

The demo reads only the rover's and the trailer's poses and the arms' own joint state.

| Variant | What it does | Score | Steps | IK misses (left / right) |
|---|---|---|---|---|
| (none), 3 runs | hold the tongue up, push the rover under, lower | 1.0, 1.0, 1.0 (seated 2.7 mm off the socket's centre; no drop in the zone) | 540 | 0 / 0 |
| `small_drop` | let go with the socket's rim level with the ball's top | 0.0 (falls 14 mm in the 0.15 s window, onto the ball, at up to 0.22 m/s: foul) | 566 | 0 / 0 |
| `tiny_drop` | let go with the rim 6 mm above the ball's top | 0.0 (falls 13 mm in the 0.15 s window, onto the ball, at up to 0.22 m/s: foul) | 564 | 0 / 0 |
| `lift_and_drop` | hold the tongue 5 cm above the ball, push the rover under, let go | 0.0 (falls 10 mm in the 0.15 s window, onto the ball, at up to 0.20 m/s: foul) | 529 | 0 / 0 |
| `overshoot_drop` | hold the tongue 5 cm above the ball, back the rover 5 cm past the socket, let go | 0.0 (the bar falls 6 mm onto the ball at up to 0.12 m/s: foul) | 536 | 0 / 0 |
| `back_up_only` | push the rover back with the tongue on the ground | 0.0 (the rover passes over the socket) | 328 | 0 / 0 |
| `idle` | nothing | 0.0 | 670 | 0 / 0 |

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- **Fouls, at any time**, the trial then scoring 0:
  - the tongue falls onto the ball: either its socket (while its rim is less than 15 cm above the ball's top, or
    below it, and its centre, the middle of its opening at its rim's height, within 2.6 cm of the ball's centre
    horizontally) or its bar's underside straight above the ball (while the ball is under the bar and that point is
    less than 15 cm above the ball's top) falls more than 4 mm over 3 control steps (0.15 s) at a peak speed over
    0.08 m/s (its sampled speed, or one control step's fall divided by the step's 0.05 s). Lowering the tongue away
    from the ball, at any speed, does not count;
  - both tyres of a rover axle more than 1 cm off the ground for 0.5 s (lifted).
- **End state**, after 2 s with the rover and the trailer at rest (under 5 mm/s, the trailer turning under 0.05 rad/s) and released (no point of either
  gripper, and no point along either arm from its shoulder joint to its palm, within 5 cm of the rover, its wheels and
  steering, or the trailer's body):
  - 1.0: the socket seated on the ball: the ball's centre within 7.5 mm of the socket's centre horizontally (inside the square cup)
    and the
    socket's rim 0 to 6 mm below the ball's centre (its cap resting on the ball);
  - 0.5: the ball's centre within 1 cm of the socket's centre (the middle of its opening, at its rim's height), not
    seated;
  - 0 otherwise.
- A full score ends the trial once it has held for 1 s.

On a real table: the end state by eye (the cup over the ball, the tongue resting on it) and the drop from an overhead
video.

## Runtime and seeds

A full 1,600-step Dream-1 trial takes about 5 minutes on an L40S, Isaac Sim's start included. The layout is fixed
(`-E object_jitter=0`); there are no seeds to choose. The demo reaches full score in 540 steps (27 s).

## Fidelity

Rigid bodies at 240 Hz physics, 20 Hz control. The rover's base is one rigid body at the crawler's published size
(`rover_spec.py`), 14 kg with its centre of mass 5.5 cm up, so an arm cannot lift an axle; its wheels turn freely with
drag (0.02 N m s/rad). The trailer is one rigid body (bed, load, tongue and socket) on two free wheels, chocked fore
and aft (the chocks are fixed); its centre of mass 1.2 cm ahead of the axle gives the tongue's tip about 5 N. The socket is a cup of boxes
over a sphere: no latch is modelled; the tongue's weight keeps it seated. Friction: tyres 1.0, track 0.8, socket 0.5,
ball 0.4, tongue 0.8.

## Real-world build

- **Track:** an earth-brown sheet of painted PVC foil 4 mm thick, taped to the table, x 14 to 46 cm, y -42 to 52.
- **Trailer:** a two-wheeled trailer of painted aluminium sheet (bed 13 x 16 cm, wheels 7 cm across on a 16 cm track)
  with its load: a lead block 12 x 7 x 7.2 cm (6.8 kg) bolted across the bed's front, from 5 cm ahead
  of the axle to 2 cm behind it (its centre 1.5 cm ahead), and a box 10 x 10 x 6 cm of empty sheet steel on the bed,
  9 kg in all, balanced so the tongue's tip bears about 5 N; the
  tongue a 1.6 x 1.4 cm aluminium bar 20 cm long ending in a square steel cup 2.6 cm across inside, 1.2 cm deep.
  Four yellow chocks 3 x 3 x 2.5 cm screwed to the track.
- **Rover:** built for the rig at the size of a 1:18 trail crawler: aluminium chassis, steel axles, the body shell of
  a Traxxas TRX-4M (painted yellow, white roof), about 13 kg of lead cast to fill the body, 14 kg in all; free wheels.
- **Tow ball:** a 1.6 cm steel ball on a 2.2 cm post screwed into the rear tow tab (a dark grey 3D-printed PLA tab
  5.0 x 1.6 x 1.6 cm).

## License

MIT (`LICENSE`), the task authors. No third-party assets: all shapes and textures are generated by code in this
folder.
