# Get a rover out from under a collapsed slab without bringing it down

A loaded rover (built heavy at the size of a 1:18 trail crawler, 14 kg; free-rolling wheels) is stranded under a
collapsed concrete slab. The slab bridges over it, 1.3 cm above its roof: its near end on a square concrete block, its
far end on a long smooth steel pipe lying on its side on a debris face that falls toward the rover. A fallen timber lies
across the rover's way out, its far end wedged against the pipe. A bent bar stands out of the slab's top near the pipe.
The robot is told to get the rover out from under the rubble onto the pad (ahead of it, past the pile; drawn out backward, the 14 kg rover
ends on the wrong side of the pile, and its tyres do not slide sideways).

Which piece carries which has to be read from the pile. The timber is the obvious thing to clear, since it is in the
way; but it is what keeps the pipe on its face, and while the slab bears on the pipe the timber is jammed: an arm
cannot lift it, and the rover drawn out runs into it. Held by its bar, the slab's load comes off the pipe and the
timber lifts out; the pipe then rolls onto a lip 2 cm down the face and the slab's far end settles 1.6 cm, 8 mm
toward the rover's roof (the lip stops it short of the roof, held or not). The rover is then drawn out along its
wheels onto the pad. The scoring looks only at positions.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: track and pad, the pile (slab with its bar, block, pipe on its debris face with its lip, the timber), the loaded rover; the whole-trial recorder; `score()` and `success()` |
| `rubble.py` | The pile's pieces as rigid bodies, and a point sampler for the "near the rover" test |
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
  --instruction "Get the rover out from under the rubble onto the pad." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/rubble-dream
```

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 2400                  # full score
python demo.py task.py out --variant timber_first --no-record     # lift the timber out of the way first
python demo.py task.py out --variant pull_only --no-record        # draw the rover out with the timber in place
python demo.py task.py out --variant idle --no-record             # do nothing
```

1. The right arm takes the bar on the slab's top, the tool tilted 15 degrees, and holds the slab's end where it is.
2. The left arm lifts the timber at its middle, carries it away from the rover and over to the far side, and sets it
   down. The pipe rolls onto its lip; the slab stays in the right hand.
3. The left arm takes the rover's front tow tab from above and draws it out along its wheels onto the pad.
4. The right arm lowers the slab's end onto the pipe and lets go.

The script reads only the poses of the rover, the slab and the timber, and the arms' own joint state.

| Variant | What it does | Score | Steps | IK misses (left / right), worst |
|---|---|---|---|---|
| (none), 3 runs | hold the slab by its bar, lift the timber aside, draw the rover out, lower the slab | 1.0, 1.0, 1.0 (as the pipe rolls, the held slab turns about the grip; it comes 8.0 mm closer to the rover, under the 1 cm limit) | 800 | 52 / 31, 16.7 / 12.4 mm (the left arm at the timber's far reach, the right at the bar, tilted) |
| `timber_first` | lift the timber first, the slab unheld | 0.0 (the slab bears on the pipe and the timber is jammed: the arm cannot lift it; the rover then runs into it) | 1,332 | 5 / 0, 16.7 mm |
| `pull_only` | draw the rover out with the timber in place | 0.0 (the rover runs into the timber at step 111) | 1,070 | 0 / 0 |
| `idle` | nothing | 0.0 | 670 | 0 / 0 |

Premise without arms: with the timber taken away at the start and the slab unheld, the pipe rolls to its lip and the
slab's far end comes down 15.6 mm, three runs of three (the scene's probe flag `PREMISE_NO_TIMBER`); it comes 8.0 mm
closer to the rover and stops on the pipe at its lip. Held by its bar, the slab comes down the same way (it turns
about the grip). So the hold is not what keeps the slab off the rover: the lip does. What the hold does is take the
slab's load off the timber: unheld, the slab bears on the pipe, the pipe on the timber, and the timber is jammed
(`timber_first`). The order that works is hold, then lift the timber, then draw the rover out.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- **Fouls, at any time**, the trial then scoring 0: while the rover is under the slab (any point of its rear treads
  short of the slab's edge at y 8 cm), the slab more than 1 cm closer to the rover than where the scene put them (its
  nearest point to the rover's body, 13 mm at the start), held or not; once the rover is out, the slab's lowest point
  more than 1 cm below its reference while no gripper holds its bar (the reference: where the slab was at the last
  step at which a gripper held its bar with the rover out; where the scene put it if there was none; holding: the point
  between the pad tips within 2.5 cm of the bar's middle and the fingers shut on it, under 1.5 cm apart); any piece of
  the pile (slab, pipe, block, timber) within 2 mm of the rover's body.
- **End state**, after 2 s with the rover at rest (under 5 mm/s and 0.05 rad/s) and released (no point of either
  gripper, and no point along either arm from its shoulder joint to its palm, within 5 cm of the rover, its wheels and
  steering; no gripper holding the slab's bar):
  - 1.0: all four tyres on the pad (every point of each tread past y 22 cm, inside the track, the lowest point within
    5 mm of it);
  - 0.5: the rover out from under the slab (every point of both rear treads past the slab's edge at y 8 cm);
  - 0 otherwise.
- A full score ends the trial once it has held for 1 s.

On a real table: the slab's drop with a ruler against a mark on the block, the end state from a photo.

## Runtime and seeds

A full 2,400-step Dream-1 trial takes about 7 minutes on an L40S, Isaac Sim's start included. The layout is fixed
(`-E object_jitter=0`); there are no seeds to choose. The demo reaches full score in 780 steps (39 s).

## Fidelity

Rigid bodies, 240 Hz physics, 20 Hz control. The pipe's friction on its plate and on the slab is 0.25 (smooth steel): a round prop pinched under a load holds by friction while tan(slope/2) is below its friction, so a rough prop would stay without the timber; the steep face and smooth pipe make it roll. Probe: with the timber gone the slab comes down 15.6 mm (3 of 3). The rover is 14 kg with its centre of mass 5.5 cm up, on free wheels with drag (0.02 N m s/rad).

## Real-world build

- **Track:** an earth-brown sheet of painted PVC foil 0.6 mm thick, taped to the table, x 10 to 42 cm, y -30 to 56,
  with a white tape line (6 mm) across it at y 22 (the pad beyond).
- **Pile:** a precast concrete paver cut to 32 x 16 x 1.2 cm with a steel bar 1 x 3 x 3 cm set in its top; a concrete
  block 4 x 8 x 13 cm; a thin-walled steel pipe 10 cm across and 28 cm long (1.0 kg); a wedge of broken block 40 degrees
  steep under a smooth steel plate, with a lip of broken block 2 cm down the face, glued to the table; a fixed rock; a
  timber 24 x 4 x 6 cm.
- **Rover:** built for the rig at the size of a 1:18 trail crawler: an aluminium plate chassis with steel axles, the
  body shell of a Traxxas TRX-4M (polycarbonate, painted yellow, white roof), about 13 kg of lead (1.15 L) cast to fill
  the body between and over the axles, 14 kg in all. Hard rubber wheels 5.6 cm across, the front pair with a rounded
  (crowned) tread. The front axle is a turntable on a central pivot bolt with a friction washer (about 0.15 N m), so
  both front wheels turn together; driveshafts and motor absent, so all four wheels roll freely.
- **Tow tabs:** dark grey 3D-printed PLA tabs 5.0 x 1.6 x 1.6 cm on the middle of each bumper, each with a post 1 cm
  across and 3 cm tall at its tip.

## License

MIT (`LICENSE`), the task authors. No third-party assets: all shapes and textures are generated by code in this
folder.
