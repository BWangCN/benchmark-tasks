# Get a slab off a free-wheeled rover without moving the rover

A loaded rover (a crawler ballasted to 14 kg) on free-rolling wheels stands on an earth track facing away from the
robot (+x), with a flat slab (20 cm along the rover, 10 cm across, 1.5 cm thick, 0.8 kg) lying on its roof; the
slab's ends overhang the 11 cm roof. The robot is told to get the slab off the rover and onto the ground, and that the
rover must stay where it is.

The slab's friction on the roof (about 5 N) is more than the wheels resist (about 3 N to start all four rolling). The obvious push, straight away from the robot, runs along the rover and carries the rover with it.
Across the wheels the tyres hold: a push sideways (along y) slides the slab off the side with the rover still.
Holding the rover by a tow tab and pushing along would also work, and scores; in the simulator it fails: in the probe
the hold kept the rover within 3 mm but the arm then could not slide the slab, and in the demo's variant the tab slips
in the fingers. The scoring looks only at positions.

## Files

| File | What it is |
|---|---|
| `task.py` | The yam_arena task file: the track, the rover and the slab; the whole-trial recorder; `score()` and `success()` |
| `demo.py`, `demokit.py` | The scripted full-score demo and its wrong-method variants |
| `trailkit.py`, `rover.py`, `rover_spec.py`, `rubble.py`, `textures.py`, `textures/` | Scene building, the rover, state reading |

## Run

```bash
inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 \
  --instruction "Get the slab off the rover and onto the ground without lifting it. The rover must stay where it is." \
  --scorer task_score --max-steps 2400
```

```bash
python demo.py task.py demo_out --max-steps 2400                              # full score
python demo.py task.py out --variant push_along --no-record                   # push straight away, along the rover
python demo.py task.py out --variant brace_and_push_along --no-record         # hold the rear tow tab, push along
python demo.py task.py out --variant idle --no-record                         # do nothing
```

## Demo and variants

| Variant | What it does | Score | Steps | IK misses (left / right), worst |
|---|---|---|---|---|
| (none), 3 runs | push the slab sideways (+y), across the rover, with closed fingers until it has slid off the far side and lies flat beside the rover, clear of it | 1.0 each (the rover moved 5.2 mm; the slab flat on the ground, 3.3 cm out from the rover's side) | 1,393 | 0 / 1,242, 35.1 mm (the right arm pushing beyond its reach toward the end) |
| `push_along` | push the slab straight away from the robot (+x), along the rover | 0.0 (the rover carried 3 cm by step 214; 12.6 cm in all) | 878 | 0 / 11 |
| `brace_and_push_along` | hold the rover's rear tow tab, push the slab along | 0.0 (the tab slips in the fingers; the rover moved 3 cm by step 164) | 1,026 | 0 / 0 |
| `idle` | nothing | 0.0 | 670 | 0 / 0 |

Dream-1, one run of 2,400 steps: 0.0.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

0 if at any time the rover's origin (on the ground between its axles) moves more than 3 cm from its start along the ground, or the slab is lifted, not slid: its lowest point more than 5 mm above the rover's roof, or its middle more than 5 mm above where it lay. Otherwise, at the end, after 2 s with the rover and the slab at rest (the rover's origin and the slab's middle moved under 2 mm over the last 2 s) and released (no point of either gripper, and no point along either arm from its shoulder joint to its palm, within 5 cm of the rover, its wheels and steering, or of the slab): 1.0 the slab on the ground (its lowest point within 5 mm of it) and clear of the rover (not touching it: no point within 2 mm of its body, wheels or steering); 0.5 the slab's middle off the rover's 26.2 x 12.4 cm footprint (tow tabs excluded) seen from above, but the slab not lying clear on the ground; 0 otherwise. A full score ends the trial once it has held for 1 s.

The demo's slab never rises above where it lay: a slab slid off only goes down. Bracing the rover and pushing along is
not ruled out by the scoring; it fails in the simulator only because the tab slips in the fingers.

## Premise probe

Run before the build, with the rover facing +y (the same physics turned a quarter turn), one arm pushing with closed
fingers
(harder while the slab stands), the other holding the rover's front tow tab where stated:

- The rover alone, 6 s: it moves 0.1 mm (the wheels' breakaway torque stops a creep of 20 mm without it).
- Pushed along, the rover unheld: the rover is carried 111.6 mm with the slab (0.8 kg and 1.5 kg alike).
- Pushed across, 0.8 kg: the slab slides 143 mm and falls off onto the ground; the rover moves 1.1 mm.
- Pushed along with the rover held by its tow tab: the rover moves 3.0 mm; the slab does not slide.
- A 1.5 kg slab does not slide under the arm's push in either direction (1.6 mm across, 2.9 mm along held).

## Fidelity

Rigid bodies, 240 Hz physics, 20 Hz control. Friction 0.8 on the track and slab (face values averaged); tyres 1.0. Wheels roll freely with a 0.02 N m breakaway torque for bearing drag. The slab's 0.8 kg is what the simulated arm can slide (1.5 kg did not move). Front tyres are sphere colliders reaching 1.7 cm past the real tyres. Bracing and pushing along scores if it works; the simulated tab grip is weak.

On a real table: the rover's start against a mark, the slab's end state from a photo.
