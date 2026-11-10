"""Every number of the rover, with its source.

The rover is a 1:18 trail crawler (Traxxas TRX-4M, Bronco body). Published figures come from the manufacturer's
data for the Bronco-bodied version (part 97074-1: "Length 10.3 in. (262 mm), Width 4.9 in. (124 mm), Height 4.6 in.
(117 mm), Wheelbase 6.10 in. (155 mm)", 468 g without battery), as quoted by retailers on 2026-11-10; numbers not
published are marked ``approx`` and are estimates from photographs, consistent with the published ones.

All lengths in metres, masses in kg. Rover frame: x forward, y left, z up, origin on the ground midway between the
axles and between the wheel tracks.
"""

from __future__ import annotations

SOURCES = {
    "pub": "manufacturer data, Bronco body 97074-1 (length 262 mm, width 124 mm, height 117 mm, wheelbase 155 mm, "
           "468 g without battery), via retailer listings, 2026-11-10",
    "approx": "estimate from photographs; not published",
}

# -- published ------------------------------------------------------------------------------------------------------
LENGTH = 0.262          # pub: overall, bumper to bumper
WIDTH = 0.124           # pub: overall, over the tyres
HEIGHT = 0.117          # pub: ground to roof
WHEELBASE = 0.155       # pub
MASS_NO_BATTERY = 0.468  # pub

# -- approximate ----------------------------------------------------------------------------------------------------
BATTERY_MASS = 0.040    # approx: 2S 650 mAh LiPo pack (replaced by a dummy of the same mass)
MASS = MASS_NO_BATTERY + BATTERY_MASS   # 0.508 kg as run
TYRE_DIAMETER = 0.056   # approx: estimate
TYRE_WIDTH = 0.022      # approx: estimate
TRACK = 0.102           # approx: WIDTH - TYRE_WIDTH, centre to centre of the tyres
GROUND_CLEARANCE = 0.025  # approx: under the axle housings
BODY_BOTTOM = 0.040     # approx: lowest edge of the body shell's sides above the ground
BODY_LENGTH = 0.225     # approx: body shell without bumpers
BODY_WIDTH = 0.100      # approx: body shell, between the tyres' inner faces plus the fender flares
ROOF_LENGTH = 0.110     # approx: cab and roof section
BUMPER_HEIGHT = 0.045   # approx: centre height of the front and rear bumpers above the ground
BUMPER_SIZE = (0.018, 0.090, 0.016)   # approx: depth (x), width (y), height (z) of each bumper bar
COM_HEIGHT = 0.048      # approx: centre of mass above the ground (low: motor and battery on the chassis rails)

# -- tyre and contact behaviour ---------------------------------------------------------------------------------------
TYRE_FRICTION = 1.0     # approx: soft rubber crawler tyre on painted board
BODY_FRICTION = 0.5     # approx: painted polycarbonate shell on board or turf
