"""Numbers of the passive 1:2 quadruped (a powered-off robot dog), each with its source.

Source of the proportions, mass shares and joint ranges: the public Unitree Go2 model in MuJoCo Menagerie
(``unitree_go2/go2.xml``, BSD-3-Clause, https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2),
read on 2026-11-11: lengths scaled by 1/2, masses by 1/8 (1/3 and 1/27 until 2026-11-12), joint ranges and signs as in the model. Only numbers are
taken; the model is built here from boxes, capsules, cylinders and spheres, with no mesh, logo or maker's colour.

Our own additions, because the handling sources name them (Boston Dynamics support, "Safe Handling for Spot"; a
laboratory protocol for the same robot): a handle at each hip, and a battery in a bay in the belly held by a light
magnetic catch. Their sizes are our choice, sized for the YAM gripper (94 mm open, pads 27 mm wide).

Robot frame: origin at the trunk's centre (the hip joints are at its mid-height); x forward (head), y to the
robot's left, z up (back) when it stands. Lengths in metres, masses in kg, angles in degrees here (radians in
quadruped.py).

License: MIT (see LICENSE next to this file).
"""

SCALE_L = 1 / 2          # lengths (1:2 since 2026-11-12: at 1:3 the calves were 8 mm rods, too fragile to handle)
SCALE_M = 1 / 8          # masses (the cube of the length scale)
L_, M_ = SCALE_L, SCALE_M

# -- trunk (go2.xml: base collision box half-sizes 0.1881 x 0.04675 x 0.057; base mass 6.921) -------------------
TRUNK_SIZE = (0.3762 * L_, 0.0935 * L_, 0.114 * L_)    # length (x), width (y), height (z): 18.8 x 4.7 x 5.7 cm
BASE_MASS_TOTAL = 6.921 * M_                           # 0.865 kg, of which the battery is BATTERY_MASS
# head: go2.xml cylinder r 0.05, half-height 0.045 at x 0.285 (vertical), sphere r 0.047 at (0.293, 0, -0.06)
HEAD_CYL = ((0.285 * L_, 0.0, 0.0), 0.05 * L_, 0.09 * L_)      # centre, radius, length (along z)
HEAD_SPHERE = ((0.293 * L_, 0.0, -0.06 * L_), 0.047 * L_)

# -- legs (go2.xml; per leg FL, FR, RL, RR) ------------------------------------------------------------------------
HIP_X, HIP_Y = 0.1934 * L_, 0.0465 * L_     # abduction joint: go2 (+-0.1934, +-0.0465, 0)
HIP_MASS = 0.678 * M_                       # the hip housing (abduction link)
HOUSING_R, HOUSING_LEN, HOUSING_Y = 0.046 * L_, 0.04 * L_, 0.08 * L_   # go2 cylinder r 0.046, half 0.02, at y 0.08
THIGH_Y = 0.0955 * L_                       # thigh joint beyond the abduction joint
THIGH_LEN = 0.213 * L_                      # thigh joint to knee: 10.65 cm
THIGH_SECTION = (0.034 * L_, 0.0245 * L_)   # depth (x), width (y): go2 box 0.034 x 0.0245
THIGH_MASS = 1.152 * M_
CALF_LEN = 0.213 * L_                       # knee to foot centre
CALF_R = 0.0125 * L_                        # go2 calf cylinders r 0.012 to 0.013: a 12.5 mm rod at 1:2
CALF_MASS = 0.241352 * M_
FOOT_R = 0.022 * L_                         # go2 foot sphere r 0.022

# joint ranges (go2.xml defaults): abduction about x, thigh and calf (knee) about y. Signs as in the model:
# abduction + moves the foot toward +y (out on the left, in on the right); thigh + swings the knee back;
# knee (calf) always negative: the leg is never straight (bent 48 degrees at least).
ABDUCTION = (-60.0, 60.0)                   # -1.0472 to 1.0472 rad
FRONT_THIGH = (-90.0, 200.0)                # -1.5708 to 3.4907 rad
REAR_THIGH = (-30.0, 260.0)                 # -0.5236 to 4.5379 rad
KNEE = (-156.0, -48.0)                      # -2.7227 to -0.83776 rad
STAND = {"thigh": 51.6, "knee": -103.1}     # go2.xml keyframe "home": thigh 0.9, calf -1.8 rad; trunk 0.27 m up
STAND_HEIGHT = 0.27 * L_

# -- limp joints (as the humanoid dummy: no friction, light viscous damping) ---------------------------------------
# The smallest gravity torque of a limb held out level is the calf's: 0.030 kg x 9.81 x 0.053 m = 16 mN m (1:2). Damping
# 0.0015 N m s/rad gives 3 mN m at 2 rad/s, under a fifth of it.
JOINT_DAMPING = 0.0015
JOINT_EFFORT = 1.0
JOINT_ARMATURE = 1e-4          # kg m^2 per hinge: steadies the 9 g calves on their stops (the motor's rotor inertia
#                                stands in: a geared servo's reflected inertia is far larger)

# -- fittings (our design; the sources name them, sizes are ours) --------------------------------------------------
# hip handles: a rung along x over each hip housing, on two brackets from the trunk's top edge
HANDLE_X = 0.075                             # rung centre, +- from the trunk centre (just inboard of the hips)
HANDLE_Y = 0.114 * L_                       # over the hip housing's inner part: 5.7 cm
HANDLE_Z = TRUNK_SIZE[2] / 2 + 0.012        # rung centre: 1.2 cm above the trunk top
HANDLE_R = 0.004                            # 8 mm rung
HANDLE_LEN = 0.045                          # between the brackets (the pads are 27 mm wide)
BRACKET = 0.005                             # bracket bar thickness

# battery bay in the belly, 2.2 cm deep from the belly face
BAY = ((-0.055, 0.055), (-0.016, 0.016), (-TRUNK_SIZE[2] / 2, -TRUNK_SIZE[2] / 2 + 0.022))
BATTERY_SIZE = (0.108, 0.030, 0.0215)       # 1 mm clear on each side, 0.5 mm at the top
BATTERY_MASS = 0.14                         # our estimate: about 1.1 kg at full size
TAB_SIZE = (0.036, 0.004, 0.010)            # grip tab standing out of the battery's outer face (along x, thin in y)
CATCH_FORCE = 3.0                           # N: the magnetic catch holds the battery seated up to this pull
CATCH_RANGE = 0.003                         # m: and lets go once it is drawn this far out

# -- surfaces and colours (our choice) -------------------------------------------------------------------------------
SHELL_FRICTION = (0.6, 0.55)                # printed shell
FOOT_FRICTION = (0.9, 0.8)                  # rubber feet (go2 foot friction 0.8)
BATTERY_FRICTION = (0.4, 0.35)
SHELL = (0.60, 0.61, 0.62)                  # grey
JOINT_COLOR = (0.27, 0.28, 0.30)            # darker hip housings and feet
HANDLE_COLOR = (0.12, 0.12, 0.13)
BATTERY_COLOR = (0.20, 0.30, 0.55)          # slate blue: stands out on earth and turf
TAB_COLOR = (0.95, 0.45, 0.05)              # orange tab

TOTAL_MASS = BASE_MASS_TOTAL + 4 * (HIP_MASS + THIGH_MASS + CALF_MASS)   # 1.90 kg (Go2 15.2 kg / 8)
