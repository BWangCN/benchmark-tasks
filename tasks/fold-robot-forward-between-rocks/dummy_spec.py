"""Numbers of the passive 1:3 humanoid dummy, each with its source.

Source of the proportions, mass shares and joint ranges: the public Unitree G1 model in MuJoCo Menagerie
(``unitree_g1/g1.xml``, BSD-3-Clause): lengths scaled by 1/3, masses by 1/27, joint ranges as in the model. Only numbers are taken; the dummy is built from boxes, capsules and spheres.

Dummy frame: origin midway between the hip joints; x forward (the chest faces +x), y to the dummy's left, z up
(toward the head) when it stands. Lengths in metres, masses in kg, angles in degrees here (radians in dummy.py).

License: MIT (see LICENSE next to this file).
"""

# -- segments (the G1 model at 1:3: lengths x 1/3, masses x 1/27) -------------------------------------------------------------
TRUNK_SIZE = (0.050, 0.072, 0.107)      # depth (x), width (y), height (z)
HEAD_SIZE = (0.047, 0.052, 0.069)
PELVIS_SIZE = (0.033, 0.043, 0.050)
TRUNK_HEAD_MASS = 0.290
PELVIS_MASS = 0.152

THIGH_LEN, THIGH_R, THIGH_MASS = 0.111, 0.015, 0.169        # hip to knee; "about 3.0 thick"
SHANK_LEN, SHANK_R, SHANK_MASS = 0.100, 0.013, 0.072        # knee to ankle; "about 2.6 thick"
FOOT_SIZE, FOOT_MASS = (0.069, 0.025, 0.020), 0.025         # length (x), width (y), height (z)
UPPER_ARM_LEN, UPPER_ARM_R, UPPER_ARM_MASS = 0.060, 0.011, 0.078   # shoulder to elbow; "about 2.5 thick" (2.2 here,
#                                                                    so the hanging arm clears the trunk's side)
FOREARM_LEN, FOREARM_R, FOREARM_MASS = 0.105, 0.011, 0.053  # forearm with hand; "about 2.2 thick"

# -- layout (hips 7.8 cm apart; standing, hips 23.0, knees 11.9, ankles 1.9 cm above ground) --
HIP_Y = 0.039
ANKLE_HEIGHT = 0.019
PELVIS_Z = (-0.015, 0.035)               # pelvis box from 1.5 cm below the hip line to its top
TRUNK_Z = (0.035, 0.142)                 # trunk on the pelvis
HEAD_Z = (0.142, 0.211)                  # head on the trunk: 0.230 + 0.211 = 44.1 cm standing height
SHOULDER_Y = 0.050                       # joint centre: the hanging upper arm clears the trunk side by 3 mm
SHOULDER_Z = 0.128                       # 1.4 cm below the top of the trunk
FOOT_FORWARD = 0.020                     # the foot box's centre ahead of the ankle

# -- housings (round, at the shoulder and hip joints; their outer edges below) -------------------------
SHOULDER_HOUSING_R = 0.012               # sphere at the shoulder joint: 2.6 cm beyond the trunk side, 1.4 cm beyond
#                                          the 7.2 cm trunk's shoulder line seen from the front
HIP_HOUSING_R = 0.014                    # sphere at the hip joint

# -- joints (the G1 model's ranges; the right side mirrors the left) ---------------------------------
SHOULDER_PITCH = (-177.0, 153.0)         # about y; negative raises the arm forward
SHOULDER_ROLL = (-91.0, 129.0)           # about x; positive takes the arm away from the side
ELBOW = (-60.0, 120.0)                   # about y; positive bends the forearm forward (flexion)
HIP_PITCH = (-145.0, 165.0)              # about y; negative swings the leg forward
HIP_ROLL = (-30.0, 170.0)                # about x; positive takes the leg away from the midline
KNEE = (-5.0, 165.0)                     # about y; positive bends the shank back (flexion)
ANKLE_PITCH = (-50.0, 30.0)              # about y; positive points the toes down

# -- limp joints (friction at most a tenth of the gravity torque of the limb held out level) ---
# The smallest such torque is the foot's: 0.025 kg x 9.81 x 0.02 m = 4.9 mN m, so joint friction is left at 0 and a
# light viscous damping (N m s/rad) settles the limbs; at 2 rad/s it gives 4 mN m, still under a tenth of the
# forearm's 26 mN m.
JOINT_DAMPING = 0.002
JOINT_EFFORT = 1.0                       # N m the damping drive may exert (it never comes near)
CONNECTOR_MASS = 0.004                   # the joint block between a 2-axis joint's two hinges (taken from the limb)

# -- surfaces (our choice: a printed shell, rubber soles) --------------------------------------------------------------------------------
SHELL_FRICTION = (0.6, 0.55)
SOLE_FRICTION = (0.9, 0.8)
SHELL = (0.62, 0.63, 0.64)               # grey shell
JOINT_COLOR = (0.28, 0.29, 0.31)         # darker joints
HATCH_COLOR = (0.95, 0.45, 0.05)         # orange back hatch
HATCH_SIZE = (0.040, 0.050)              # width (y) x height (z) on the back of the trunk
HATCH_Z = 0.090                          # its centre, on the trunk's back

TOTAL_MASS = (TRUNK_HEAD_MASS + PELVIS_MASS + 2 * (THIGH_MASS + SHANK_MASS + FOOT_MASS + UPPER_ARM_MASS
                                                   + FOREARM_MASS))
