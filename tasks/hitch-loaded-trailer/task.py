"""On the station table: a loaded two-wheeled trailer stands on an earth track on the robot's left, chocked, its tongue
lying on the ground with a socket at its tip. A loaded trail crawler (a rover built heavy, 14 kg; free-rolling
wheels; steering straight) stands 12 cm ahead of the tongue's tip, its back to the trailer, with a tow ball on a post
at its rear. The robot must hitch the trailer to the rover.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import rover as rv  # noqa: E402
import trailkit as tk  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Hitch the trailer to the rover."

TOP = tk.TOP
TEX = HERE / "textures"

# -- layout (world frame: x forward from the arm bases, y to the robot's left, metres) ---------------------------
TRACK_X, TRACK_Y = (0.14, 0.46), (-0.42, 0.52)
DECK_T = 0.004
GROUND = TOP + DECK_T
LINE_X = 0.30                     # the trailer's and the rover's centre line (x)
EARTH = (0.42, 0.31, 0.21)

# the tow ball: a sphere on a post at the rover's rear tab
BALL_R = 0.008
BALL_POST = (0.005, 0.022)        # radius, height above the tab's top
BALL_LOCAL = (-rv.TOW_POST_X, 0.0, rv.S.BUMPER_HEIGHT + rv.TOW_TAB[2] / 2 + BALL_POST[1] + BALL_R)

# the trailer: a bed with a load on one axle; the tongue runs forward (-y) to a socket at its tip
T_AXLE_Y = 0.3685                 # the trailer's axle (y), its wheels touching the chocks; the tongue points -y
T_WHEEL_R, T_TRACK, T_TYRE_W = 0.035, 0.16, 0.025
TONGUE_L = 0.20                   # from the axle to the socket's centre
T_MASS = 9.0                      # bed, lead load, wheels: the tongue's tip bears about 5 N (the load sits nearly over the axle)
T_COM = (0.0, -0.012, 0.04)       # centre of mass, from the axle's middle on the ground: 1.2 cm ahead of the axle
SOCKET_IN, SOCKET_H, SOCKET_WALL = 0.013, 0.012, 0.004    # the socket: a cup 2.6 cm across inside, open downward
TRAILER_COLOR = (0.30, 0.42, 0.58)
CHOCK = (0.030, 0.030, 0.025)


def _track():
    cx, cy = (TRACK_X[0] + TRACK_X[1]) / 2, (TRACK_Y[0] + TRACK_Y[1]) / 2
    hx, hy = (TRACK_X[1] - TRACK_X[0]) / 2, (TRACK_Y[1] - TRACK_Y[0]) / 2
    return tk.body("track", pos=(cx, cy, TOP), shapes=[
        tk.Shape.span((-hx, -hy, 0), (hx, hy, DECK_T), EARTH, friction=(0.9, 0.8)),
        tk.Shape.tile((0, 0, DECK_T + 0.0004), (2 * hx, 2 * hy), str(TEX / "earth.png"), texture_m=0.12)])


def _socket_shapes(c):
    """A cup open downward at ``c`` (trailer frame): a cap and four walls round a 2.6 cm hole."""
    x, y, z = c
    w, h, t = SOCKET_IN, SOCKET_H, SOCKET_WALL
    col = (0.20, 0.20, 0.22)
    return [tk.Shape.box((x, y, z + h / 2 + t / 2), (2 * w + 2 * t, 2 * w + 2 * t, t), col, friction=(0.5, 0.4), tag="socket"),
            tk.Shape.box((x + w + t / 2, y, z), (t, 2 * w + 2 * t, h), col, friction=(0.5, 0.4), tag="socket"),
            tk.Shape.box((x - w - t / 2, y, z), (t, 2 * w + 2 * t, h), col, friction=(0.5, 0.4), tag="socket"),
            tk.Shape.box((x, y + w + t / 2, z), (2 * w, t, h), col, friction=(0.5, 0.4), tag="socket"),
            tk.Shape.box((x, y - w - t / 2, z), (2 * w, t, h), col, friction=(0.5, 0.4), tag="socket")]


def trailer_shapes(pitch_tip_height: float):
    """Trailer frame: origin on the ground under the axle's middle, x across, the tongue along -y, z up."""
    shapes = [tk.Shape.box((0, 0.03, T_WHEEL_R + 0.03), (0.13, 0.16, 0.012), TRAILER_COLOR, friction=(0.6, 0.5), tag="bed"),
              tk.Shape.box((0, 0.03, T_WHEEL_R + 0.065), (0.10, 0.10, 0.06), (0.55, 0.45, 0.30), friction=(0.6, 0.5),
                           tag="load"),
              tk.Shape.box((0, (0.02 - (TONGUE_L - SOCKET_IN - SOCKET_WALL + 0.002)) / 2, T_WHEEL_R + 0.02),
                           (0.016, TONGUE_L - SOCKET_IN - SOCKET_WALL + 0.002 + 0.02, 0.014), TRAILER_COLOR,
                           friction=(0.8, 0.7), tag="tongue")]          # from 2 cm behind the axle to the cup's near wall
    shapes += _socket_shapes((0.0, -TONGUE_L, T_WHEEL_R + 0.02 - SOCKET_H / 2 + 0.007))
    return shapes


def trailer_body():
    """The trailer as one articulated body: the bed, load, tongue and socket on the base; two free wheels."""
    shapes = trailer_shapes(0.0)
    links = []
    for side, y in (("l", T_TRACK / 2), ("r", -T_TRACK / 2)):
        links.append(tk.Link(f"twheel_{side}", "base", "revolute", "X", (y, 0.0, T_WHEEL_R), mass=0.25, damping=0.02,
                             effort=5.0, shapes=(tk.Shape.cylinder((0, 0, 0), T_WHEEL_R, T_TYRE_W, axis="X",
                                                                   color=(0.08, 0.08, 0.08), friction=(1.0, 0.9),
                                                                   tag="ttyre"),)))
    return shapes, tuple(links)


# the tongue's tip rests on the ground: pitch the trailer about its axle until the socket's lowest corner touches
_tip_drop = T_WHEEL_R + 0.02 - SOCKET_H / 2 + 0.007 - SOCKET_H / 2      # the socket's rim above the base frame
_far = TONGUE_L + SOCKET_IN + SOCKET_WALL                               # the cup's far wall, from the axle
PITCH0 = math.atan2(_tip_drop, _far)                                    # nose down until the far wall's foot touches
TRAILER_Q = tk.quat_xyzw_from_axis_angle((1, 0, 0), PITCH0)            # the tongue (-y) tipped down to the ground
_shapes, _links = trailer_body()
TRAILER = tk.body("trailer", pos=(LINE_X, T_AXLE_Y, GROUND + 0.001), rot_xyzw=TRAILER_Q, shapes=_shapes,
                  mass=T_MASS - 0.5, com=T_COM, links=_links, color=TRAILER_COLOR, angular_damping=0.1)
TRAILER0 = tk.state_at(TRAILER, TRAILER.pos, TRAILER.spawn_xyzw)


def socket_centre(st: tk.BodyState) -> np.ndarray:
    """The socket's centre (world): the middle of its hole at its rim's height."""
    return st.to_world((0.0, -TONGUE_L, T_WHEEL_R + 0.02 - SOCKET_H / 2 + 0.007 - SOCKET_H / 2))


TIP0 = socket_centre(TRAILER0)
GAP0 = 0.12                       # from the ball to the socket at the start (along y)
BALL_SHAPES = (tk.Shape.cylinder((BALL_LOCAL[0], 0.0, BALL_LOCAL[2] - BALL_R - BALL_POST[1] / 2), BALL_POST[0],
                                 BALL_POST[1], axis="Z", color=(0.75, 0.75, 0.78), tag="ball_post"),
               tk.Shape.sphere(BALL_LOCAL, BALL_R, color=(0.80, 0.80, 0.84), friction=(0.4, 0.3), tag="ball"))
# the rover faces -y (its back to the trailer); its ball 12 cm short of the socket
_ry = TIP0[1] - GAP0 - rv.TOW_POST_X
ROVER = rv.rover_l4("rover", pos=(LINE_X, _ry, GROUND + 0.0008), rot_xyzw=rv.upright(-math.pi / 2),
                    extra_shapes=BALL_SHAPES)
ROVER0 = tk.state_at(ROVER, ROVER.pos, ROVER.spawn_xyzw)
_WY = float(TRAILER0.links["twheel_l"][0][1])                         # the wheels' centres (y) as spawned, pitched
CHOCKS = [tk.body(f"chock_{s}", pos=(LINE_X + sgn * T_TRACK / 2, _WY + T_WHEEL_R + CHOCK[1] / 2 + 0.001,
                                     GROUND + CHOCK[2] / 2),
                  shapes=[tk.Shape.box((0, 0, 0), CHOCK, (0.85, 0.75, 0.10), friction=(1.0, 0.9))])
          for s, sgn in (("l", 1), ("r", -1))] + \
         [tk.body(f"chock_{s}2", pos=(LINE_X + sgn * T_TRACK / 2, _WY - T_WHEEL_R - CHOCK[1] / 2 - 0.001,
                                      GROUND + CHOCK[2] / 2),
                  shapes=[tk.Shape.box((0, 0, 0), CHOCK, (0.85, 0.75, 0.10), friction=(1.0, 0.9))])
          for s, sgn in (("l", 1), ("r", -1))]
OBJECTS = [_track(), TRAILER, ROVER, *CHOCKS]


def ball_centre(st: tk.BodyState) -> np.ndarray:
    return st.to_world(BALL_LOCAL)


# -- scoring -------------------------------------------------------------------------------------------------------
SEAT_XY = 0.0075         # m: the ball's centre within this of the socket's centre (horizontally): inside the square cup,
#                          corners included
SEAT_Z = (-0.006, 0.0)   # m: seated, the socket's cap rests on the ball and its rim is 0 to 6 mm below the ball's centre
#                          (resting on its rim instead, the rim would be 4 to 8 mm above it)
NEAR = 0.010             # m: for the partial score, the ball's centre within this of the socket's centre (3D)
DROP_SPEED = 0.08        # m/s: peak downward speed of the socket for a drop. The demo lowers it at 1 cm/s
LIFT_CLEAR, LIFT_STEPS = 0.010, 10
REST_SPEED, REST_SPIN = 0.005, 0.05
REST_STEPS = 40
RELEASE = 0.05
DROP_ZONE_UP = 0.15      # m: the drop test applies while the socket's rim is less than this above the ball's top (or
#                          below it)...
DROP_ZONE = 0.026        # m: ...and its centre within this of the ball's centre horizontally (the cup's half-width plus the ball's radius)
DROP_WINDOW = 3          # control steps (0.15 s: 4 samples) over which a drop is judged
TONGUE_W = 0.016          # m: the tongue bar's width
BAR_Y = (-(TONGUE_L - SOCKET_IN - SOCKET_WALL + 0.002), 0.02)   # the bar's extent along the trailer (y): cup's near wall to its root
BAR_UNDER = T_WHEEL_R + 0.02 - 0.007   # the bar's underside (trailer frame z)
DROP_FALL = 0.004        # m: a drop: the socket fell more than this over the window, at a peak over DROP_SPEED

REC = tk.Recorder()


def seated(rover: tk.BodyState, trailer: tk.BodyState) -> tuple[bool, float, float]:
    b, s = ball_centre(rover), socket_centre(trailer)
    dxy = float(np.hypot(*(b[:2] - s[:2])))
    dz = float(s[2] - b[2])
    return dxy <= SEAT_XY and SEAT_Z[0] <= dz <= SEAT_Z[1], dxy, dz


def _drop_window(rec: tk.Recorder, key: str, part: str, near: bool, v_point, point, dt: float) -> None:
    """The drop test on one point of the tongue: over DROP_WINDOW control steps, all in the zone, a fall of more than
    DROP_FALL at a peak (sampled speed, or a step's fall over the step's time) over DROP_SPEED."""
    hist = rec.values.setdefault(key, [])                          # (in the zone, downward speed, height) per step
    hist.append((near, max(0.0, -float(v_point[2])), float(point[2])))
    del hist[:-(DROP_WINDOW + 1)]
    if len(hist) == DROP_WINDOW + 1 and all(h[0] for h in hist):
        steps = [(hist[i][2] - hist[i + 1][2]) / dt for i in range(DROP_WINDOW)]
        peak = max([h[1] for h in hist] + steps)
        fell = hist[0][2] - hist[-1][2]
        rec.extreme("drop_speed", peak if fell > DROP_FALL else 0.0)
        rec.latch("dropped", peak > DROP_SPEED and fell > DROP_FALL,
                  f"the tongue ({part}) fell {1000 * fell:.0f} mm onto the ball at up to {peak:.2f} m/s")


def track_step(rec: tk.Recorder, rover: tk.BodyState, trailer: tk.BodyState, dt: float = 0.05) -> None:
    s = socket_centre(trailer)
    v_point = np.asarray(trailer.lin_vel, float) + np.cross(np.asarray(trailer.ang_vel, float), s - trailer.pos)
    b = ball_centre(rover)
    near = (s[2] - (b[2] + BALL_R) < DROP_ZONE_UP) and float(np.hypot(*(s[:2] - b[:2]))) < DROP_ZONE
    _drop_window(rec, "drop_hist", "socket", near, v_point, s, dt)
    # the tongue's bar over the ball: its underside straight above the ball's centre
    loc = trailer.to_local(b)
    under_bar = abs(float(loc[0])) < TONGUE_W / 2 + BALL_R and BAR_Y[0] <= float(loc[1]) <= BAR_Y[1]
    pb = trailer.to_world((0.0, float(loc[1]) if under_bar else BAR_Y[0], BAR_UNDER))
    v_bar = np.asarray(trailer.lin_vel, float) + np.cross(np.asarray(trailer.ang_vel, float), pb - trailer.pos)
    _drop_window(rec, "bar_hist", "bar", under_bar and pb[2] - (b[2] + BALL_R) < DROP_ZONE_UP, v_bar, pb, dt)
    rec.values["socket_prev"] = s.copy()
    lows = {n: float(p[:, 2].min() - GROUND) for n, p in rv.l3_tyre_points(ROVER, rover).items()}
    axle_up = any(lows[a] > LIFT_CLEAR and lows[b] > LIFT_CLEAR for a, b in (("wheel_fl", "wheel_fr"), ("wheel_rl", "wheel_rr")))
    run = rec.values.get("clear_run", 0) + 1 if axle_up else 0
    rec.values["clear_run"] = run
    rec.latch("lifted", run >= LIFT_STEPS, "both tyres of the rover's axle more than 1 cm off the ground")


@tk.on_reset
def _reset(environment=None) -> None:
    REC.reset()


@tk.on_step
def _record(environment) -> None:
    REC.steps += 1
    rover = tk.body_state(environment, "rover")
    trailer = tk.body_state(environment, "trailer")
    track_step(REC, rover, trailer)
    REC.window("rest", rover.speed() < REST_SPEED and trailer.speed() < REST_SPEED and trailer.spin() < REST_SPIN,
               REST_STEPS)
    grips = tk.grippers(environment)
    REC.window("released", rv.l4_released(ROVER, rover, grips, RELEASE) and tk.released(TRAILER, trailer, grips, RELEASE),
               REST_STEPS)


def evaluate(rover: tk.BodyState, trailer: tk.BodyState, rec: tk.Recorder = REC) -> tuple[float, str]:
    for foul in ("dropped", "lifted"):
        if rec.flag(foul):
            return 0.0, rec.why.get(foul, foul)
    if not (rec.full("rest", REST_STEPS) and all(rec.windows["rest"])):
        return 0.0, "not at rest for 2 s"
    if not (rec.full("released", REST_STEPS) and all(rec.windows["released"])):
        return 0.0, "a gripper or an arm within 5 cm of the rover or the trailer in the last 2 s"
    ok, dxy, dz = seated(rover, trailer)
    if ok:
        return 1.0, f"socket seated on the ball ({1000 * dxy:.1f} mm off its centre)"
    if float(np.linalg.norm(ball_centre(rover) - socket_centre(trailer))) <= NEAR:
        return 0.5, f"ball within 1 cm of the socket, not seated ({1000 * dxy:.1f} mm, {1000 * dz:.1f} mm)"
    return 0.0, "not hitched"


_LAST: dict = {}


def score(objects) -> float:
    """0 if at any time the tongue falls onto the ball: its socket (rim less than 15 cm above the ball's top, or
    below it, and its centre within 2.6 cm of the ball's centre horizontally; the socket's centre: the middle of its
    opening, at its rim's height) or its bar's underside straight above the ball (the ball under the bar, that point
    less than 15 cm above the ball's top) falling more than 4 mm over 3 control steps (0.15 s) at a peak speed over
    0.08 m/s (its sampled speed or one step's fall over 0.05 s) (the tongue dropped
    onto the ball), or both tyres of a rover axle are
    more than 1 cm off the ground for 0.5 s. Otherwise, at the end, after 2 s with the rover and the trailer at rest and
    the arms released (5 cm from both): 1.0 the socket seated on the ball (the ball's centre within 7.5 mm of the socket's
    centre horizontally and the socket's rim 0 to 6 mm below the ball's centre: the cup over the ball, the tongue resting on it); 0.5 the ball's centre within 1 cm of the
    socket's centre but not seated; 0 otherwise."""
    s, why = evaluate(objects["rover"], objects["trailer"])
    _LAST.update(score=s, why=why)
    return s


def success(objects) -> bool:
    return evaluate(objects["rover"], objects["trailer"])[0] == 1.0


def report() -> dict:
    return {"last": dict(_LAST), "fouls": {k: REC.why[k] for k in REC.flags if REC.flags[k]},
            "max_drop_speed": round(REC.extremes.get("drop_speed", 0.0), 3), "steps": REC.steps}
