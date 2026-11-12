"""Offline tests of the robot-dog battery scorer on synthetic states (no simulator):  python -m pytest test_scoring.py"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import quadruped as Q  # noqa: E402
import quadruped_spec as S  # noqa: E402
import task as T  # noqa: E402
import trailkit as tk  # noqa: E402


def dog():
    return tk.state_at(T.DOG, T.START_POS, T.START_ROT, joints=T.START_POSE)


def seated(d):
    return tk.state_at(T.BATTERY, d.to_world(Q.SEAT), d.quat_xyzw)


def on_pad(dx=0.0, dy=0.0, dz=0.0):
    """The battery lying on the charging pad, its tab up, long side along y."""
    c = np.array([(T.CHARGER[0][0] + T.CHARGER[0][1]) / 2 + dx, (T.CHARGER[1][0] + T.CHARGER[1][1]) / 2 + dy,
                  T.GROUND + T.CHARGER_H + S.BATTERY_SIZE[2] / 2 + dz])
    R = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])        # its x along world y, its -z (tab) up
    return tk.state_at(T.BATTERY, c, tk.quat_from_rot(R))


def rec(rest=True, released=True, lifted=False, fell_out=False):
    r = tk.Recorder()
    r.latch("fell_out", fell_out)
    for _ in range(T.REST_STEPS):
        r.window("rest", rest, T.REST_STEPS)
        r.window("released", released, T.REST_STEPS)
    r.latch("lifted", lifted)
    return r


def test_idle_is_zero():
    d = dog()
    assert T.evaluate(d, seated(d), rec())[0] == 0.0


def test_battery_on_the_charging_pad_is_full():
    s, why = T.evaluate(dog(), on_pad(), rec())
    assert s == 1.0, why


def test_battery_out_but_elsewhere_is_half():
    b = tk.state_at(T.BATTERY, (0.30, 0.20, T.GROUND + S.BATTERY_SIZE[2] / 2), (0, 0, 0, 1))
    assert T.evaluate(dog(), b, rec())[0] == 0.5


def test_battery_hanging_over_the_pad_edge_is_half():
    assert T.evaluate(dog(), on_pad(dy=0.04), rec())[0] == 0.5          # its end past the pad's outline


def test_battery_held_above_the_pad_does_not_count():
    assert T.evaluate(dog(), on_pad(dz=0.02), rec())[0] == 0.5          # 2 cm up: out of the bay, not on the pad


@pytest.mark.parametrize("kw", [dict(lifted=True), dict(rest=False), dict(released=False), dict(fell_out=True)])
def test_foul_rest_and_release(kw):
    assert T.evaluate(dog(), on_pad(), rec(**kw))[0] == 0.0


def test_lift_foul_reads_the_trunk():
    d = dog()
    assert T.robot_lowest(d) < 0.03                                       # the start pose: trunk near the ground
    up = tk.state_at(T.DOG, np.asarray(T.START_POS) + (0, 0, 0.08), T.START_ROT, joints=T.START_POSE)
    assert T.robot_lowest(up) > T.LIFT_FOUL


def test_seated_battery_over_the_pad_scores_nothing():
    """The robot dragged belly-down onto the charging pad with the battery still in its bay: 0."""
    c = np.array([(T.CHARGER[0][0] + T.CHARGER[0][1]) / 2, (T.CHARGER[1][0] + T.CHARGER[1][1]) / 2])
    d = tk.state_at(T.DOG, (c[0], c[1], T.START_POS[2]), T.START_ROT, joints=T.START_POSE)
    assert T.evaluate(d, seated(d), rec())[0] == 0.0


def test_scoring_text_matches_constants():
    doc = " ".join(T.score.__doc__.split())
    for text in ("4 mm from its seat", "within 2 cm of it", "6 cm above the ground", "2 mm", "5 cm", "20 control steps", "5 mm", "3 cm from its seat"):
        assert text in doc
