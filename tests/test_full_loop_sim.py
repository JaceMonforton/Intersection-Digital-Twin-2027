"""The one-command lab-car loop passes its fault options on to the fake car. Nothing runs."""
from argparse import Namespace

from full_loop_sim import fake_car_args


def options(**overrides):
    return Namespace(**{"drop": 0.0, "delay": None, "replay": 0.0, "outage": [], **overrides})


def test_a_clean_run_asks_for_no_faults():
    assert fake_car_args(options()) == ["scripts/fake_lab_car.py"]


def test_each_fault_is_passed_on_and_outages_can_repeat():
    command = fake_car_args(options(drop=0.1, delay="0.02,0.15", replay=0.05, outage=["20,22", "40,41"]))
    assert command == ["scripts/fake_lab_car.py", "--drop", "0.1", "--delay", "0.02,0.15", "--replay", "0.05",
                       "--outage", "20,22", "--outage", "40,41"]
