import numpy as np

from conftest import make_traj
from vlmrl.types import Obs, StepResult, Trajectory, Turn


def test_trajectory_properties():
    t = make_traj(dist=0.2, success=True, n_turns=3, invalid=2)
    assert t.success and t.env_return == 1.0 and t.num_invalid == 2


def test_empty_trajectory():
    t = Trajectory(group_id=0, start_state={})
    assert not t.success and t.env_return == 0 and t.frames == []


def test_frames_include_final_observation():
    img = lambda v: np.full((2, 2, 3), v, np.uint8)
    t = Trajectory(group_id=0, start_state={})
    t.turns = [Turn(Obs(image=img(0)), "", None, StepResult(Obs(image=img(1)))),
               Turn(Obs(image=img(1)), "", None, StepResult(Obs(image=img(2))))]
    assert [f[0, 0, 0] for f in t.frames] == [0, 1, 2]
