"""Metaworld environment (vlmrl/envs/metaworld.py). Needs the `metaworld` extra (mujoco 3.3.0) and a headless
renderer (EGL or OSMesa); skipped otherwise."""
import numpy as np
import pytest

mw = pytest.importorskip("metaworld")
from vlmrl.envs import metaworld as M  # noqa: E402


def _state(bench, split, name, task=0, seed=0):
    return {"benchmark": bench, "split": split, "env_name": name, "task": task, "seed": seed}


@pytest.fixture(scope="module")
def renderable():
    try:
        env = M.MetaWorldVecEnv("ML1:reach-v3", image_size=32)
        env.reset([_state("ML1:reach-v3", "train", "reach-v3")])
        env.close()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no headless MuJoCo renderer: {e}")


def test_benchmark_splits_are_disjoint_and_sized():
    ml10, ml45 = M.benchmark("ML10"), M.benchmark("ML45")
    for b, n_train, n_test in ((ml10, 10, 5), (ml45, 45, 5)):
        tr, te = M.split_tasks(b, "train"), M.split_tasks(b, "test")
        assert len(tr) == n_train and len(te) == n_test and not set(tr) & set(te)
    ml1 = M.split_tasks(M.benchmark("ML1:push-v3"), "test")
    assert list(ml1) == ["push-v3"] and len(ml1["push-v3"]) == 50


def test_every_task_has_an_instruction():
    names = set(M.split_tasks(M.benchmark("ML45"), "train")) | set(M.split_tasks(M.benchmark("ML45"), "test"))
    assert len(names) == 50 and all(M.task_text(n) for n in names)
    with pytest.raises(ValueError):
        M.benchmark("ML2")


def test_sampler_draws_from_its_split(rng):
    s = M.TaskSampler("ML10", "test").sample(40, rng)
    assert {x["env_name"] for x in s} <= set(M.split_tasks(M.benchmark("ML10"), "test"))
    assert all(x["split"] == "test" and 0 <= x["task"] < 50 for x in s)


def test_reset_is_exact_and_hand_moves_toward_target(renderable):
    env = M.MetaWorldVecEnv("ML1:reach-v3", image_size=32)
    s = _state("ML1:reach-v3", "train", "reach-v3", task=3, seed=7)
    a, b = env.reset([s])[0], env.reset([s])[0]
    assert np.allclose(a.state, b.state) and np.array_equal(a.image, b.image) and a.image.shape == (32, 32, 3)
    start = a.state[:3].copy()
    r = env.step([0], [np.array([0.0, 0.0, 5.0, 0.0])])[0]          # 5 cm up
    assert r.info["stop"] in ("reached", "steps", "success") and r.obs.state[2] - start[2] > 3.5
    assert abs(r.obs.state[0] - start[0]) < 1.0 and abs(r.obs.state[1] - start[1]) < 1.0
    r = env.step([0], [np.array([50.0, 0.0, 0.0, 0.0])])[0]         # clipped to max_delta (10 cm)
    assert np.linalg.norm(r.obs.state[:3] - (start + [0, 0, 5])) < 11.0
    env.close()


def test_gripper_closes_and_opens(renderable):
    env = M.MetaWorldVecEnv("ML1:pick-place-v3", image_size=32, max_substeps=15)
    o = env.reset([_state("ML1:pick-place-v3", "train", "pick-place-v3")])[0]
    assert o.state[3] > 0.9
    closed = env.step([0], [np.array([0.0, 0.0, 0.0, -1.0])])[0].obs.state[3]
    opened = env.step([0], [np.array([0.0, 0.0, 0.0, 1.0])])[0].obs.state[3]
    assert closed < 0.5 < opened
    env.close()


@pytest.mark.parametrize("style", ["pear", "compact"])
def test_parse_format_roundtrip(style):
    itf = M.MetaWorldTextInterface(prompt_style=style)
    a = np.array([2.5, -1.0, 0.5, -1.0])
    back = itf.parse_action(itf.format_action(a))
    assert np.allclose(back[:3], a[:3], atol=0.05) and np.sign(back[3]) == -1
    assert itf.parse_action("no action here") is None
    sp = itf.system_prompt(_state("ML10", "train", "push-v3"))
    assert M.task_text("push-v3") in sp


def test_compact_grip_words():
    itf = M.MetaWorldTextInterface(prompt_style="compact")
    assert itf.parse_action('{"x": 1, "y": 2, "z": 3, "grip": "open"}')[3] == 1.0
    assert itf.parse_action('{"x": 1, "y": 2, "z": 3}')[3] == 0.0          # no grip word: keep the command


def test_scripted_expert_solves_push_through_the_text_protocol(renderable):
    b = M.make_metaworld(benchmark="ML1:push-v3", image_size=32, max_turns=15)
    env, itf = b.make_vec_env(), b.interface
    wins = 0
    for task in range(3):
        env.reset([_state("ML1:push-v3", "test", "push-v3", task=task, seed=task)])
        for _ in range(15):
            r = env.step([0], [itf.parse_action(itf.format_action(M.oracle_action(env, 0)))])[0]
            if r.terminated or r.truncated:
                break
        wins += int(r.info["success"])
    assert wins >= 2
    env.close()


def test_factory_via_registry():
    from vlmrl.trainer import make_env_bundle
    b = make_env_bundle("metaworld", {"benchmark": "ML10", "eval_split": "test", "max_turns": 9})
    assert b.max_turns == 9 and b.eval_sampler.split == "test" and len(b.train_sampler.env_names) == 10


def test_frames_are_rendered_large_and_cropped(renderable):
    for crop in ("center", "resize"):
        env = M.MetaWorldVecEnv("ML1:push-v3", crop=crop)          # default: render 640x480, crop to 224x224
        o = env.reset([_state("ML1:push-v3", "train", "push-v3")])[0]
        assert o.image.shape == (224, 224, 3) and env.render_size == (640, 480)
        assert o.image.std() > 10                                    # a real scene, not a blank frame
        env.close()
