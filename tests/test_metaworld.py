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
    assert 0.0 <= a.extras["mw_progress"] <= 10.0
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


@pytest.mark.parametrize("style", ["pear", "compact", "compact_rel"])
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
    assert np.allclose(itf.parse_action('{"grip": "close"}'), [0, 0, 0, -1])          # gripper-only: no move
    assert np.allclose(itf.parse_action('{"x": 2.5}'), [2.5, 0, 0, 0])
    assert itf.parse_action('{"x": 1, "y": 2, "z": 3}')[3] == 0.0          # no grip word: keep the command


def test_scripted_expert_solves_push_through_the_text_protocol(renderable):
    b = M.make_metaworld(benchmark="ML1:push-v3", image_size=32, max_turns=15)
    env, itf = b.make_vec_env(), b.interface
    wins = 0
    for task in range(3):
        d0 = env.reset([_state("ML1:push-v3", "test", "push-v3", task=task, seed=task)])[0].extras["mw_obj_dist"]
        assert 0.05 < d0 < 0.5                      # the puck starts away from the goal (m)
        for _ in range(15):
            r = env.step([0], [itf.parse_action(itf.format_action(M.oracle_action(env, 0)))])[0]
            if r.terminated or r.truncated:
                break
        wins += int(r.info["success"])
        if r.info["success"]:                       # Metaworld's dense reward at the final state, for pref_mw
            assert r.obs.extras["mw_progress"] > 5.0
            assert r.obs.extras["mw_obj_dist"] <= 0.05    # puck within push's TARGET_RADIUS, for pref_mw_obj
    assert wins == 3                                # gain 0.5 (1.0 overshoots: expert push success 0.63)
    env.close()


def test_controller_gain_reaches_the_env():
    assert M.make_metaworld(benchmark="ML1:push-v3").make_vec_env().gain == 0.5
    assert M.make_metaworld(benchmark="ML1:push-v3", gain=1.0).make_vec_env().gain == 1.0


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


def test_simulators_of_one_task_share_a_renderer(renderable):
    """One GL context per task class, not per simulator (~128 MiB of GPU memory each)."""
    env = M.MetaWorldVecEnv("ML10", image_size=32)
    ss = [_state("ML10", "train", n, task=i, seed=i) for i, n in enumerate(["reach-v3", "push-v3"] * 4)]
    obs = env.reset(ss)
    assert len(env._renderers) == 2 and all(o.image.std() > 10 for o in obs)
    assert not np.array_equal(obs[0].image, obs[2].image)          # same task, other variation: its own scene
    env.close()


def test_corner_view_is_rotated_not_mirrored(renderable):
    """The corner camera is rolled 180 degrees; the frame must be its 180-degree rotation. Checked against the camera's
    own projection: in a correct (unmirrored) view, world +z goes up the image and the camera's world "right" axis
    goes right."""
    import mujoco
    env = M.MetaWorldVecEnv("ML1:reach-v3", image_size=224)
    env.reset([_state("ML1:reach-v3", "train", "reach-v3")])
    e = env.raw(0)
    cid = mujoco.mj_name2id(e.model, mujoco.mjtObj.mjOBJ_CAMERA, "corner")
    R = e.data.cam_xmat[cid].reshape(3, 3)
    assert R[2, 1] < -0.5                                     # camera "up" points down in the world: rolled
    raw = env._renderers["reach-v3"].render()                 # raw camera image (upside down)
    view = env._view(0).image
    w, h = env.render_size
    s = env.image_size
    rot = raw[::-1, ::-1][(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    assert np.array_equal(view, rot)
    # and +x (world) projects to the right in the rotated view: camera x axis (raw right) has a negative world-x part
    assert R[0, 0] < 0
    env.close()


def test_compact_rel_has_no_absolute_coordinates(renderable):
    b = M.make_metaworld(benchmark="ML1:reach-v3", image_size=32, prompt_style="compact_rel")
    env, itf = b.make_vec_env(), b.interface
    s = _state("ML1:reach-v3", "train", "reach-v3")
    o = env.reset([s])[0]
    first = itf.observation_message(o, 0, False, False)[1]["text"]
    assert first == "Gripper: open. Move:"
    r = env.step([0], [itf.parse_action('{"dx": 4.0, "dy": 0.0, "dz": -2.0, "grip": "open"}')])[0]
    text = itf.observation_message(r.obs, 1, False, False)[1]["text"]
    assert text.startswith("Gripper: open. Last move: dx=+") and "dz=-" in text and "y=6" not in text
    assert "+dx moves the hand to the right" in itf.system_prompt(s)
    assert itf.parse_action('{"x": 1, "y": 2, "z": 3}') is None            # absolute-looking keys are not moves
    assert np.allclose(itf.parse_action('{"grip": "open"}'), [0, 0, 0, 1])
    env.close()


def test_expert_rollout_solves_push_for_behaviour_cloning(renderable):
    from vlmrl.rollout import expert_rollout
    b = M.make_metaworld(benchmark="ML1:push-v3", image_size=32, prompt_style="compact_rel", max_turns=15)
    env = b.make_vec_env()
    trajs = expert_rollout(env, b.interface, b.expert, [_state("ML1:push-v3", "train", "push-v3", task=i, seed=i)
                                                        for i in range(3)], [0, 1, 2], 15)
    assert sum(t.success for t in trajs) >= 2
    assert all(t.turns[0].response.startswith('{"dx": ') for t in trajs)
    env.close()


def test_extra_training_seeds_exclude_held_out_positions():
    """ML1 seed 1's training positions are seed 0's held-out positions: pooling seeds must drop them."""
    one = M.TaskSampler("ML1:push-v3", "train", 0)
    many = M.TaskSampler("ML1:push-v3", "train", 0, n_seeds=3)
    held = [M._rand_vec(t) for t in M.split_tasks(M.benchmark("ML1:push-v3", 0), "test")["push-v3"]]
    assert one.pool is None and many.dropped == 50 and len(many.pool["push-v3"]) == 100
    for st in many.sample(64, np.random.default_rng(0)):
        t = M.split_tasks(M.benchmark("ML1:push-v3", st["bench_seed"]), "train")["push-v3"][st["task"]]
        assert min(np.linalg.norm(M._rand_vec(t) - h) for h in held) > 0.01
    assert M.make_metaworld(benchmark="ML1:push-v3", train_seeds=3).train_sampler.pool is not None


def test_reset_uses_the_start_states_benchmark_seed(renderable):
    env = M.MetaWorldVecEnv("ML1:push-v3", image_size=32, render_size=(64, 48))
    st = _state("ML1:push-v3", "train", "push-v3", task=0, seed=0)
    d0 = env.reset([st])[0].extras["mw_obj_dist"]
    d2 = env.reset([{**st, "bench_seed": 2}])[0].extras["mw_obj_dist"]
    assert d0 != d2                                  # another seed's task 0: other puck / goal positions
    env.close()


def test_compact_rel_reason_reads_the_last_json_only():
    itf = M.MetaWorldTextInterface(prompt_style="compact_rel_reason")
    reply = ('The gripper is left of the puck, so dx should be positive (dx=9 would overshoot).\n'
             '{"dx": 3.0, "dy": -1.5, "dz": 0.0, "grip": "close"}')
    assert np.allclose(itf.parse_action(reply), [3.0, -1.5, 0.0, -1.0])
    assert itf.parse_action("Move right by dx=3 and down.") is None           # reasoning without a JSON move: invalid
    assert "short sentences" in itf.system_prompt({"env_name": "push-v3"})
    assert np.allclose(itf.parse_action(itf.format_action(np.array([1.0, 2.0, -3.0, 1.0]))), [1.0, 2.0, -3.0, 1.0])


def test_swap_instructions_gives_every_task_another_tasks_instruction():
    names = sorted(M.TASK_TEXT)
    swapped = [M.task_text(n, swap=True) for n in names]
    assert all(s != M.TASK_TEXT[n] for n, s in zip(names, swapped))      # always a different instruction
    itf = M.MetaWorldTextInterface(prompt_style="compact_rel", swap_instructions=True)
    assert M.TASK_TEXT["drawer-open"] in itf.system_prompt({"env_name": "drawer-close-v3"})
    assert M.make_metaworld(benchmark="ML10", swap_instructions=True).interface.swap_instructions


def test_goal_marker_scale_enlarges_the_goal_sites(renderable):
    small = M.MetaWorldVecEnv("ML1:reach-v3", image_size=32, render_size=(64, 48))
    big = M.MetaWorldVecEnv("ML1:reach-v3", image_size=32, render_size=(64, 48), goal_marker_scale=2.0)
    st = _state("ML1:reach-v3", "train", "reach-v3")
    small.reset([st]); big.reset([st])
    gid = [i for i in range(small._envs[0].model.nsite) if "goal" in small._envs[0].model.site(i).name][0]
    assert np.allclose(big._envs[0].model.site_size[gid], 2.0 * small._envs[0].model.site_size[gid])
    assert M.make_metaworld(benchmark="ML10", goal_marker_scale=2.0).make_vec_env().goal_marker_scale == 2.0
    small.close(); big.close()


def test_compact_rel_goal_states_the_goal_offset_once_at_the_start():
    from vlmrl.types import Obs
    itf = M.MetaWorldTextInterface(prompt_style="compact_rel_goal")
    obs = Obs(image=np.zeros((8, 8, 3), np.uint8), state=np.zeros(4), extras={"goal_offset": [12.0, -3.5, 4.0]})
    first = itf.observation_message(obs, 0, None, False)[-1]["text"]
    later = itf.observation_message(obs, 1, None, False)[-1]["text"]
    assert "Goal: dx=+12.0, dy=-3.5, dz=+4.0 from the hand." in first and "Goal" not in later
    assert "told where the goal is" in itf.system_prompt({"env_name": "push-v3"})
    assert np.allclose(itf.parse_action('{"dx": 1, "dy": 2, "dz": 3, "grip": "open"}'), [1, 2, 3, 1])


def test_goal_offset_is_target_minus_hand_in_cm(renderable):
    env = M.MetaWorldVecEnv("ML1:reach-v3", image_size=32, render_size=(64, 48))
    o = env.reset([_state("ML1:reach-v3", "train", "reach-v3")])[0]
    sim = env._envs[0]
    expect = (np.asarray(sim._target_pos)[:3] - env._obs[0][:3]) * 100.0
    assert np.allclose(o.extras["goal_offset"], expect, atol=0.06)
    env.close()


def test_compact_rel_locate_states_the_goal_then_the_move():
    """The policy is told nothing about the goal; its reply states the goal offset, then the move (the last JSON). The
    expert's training reply carries the true offset of the observation it answers."""
    from vlmrl.types import Obs
    itf = M.MetaWorldTextInterface(prompt_style="compact_rel_locate")
    obs = Obs(image=np.zeros((8, 8, 3), np.uint8), state=np.zeros(4), extras={"goal_offset": [12.0, -3.5, 4.0]})
    assert "Goal" not in itf.observation_message(obs, 0, None, False)[-1]["text"]
    assert "say where the task's goal is" in itf.system_prompt({"env_name": "push-v3"})
    reply = itf.expert_reply(np.array([3.0, -2.0, -1.0, -1.0]), obs)
    assert reply.startswith("Goal: dx=+12.0, dy=-3.5, dz=+4.0. {")
    assert np.allclose(itf.parse_action(reply), [3.0, -2.0, -1.0, -1.0])          # the goal's dx/dy/dz are not the move
    assert np.allclose(itf.goal_estimate(reply), [12.0, -3.5, 4.0])
    assert itf.parse_action("Goal: dx=+12.0, dy=-3.5, dz=+4.0.") is None           # a goal without a move: invalid
    assert itf.goal_estimate('{"dx": 1, "dy": 2, "dz": 3}') is None
    assert itf.expert_reply(np.zeros(4), Obs()).startswith("Goal: unknown. {")
    assert M.MetaWorldTextInterface(prompt_style="compact_rel").expert_reply(np.zeros(4), obs).startswith("{")


def test_expert_rollout_with_compact_rel_locate_states_the_true_goal(renderable):
    from vlmrl.rollout import expert_rollout
    b = M.make_metaworld(benchmark="ML1:reach-v3", prompt_style="compact_rel_locate", image_size=32,
                         render_width=64, render_height=48)
    env = b.make_vec_env()
    trajs = expert_rollout(env, b.interface, b.expert, [_state("ML1:reach-v3", "train", "reach-v3")], [0], 3)
    for t in trajs[0].turns:
        assert np.allclose(b.interface.goal_estimate(t.response), t.obs.extras["goal_offset"], atol=0.06)
        assert t.action is not None
    env.close()


def test_train_tasks_restricts_training_and_the_training_eval_not_held_out():
    sub = ["reach", "push-v3"]
    b = M.make_metaworld(benchmark="ML10", train_tasks=sub)
    assert b.train_sampler.env_names == ["push-v3", "reach-v3"]
    assert {s["env_name"] for s in b.train_sampler.sample(40, np.random.default_rng(0))} == {"push-v3", "reach-v3"}
    assert M.make_metaworld(benchmark="ML10", train_tasks=sub, eval_split="train").eval_sampler.env_names == ["push-v3", "reach-v3"]
    assert len(M.make_metaworld(benchmark="ML10", train_tasks=sub, eval_split="test").eval_sampler.env_names) == 5
    seeded = M.TaskSampler("ML10", "train", n_seeds=2, tasks=sub)
    assert sorted(seeded.pool) == ["push-v3", "reach-v3"]
    with pytest.raises(AssertionError):
        M.TaskSampler("ML10", "train", tasks=["door-close"])                 # a held-out task, not a training one
