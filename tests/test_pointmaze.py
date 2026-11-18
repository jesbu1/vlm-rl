import json

import numpy as np
import pytest

from vlmrl.envs.pointmaze import (GOAL_RADIUS, OfflineStartSampler, PointMazeConfig, PointMazeSimVecEnv,
                                  PointMazeTextInterface, RandomStartSampler, _walls, get_maze_map,
                                  make_pointmaze, oracle_delta, render_topdown, xy_to_pixel)
from vlmrl.types import Obs


# ----------------------------------------------------------------------------- maps / rendering

def test_unknown_maze_raises():
    with pytest.raises(ValueError):
        get_maze_map("nope")


def test_xy_to_pixel_corners_and_centre():
    mm = get_maze_map("umaze")  # 5x5
    assert np.allclose(xy_to_pixel((0, 0), mm, 100), (50, 50))
    assert np.allclose(xy_to_pixel((-2.5, 2.5), mm, 100), (0, 0))       # top-left
    assert np.allclose(xy_to_pixel((2.5, -2.5), mm, 100), (100, 100))   # bottom-right


def test_xy_to_pixel_non_square_letterboxed():
    mm = get_maze_map("medium")  # 8x8
    lg = get_maze_map("large")   # 9 rows x 12 cols
    assert np.allclose(xy_to_pixel((0, 0), lg, 120), (60, 60))
    col, row = xy_to_pixel((-6, 4.5), lg, 120)
    assert np.isclose(col, 0) and np.isclose(row, 15)  # vertical letterbox of (120 - 9*10)/2
    assert len(mm) == 8


def test_render_topdown_draws_agent_and_goal():
    mm = get_maze_map("open")
    img = render_topdown(mm, (-1.0, 0.0), (1.5, 0.5), size=210)
    assert img.shape == (210, 210, 3) and img.dtype == np.uint8
    ac, ar = map(int, xy_to_pixel((-1.0, 0.0), mm, 210))
    gc, gr = map(int, xy_to_pixel((1.5, 0.5), mm, 210))
    assert img[ar, ac, 2] > 150 and img[ar, ac, 0] < 100     # blue agent
    assert img[gr, gc, 1] > 120 and img[gr, gc, 0] < 100     # green goal
    wall = img[2, 2]
    assert wall.max() < 100                                   # border is wall (dark)


def test_render_is_deterministic_and_trail_changes_pixels():
    mm = get_maze_map("umaze")
    a = render_topdown(mm, (1, 1), (-1, -1), 64)
    b = render_topdown(mm, (1, 1), (-1, -1), 64)
    c = render_topdown(mm, (1, 1), (-1, -1), 64, trail=[(1, 1), (1, -1), (0, -1)])
    assert np.array_equal(a, b) and not np.array_equal(a, c)


# ----------------------------------------------------------------------------- dynamics

def _start(qpos, goal, qvel=(0, 0)):
    return {"qpos": np.array(qpos, float), "qvel": np.array(qvel, float), "goal": np.array(goal, float)}


def test_reset_to_exact_state():
    env = PointMazeSimVecEnv("umaze", image_size=32)
    obs = env.reset([_start((1.0, 1.0), (-1, -1), (0.3, -0.2)), _start((-1.0, 1.0), (1, -1))])
    assert len(obs) == 2
    assert np.allclose(obs[0].state, [1, 1, 0.3, -0.2])
    assert np.allclose(obs[1].extras["goal"], [1, -1])
    # resetting to fewer slots reuses envs
    obs = env.reset([_start((0.0, -1.0), (1, 1))])
    assert len(obs) == 1 and np.allclose(obs[0].state[:2], [0, -1])
    env.close()


def test_delta_controller_tracks_waypoint():
    env = PointMazeSimVecEnv("open", image_size=32, settle=False)
    env.reset([_start((0, 0), (2.4, 1.4))])
    r = env.step([0], [np.array([1.0, 0.0])])[0]
    assert np.linalg.norm(r.obs.state[:2] - [1.0, 0.0]) < 0.1
    assert r.info["low_level_actions"].shape == (40, 2)
    assert np.abs(r.info["low_level_actions"]).max() <= 1.0


def test_delta_is_clipped_to_max_delta():
    env = PointMazeSimVecEnv("open", image_size=32, max_delta=0.5, settle=False)
    env.reset([_start((-1, 0), (2.4, 1.4))])
    r = env.step([0], [np.array([5.0, 0.0])])[0]
    assert abs(r.obs.state[0] - (-0.5)) < 0.1


def test_walls_block_motion():
    env = PointMazeSimVecEnv("umaze", image_size=32)
    # (1, 0) in umaze is the middle row's right free cell; row 2 cols 0..2 are walls to its left
    env.reset([_start((0.0, 1.0), (-1, -1))])
    r = env.step([0], [np.array([0.0, -1.0])])[0]   # straight down from top corridor into a wall
    assert r.obs.state[1] > 0.4


def test_force_mode_moves_and_clips():
    env = PointMazeSimVecEnv("open", action_mode="force", action_repeat=10, image_size=32)
    env.reset([_start((0, 0), (2.4, 1.4))])
    r = env.step([0], [np.array([3.0, 0.0])])[0]
    assert r.obs.state[2] > 1.0 and np.allclose(r.info["low_level_actions"], [[1.0, 0.0]] * 10)


def test_success_terminates_early_with_reward():
    env = PointMazeSimVecEnv("open", image_size=32)
    env.reset([_start((0, 0), (0.6, 0.0))])
    r = env.step([0], [np.array([0.6, 0.0])])[0]
    assert r.terminated and r.reward == 1.0 and r.info["success"]
    assert r.info["dist"] <= GOAL_RADIUS and r.info["stop"] == "success"
    assert len(r.info["low_level_actions"]) < 40


def test_settle_runs_until_at_rest():
    env = PointMazeSimVecEnv("open", image_size=32, settle=True)
    env.reset([_start((0, 0), (2.4, 1.4))])
    r = env.step([0], [np.array([0.8, 0.4])])[0]       # |d| < max_delta, so no clipping
    assert r.info["stop"] == "reached"
    assert np.linalg.norm(r.obs.state[:2] - [0.8, 0.4]) < 0.05 and np.linalg.norm(r.obs.state[2:]) < 0.1


def test_settle_stops_when_blocked():
    env = PointMazeSimVecEnv("umaze", image_size=32, settle=True)
    env.reset([_start((1.0, 0.0), (-1, -1))])
    r = env.step([0], [np.array([-1.0, 0.0])])[0]   # wall cell spans x in [-0.5, 0.5] on this row
    assert r.info["stop"] == "blocked" and 0.5 < r.obs.state[0] < 0.7
    assert len(r.info["low_level_actions"]) < 200


def test_settle_capped_by_max_substeps():
    env = PointMazeSimVecEnv("open", image_size=32, settle=True, max_substeps=5)
    env.reset([_start((0, 0), (2.4, 1.4))])
    r = env.step([0], [np.array([1.0, 0.0])])[0]
    assert r.info["stop"] == "steps" and len(r.info["low_level_actions"]) == 5


def test_obs_extras_include_distance():
    env = PointMazeSimVecEnv("open", image_size=32)
    o = env.reset([_start((0, 0), (0.3, 0.4))])[0]
    assert np.isclose(o.extras["dist"], 0.5)


def test_step_subset_of_slots_leaves_others_untouched():
    env = PointMazeSimVecEnv("open", image_size=32)
    env.reset([_start((0, 0), (2, 1)), _start((-1, -1), (2, 1))])
    env.step([1], [np.array([1.0, 0.0])])
    assert np.allclose(env._obs(0).state[:2], [0, 0])
    assert not np.allclose(env._obs(1).state[:2], [-1, -1])


@pytest.mark.parametrize("maze", ["umaze", "medium", "large"])
def test_oracle_solves_maze(maze, rng):
    b = make_pointmaze(PointMazeConfig(maze=maze, image_size=32))
    env = b.make_vec_env()
    for s in b.train_sampler.sample(4, rng):
        env.reset([s])
        for _ in range(30):
            o = env._obs(0)
            r = env.step([0], [oracle_delta(env.maze_map, o.state[:2], o.extras["goal"])])[0]
            if r.terminated:
                break
        assert r.terminated, f"oracle failed on {maze} from {s}"


# ----------------------------------------------------------------------------- interface

@pytest.fixture
def itf():
    return PointMazeTextInterface(get_maze_map("umaze"), prompt_style="simple")


@pytest.fixture
def pear():
    return PointMazeTextInterface(get_maze_map("umaze"))  # default = PEAR protocol


@pytest.mark.parametrize("text,want", [
    ('think <action>{"dx": 0.5, "dy": -1}</action>', [0.5, -1]),
    ('<action>{"dx":1,"dy":1}</action> later <action>{"dx": 0.1, "dy": 0.2}</action>', [0.1, 0.2]),
    ("<action>0.3, -0.4</action>", [0.3, -0.4]),
    ("<ACTION> {\"dy\": 2, \"dx\": -3} </ACTION>", [-3, 2]),
    ("<action>{\"dx\": 1e-1, \"dy\": -2.5e0}</action>", [0.1, -2.5]),
])
def test_parse_valid(itf, text, want):
    assert np.allclose(itf.parse_action(text), want)


@pytest.mark.parametrize("text", ["no tag", "<action></action>", "<action>{\"dx\": 1}</action>",
                                  "<action>{\"dx\": NaN, \"dy\": 1}</action>", "<action>1</action>",
                                  '<action>{"dx": 1, "dy": 2}'])
def test_parse_invalid(itf, text):
    assert itf.parse_action(text) is None


def test_force_mode_parse_clips():
    f = PointMazeTextInterface(get_maze_map("umaze"), action_mode="force", prompt_style="simple")
    assert np.allclose(f.parse_action('<action>{"fx": 4, "fy": -0.5}</action>'), [1, -0.5])
    assert f.parse_action('<action>{"dx": 4, "dy": -0.5}</action>') is not None  # numeric fallback
    assert "fx" in f.system_prompt({})


def test_prompt_and_observation_message(itf):
    sp = itf.system_prompt({})
    assert "5 cells wide" in sp and "<action>" in sp and '"dx"' in sp
    obs = Obs(image=np.zeros((8, 8, 3), np.uint8), state=np.array([1.0, 2.0, 0.0, 0.0]),
              extras={"goal": np.array([-1.0, 0.5])})
    msg = itf.observation_message(obs, 2, None, parse_error=True)
    assert msg[0]["type"] == "image" and msg[0]["image"].size == (8, 8)
    assert "Round 3" in msg[1]["text"] and "(1.00, 2.00)" in msg[1]["text"] and "(-1.00, 0.50)" in msg[1]["text"]
    assert "no valid <action>" in msg[1]["text"]
    no_state = PointMazeTextInterface(get_maze_map("umaze"), include_state=False, prompt_style="simple")
    assert "position" not in no_state.observation_message(obs, 0, None, False)[1]["text"]
    assert np.array_equal(itf.noop_action(), [0, 0])


@pytest.mark.parametrize("style", ["compact", "compact_rel"])
def test_compact_observation_goal_absolute_or_relative(style):
    """compact gives the goal's absolute position; compact_rel gives it relative to the ball, so echoing the
    numbers back is the straight-line move (and parses to exactly goal - ball)."""
    itf = PointMazeTextInterface(get_maze_map("umaze"), prompt_style=style, max_delta=3.0)
    obs = Obs(image=np.zeros((8, 8, 3), np.uint8), state=np.array([1.0, 2.0, 0.0, 0.0]),
              extras={"goal": np.array([-1.0, 0.5])})
    text = itf.observation_message(obs, 0, None, parse_error=False)[1]["text"]
    assert text.startswith("Ball: x=1.00, y=2.00. ") and text.endswith("Move:")
    goal = text.split("Goal: ")[1]
    if style == "compact":
        assert goal.startswith("x=-1.00, y=0.50.")
    else:
        assert goal.startswith("x=-2.00, y=-1.50 from the ball.")
        assert np.allclose(itf.parse_action(goal), [-2.0, -1.5])
    assert '{"x": 1.0, "y": -0.5}' in itf.system_prompt({})
    assert np.allclose(itf.parse_action(itf.format_action(np.array([0.5, -2.0]))), [0.5, -2.0])


# ----------------------------------------------------------------------------- start states

def test_random_sampler_free_cells_and_min_dist(rng):
    mm = get_maze_map("medium")
    walls = _walls(mm)
    s = RandomStartSampler(mm, min_dist=2.0).sample(50, rng)
    h, w = walls.shape
    for st in s:
        for xy in (st["qpos"], st["goal"]):
            i, j = int(np.floor(h / 2 - xy[1])), int(np.floor(xy[0] + w / 2))
            assert not walls[i, j]
        assert np.linalg.norm(st["qpos"] - st["goal"]) >= 2.0
        assert np.array_equal(st["qvel"], [0, 0])


def test_offline_sampler_from_npz(tmp_path, rng):
    n = 100
    qpos = rng.uniform(-1, 1, (n, 2))
    goal = qpos.copy()
    goal[: n // 2] += 3.0  # only first half is far from goal
    p = tmp_path / "starts.npz"
    np.savez(p, qpos=qpos, qvel=np.zeros((n, 2)), goal=goal, episode=np.arange(n))
    s = OfflineStartSampler(str(p), min_goal_dist=1.0)
    assert s.n == n // 2
    out = s.sample(10, rng)
    assert all(set(o) == {"qpos", "qvel", "goal", "episode"} for o in out)
    assert all(o["episode"] < n // 2 for o in out)
    with pytest.raises(AssertionError):
        OfflineStartSampler(str(p), min_goal_dist=100.0)


def test_factory_bundle(rng):
    b = make_pointmaze(maze="umaze", max_turns=7, image_size=48)
    assert b.max_turns == 7
    env = b.make_vec_env()
    obs = env.reset(b.train_sampler.sample(3, rng))
    assert obs[0].image.shape == (48, 48, 3)


# ----------------------------------------------------------------------------- PEAR protocol interface

def test_pear_prompt_follows_paper_template(pear):
    sp = pear.system_prompt({"goal": np.array([1.0, -1.25])})
    for section in ("## OBSERVATION", "## ACTION", "## EXECUTION", "## RESPONSE FORMAT", "## ENVIRONMENT", "## TASK"):
        assert section in sp
    assert sp.startswith("You are controlling a robot.")
    assert "Lengths are in maze cells" in sp and "```json" in sp
    assert "centred at x=1.00, y=-1.25" in sp
    assert "centred" not in PointMazeTextInterface(get_maze_map("umaze"), include_state=False).system_prompt({"goal": [0, 0]})


def test_pear_observation_state_json_then_image(pear):
    obs = Obs(image=np.zeros((8, 8, 3), np.uint8), state=np.array([1.0, 2.0, 0.3, 0.0]), extras={"goal": np.zeros(2)})
    msg = pear.observation_message(obs, 0, None, parse_error=False)
    assert [c["type"] for c in msg] == ["text", "image"]
    assert json.loads(msg[0]["text"]) == {"ball": {"x": 1.0, "y": 2.0}}
    msg = pear.observation_message(obs, 1, None, parse_error=True)
    assert "no valid action" in msg[0]["text"]


@pytest.mark.parametrize("text,want", [
    ('I will go right.\n```json\n{"ball": {"x": 0.5, "y": -1}}\n```', [0.5, -1]),
    ('```json\n{"ball": {"x": 9, "y": 9}}\n``` then ```json\n{"ball": {"y": 0.2, "x": 0.1}}\n```', [0.1, 0.2]),
    ('<action>{"ball": {"x": 0.3, "y": 0}}</action>', [0.3, 0]),          # tag fallback, same schema
])
def test_pear_parse_valid(pear, text, want):
    assert np.allclose(pear.parse_action(text), want)


@pytest.mark.parametrize("text", ['```json\n{"ball": {"x": 1}}\n```',      # missing quantity
                                  '```json\n{"dx": 1, "dy": 0}\n```',      # wrong schema
                                  '```json\n{"ball": {"x": 1, "y": }}\n```',
                                  "no action at all", '```json\n{"ball": {"x": NaN, "y": 0}}\n```'])
def test_pear_parse_invalid(pear, text):
    assert pear.parse_action(text) is None


def test_format_action_roundtrip(pear, itf):
    for interface in (pear, itf):
        assert np.allclose(interface.parse_action(interface.format_action(np.array([0.25, -0.5]))), [0.25, -0.5])


def test_pear_requires_delta_mode():
    with pytest.raises(AssertionError):
        PointMazeTextInterface(get_maze_map("umaze"), action_mode="force", prompt_style="pear")


def test_think_cue_defaults_and_overrides():
    m = get_maze_map("umaze")
    sp = lambda **kw: PointMazeTextInterface(m, **kw).system_prompt({})  # noqa: E731
    assert "Think briefly" not in sp() and "Think briefly, then end your response with" in sp(think=True)
    assert "Think briefly, then end your reply" in sp(prompt_style="simple")
    simple_no = sp(prompt_style="simple", think=False)
    assert "Think briefly" not in simple_no and "End your reply with exactly one action tag" in simple_no


# ----------------------------------------------------------------------------- procedural mazes + split

def _components(m):
    from collections import deque
    free = {(i, j) for i, r in enumerate(m) for j, c in enumerate(r) if c == 0}
    seen, comps = set(), 0
    for s in free:
        if s in seen:
            continue
        comps += 1
        q = deque([s])
        seen.add(s)
        while q:
            i, j = q.popleft()
            for n in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if n in free and n not in seen:
                    seen.add(n)
                    q.append(n)
    edges = sum(1 for (i, j) in free for n in ((i + 1, j), (i, j + 1)) if n in free)
    return comps, len(free), edges


@pytest.mark.parametrize("rows,cols,seed", [(11, 11, 0), (11, 11, 1), (7, 13, 5), (15, 15, 3)])
def test_generated_maze_is_connected_bordered_and_has_loops(rows, cols, seed):
    from vlmrl.envs.pointmaze import generate_maze
    m = generate_maze(rows, cols, seed)
    assert len(m) == rows and all(len(r) == cols for r in m)
    assert all(c == 1 for c in m[0] + m[-1]) and all(r[0] == 1 and r[-1] == 1 for r in m)
    comps, nodes, edges = _components(m)
    assert comps == 1                       # every free cell reachable
    assert edges > nodes - 1                # more edges than a tree: there are loops
    assert m == generate_maze(rows, cols, seed) and m != generate_maze(rows, cols, seed + 1)
    assert generate_maze(rows, cols, seed, loop_frac=0.0) != m


def test_generated_maze_names():
    from vlmrl.envs.pointmaze import maze_family
    assert np.array(get_maze_map("gen7x9s3")).shape == (7, 9)
    assert maze_family("gen11x11", 3, 10) == ["gen11x11s10", "gen11x11s11", "gen11x11s12"]
    with pytest.raises(ValueError):
        maze_family("umaze", 2)
    with pytest.raises(AssertionError):
        get_maze_map("gen10x11s0")          # even size


def test_maze_split_train_and_heldout_layouts_are_disjoint(rng):
    from vlmrl.envs.pointmaze import TEST_SEED_OFFSET
    b = make_pointmaze(maze="gen9x9", train_mazes=6, eval_mazes=3, image_size=32)
    tr = {s["maze"] for s in b.train_sampler.sample(200, rng)}
    te = {s["maze"] for s in b.eval_sampler.sample(200, rng)}
    assert tr == {f"gen9x9s{i}" for i in range(6)}
    assert te == {f"gen9x9s{TEST_SEED_OFFSET + i}" for i in range(3)} and not tr & te
    in_dist = make_pointmaze(maze="gen9x9", train_mazes=6, eval_split="train", image_size=32)
    assert {s["maze"] for s in in_dist.eval_sampler.sample(100, rng)} <= tr
    with pytest.raises(AssertionError):
        make_pointmaze(maze="gen9x9", start_states="D4RL/pointmaze/umaze-v2")


def test_vec_env_follows_each_episodes_maze_and_reuses_envs(rng):
    b = make_pointmaze(maze="gen9x9", train_mazes=4, image_size=48)
    env = b.make_vec_env()
    starts = b.train_sampler.sample(8, rng)
    obs = env.reset(starts)
    for k, s in enumerate(starts):
        assert env.maze_map_of(k) == get_maze_map(s["maze"])
        np.testing.assert_array_equal(obs[k].image, render_topdown(get_maze_map(s["maze"]), s["qpos"], s["goal"], 48))
    made = env._n_made
    assert made == len({s["maze"] for s in starts})   # one simulator per maze, not per slot
    env.reset(starts)                       # same mazes again: no new simulators
    assert env._n_made == made
    env.reset([{**starts[0], "maze": None}])  # no maze name -> the env's default maze
    assert env.maze_map_of(0) == env.maze_map
    env.close()


def test_idle_simulators_are_bounded(rng):
    b = make_pointmaze(maze="gen7x7", train_mazes=40, image_size=16)
    env = b.make_vec_env()
    env.max_sims = 10
    for _ in range(6):
        env.reset(b.train_sampler.sample(12, rng))
        assert len(env._sims) <= max(10, len({m for m in env._slot_maze}))
    env.close()


def test_slots_sharing_a_simulator_keep_independent_state(rng):
    """Slots on the same maze share one MuJoCo env; stepping one must not move the others, and the result must
    match an env that holds only that slot."""
    b = make_pointmaze(maze="umaze", image_size=16)
    starts = b.train_sampler.sample(6, rng)
    shared, solo = b.make_vec_env(), b.make_vec_env()
    shared.reset(starts)
    acts = [rng.uniform(-1, 1, 2) for _ in starts]
    for t in range(3):
        res = shared.step(list(range(6)), acts)
        if t == 0:
            before = [r.obs.state.copy() for r in res]
    shared.step([0], [np.zeros(2)])
    for k in range(1, 6):
        solo.reset([starts[k]])
        for _ in range(3):
            r = solo.step([0], [acts[k]])[0]
        np.testing.assert_allclose(r.obs.state, res[k].obs.state, atol=1e-6)
        np.testing.assert_allclose(shared._qpos[k][:2], res[k].obs.state[:2], atol=1e-9)
    assert shared._n_made == 1 and before


@pytest.mark.parametrize("split", ["train", "heldout"])
def test_oracle_solves_procedural_mazes(split, rng):
    b = make_pointmaze(maze="gen11x11", train_mazes=8, eval_mazes=4, max_delta=3.0, min_goal_dist=3.0, image_size=16)
    env = b.make_vec_env()
    for s in (b.train_sampler if split == "train" else b.eval_sampler).sample(6, rng):
        env.reset([s])
        for _ in range(20):
            o = env._obs(0)
            r = env.step([0], [oracle_delta(env.maze_map_of(0), o.state[:2], o.extras["goal"], 3.0)])[0]
            if r.terminated:
                break
        assert r.terminated, f"oracle failed on {s['maze']}"


def test_prompt_describes_the_episodes_maze():
    itf = PointMazeTextInterface(get_maze_map("umaze"))
    assert "5 cells wide and 5 cells tall" in itf.system_prompt({"goal": [0, 0]})
    assert "13 cells wide and 7 cells tall" in itf.system_prompt({"goal": [0, 0], "maze": "gen7x13s1"})
    simple = PointMazeTextInterface(get_maze_map("umaze"), prompt_style="simple")
    assert "13 cells wide and 7 cells tall" in simple.system_prompt({"maze": "gen7x13s1"})


def test_procgen_task_preset_composes():
    from vlmrl.config import compose
    c = compose(["experiment=pointmaze_q3vl4b_vllm", "task=pointmaze_procgen"])
    assert c.env_kwargs["maze"] == "gen11x11" and c.env_kwargs["eval_split"] == "heldout"
    b = make_pointmaze(PointMazeConfig(**{**c.env_kwargs, "image_size": 16}))
    assert b.max_turns == 20 and len(b.eval_sampler.mazes) == 16


def test_procgen_experiment_eval_sets_build_and_are_solvable(rng):
    """Every eval set of experiment=pointmaze_procgen builds (as Trainer.build_eval_sets does) and the oracle
    solves its starts within that set's round limit, including the larger held-out layouts."""
    from vlmrl.config import compose
    c = compose(["experiment=pointmaze_procgen"])
    seen = {}
    for name, spec in c.eval_sets.items():
        kw = {**c.env_kwargs, **spec, "image_size": 16}
        b = make_pointmaze(PointMazeConfig(**kw))
        seen[name] = {s["maze"] for s in b.eval_sampler.sample(64, rng)}
        env = b.make_vec_env()
        for s in b.eval_sampler.sample(3, rng):
            env.reset([s])
            for _ in range(b.max_turns):
                o = env._obs(0)
                r = env.step([0], [oracle_delta(env.maze_map_of(0), o.state[:2], o.extras["goal"], kw["max_delta"])])[0]
                if r.terminated:
                    break
            assert r.terminated, f"{name}: oracle failed on {s['maze']} within {b.max_turns} rounds"
        env.close()
    assert seen["id_layouts"] <= {f"gen11x11s{i}" for i in range(64)}
    assert not seen["ood_layouts"] & seen["id_layouts"] and all(m.startswith("gen15x15") for m in seen["ood_larger"])
