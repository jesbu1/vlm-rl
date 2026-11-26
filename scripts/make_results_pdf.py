# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8", "numpy", "wandb"]
# ///
"""Build reports/vlmrl_results.pdf: key results and takeaways, with figures drawn from the wandb runs.

    uv run scripts/make_results_pdf.py             # figures + tables from the cache, then pdflatex (offline)
    uv run scripts/make_results_pdf.py --fetch     # first refresh reports/data/runs.json from wandb

The cache in the repo holds the runs behind every figure, so the PDF rebuilds anywhere without wandb access. --fetch
reads your wandb (WANDB_API_KEY; WANDB_BASE_URL for a self-hosted server) under --entity / --metaworld_project /
--maze_project (defaults: $WANDB_ENTITY or "tri", and our project names). It updates the runs it can read and keeps the
cached copy of any it cannot, so registering your own runs in RUNS and fetching mixes them with the cached ones. Text lives in reports/vlmrl_results.tex; the numbers in its tables are
generated here (reports/tables/*.tex) so they follow the data.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports")
DATA = os.path.join(OUT, "data", "runs.json")
FIG = os.path.join(OUT, "figures")
TAB = os.path.join(OUT, "tables")
ENTITY = os.environ.get("WANDB_ENTITY", "tri")
MAZE, MW = "vlm-rl-pointmaze", "vlm-rl-metaworld"      # project keys in RUNS; --maze_project / --metaworld_project map them

# key -> (project, run id). Only runs from after every bug fix; restarts and crashed duplicates are not listed.
RUNS = {
    # procedural mazes (40 steps, eval on training / held-out 11x11 / held-out 15x15 layouts)
    "pg_qwen2b": (MAZE, "7273b286"), "pg_minicpm": (MAZE, "23055312"), "pg_gemma": (MAZE, "3e97482a"),
    "pg_smol22_traj": (MAZE, "e320a919"),
    # single fixed mazes, 100 steps
    "fx_500m_umaze": (MAZE, "da33c51b"), "fx_500m_medium": (MAZE, "86e6b888"),
    "fx_22_tok_umaze": (MAZE, "39d8029d"), "fx_22_tok_medium": (MAZE, "e532811d"),
    "fx_22_traj_umaze": (MAZE, "73150cd4"), "fx_22_traj_medium": (MAZE, "1eb55355"),
    "fx_22_trajkl_umaze": (MAZE, "f118986b"), "fx_22_trajkl_medium": (MAZE, "077253d8"),
    # Metaworld reach
    "r_q35": (MW, "f1bc2e04"), "r_q3vl4b_bc": (MW, "726835b5"), "r_gemma_bc": (MW, "d092d258"),
    "r_q3vl2b_t16": (MW, "69f4f4fc"), "r_gemma_t16": (MW, "8cea117b"),
    "r_q3vl4b_t12": (MW, "b39008c2"), "r_q3vl2b_t12": (MW, "448f683a"), "r_gemma_t12": (MW, "d6e2f340"),
    "r_q3vl4b_t16": (MW, "7bf0dea1"), "r_q3vl4b_t20": (MW, "ccb99dd9"),
    # Metaworld push, controller gain 1.0
    "p_rl_from_reach": (MW, "4a80c58d"), "p_rl_from_reach_t16": (MW, "47188e19"),
    "p1_bc_prefmw": (MW, "a6c19ea2"), "p1_bc_prefobj": (MW, "ed31ad85"), "p1_q3vl4b_bc": (MW, "a1aeb101"),
    "p1_dagger": (MW, "b8487691"), "p1_dart1": (MW, "51e3aa75"), "p1_dart2": (MW, "7fecc5dd"),
    "p1_dart1_t08": (MW, "ca608d8c"), "p1_dart1_t10": (MW, "ed962f90"),
    # Metaworld push, controller gain 0.5
    "p5_bc30": (MW, "223fd849"), "p5_dart30": (MW, "65571a85"), "p5_q3vl4b_bc30": (MW, "00238be4"),
    "p5_bc100": (MW, "a80a57d1"), "p5_q3vl4b_bc100": (MW, "782211cf"), "p5_seeds10": (MW, "9f6f5f04"),
    # ML10
    "m_rl": (MW, "147bb6b9"), "m_from_reach": (MW, "3e049c7c"), "m1_bc_prefmw": (MW, "ddd405a9"),
    "m1_bc_prefobj": (MW, "2e436798"), "m1_dagger": (MW, "16467d2b"), "m1_dart": (MW, "c7413f19"),
    "m5_bc30": (MW, "337e1e20"), "m5_bc100": (MW, "351850a4"), "m5_bc200": (MW, "971c5b8e"),
    # the generic recipe (gain 0.5, BC on the scripted experts, train_seeds=10, RL with pref_mw_obj); "group:" = the
    # group's newest run, so runs that have not started yet are picked up on the next --fetch
    "rc_q35_ml10": (MW, "group:metaworld-recipe-q35-ml10"), "rc_q35_ml45": (MW, "group:metaworld-recipe-q35-ml45"),
    "rc_q3vl4b_ml45": (MW, "group:metaworld-recipe-q3vl4b-ml45"),
    "rc_q35_ml10_rl2e6": (MW, "group:metaworld-recipe-q35-ml10-rl2e-6"),
    "rc_q35_ml45_rl2e6": (MW, "group:metaworld-recipe-q35-ml45-rl2e-6"),
    "rc_q35_ml45_bc800": (MW, "group:metaworld-recipe-q35-ml45-bc800"), "rc_q35_ml10_bc400": (MW, "group:metaworld-recipe-q35-ml10-bc400"),
    "rc_q35_ml10_bc800": (MW, "group:metaworld-recipe-q35-ml10-bc800"),
    # eval-only per-task evals of saved clones (steps=0; 270/100 episodes on ML45, 200/100 on ML10)
    "pt_ml45_400": (MW, "group:metaworld-pertask-ml45-clone400"), "pt_ml45_800": (MW, "group:metaworld-pertask-ml45-clone800"),
    "pt_ml10_400": (MW, "group:metaworld-pertask-ml10-clone400"),
    # cloning with reasons (bc_rationale, prompt_style compact_rel_reason), then RL with reasoning
    "rb_ml10": (MW, "group:metaworld-reasonbc-q35-ml10"), "rb_ml45": (MW, "group:metaworld-reasonbc-q35-ml45"),
    "rr_ml10": (MW, "group:metaworld-rlreason-q35-ml10"),     # RL only with reasoning, from the base model
    "rr_ml45": (MW, "group:metaworld-rlreason-q35-ml45"),
    "rb2_ml10": (MW, "group:metaworld-reasonbc2-q35-ml10"), "rb2_ml45": (MW, "group:metaworld-reasonbc2-q35-ml45"),
    # REINFORCE++-baseline (adv_mode=reinforce_pp) vs GRPO (zscore) twins, same settings
    "rpp_ml10": (MW, "group:metaworld-adv-reinforce_pp-ml10-clone400"), "grpo_ml10": (MW, "group:metaworld-adv-zscore-ml10-clone400"),
    "rpp_ml45": (MW, "group:metaworld-adv-reinforce_pp-ml45-clone400"), "grpo_ml45": (MW, "group:metaworld-adv-zscore-ml45-clone400"),
    "rpp_rr_ml10": (MW, "group:metaworld-adv-reinforce_pp-ml10-rlreason"),
    # model size (Qwen3.5-9B, same recipe) and the swapped-instruction diagnostic of the 4B clones
    "q9_ml45": (MW, "group:metaworld-recipe-q35-9b-ml45"), "q9_ml10": (MW, "group:metaworld-recipe-q35-9b-ml10"),
    "sw_ml45": (MW, "group:metaworld-swap-ml45-clone400"), "sw_ml10": (MW, "group:metaworld-swap-ml10-clone400"),
    # hidden goal markers (eval only) and the goal offset in text with the text history
    "nm_ml45": (MW, "group:metaworld-nomarker-ml45-clone400"),
    "gh_ml45": (MW, "group:metaworld-goalhist-q35-ml45"), "gh_ml10": (MW, "group:metaworld-goalhist-q35-ml10"),
    # RL only (no cloning) with twice the groups per step
    "ro_ml10": (MW, "group:metaworld-rlonly-q35-ml10-g32"),
    # text history without the goal in text (ablation of gh_ml10)
    "ho_ml10": (MW, "group:metaworld-histonly-q35-ml10"),
    # RL from good clones: lower lr / only above-average rollouts (adv_min=0); baselines p5_seeds10, m5_bc200 (lr 1e-5)
    "rl_push_lr": (MW, "group:metaworld-rlfix-push450-lr2e-6"), "rl_push_pos": (MW, "group:metaworld-rlfix-push450-posadv"),
    "rl_ml10_lr": (MW, "group:metaworld-rlfix-ml10-lr2e-6"), "rl_ml10_pos": (MW, "group:metaworld-rlfix-ml10-posadv"),
}
CONFIG_KEYS = ["model", "lr", "temperature", "reward", "loss_agg", "kl_coef", "bc_steps", "bc_noise", "bc_dagger_steps",
               "groups_per_step", "group_size", "eval_episodes", "wandb_group"]

# Measured outside wandb (scripted expert driven through the text protocol, 15 turns; scripts in the PDF's appendix):
# success per task at controller gain 1.0 and 0.5. Push: 30 episodes on training positions; ML10: 8 per task.
EXPERT = {
    "push (ML1, 30 ep.)": (0.63, 0.97),
    "ML10 train: reach": (1.0, 1.0), "push": (0.625, 0.875), "pick-place": (0.75, 1.0), "door-open": (0.875, 0.625),
    "drawer-close": (1.0, 1.0), "button-press-topdown": (1.0, 1.0), "peg-insert-side": (0.75, 0.75),
    "window-open": (1.0, 1.0), "sweep": (0.875, 1.0), "basketball": (0.625, 0.875),
    "ML10 held-out: drawer-open": (1.0, 1.0), "door-close": (1.0, 1.0), "shelf-place": (0.0, 0.75),
    "sweep-into": (0.0, 0.625), "lever-pull": (0.125, 0.25),
}

# Scripted-expert success per task at controller gain 0.5 through the text protocol (15 turns; ML45: 6 episodes per task,
# ML10: 8), the ceiling for cloning. Measured outside wandb like EXPERT above.
EXPERT45 = {"handle-pull-side": 0.0, "lever-pull": 0.0, "stick-push": 0.0, "stick-pull": 0.17, "peg-unplug-side": 0.33,
            "soccer": 0.33, "basketball": 0.5, "disassemble": 0.5, "handle-pull": 0.5, "peg-insert-side": 0.5,
            "coffee-push": 0.67, "sweep-into": 0.67, "door-open": 0.83, "hammer": 0.83, "push-back": 0.83,
            "bin-picking": 0.83}          # every other ML45 task (training and held-out): 1.0
EXPERT10 = {"reach": 1.0, "push": 0.875, "pick-place": 1.0, "door-open": 0.625, "drawer-close": 1.0,
            "button-press-topdown": 1.0, "peg-insert-side": 0.75, "window-open": 1.0, "sweep": 1.0, "basketball": 0.875,
            "drawer-open": 1.0, "door-close": 1.0, "shelf-place": 0.75, "sweep-into": 0.625, "lever-pull": 0.25}

# palette: validated categorical slots (light surface), text tokens, recessive grid
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"


# ----------------------------------------------------------------------------- fetch

def fetch(entity=ENTITY, projects=None):
    import wandb
    projects = projects or {}
    try:
        api = wandb.Api(timeout=120)
    except Exception as e:  # noqa: BLE001 -- no API key / server unreachable: build from the cache as it is
        print(f"wandb not reachable ({type(e).__name__}: {e}); keeping reports/data/runs.json as it is", flush=True)
        return
    out = load() if os.path.exists(DATA) else {}           # keep what we cannot (re)read
    for key, (proj_key, rid) in RUNS.items():
        proj = projects.get(proj_key, proj_key)
        try:
            if rid.startswith("group:"):
                found = sorted(api.runs(f"{entity}/{proj}", filters={"group": rid[6:]}), key=lambda x: x.created_at)
                if not found:
                    print(f"{key:22s} {rid}: no run yet", flush=True)
                    continue
                rid = found[-1].id
            r = api.run(f"{entity}/{proj}/{rid}")
        except Exception as e:  # noqa: BLE001 -- no access (someone else's wandb), deleted run, network
            print(f"{key:22s} {rid}: not readable ({type(e).__name__}); keeping the cached copy" if key in out
                  else f"{key:22s} {rid}: not readable ({type(e).__name__})", flush=True)
            continue
        rows = []
        for row in r.scan_history():
            keep = {k: float(v) for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                    and (k == "_step" or k.startswith(("rollout/success", "rollout/invalid_frac", "rollout/resp_chars",
                                                        "train/mean_logp", "eval/", "bc/")))}
            if len(keep) > 1:
                rows.append(keep)
        cfg = {k: r.config.get(k) for k in CONFIG_KEYS}
        cfg["benchmark"] = (r.config.get("env_kwargs") or {}).get("benchmark")
        cfg["maze"] = (r.config.get("env_kwargs") or {}).get("maze")
        out[key] = {"project": proj_key, "id": rid, "group": r.group, "state": r.state, "config": cfg, "rows": rows}
        print(f"{key:22s} {rid} {r.state:9s} {len(rows)} rows", flush=True)
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    with open(DATA, "w") as f:
        json.dump(out, f, indent=0)


# ----------------------------------------------------------------------------- data helpers

def load():
    with open(DATA) as f:
        return json.load(f)


def series(run, key):
    """(steps, values) of one metric, sorted by step."""
    pts = sorted((r["_step"], r[key]) for r in run["rows"] if key in r)
    return [p[0] for p in pts], [p[1] for p in pts]


def at(run, key, step):
    xs, ys = series(run, key)
    return dict(zip(xs, ys)).get(step)


def smooth(ys, w=5):
    """Trailing mean over the last w points (the start uses what exists)."""
    return [sum(ys[max(0, i - w + 1):i + 1]) / len(ys[max(0, i - w + 1):i + 1]) for i in range(len(ys))]


def wilson(p, n, z=1.96):
    """95% Wilson interval of a success rate p over n episodes."""
    if n == 0:
        return 0.0, 0.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ----------------------------------------------------------------------------- plotting helpers

def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 8.5, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-", "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.fontsize": 7.5, "legend.frameon": False, "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round", "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
        "text.color": INK,
    })


def end_labels(ax, items, xpad=1.5, min_gap=0.055):
    """Direct labels at line ends: items = [(x, y, text, color)]. Labels are spread vertically so they don't collide
    (in axis-fraction units); a small colored dot beside each carries the identity, the text stays in ink."""
    if not items:
        return
    lo, hi = ax.get_ylim()
    span = hi - lo
    ys = sorted([(y, i) for i, (_, y, _, _) in enumerate(items)])
    placed, last = {}, -1e9
    for y, i in ys:
        f = max((y - lo) / span, last + min_gap)
        placed[i], last = f, f
    over = last - 1.0
    if over > 0:
        placed = {i: f - over for i, f in placed.items()}
    for i, (x, y, text, color) in enumerate(items):
        yy = lo + placed[i] * span
        ax.annotate(text, xy=(x, y), xytext=(x + xpad, yy), fontsize=7.5, color=INK2, va="center",
                    annotation_clip=False)
        ax.plot([x], [y], "o", ms=4.2, color=color, mec="white", mew=1.2, zorder=5, clip_on=False)


def line(ax, xs, ys, color, label, raw=None, lw=1.6, marker=None):
    if raw is not None:
        ax.plot(raw[0], raw[1], color=color, lw=0.8, alpha=0.28)
    ax.plot(xs, ys, color=color, lw=lw, label=label, marker=marker, ms=3.2 if marker else 0,
            mec="white", mew=0.6)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name))
    if os.environ.get("PREVIEW_DIR"):
        fig.savefig(os.path.join(os.environ["PREVIEW_DIR"], name.replace(".pdf", ".png")), dpi=200)
    import matplotlib.pyplot as plt
    plt.close(fig)


# ----------------------------------------------------------------------------- figures

def fig_maze_procgen(D):
    import matplotlib.pyplot as plt
    runs = [("pg_qwen2b", "Qwen3-VL-2B"), ("pg_minicpm", "MiniCPM-V 4.6"), ("pg_gemma", "Gemma 4 E2B"),
            ("pg_smol22_traj", "SmolVLM2-2.2B")]
    sets = [("eval/id_layouts/success", "Training layouts"), ("eval/ood_layouts/success", "Held-out 11×11 layouts"),
            ("eval/ood_larger/success", "Held-out 15×15 layouts")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
    for ax, (key, title) in zip(axes, sets):
        ends = []
        for i, (k, name) in enumerate(runs):
            xs, ys = series(D[k], key)
            line(ax, xs, ys, C[i], name, marker="o")
            if xs:
                ends.append((xs[-1], ys[-1], f"{ys[-1]:.2f}", C[i]))
        ax.set_title(title)
        ax.set_xlabel("RL step")
        ax.set_ylim(0, 0.6)
        ax.set_xlim(-1, 41)
        end_labels(ax, ends, xpad=1.0, min_gap=0.075)
    axes[0].set_ylabel("Eval success (64 episodes)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(w_pad=2.2)
    save(fig, "maze_procgen.pdf")


def fig_maze_fixed(D):
    import matplotlib.pyplot as plt
    runs = [("fx_500m_{}", "SmolVLM2-500M"), ("fx_22_traj_{}", "SmolVLM2-2.2B, traj_mean"),
            ("fx_22_trajkl_{}", "SmolVLM2-2.2B, traj_mean + KL 0.05")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, maze, title in ((axes[0], "umaze", "UMaze: eval success"), (axes[1], "medium", "Medium maze: eval success")):
        ends = []
        for i, (k, name) in enumerate(runs):
            xs, ys = series(D[k.format(maze)], "eval/success")
            line(ax, xs, ys, C[i], name, marker="o")
            ends.append((xs[-1], ys[-1], f"{ys[-1]:.2f}", C[i]))
        ax.set_title(title)
        ax.set_xlabel("RL step")
        ax.set_ylim(0, 1.05)
        end_labels(ax, ends, xpad=2.0)
    axes[0].set_ylabel("Success")
    ax = axes[2]
    for i, (k, name) in ((3, ("fx_22_tok_umaze", "token-mean loss")), (1, ("fx_22_traj_umaze", "traj_mean"))):
        xs, ys = series(D[k], "rollout/success")
        line(ax, xs[:15], ys[:15], C[i], name, marker="o")
    ax.set_title("SmolVLM2-2.2B, UMaze: rollouts")
    ax.set_xlabel("RL step (lr 5e-6)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.36, -0.09))
    fig.tight_layout(w_pad=2.5)
    save(fig, "maze_fixed.pdf")


def fig_reach_models(D):
    import matplotlib.pyplot as plt
    runs = [("r_q35", "Qwen3.5-4B, RL only"), ("r_q3vl4b_bc", "Qwen3-VL-4B, BC then RL"),
            ("r_q3vl2b_t16", "Qwen3-VL-2B, RL only (T=1.6)"), ("r_gemma_bc", "Gemma 4 E2B, BC then RL")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, key, title in ((axes[0], "eval/heldout_positions/success", "Eval on held-out goal positions"),
                           (axes[1], "rollout/success", "Training rollout success (5-step mean)")):
        ends = []
        for i, (k, name) in enumerate(runs):
            xs, ys = series(D[k], key)
            if key == "rollout/success":
                line(ax, xs, smooth(ys), C[i], name, raw=(xs, ys))
            else:
                line(ax, xs, smooth(ys, 3), C[i], name, raw=(xs, ys))
        ax.set_title(title)
        ax.set_xlabel("RL step")
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Success")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(w_pad=3)
    save(fig, "reach_models.pdf")


def fig_reach_exploration(D):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axes[0]
    rows = [("r_q35", "Qwen3.5-4B (learned)"), ("r_q3vl4b_t12", "Qwen3-VL-4B (stuck)"),
            ("r_q3vl2b_t12", "Qwen3-VL-2B (stuck)"), ("r_gemma_t12", "Gemma 4 E2B (stuck)")]
    vals = [at(D[k], "train/mean_logp", 1) for k, _ in rows]
    y = list(range(len(rows)))[::-1]
    ax.barh(y, vals, height=0.45, color=C[0])
    for yy, v in zip(y, vals):
        ax.text(v - 0.015, yy, f"{v:.2f}", va="center", ha="right", fontsize=7.5, color=INK2)
    ax.set_yticks(y, [n for _, n in rows])
    ax.set_xlim(min(vals) * 1.35, 0)
    ax.set_title("Step 1: mean token log-prob")
    ax.set_xlabel("log-prob per sampled token, T = 1.2")
    ax.grid(axis="y", visible=False)
    for ax, (k12, k16, name) in zip(axes[1:], (("r_q3vl2b_t12", "r_q3vl2b_t16", "Qwen3-VL-2B"),
                                               ("r_gemma_t12", "r_gemma_t16", "Gemma 4 E2B"))):
        for i, (k, lab) in enumerate(((k12, "T = 1.2"), (k16, "T = 1.6"))):
            xs, ys = series(D[k], "rollout/success")
            xs, ys = xs[:30], ys[:30]
            line(ax, xs, smooth(ys), C[i], lab, raw=(xs, ys))
        ax.set_title(f"{name} rollouts")
        ax.set_xlabel("RL step")
        ax.set_ylim(0, 0.5)
        ax.legend(loc="upper left")
    fig.tight_layout(w_pad=2)
    save(fig, "reach_exploration.pdf")


def fig_push_reward(D):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.3))
    ends = []
    for i, (k, name) in enumerate((("p1_bc_prefmw", "pref_mw (dense reward)"), ("p1_bc_prefobj", "pref_mw_obj (object first)"))):
        xs, ys = series(D[k], "rollout/success")
        xs, ys = xs[:50], ys[:50]
        line(ax, xs, smooth(ys), C[i], name, raw=(xs, ys))
        ends.append((xs[-1], smooth(ys)[-1], f"{smooth(ys)[-1]:.2f}", C[i]))
    ax.set_title("Push from the same clone: rollout success")
    ax.set_xlabel("RL step")
    ax.set_ylabel("Success (5-step mean)")
    ax.set_ylim(0, 0.25)
    ax.legend(loc="upper right")
    end_labels(ax, ends, xpad=1.0)
    fig.tight_layout()
    save(fig, "push_reward.pdf")


CLONES = [  # (label, [run keys whose step-0 eval measures this clone], note)
    ("gain 1.0 · BC 30 steps", ["p1_bc_prefmw", "p1_bc_prefobj"], "two clones, same recipe"),
    ("gain 1.0 · + DAgger 20 steps", ["p1_dagger"], ""),
    ("gain 1.0 · + DART 1 cm, 20 steps", ["p1_dart1", "p1_dart1_t08", "p1_dart1_t10"], "one clone, three evals"),
    ("gain 1.0 · + DART 2 cm, 20 steps", ["p1_dart2"], ""),
    ("gain 0.5 · BC 30 steps", ["p5_bc30"], ""),
    ("gain 0.5 · DART 1 cm, 30 steps", ["p5_dart30"], ""),
    ("gain 0.5 · BC 100 steps", ["p5_bc100"], ""),
    ("gain 0.5 · BC 100 steps, 450 positions", ["p5_seeds10"], "ablation"),
    ("gain 0.5 · Qwen3-VL-4B, BC 100 steps", ["p5_q3vl4b_bc100"], ""),
]


def clone_stats(D, keys, key):
    """Pooled step-0 eval (success count, episodes) over the listed runs; None if not evaluated yet."""
    wins = n = 0
    for k in keys:
        v = at(D[k], key, 0)
        if v is None:
            continue
        eps = int(D[k]["config"].get("eval_episodes") or 32)
        wins += round(v * eps)
        n += eps
    return (wins, n) if n else None


def fig_push_clones(D):
    import matplotlib.pyplot as plt
    rows = [c for c in CLONES if clone_stats(D, c[1], "eval/train_positions/success")]
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 0.33 * len(rows) + 0.9))
    y = list(range(len(rows)))[::-1]
    for j, (key, lab, off) in enumerate((("eval/train_positions/success", "training positions", 0.13),
                                         ("eval/heldout_positions/success", "held-out positions", -0.13))):
        for yy, (name, keys, _) in zip(y, rows):
            w, n = clone_stats(D, keys, key)
            p = w / n
            lo, hi = wilson(p, n)
            ax.plot([lo, hi], [yy + off] * 2, color=C[j], lw=1.2, alpha=0.55, solid_capstyle="butt")
            ax.plot([p], [yy + off], "o", ms=5, color=C[j], mec="white", mew=1.2, zorder=4,
                    label=lab if yy == y[0] else None)
            ax.text(hi + 0.012, yy + off, f"{p:.2f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Success of the cloned policy before RL (dots; bars: 95% interval)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0.0, 1.0))
    fig.tight_layout()
    save(fig, "push_clones.pdf")


def fig_expert_gain():
    import matplotlib.pyplot as plt
    train = [k for k in EXPERT if not k.startswith(("push (ML1", "ML10 held-out")) and
             list(EXPERT).index(k) < list(EXPERT).index("ML10 held-out: drawer-open")]
    held = list(EXPERT)[list(EXPERT).index("ML10 held-out: drawer-open"):]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [10, 5.6]})
    for ax, names, title in ((axes[0], train, "ML10 training tasks"), (axes[1], held, "ML10 held-out tasks")):
        y = list(range(len(names)))[::-1]
        for yy, n in zip(y, names):
            a, b = EXPERT[n]
            ax.plot([a, b], [yy, yy], color=GRID, lw=2.2, zorder=1, solid_capstyle="butt")
        ax.plot([EXPERT[n][0] for n in names], y, "o", ms=5, color=C[1], mec="white", mew=1.2, label="gain 1.0 (old)", zorder=3)
        ax.plot([EXPERT[n][1] for n in names], y, "o", ms=5, color=C[0], mec="white", mew=1.2, label="gain 0.5 (new)", zorder=4)
        ax.set_yticks(y, [n.split(": ")[-1] for n in names])
        ax.set_xlim(-0.04, 1.06)
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
        ax.set_xlabel("Expert success (8 episodes per task)")
    axes[0].legend(loc="lower left", ncol=2, bbox_to_anchor=(0.0, 1.08))
    fig.tight_layout(w_pad=2)
    save(fig, "expert_gain.pdf")


def fig_push_rl(D):
    import matplotlib.pyplot as plt
    runs = [("p5_bc100", "Qwen3.5-4B, BC 100"), ("p5_q3vl4b_bc100", "Qwen3-VL-4B, BC 100"),
            ("p5_bc30", "Qwen3.5-4B, BC 30"), ("p1_bc_prefobj", "Qwen3.5-4B, BC 30, gain 1.0"),
            ("p5_seeds10", "Qwen3.5-4B, BC 100, 450 positions")]
    runs = [r for r in runs if series(D[r[0]], "rollout/success")[0]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, key, title, w in ((axes[0], "rollout/success", "Training rollout success (5-step mean)", 5),
                              (axes[1], "eval/heldout_positions/success", "Eval on held-out positions (3-point mean)", 3)):
        ends = []
        for i, (k, name) in enumerate(runs):
            xs, ys = series(D[k], key)
            if not xs:
                continue
            line(ax, xs, smooth(ys, w), C[i], name, raw=(xs, ys))
            ends.append((xs[-1], smooth(ys, w)[-1], f"{smooth(ys, w)[-1]:.2f}", C[i]))
        ax.set_title(title)
        ax.set_xlabel("RL step")
        ax.set_ylim(0, 1.0)
        end_labels(ax, ends, xpad=1.5)
    axes[0].set_ylabel("Success")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(w_pad=4)
    save(fig, "push_rl.pdf")


def fig_ml10(D):
    import matplotlib.pyplot as plt
    runs = [("m_rl", "RL only"), ("m_from_reach", "RL from the reach adapter"), ("m5_bc30", "BC 30, gain 0.5"),
            ("m5_bc100", "BC 100, gain 0.5"), ("m5_bc200", "BC 200, gain 0.5")]
    runs = [r for r in runs if series(D[r[0]], "eval/train_tasks/success")[0]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, key, title in ((axes[0], "eval/train_tasks/success", "Eval on its 10 training tasks (3-point mean)"),
                           (axes[1], "eval/heldout_tasks/success", "Eval on its 5 held-out tasks")):
        ends = []
        for i, (k, name) in enumerate(runs):
            xs, ys = series(D[k], key)
            line(ax, xs, smooth(ys, 3), C[i], name, raw=(xs, ys))
            ends.append((xs[-1], smooth(ys, 3)[-1], f"{smooth(ys, 3)[-1]:.2f}", C[i]))
        ax.set_title(title)
        ax.set_xlabel("RL step")
        ax.set_ylim(0, 0.6)
        end_labels(ax, ends, xpad=2)
    axes[0].set_ylabel("Success (48 episodes)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(w_pad=4)
    save(fig, "ml10.pdf")


def fig_rl_fixes(D):
    """RL from good clones under three settings: held-out push positions, ML10 training tasks (3-eval means)."""
    import matplotlib.pyplot as plt
    sets = [("Push, 450-position clone: held-out positions", "eval/heldout_positions/success",
             [("p5_seeds10", "lr 1e-5"), ("rl_push_lr", "lr 2e-6"), ("rl_push_pos", "lr 1e-5, adv_min = 0")]),
            ("ML10, 200-step clone: training tasks", "eval/train_tasks/success",
             [("m5_bc200", "lr 1e-5"), ("rl_ml10_lr", "lr 2e-6"), ("rl_ml10_pos", "lr 1e-5, adv_min = 0")])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for ax, (title, key, runs) in zip(axes, sets):
        ends = []
        for i, (k, name) in enumerate(runs):
            if k not in D:
                continue
            xs, ys = series(D[k], key)
            xs, ys = [x for x in xs if x <= 60], ys[:len([x for x in xs if x <= 60])]
            if not xs:
                continue
            line(ax, xs, smooth(ys, 3), C[i], name, raw=(xs, ys))
            ends.append((xs[-1], smooth(ys, 3)[-1], f"{smooth(ys, 3)[-1]:.2f}", C[i]))
        ax.set_title(title)
        ax.set_xlabel("RL step")
        ax.set_ylim(0, 0.9)
        end_labels(ax, ends, xpad=1.5)
    axes[0].set_ylabel("Eval success (3-eval mean)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout(w_pad=4)
    save(fig, "rl_fixes.pdf")


def fig_recipe(D):
    """The generic recipe on ML10 and ML45: training- and held-out-task evals after cloning (step 0) and during RL."""
    import matplotlib.pyplot as plt
    runs = [("rc_q35_ml10", "ML10, Qwen3.5-4B, RL lr 1e-5"), ("rc_q35_ml10_rl2e6", "ML10, Qwen3.5-4B, RL lr 2e-6"),
            ("rc_q35_ml45", "ML45, Qwen3.5-4B, RL lr 1e-5"), ("rc_q35_ml45_rl2e6", "ML45, Qwen3.5-4B, RL lr 2e-6"),
            ("rc_q3vl4b_ml45", "ML45, Qwen3-VL-4B, RL lr 1e-5"), ("rc_q35_ml45_bc800", "ML45, Qwen3.5-4B, BC 800, RL lr 2e-6"),
            ("rc_q35_ml10_bc400", "ML10, Qwen3.5-4B, BC 400, RL lr 2e-6"), ("rc_q35_ml10_bc800", "ML10, Qwen3.5-4B, BC 800, RL lr 2e-6")]
    runs = [(i, k, n) for i, (k, n) in enumerate(runs) if k in D and series(D[k], "eval/train_tasks/success")[0]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, key, title in ((axes[0], "eval/train_tasks/success", "Training tasks (3-eval mean)"),
                           (axes[1], "eval/heldout_tasks/success", "Held-out tasks (3-eval mean)")):
        ends = []
        for i, k, name in runs:
            xs, ys = series(D[k], key)
            line(ax, xs, smooth(ys, 3), C[i], name, raw=(xs, ys), marker="o" if len(xs) == 1 else None)
            ends.append((xs[-1], smooth(ys, 3)[-1], f"{smooth(ys, 3)[-1]:.2f}", C[i]))
        ax.set_title(title)
        ax.set_xlabel("RL step (0 = the clone)")
        ax.set_ylim(0, 0.9 if "train" in key else 0.3)
        end_labels(ax, ends, xpad=1.5)
    axes[0].set_ylabel("Eval success")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout(w_pad=4)
    save(fig, "recipe.pdf")


def _per_task(run, split):
    row = next((r for r in run["rows"] if "eval/success" in r), {})
    pre = f"eval/{split}/task/"
    return {k[len(pre):].replace("-v3/success", ""): v for k, v in row.items() if k.startswith(pre)}


def fig_pertask(D):
    """Per task: the scripted expert's success (the ceiling) and the cloned policies'. ML45 training tasks; then the
    held-out tasks of ML45 and ML10 and ML10's training tasks."""
    import matplotlib.pyplot as plt
    if not all(k in D for k in ("pt_ml45_400", "pt_ml45_800", "pt_ml10_400")):
        return
    a, b = _per_task(D["pt_ml45_400"], "train_tasks"), _per_task(D["pt_ml45_800"], "train_tasks")
    names = sorted(a, key=lambda n: (EXPERT45.get(n, 1.0), b.get(n, 0), a.get(n, 0)))
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 0.17 * len(names) + 0.9))
    y = list(range(len(names)))
    ax.plot([EXPERT45.get(n, 1.0) for n in names], y, "o", ms=7, mfc="none", mec=MUTED, mew=1.2, label="scripted expert")
    ax.plot([a[n] for n in names], y, "o", ms=4.5, color=C[0], mec="white", mew=0.8, label="clone, 400 cloning steps")
    ax.plot([b.get(n, 0) for n in names], y, "o", ms=4.5, color=C[1], mec="white", mew=0.8, label="clone, 800 cloning steps")
    ax.set_yticks(y, names, fontsize=7)
    ax.set_xlim(-0.04, 1.06)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Success on the task (6 episodes per task; 3-4 for the clones)")
    ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0.0, 1.0))
    fig.tight_layout()
    save(fig, "pertask_ml45.pdf")

    fig, axes = plt.subplots(3, 1, figsize=(5.6, 5.0), sharex=True, gridspec_kw={"height_ratios": [5, 5, 10]})
    panels = [("ML45 held-out tasks", [("pt_ml45_400", "heldout_tasks", C[0], "400 steps"), ("pt_ml45_800", "heldout_tasks", C[1], "800 steps")], EXPERT45),
              ("ML10 held-out tasks", [("pt_ml10_400", "heldout_tasks", C[0], "400 steps")], EXPERT10),
              ("ML10 training tasks", [("pt_ml10_400", "train_tasks", C[0], "400 steps")], EXPERT10)]
    for ax, (title, series_, ex) in zip(axes, panels):
        base = _per_task(D[series_[0][0]], series_[0][1])
        names = sorted(base, key=lambda n: base[n])
        y = list(range(len(names)))
        ax.plot([ex.get(n, 1.0) for n in names], y, "o", ms=7, mfc="none", mec=MUTED, mew=1.2, label="expert")
        for k, split, col, lab in series_:
            v = _per_task(D[k], split)
            ax.plot([v.get(n, 0) for n in names], y, "o", ms=4.5, color=col, mec="white", mew=0.8, label=f"clone, {lab}")
        ax.set_yticks(y, names, fontsize=7)
        ax.set_xlim(-0.04, 1.06)
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
    axes[-1].set_xlabel("Success on the task (20 episodes per task for the clones; 8 for the expert on ML10)")
    axes[0].legend(loc="lower left", ncol=3, bbox_to_anchor=(0.0, 1.18), fontsize=7)
    fig.tight_layout(h_pad=1.0)
    save(fig, "pertask_heldout.pdf")


def fig_reasoning(D):
    """ML10 with reasoning in every reply: RL only from the base model vs cloning (with and without reasons)."""
    import matplotlib.pyplot as plt
    runs = [("rr_ml10", "RL only, with reasoning (from the base model)"), ("rb_ml10", "BC with reasons (400), then RL"),
            ("rb2_ml10", "BC with one-sentence reasons (400), then RL")]
    runs = [(i, k, n) for i, (k, n) in enumerate(runs) if k in D and series(D[k], "eval/train_tasks/success")[0]]
    if not runs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, key, title in ((axes[0], "eval/train_tasks/success", "Training tasks"),
                           (axes[1], "eval/heldout_tasks/success", "Held-out tasks"),
                           (axes[2], "eval/heldout_tasks/task/door-close-v3/success", "Held-out: door-close")):
        ends = []
        for i, k, name in runs:
            xs, ys = series(D[k], key)
            if not xs:
                continue
            line(ax, xs, ys, C[i], name, marker="o")
            ends.append((xs[-1], ys[-1], f"{ys[-1]:.2f}", C[i]))
        ax.axhline({"eval/train_tasks/success": 0.61, "eval/heldout_tasks/success": 0.02}.get(key, 0.04), color=MUTED,
                   lw=1.0, zorder=0)
        ax.set_title(title)
        ax.set_xlabel("RL step (0 = start)")
        ax.set_ylim(0, 1.0 if "door" in key or "train" in key else 0.3)
        end_labels(ax, ends, xpad=2)
    axes[0].set_ylabel("Eval success")
    axes[2].text(0.02, 0.96, "gray line: answer-only clone", transform=axes[2].transAxes, fontsize=6.5, color=MUTED, va="top")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12), fontsize=7)
    fig.tight_layout(w_pad=2)
    save(fig, "reasoning_ml10.pdf")


def fig_advmode(D):
    """GRPO (per-group std) vs REINFORCE++-baseline (one std over the batch), same clones and settings."""
    import matplotlib.pyplot as plt
    pairs = [("ML10", "grpo_ml10", "rpp_ml10"), ("ML45", "grpo_ml45", "rpp_ml45")]
    if not all(k in D for _, a, b in pairs for k in (a, b)):
        return
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2))
    for col, (bench, kg, kr) in enumerate(pairs):
        for ax, key, title in ((axes[2 * col], "eval/train_tasks/success", f"{bench}: training tasks"),
                               (axes[2 * col + 1], "eval/heldout_tasks/success", f"{bench}: held-out tasks")):
            ends = []
            for i, (k, name) in enumerate(((kg, "GRPO"), (kr, "REINFORCE++"))):
                xs, ys = series(D[k], key)
                line(ax, xs, ys, C[i], name, marker="o")
                ends.append((xs[-1], ys[-1], f"{ys[-1]:.2f}", C[i]))
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("RL step")
            ax.set_ylim(0, 0.9 if "train" in key else 0.3)
            end_labels(ax, ends, xpad=2)
    axes[0].set_ylabel("Eval success")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(w_pad=1.2)
    save(fig, "advmode.pdf")


def _per_task_evals(run, split):
    """Per task: the success of every eval in the run (RL steps 0, 10, ...)."""
    pre, out = f"eval/{split}/task/", {}
    for r in sorted((r for r in run["rows"] if "eval/success" in r), key=lambda r: r["_step"]):
        for k, v in r.items():
            if k.startswith(pre):
                out.setdefault(k[len(pre):].replace("-v3/success", ""), []).append(v)
    return out


def fig_goalhist(D):
    """Held-out tasks per task: the one-frame clones vs clones that see the goal offset once in text and keep the text
    history (and, once it runs, the history without the goal text)."""
    import matplotlib.pyplot as plt
    if not all(k in D for k in ("pt_ml10_400", "pt_ml45_400", "gh_ml10", "gh_ml45")):
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.0), sharex=True)
    for ax, bench, base, others in (
            (axes[0], "ML10", "pt_ml10_400", [("gh_ml10", C[2], "goal once in text + text history"),
                                              ("ho_ml10", C[1], "text history only")]),
            (axes[1], "ML45", "pt_ml45_400", [("gh_ml45", C[2], "goal once in text + text history")])):
        b = _per_task(D[base], "heldout_tasks")
        names = sorted(b, key=lambda n: -b[n])
        y = list(range(len(names)))
        ax.plot([b[n] for n in names], y, "o", ms=5, color=C[0], mec="white", mew=0.8, label="one frame, no history")
        for j, (k, col, lab) in enumerate(others):
            if k not in D or not series(D[k], "eval/heldout_tasks/success")[0]:
                continue
            ev = _per_task_evals(D[k], "heldout_tasks")
            off = 0.18 * (j + 1)
            for i, n in enumerate(names):
                v = ev.get(n, [0])
                ax.plot([min(v), max(v)], [i + off] * 2, color=col, lw=1.4, alpha=0.55, solid_capstyle="round")
            ax.plot([sum(ev.get(n, [0])) / len(ev.get(n, [0])) for n in names], [i + off for i in y], "o", ms=5,
                    color=col, mec="white", mew=0.8, label=lab)
        ax.set_yticks(y, names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(-0.04, 1.04)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{bench} held-out tasks")
        ax.set_xlabel("Success (bar: range over evals)")
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.16), fontsize=7)
    fig.tight_layout(w_pad=3)
    save(fig, "goalhist.pdf")


def fig_rlonly(D):
    """RL from the base model (no cloning) on ML10: 16 vs 32 groups per step, and with reasoning."""
    import matplotlib.pyplot as plt
    runs = [("m_rl", "16 groups/step (gain 1.0)"), ("ro_ml10", "32 groups/step"),
            ("rr_ml10", "16 groups/step, with reasoning (T 1.0)")]
    runs = [(i, k, n) for i, (k, n) in enumerate(runs) if k in D and series(D[k], "eval/train_tasks/success")[0]]
    if not runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3))
    for ax, key, title in ((axes[0], "eval/train_tasks/success", "ML10 training tasks"),
                           (axes[1], "rollout/success", "Training rollouts (5-step mean)")):
        ends = []
        for i, k, name in runs:
            xs, ys = series(D[k], key)
            if not xs:
                continue
            sm = smooth(ys, 5) if key.startswith("rollout") else ys
            line(ax, xs, sm, C[i], name, raw=(xs, ys) if key.startswith("rollout") else None,
                 marker=None if key.startswith("rollout") else "o")
            ends.append((xs[-1], sm[-1], f"{sm[-1]:.2f}", C[i]))
        ax.axhline(0.69, color=MUTED, lw=1.0, zorder=0)
        ax.set_title(title)
        ax.set_xlabel("RL step")
        ax.set_ylim(0, 0.75)
        end_labels(ax, ends, xpad=3)
    axes[0].set_ylabel("Success")
    axes[0].text(2, 0.66, "gray line: 400-step clone (0.69)", fontsize=6.5, color=MUTED, va="top")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12), fontsize=7)
    fig.tight_layout(w_pad=3)
    save(fig, "rlonly_ml10.pdf")


# ----------------------------------------------------------------------------- tables

def tex_escape(s):
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def table_clones(D):
    lines = [r"\begin{tabular}{lrrl}", r"\toprule", r"Clone & Training pos. & Held-out pos. & Evals \\", r"\midrule"]
    for name, keys, note in CLONES:
        tr = clone_stats(D, keys, "eval/train_positions/success")
        he = clone_stats(D, keys, "eval/heldout_positions/success")
        if not tr:
            lines.append(f"{tex_escape(name.replace(' · ', ', '))} & \\multicolumn{{2}}{{c}}{{running}} & \\\\")
            continue
        lines.append(f"{tex_escape(name.replace(' · ', ', '))} & {tr[0]}/{tr[1]} ({tr[0]/tr[1]:.2f}) & "
                     f"{he[0]}/{he[1]} ({he[0]/he[1]:.2f}) & {tex_escape(note)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_runs(D):
    lines = [r"\begin{longtable}{@{}llll@{}}", r"\toprule", r"Key & Run & wandb group & Model \\", r"\midrule", r"\endhead"]
    for k, run in D.items():
        m = (run["config"].get("model") or "").split("/")[-1].replace("-Video-Instruct", "").replace("-Instruct", "")
        m = m.replace("-it", "")
        proj = "maze" if run["project"].endswith("pointmaze") else "mw"
        lines.append(f"\\texttt{{{tex_escape(k)}}} & \\texttt{{{run['id']}}} & "
                     f"\\texttt{{{proj}/{tex_escape(run['group'] or '')}}} & {tex_escape(m)} \\\\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    return "\n".join(lines)


def write_tables(D):
    os.makedirs(TAB, exist_ok=True)
    for name, fn in (("clones", table_clones), ("runs", table_runs)):
        with open(os.path.join(TAB, f"{name}.tex"), "w") as f:
            f.write(fn(D) + "\n")


# ----------------------------------------------------------------------------- build

def build():
    D = load()
    style()
    for fn in (fig_maze_procgen, fig_maze_fixed, fig_reach_models, fig_reach_exploration, fig_push_reward,
               fig_push_clones, fig_push_rl, fig_ml10, fig_rl_fixes, fig_recipe, fig_pertask, fig_reasoning, fig_advmode,
               fig_goalhist, fig_rlonly):
        fn(D)
    fig_expert_gain()
    write_tables(D)
    for _ in range(2):   # twice: table of contents and references
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "vlmrl_results.tex"], cwd=OUT,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:])
            raise SystemExit("pdflatex failed")
    for ext in ("aux", "log", "out", "toc"):
        p = os.path.join(OUT, f"vlmrl_results.{ext}")
        if os.path.exists(p):
            os.remove(p)
    print("wrote", os.path.join(OUT, "vlmrl_results.pdf"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="pull run histories from wandb first")
    ap.add_argument("--entity", default=ENTITY, help="wandb entity to fetch from (default: $WANDB_ENTITY or tri)")
    ap.add_argument("--metaworld_project", default=MW)
    ap.add_argument("--maze_project", default=MAZE)
    ap.add_argument("--figures_only", action="store_true", help="skip pdflatex")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.entity, {MW: a.metaworld_project, MAZE: a.maze_project})
    if a.figures_only:
        D = load()
        style()
        for fn in (fig_maze_procgen, fig_maze_fixed, fig_reach_models, fig_reach_exploration, fig_push_reward,
                   fig_push_clones, fig_push_rl, fig_ml10, fig_rl_fixes, fig_recipe, fig_pertask, fig_reasoning, fig_advmode,
               fig_goalhist, fig_rlonly):
            fn(D)
        fig_expert_gain()
        write_tables(D)
    else:
        build()
