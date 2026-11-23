"""wandb Report for the SmolVLM-256M procedural-maze runs (group smolvlm-procgen-v1). Text is bullets; panels are live.

    python scripts/make_smolvlm_procgen_report.py            # rewrite the existing report (REPORT_URL) in place
    python scripts/make_smolvlm_procgen_report.py --url ''   # create a new report instead

Sections: takeaways (scripts/report_notes/smolvlm_procgen_takeaways.md), prompts compared (every prompt style as the
model reads it, rendered from the code, what differs, what the models actually wrote, and
scripts/report_notes/prompts_results.md), related runs from the AWS Batch agent (scripts/report_notes/
smolvlm_procgen_related.md + live panels over --related_groups; `--related ''` drops it), then live panels: headline,
by prompt, generalization (per eval set), learning dynamics, behaviour, episodes (video + transcript + samples table
per split), cost, setup and a run comparer.

Every panel queries the runs live, so curves keep updating while runs train; the markdown text is static (rerun the
script, or edit the report in the UI, to refresh it).
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from urllib.parse import urlparse

import wandb

SETS = {"id_layouts": "the 64 training layouts (new start/goal pairs)",
        "ood_layouts": "16 held-out 11x11 layouts",
        "ood_larger": "16 held-out 15x15 layouts, 30 turns"}

SETUP = """\
**Task**
* PointMaze on procedurally generated mazes (`experiment=pointmaze_procgen`).
* Each episode: an 11x11 layout from 64 training seeds, random start, goal >= 3 cells away, 20 turns.
* Each turn: top-down image + one text line; reply = a move `{"x": dx, "y": dy}` (<= 3 cells; the ball stops at walls).

**Training**
* SmolVLM-256M-Instruct, full fine-tuning (fp32 master weights), GRPO + DAPO group filter; no KL to the reference.
* **Sparse success reward only** (1 if the ball reaches the goal, else 0).
* 64 groups x 16 rollouts per step; groups whose 16 rollouts all got the same reward carry no signal and are resampled
  (up to 3 rounds, so <= 3072 episodes / step).
* Rollouts sampled at T=1 by vLLM; an unparseable reply is a no-op (`on_invalid=noop`).
* One run per A6000 (trainer + vLLM server on the same GPU).

**Eval** (greedy, every 5 steps, 64 episodes per set)
""" + "\n".join(f"* `{k}`: {v}." for k, v in SETS.items()) + """
* `eval/success` = unweighted mean of the three. SE ~5 points at 25% success; one seed per config.
"""

TAKEAWAYS = "scripts/report_notes/smolvlm_procgen_takeaways.md"
RELATED = "scripts/report_notes/smolvlm_procgen_related.md"
PROMPT_RESULTS = "scripts/report_notes/prompts_results.md"
RELATED_GROUPS = "pointmaze-small-fft-batch-v1,pointmaze-small-fft-batch-v2,pointmaze-gemma4-fft-batch-v1"
RELATED_PROCGEN_GROUPS = "procgen-fft-larger-v1,procgen-fft-larger-v1-lr1e-6"   # before the loss-mask fix
FIXED_PROCGEN_GROUPS = "procgen-fft-v2,procgen-fft-larger-v2-maskfix"           # after it (commit 2e2b7d2)
MASK_NOTE = "scripts/report_notes/mask_bug.md"
SUMMARY = "scripts/report_notes/summary.md"
METAWORLD = "scripts/report_notes/metaworld.md"
METAWORLD_PROJECT = "vlm-rl-metaworld"
# (runset label, wandb group, optional run-ID filter): the runs that learned from 0 (corrected camera, compact_rel,
# pref_mw) and the live v3 runs (reach + ML10)
METAWORLD_REACH = [("Qwen3.5-4B, RL only (lr 1e-5)", "metaworld-q35-v3-lr1e-5", ["f1bc2e04"]),
                   ("Qwen3.5-4B, RL only (lr 2e-5: degrades)", "metaworld-ml1-q35-mlp-v2-lr2e-5", ["3737934e"]),
                   ("Qwen3-VL-4B, BC then RL", "metaworld-bc2-q3vl4b-reach", None),
                   ("Gemma 4 E2B, BC then RL", "metaworld-bc2-gemma4-reach", None),
                   ("Gemma 4 E2B, RL only (T=1.6)", "metaworld-reach-gemma4-t16", None),
                   ("Qwen3-VL-2B, RL only (T=1.6)", "metaworld-reach-q3vl2b-t16", None)]
METAWORLD_PUSH = [("gain 0.5: Qwen3.5-4B, BC (30 steps) then RL", "metaworld-g05-q35-push", None),
                  ("gain 0.5: Qwen3.5-4B, BC (100 steps) then RL", "metaworld-g05-q35-push-bc100", None),
                  ("gain 0.5: Qwen3-VL-4B, BC (100 steps) then RL", "metaworld-g05-q3vl4b-push-bc100", None),
                  ("gain 0.5: Qwen3.5-4B, 450 training positions (10 seeds), BC (100) then RL", "metaworld-g05-q35-push-seeds10", None),
                  ("450-position clone, RL lr 2e-6", "metaworld-rlfix-push450-lr2e-6", None),
                  ("450-position clone, RL adv_min=0 (positive only)", "metaworld-rlfix-push450-posadv", None),
                  ("gain 1.0: Qwen3.5-4B, BC then RL, pref_mw (RL undoes BC)", "metaworld-bc2-q35-push", None),
                  ("gain 1.0: Qwen3.5-4B, BC then RL, pref_mw_obj", "metaworld-bc3-q35-push", None)]
METAWORLD_ML45 = [("recipe: Qwen3.5-4B, BC 400 then RL (train_seeds=10)", "metaworld-recipe-q35-ml45", None),
                  ("recipe: Qwen3-VL-4B, BC 400 then RL (train_seeds=10)", "metaworld-recipe-q3vl4b-ml45", None)]
METAWORLD_ML10 = [("recipe: Qwen3.5-4B, BC 200 then RL (train_seeds=10)", "metaworld-recipe-q35-ml10", None),
                  ("recipe: Qwen3.5-4B, BC 200, RL lr 2e-6", "metaworld-recipe-q35-ml10-rl2e-6", None),
                  ("gain 0.5: Qwen3.5-4B, BC (200 steps) then RL", "metaworld-g05-q35-ml10-bc200", None),
                  ("200-step clone, RL lr 2e-6", "metaworld-rlfix-ml10-lr2e-6", None),
                  ("200-step clone, RL adv_min=0 (positive only)", "metaworld-rlfix-ml10-posadv", None),
                  ("gain 0.5: Qwen3.5-4B, BC (100 steps) then RL", "metaworld-g05-q35-ml10-bc100", None),
                  ("gain 0.5: Qwen3.5-4B, BC (30 steps) then RL", "metaworld-g05-q35-ml10", None),
                  ("gain 1.0: Qwen3.5-4B, RL only", "metaworld-q35-v3-lr1e-5", ["147bb6b9"]),
                  ("gain 1.0: Qwen3.5-4B, RL from the reach adapter", "metaworld-q35-v4-ml10-from-reach", None)]
METAWORLD_BEST_RUN = ("metaworld-q35-v3-lr1e-5", "f1bc2e04")   # Qwen3.5-4B reach, lr 1e-5: held-out 0.41 at step 30
# Restarts that crashed or failed within a few steps and were rerun under the same name (infrastructure: vLLM reload
# errors, port collisions); hidden so every panel shows one line per configuration.
HIDDEN_RUNS = ["0dcb561c", "101b2806", "f4c4f00a", "07e34721", "3539515a", "9a558a4c", "f7aa5df3", "da16b80f",
               "67ca3bc6", "b4811125"]
REPORT_URL = ("https://forge.coreweave.com/wandb/tri/vlm-rl-pointmaze/reports/"
              "SmolVLM-256M-learns-procedural-mazes-from-sparse-reward--VmlldzoxODMzMjEwNA==")

PROMPT_STYLES = {  # style -> (who it is for, where the goal is given)
    "pear": ("PEAR paper protocol, App. H; 4B-class models", "absolute, once, in the system prompt's TASK section"),
    "simple": ("short custom prompt; ~2B models", "absolute, every turn (with position and velocity)"),
    "compact": ("< 1B models", "absolute, every turn"),
    "compact_rel": ("< 1B models", "**relative to the ball**, every turn"),
}
OBSERVED = {  # what the models actually wrote (from the run transcripts)
    "pear": "small models fail the format: Qwen3-VL-2B echoes the state JSON; SmolVLM-256M / MiniCPM-V ~100% invalid",
    "simple": "~2B models learn the tag in 5-10 steps; SmolVLM-256M stays ~100% invalid (prose, no tag)",
    "compact": "after training, `x=2.24, y=-4.10. Move right.` every turn: the goal's absolute position, whatever "
               "the ball's position",
    "compact_rel": "after training, `Goal: x=-1.78, y=-2.98 from the ball.` -> `x=-1.78, y=-2.98.`: the "
                   "straight-line move; lr=3e-5 later degenerated to `X=+ Haguepite facequeenue...`",
}


def prompt_md() -> str:
    """Every prompt style as the model reads it, rendered from the code on one fixed procedural-maze state (first
    turn), so the report always shows the prompts the runs actually used."""
    import re
    import sys

    import numpy as np
    sys.path.insert(0, ".")
    from vlmrl.envs.pointmaze import make_pointmaze

    def code(t):
        return f"`` {t} ``" if "`" in t else f"`{t}`"

    out = ["**What differs between the prompts**",
           "* **Where the goal is**: `pear` gives the absolute goal once, in the system prompt; `simple` and `compact` "
           "repeat the absolute goal every turn; `compact_rel` gives it **relative to the ball** every turn.",
           "* **Length**: `pear` ~1.8k-char structured system prompt; `simple` ~0.9k; `compact` / `compact_rel` "
           "~0.36k with identical system prompts (only the per-turn line differs).",
           "* **Reply format**: `pear` a fenced json block with the change of every quantity; `simple` an "
           "`<action>` tag; `compact` / `compact_rel` bare JSON, parsed leniently (first x and y values anywhere).",
           "* **Why `compact_rel` works for a 256M model**: copying the numbers it is given is already a good move, "
           "so RL only has to refine it.",
           "",
           "**Each prompt as the model reads it** (same ball, goal and 11x11 maze; first turn; same image in all)"]
    seen: dict[str, str] = {}
    for style, (who, goal) in PROMPT_STYLES.items():
        b = make_pointmaze(maze="gen11x11", train_mazes=1, max_delta=3.0, min_goal_dist=3.0, prompt_style=style,
                           image_size=16)
        s = b.train_sampler.sample(1, np.random.default_rng(0))[0]
        env = b.make_vec_env()
        obs = env.reset([s])[0]
        env.close()
        sp = b.interface.system_prompt(s)
        heads = re.findall(r"^## (\w[\w ]*)$", sp, flags=re.M)
        parts = b.interface.observation_message(obs, 0, None, False)
        turn = " + ".join("[image]" if c["type"] == "image" else code(c["text"].replace("\n", " ")) for c in parts)
        reply = b.interface.format_action(np.array([0.5, -1.0])).replace("\n", " ")
        same = [k for k, v in seen.items() if v == sp]
        seen[style] = sp
        if same:
            system = f"identical to `{same[0]}`'s"
        elif heads:
            system = f"{len(sp):,} chars, sections {' / '.join(heads)}; goal: {code(b.interface.instruction(s))}"
        else:
            system = f"{len(sp):,} chars: " + code(sp.replace("\n", " ") if len(sp) < 500 else sp.split("\n")[0])
        out += [f"* **`{style}`** ({who}); goal given {goal}",
                f"  * system prompt: {system}",
                f"  * each turn: {turn}",
                f"  * expected reply: {code(reply)}",
                f"  * observed: {OBSERVED[style]}"]
    return "\n".join(out)


def tile(panels, per_row=4, h=8):
    """Lay panels out left to right, `per_row` per row, filling the 24-column grid (W&B's default is 3 narrow panels
    per row, which leaves most of a wide screen empty). Panels with an explicit layout keep it."""
    import wandb_workspaces.reports.v2 as wr
    per_row = max(1, min(per_row, len(panels)))
    w = 24 // per_row
    for i, p in enumerate(panels):
        if getattr(p, "layout", None) is None or (p.layout.x, p.layout.y, p.layout.w, p.layout.h) == (0, 0, 8, 6):
            p.layout = wr.Layout(x=(i % per_row) * w, y=(i // per_row) * h, w=w, h=h)
    return panels


def hide(flt: str) -> str:
    return flt + " and ID not in [" + ", ".join(f"'{r}'" for r in HIDDEN_RUNS) + "]"


def prompt_blocks(a, wr) -> list:
    if not a.prompts:
        return []
    md = prompt_md()
    if os.path.exists(PROMPT_RESULTS):
        md += "\n\n" + open(PROMPT_RESULTS).read()
    return [wr.H2("Prompts compared (exact text)"), wr.MarkdownBlock(md)]


RELATED_PROCGEN = [("Qwen3-VL-2B, lr 5e-6", "procgen-fft-v2-diag", ["7273b286"]),
                   ("MiniCPM-V 4.6, lr 1e-6", "procgen-fft-v3-exact", ["23055312"]),
                   ("Gemma 4 E2B, lr 1e-6", "procgen-fft-v2", ["3e97482a"]),
                   ("SmolVLM2-2.2B, lr 5e-6, traj_mean", "procgen-fft-v3-exact-trajmean", ["e320a919"])]
RELATED_FIXED = [("SmolVLM2-500M (UMaze, medium)", "smolvlm2-fixed-maze-v1", ["da33c51b", "86e6b888"]),
                 ("SmolVLM2-2.2B traj_mean (UMaze, medium)", "smolvlm2-fixed-maze-v1-trajmean", ["73150cd4", "1eb55355"]),
                 ("SmolVLM2-2.2B traj_mean + KL (UMaze, medium)", "smolvlm2-fixed-maze-v1-trajmean-kl",
                  ["f118986b", "077253d8"])]


def pinned(a, wr, label, group, run_ids=None, project=None):
    """A runset of one wandb group, optionally pinned to run IDs (groups can mix models / restarts)."""
    f = f"Group == '{group}'" + (" and ID in [" + ", ".join(f"'{i}'" for i in run_ids) + "]" if run_ids else "")
    return wr.Runset(entity=a.entity, project=project or a.project, name=label, filters=f)


def related_blocks(a, wr) -> list:
    """Other models on mazes (after the loss-mask fix): short notes + one grid each for procedural and fixed mazes."""
    if not a.related or not os.path.exists(a.related):
        return []
    set_keys = [("eval/success", "Eval success (mean of 3 sets)"), ("eval/id_layouts/success", "training layouts"),
                ("eval/ood_layouts/success", "held-out 11x11"), ("eval/ood_larger/success", "held-out 15x15"),
                ("rollout/invalid_frac", "Invalid replies"), ("train/mean_logp", "Mean log-prob of trained tokens")]
    fixed_keys = [("eval/success", "Eval success (fixed maze)"), ("rollout/success", "Training rollout success"),
                  ("rollout/invalid_frac", "Invalid replies"), ("train/mean_logp", "Mean log-prob of trained tokens")]
    line = lambda k, t: wr.LinePlot(title=t, x="Step", y=[k], max_runs_to_show=30)
    return [
        wr.H1("Other models on mazes (AWS Batch)"),
        wr.MarkdownBlock(open(a.related).read()),
        wr.H2("Procedural mazes"),
        wr.PanelGrid(runsets=[pinned(a, wr, *x) for x in RELATED_PROCGEN], panels=tile([line(*x) for x in set_keys], 3)),
        wr.H2("Single fixed mazes, 100 steps"),
        wr.PanelGrid(runsets=[pinned(a, wr, *x) for x in RELATED_FIXED], panels=tile([line(*x) for x in fixed_keys])),
    ]


def metaworld_blocks(a, wr) -> list:
    """Metaworld runs (another wandb project): notes, eval / training curves, and episode videos of the best run."""
    if not a.metaworld or not os.path.exists(METAWORLD):
        return []

    def rs(label, group, run_ids=None):
        return pinned(a, wr, label, group, run_ids, project=METAWORLD_PROJECT)
    line = lambda k, t: wr.LinePlot(title=t, x="Step", y=[k], max_runs_to_show=30)
    ml1 = [("eval/heldout_positions/success", "Eval: held-out goal positions"),
           ("eval/train_positions/success", "Eval: training positions"),
           ("rollout/success", "Training rollout success (sampled)"),
           ("train/mean_logp", "Mean log-prob of trained tokens (sharpening)")]
    ml10 = [("eval/train_tasks/success", "Eval: its 10 training tasks"),
            ("eval/heldout_tasks/success", "Eval: its 5 held-out tasks"),
            ("rollout/success", "Training rollout success (sampled)")]
    best = rs("best run: Qwen3.5-4B reach (lr 1e-5)", METAWORLD_BEST_RUN[0], [METAWORLD_BEST_RUN[1]])
    return [
        wr.H1("Metaworld: reach learns, push and ML10 start with behaviour cloning"),
        wr.MarkdownBlock(open(METAWORLD).read()),
        wr.H2("Reach, by model (live; BC runs start at their clone's score)"),
        wr.PanelGrid(runsets=[rs(*x) for x in METAWORLD_REACH], panels=tile([line(*x) for x in ml1])),
        wr.H2("Push: behaviour cloning, then RL with each preference reward"),
        wr.PanelGrid(runsets=[rs(*x) for x in METAWORLD_PUSH], panels=tile([line(*x) for x in ml1])),
        wr.H2("ML10 (10 training tasks, 5 held-out tasks)"),
        wr.PanelGrid(runsets=[rs(*x) for x in METAWORLD_ML10], panels=tile([line(*x) for x in ml10], 3)),
        wr.H2("ML45 (45 training tasks, 5 held-out tasks): the generic recipe"),
        wr.PanelGrid(runsets=[rs(*x) for x in METAWORLD_ML45],
                     panels=tile([line(k, t.replace("10 training", "45 training")) for k, t in ml10], 3)),
        wr.H2("Episodes of the best reach run: video (step slider) and transcript"),
        wr.PanelGrid(runsets=[best], panels=[
            wr.MediaBrowser(title="held-out positions: video", media_keys=["eval/heldout_positions/episode"],
                            num_columns=1, layout=wr.Layout(x=0, y=0, w=8, h=11)),
            wr.MediaBrowser(title="held-out positions: transcript", media_keys=["eval/heldout_positions/transcript"],
                            num_columns=1, layout=wr.Layout(x=8, y=0, w=8, h=11)),
            wr.MediaBrowser(title="training positions: video", media_keys=["eval/train_positions/episode"],
                            num_columns=1, layout=wr.Layout(x=16, y=0, w=8, h=11))]),
    ]


def bullets(*items) -> "object":
    import wandb_workspaces.reports.v2 as wr
    return wr.MarkdownBlock("\n".join(f"* {x}" for x in items))


def build(a) -> str:
    import wandb_workspaces.reports.v2 as wr
    flt = f"Group == '{a.group}'"

    def runs(name="runs", groupby=()):
        return wr.Runset(entity=a.entity, project=a.project, name=name, filters=flt, groupby=list(groupby))

    def lines(keys, titles=None, **kw):
        titles = titles or keys
        return [wr.LinePlot(title=t, x="Step", y=[k], max_runs_to_show=20, **kw) for k, t in zip(keys, titles)]

    def grid(panels, runset=None):
        return wr.PanelGrid(runsets=[runset or runs()], panels=tile(panels))

    def episode_blocks(split, title):
        tag = split.replace("/", " ")
        return [
            wr.H3(title),
            grid([wr.MediaBrowser(title=f"{tag} video", media_keys=[f"{split}/episode"], num_columns=1,
                                  layout=wr.Layout(x=0, y=0, w=10, h=10)),
                  wr.MediaBrowser(title=f"{tag} transcript", media_keys=[f"{split}/transcript"], num_columns=1,
                                  layout=wr.Layout(x=10, y=0, w=14, h=10)),
                  wr.WeavePanelSummaryTable(table_name=f"{split}/samples", layout=wr.Layout(x=0, y=10, w=24, h=12))]),
        ]

    by_prompt = runs("by prompt", ["config.env_kwargs.prompt_style", "config.model"])
    set_keys = [f"eval/{s}/success" for s in SETS]
    takeaways = open(a.takeaways).read() if a.takeaways else "_(to be written)_"
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    blocks = [
        wr.TableOfContents(),
        *([wr.H1("Summary"), wr.MarkdownBlock(open(SUMMARY).read())] if os.path.exists(SUMMARY) else []),
        *metaworld_blocks(a, wr),
        wr.H1("Procedural mazes: SmolVLM-256M (main study)"),
        wr.MarkdownBlock(f"_Text written {stamp}; the panels below are live and may have moved on since._\n\n"
                         + takeaways),
        wr.H2("Eval success and generalization"),
        bullets("One line per run; run names give prompt and lr (runs named only `lr=...` use `compact`). Eval = "
                "greedy, mean of 3 sets; rollout = T=1 training episodes.",
                "Memorizing the training layouts would show as `id_layouts` >> the held-out sets; similar curves = the "
                "skill transfers to unseen (and larger) mazes."),
        grid(lines(["eval/success", "rollout/success"], ["Eval success (greedy, mean of 3 sets)",
                                                         "Training rollout success (T=1)"])
             + lines(set_keys, [f"{s}: {d}" for s, d in SETS.items()])
             + [wr.BarPlot(title="Latest eval success per set", metrics=set_keys, orientation="h",
                           max_runs_to_show=20)]),
        wr.H2("By prompt: absolute vs relative goal"),
        bullets("Grouped by `env_kwargs.prompt_style` and model: line = mean over runs, band = min/max. `compact` "
                "echoes the absolute goal (< 7%); `compact_rel` turns the echo into the straight-line move (~20-25%)."),
        grid(lines(["eval/success", "rollout/success", "rollout/invalid_frac", "train/mean_logp"],
                   ["Eval success by prompt", "Rollout success by prompt", "Invalid replies by prompt",
                    "Mean log-prob of sampled tokens by prompt"],
                   groupby_aggfunc="mean", groupby_rangefunc="minmax"), by_prompt),
        wr.H2("Training dynamics, behaviour and cost"),
        bullets("`groups_kept`: groups with mixed outcomes (the only ones with a GRPO signal). `mean_logp` falling "
                "steadily is the early warning of a collapse; `grad_norm` spikes = unstable updates (clipped at 1.0).",
                "Turns: lower = faster solves, 20 (30 on `ood_larger`) = timed out. Reply length jumps when replies "
                "degenerate into free text. Time: seconds per step for rollouts and the update."),
        grid(lines(["rollout/groups_kept", "train/grad_norm", "train/mean_logp", "rollout/invalid_frac",
                    "rollout/turns", "rollout/resp_chars", "time/rollout", "time/train"])),
        wr.H2("Episodes: videos, transcripts and samples"),
        bullets("Video (step slider; caption = success / turns / end reason) next to its transcript; the samples "
                "table below has one row per logged episode across runs and steps (sort by success)."),
        *[blk for name, title in [("train", "Training rollouts (T=1)"),
                                  ("eval/ood_larger", f"Eval ood_larger: {SETS['ood_larger']}")]
          for blk in episode_blocks(name, title)],
        *related_blocks(a, wr),
        wr.H1("Reference"),
        *prompt_blocks(a, wr),
        wr.H2("Setup"),
        wr.MarkdownBlock(SETUP),
        wr.H2("Runs"),
        grid([wr.RunComparer(diff_only=True)]),
    ]
    report = wr.Report(entity=a.entity, project=a.project, title=a.title, blocks=blocks, width="fluid",
                       description=f"SmolVLM-256M full-FT GRPO on procedural mazes, sparse reward (group {a.group}).")
    if a.url:
        old = wr.Report.from_url(a.url.replace("/wandb/", "/", 1))  # from_url wants entity/project/reports/name
        old.blocks, old.title, old.description = report.blocks, report.title, report.description
        old.width = "fluid"
        report = old
    report.save(draft=a.draft)
    # Report.url drops the app URL's path prefix (e.g. https://forge.coreweave.com/wandb/), so rebuild it
    from wandb_workspaces._graphql import get_app_url
    return get_app_url(wandb.Api()).rstrip("/") + urlparse(report.url).path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="smolvlm-procgen-v2")   # v1 predates the loss-mask fix
    ap.add_argument("--entity", default="tri")
    ap.add_argument("--project", default="vlm-rl-pointmaze")
    ap.add_argument("--title", default="VLM RL on mazes and Metaworld: what learns, what doesn't, and why")
    ap.add_argument("--takeaways", default=TAKEAWAYS, help="markdown file for the takeaways section")
    ap.add_argument("--url", default=None,
                    help="report to update in place (default: REPORT_URL, or a new report with --draft; '' = new)")
    ap.add_argument("--draft", action="store_true",
                    help="save as a new unpublished draft (never over the live report unless --url is given)")
    ap.add_argument("--related", default=RELATED, help="markdown for the related-runs section ('' = omit)")
    ap.add_argument("--no_prompts", dest="prompts", action="store_false", help="omit the prompts section")
    ap.add_argument("--no_metaworld", dest="metaworld", action="store_false", help="omit the Metaworld section")
    ap.add_argument("--related_groups", default=RELATED_GROUPS, help="comma-separated wandb groups for its panels")
    ap.add_argument("--related_procgen_groups", default=RELATED_PROCGEN_GROUPS,
                    help="related wandb groups on the procedural setup (per-eval-set panels)")
    ap.add_argument("--fixed_procgen_groups", default=FIXED_PROCGEN_GROUPS,
                    help="procedural-setup groups trained after the loss-mask fix")
    a = ap.parse_args()
    if a.url is None:   # a draft is a separate report; only an explicit --url targets an existing one
        a.url = "" if a.draft else REPORT_URL
    print(build(a))


if __name__ == "__main__":
    main()
