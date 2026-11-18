"""wandb Report for the SmolVLM-256M procedural-maze runs (group smolvlm-procgen-v1): live panels for the prompt
comparison (compact vs compact_rel), layout generalization (id / ood eval sets), learning dynamics, behaviour,
episode videos and cost.

    python scripts/make_smolvlm_procgen_report.py --takeaways takeaways.md [--group smolvlm-procgen-v1]

A "prompt styles at a glance" section (every prompt as the model reads it, rendered from the code, plus what happened
with each; scripts/report_notes/prompts_results.md) and a "related runs" section (other models / fixed mazes, run by another agent) is included by default: its text is
scripts/report_notes/smolvlm_procgen_related.md and its panels query --related_groups (`--related ''` drops it).

Every panel queries the runs live, so curves keep updating while runs train; only the takeaways and setup text are
static (rerun the script, or edit the report in the UI, to refresh them).
"""
from __future__ import annotations

import argparse
from datetime import datetime
from urllib.parse import urlparse

import wandb

SETS = {"id_layouts": "the 64 training layouts (new start/goal pairs)",
        "ood_layouts": "16 held-out 11x11 layouts",
        "ood_larger": "16 held-out 15x15 layouts, 30 turns"}

SETUP = """\
**Task.** PointMaze on procedurally generated mazes (`experiment=pointmaze_procgen`): every episode draws an 11x11
layout from 64 training seeds, a random start and a goal at least 3 cells away. The model sees a top-down image plus a
short text line each turn and replies with a move `{"x": dx, "y": dy}` (at most 3 cells per turn, the ball stops at
walls); 20 turns per episode.

**Prompts** (`env_kwargs.prompt_style`):
* `compact`: `Ball: x=0.31, y=1.32. Goal: x=0.07, y=0.92. Move:` (absolute goal position).
* `compact_rel`: `Ball: x=0.31, y=1.32. Goal: x=-0.24, y=-0.40 from the ball. Move:` (goal relative to the ball).

**Training.** SmolVLM-256M-Instruct, full fine-tuning, GRPO with the **sparse success reward only** (1 if the ball
reaches the goal, else 0). 64 groups x 16 rollouts per step with DAPO filtering (groups where all 16 rollouts got the
same reward carry no signal and are resampled, up to 3 rounds, so <= 3072 episodes / step). Rollouts are sampled at
T=1 with vLLM; an unparseable reply is a no-op (`on_invalid=noop`). One run per GPU, trainer and vLLM server on the
same A6000.

**Eval.** Greedy (T=0) every 5 steps on three fixed sets of 64 episodes each:
""" + "\n".join(f"* `{k}`: {v}." for k, v in SETS.items()) + """

`eval/success` is the unweighted mean of the three. One seed per config; a 64-episode set has a standard error of
about 5 points at 25% success, so differences of a few points are noise.
"""


RELATED = "scripts/report_notes/smolvlm_procgen_related.md"
PROMPT_RESULTS = "scripts/report_notes/prompts_results.md"
PROMPT_STYLES = {  # style -> (who uses it, where the goal is given)
    "pear": ("PEAR paper protocol, App. H", "absolute, once in the system prompt (TASK)"),
    "simple": ("short custom prompt", "absolute, every turn"),
    "compact": ("< 1B models", "absolute, every turn"),
    "compact_rel": ("< 1B models", "**relative to the ball**, every turn"),
}


def prompt_table() -> str:
    """'What the model reads' for every prompt style, rendered from the code on one fixed procedural-maze state, so
    the report always shows the prompts the runs actually used."""
    import re
    import sys

    import numpy as np
    sys.path.insert(0, ".")
    from vlmrl.envs.pointmaze import make_pointmaze
    rows = ["| prompt | system prompt | each turn the model reads | it replies | goal given as |",
            "|---|---|---|---|---|"]
    for style, (who, goal) in PROMPT_STYLES.items():
        b = make_pointmaze(maze="gen11x11", train_mazes=1, max_delta=3.0, min_goal_dist=3.0, prompt_style=style,
                           image_size=16)
        s = b.train_sampler.sample(1, np.random.default_rng(0))[0]
        env = b.make_vec_env()
        obs = env.reset([s])[0]
        env.close()
        sp = b.interface.system_prompt(s)
        heads = re.findall(r"^## (\w[\w ]*)$", sp, flags=re.M)
        first = sp.split(". ")[0].split("\n")[0] + "."
        desc = f"{len(sp):,} chars; " + (f"sections {' / '.join(heads)}" if heads else f"\"{first}\"")
        parts = b.interface.observation_message(obs, 0, None, False)
        text = " ".join(c["text"] for c in parts if c["type"] == "text").replace("\n", " ")
        order = " + ".join("image" if c["type"] == "image" else "text" for c in parts)
        reply = b.interface.format_action(np.array([0.5, -1.0])).replace("\n", " ")
        code = (lambda t: f"`` {t} ``" if "`" in t else f"`{t}`")
        if style == "pear":
            goal += f": {code(b.interface.instruction(s))}"
        rows.append(f"| `{style}` ({who}) | {desc} | {order}: {code(text)} | {code(reply)} | {goal} |")
    return "\n".join(rows)


def prompt_blocks(a, wr) -> list:
    import os
    if not a.prompts:
        return []
    md = ("Same ball, goal and maze for every style (procedural 11x11 layout, first turn). The image is the same "
          "top-down render in all of them.\n\n" + prompt_table())
    if os.path.exists(PROMPT_RESULTS):
        md += "\n\n" + open(PROMPT_RESULTS).read()
    return [wr.H1("Prompt styles at a glance"), wr.MarkdownBlock(md)]
RELATED_GROUPS = "pointmaze-small-fft-batch-v1,pointmaze-small-fft-batch-v2"


def related_blocks(a, wr) -> list:
    """Companion experiments (other models, fixed mazes) with their own live panels, after the takeaways."""
    import os
    if not a.related or not os.path.exists(a.related):
        return []
    groups = [g for g in a.related_groups.split(",") if g]

    def rs(g):
        return wr.Runset(entity=a.entity, project=a.project, name=g, filters=f"Group == '{g}'")
    keys = [("eval/success", "Eval success (greedy, 32 episodes)"), ("rollout/invalid_frac", "Invalid replies"),
            ("rollout/success", "Training rollout success (T=1)"), ("train/grad_norm", "Gradient norm")]
    return [
        wr.H1("Related runs: other models and fixed mazes (AWS Batch)"),
        wr.MarkdownBlock(open(a.related).read()),
        wr.PanelGrid(runsets=[rs(g) for g in groups],
                     panels=[wr.LinePlot(title=t, x="Step", y=[k], max_runs_to_show=30) for k, t in keys]
                     + [wr.BarPlot(title="Latest eval success", metrics=["eval/success"], orientation="h",
                                   max_runs_to_show=30)]),
    ]


def build(a) -> str:
    import wandb_workspaces.reports.v2 as wr
    flt = f"Group == '{a.group}'"

    def runs(name="runs", groupby=()):
        return wr.Runset(entity=a.entity, project=a.project, name=name, filters=flt, groupby=list(groupby))

    def lines(keys, titles=None, **kw):
        titles = titles or keys
        return [wr.LinePlot(title=t, x="Step", y=[k], max_runs_to_show=20, **kw) for k, t in zip(keys, titles)]

    def grid(panels, runset=None):
        return wr.PanelGrid(runsets=[runset or runs()], panels=panels)

    def episode_blocks(split, title):
        tag = split.replace("/", " ")
        return [
            wr.H2(title),
            grid([wr.MediaBrowser(title=f"{tag} video", media_keys=[f"{split}/episode"], num_columns=1,
                                  layout=wr.Layout(x=0, y=0, w=10, h=10)),
                  wr.MediaBrowser(title=f"{tag} transcript", media_keys=[f"{split}/transcript"], num_columns=1,
                                  layout=wr.Layout(x=10, y=0, w=14, h=10)),
                  wr.WeavePanelSummaryTable(table_name=f"{split}/samples", layout=wr.Layout(x=0, y=10, w=24, h=12))]),
        ]

    by_prompt = runs("by prompt", ["config.env_kwargs.prompt_style"])
    set_keys = [f"eval/{s}/success" for s in SETS]
    takeaways = open(a.takeaways).read() if a.takeaways else "_(to be written)_"
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    blocks = [
        wr.TableOfContents(),
        wr.H1("Takeaways"),
        wr.MarkdownBlock(f"_Written {stamp}; the panels below are live and may have moved on since._\n\n" + takeaways),
        *prompt_blocks(a, wr),
        *related_blocks(a, wr),
        wr.H1("Headline: greedy eval success"),
        wr.P("Mean over the three eval sets. Each line is one run; the run name gives the prompt and learning rate "
             "(the two runs named only lr=... use the compact prompt and were stopped early)."),
        grid(lines(["eval/success", "rollout/success"],
                   ["Eval success (greedy, mean of 3 sets)", "Training rollout success (T=1)"])),
        wr.H1("Prompt: absolute vs relative goal"),
        wr.P("Runs grouped by prompt (line = mean over learning rates, band = min/max). With the absolute goal "
             "(compact) the policy collapsed to echoing the goal's coordinates as its move; with the goal relative "
             "to the ball (compact_rel) the same echo is the straight-line move."),
        grid(lines(["eval/success", "rollout/success", "rollout/invalid_frac"],
                   ["Eval success by prompt", "Rollout success by prompt", "Invalid replies by prompt"],
                   groupby_aggfunc="mean", groupby_rangefunc="minmax"), by_prompt),
        wr.H1("Generalization: in vs out of distribution"),
        wr.P("Greedy success on each eval set. A model that memorized the training layouts would do much better on "
             "id_layouts than on the held-out sets; similar curves mean the skill transfers to unseen mazes."),
        grid(lines(set_keys, [f"{s}: {d}" for s, d in SETS.items()])
             + [wr.BarPlot(title="Latest eval success per set", metrics=set_keys, orientation="h",
                           max_runs_to_show=20)]),
        wr.H1("Learning dynamics"),
        wr.P("groups_kept = groups out of 64 whose 16 rollouts had mixed outcomes (the only ones with a GRPO "
             "signal; few means the task is too hard or already solved). grad_norm spikes and falling mean_logp "
             "mark unstable updates (compact_rel at lr=3e-5 had one at step 5 and recovered)."),
        grid(lines(["rollout/groups_kept", "train/grad_norm", "train/mean_logp", "rollout/invalid_frac"])),
        wr.H1("Behaviour"),
        wr.P("Turns per episode (lower = faster solves; 20 = timed out), how episodes end, and reply length. "
             "A policy stuck against a wall repeats the same move until max_turns."),
        grid(lines(["eval/turns", "rollout/turns", "rollout/end_success", "rollout/end_max_turns",
                    "rollout/resp_chars"])),
        wr.H1("Episodes: videos, transcripts and sampled actions"),
        wr.P("What the models actually do. For each split: the logged episode video (top-down frames, use the step "
             "slider; the caption gives success / turns / end reason) next to its transcript (exact prompts and "
             "replies), then the samples table with one row per logged episode (success, turns, reply length, "
             "transcript, video) across all runs and steps. Training episodes are sampled at T=1; eval episodes "
             "are greedy. The compact runs' transcripts show the goal-echo collapse (the same reply every turn); "
             "compact_rel replies copy the relative offset."),
        *[blk for name, title in [("train", "Training rollouts (T=1)")] + [(f"eval/{s}", f"Eval {s}: {d}")
                                                                      for s, d in SETS.items()]
          for blk in episode_blocks(name, title)],
        wr.H1("Cost"),
        wr.P("Seconds per step for rollouts (incl. DAPO resampling) and the update, and sequences trained per step."),
        grid(lines(["time/rollout", "time/train", "train/sequences", "rollout/rounds"])),
        wr.H1("Setup"),
        wr.MarkdownBlock(SETUP),
        wr.H1("Runs"),
        grid([wr.RunComparer(diff_only=True)]),
    ]
    report = wr.Report(entity=a.entity, project=a.project, title=a.title, blocks=blocks,
                       description=f"SmolVLM-256M full-FT GRPO on procedural mazes, sparse reward (group {a.group}).")
    if a.url:
        old = wr.Report.from_url(a.url.replace("/wandb/", "/", 1))  # from_url wants entity/project/reports/name
        old.blocks, old.title, old.description = report.blocks, report.title, report.description
        report = old
    report.save(draft=a.draft)
    # Report.url drops the app URL's path prefix (e.g. https://forge.coreweave.com/wandb/), so rebuild it
    from wandb_workspaces._graphql import get_app_url
    return get_app_url(wandb.Api()).rstrip("/") + urlparse(report.url).path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="smolvlm-procgen-v1")
    ap.add_argument("--entity", default="tri")
    ap.add_argument("--project", default="vlm-rl-pointmaze")
    ap.add_argument("--title", default="SmolVLM-256M learns procedural mazes from sparse reward")
    ap.add_argument("--takeaways", default=None, help="markdown file for the takeaways section")
    ap.add_argument("--url", default=None, help="update this existing report in place instead of creating one")
    ap.add_argument("--draft", action="store_true", help="save as an unpublished draft")
    ap.add_argument("--related", default=RELATED, help="markdown for the related-runs section ('' = omit)")
    ap.add_argument("--no_prompts", dest="prompts", action="store_false", help="omit the prompt-styles section")
    ap.add_argument("--related_groups", default=RELATED_GROUPS, help="comma-separated wandb groups for its panels")
    print(build(ap.parse_args()))


if __name__ == "__main__":
    main()
