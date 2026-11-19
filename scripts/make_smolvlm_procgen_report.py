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
RELATED_PROCGEN_GROUPS = "procgen-fft-larger-v1,procgen-fft-larger-v1-lr1e-6"
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


def prompt_blocks(a, wr) -> list:
    if not a.prompts:
        return []
    md = prompt_md()
    if os.path.exists(PROMPT_RESULTS):
        md += "\n\n" + open(PROMPT_RESULTS).read()
    return [wr.H1("Prompts compared"), wr.MarkdownBlock(md)]


def related_blocks(a, wr) -> list:
    """Companion experiments (other models, fixed mazes) with their own live panels, after the prompts."""
    if not a.related or not os.path.exists(a.related):
        return []
    groups = [g for g in a.related_groups.split(",") if g]

    def rs(g):
        return wr.Runset(entity=a.entity, project=a.project, name=g, filters=f"Group == '{g}'")
    keys = [("eval/success", "Eval success (greedy, 32 episodes)"), ("rollout/invalid_frac", "Invalid replies"),
            ("rollout/success", "Training rollout success (T=1)"), ("train/grad_norm", "Gradient norm")]
    procgen = [g for g in a.related_procgen_groups.split(",") if g]
    set_keys = [("eval/success", "Eval success (mean of 3 sets)"), ("eval/id_layouts/success", "id_layouts"),
                ("eval/ood_layouts/success", "ood_layouts (held-out 11x11)"),
                ("eval/ood_larger/success", "ood_larger (held-out 15x15)"),
                ("rollout/invalid_frac", "Invalid replies"), ("rollout/groups_kept", "Groups with signal")]
    return [
        wr.H1("Related runs: other models and fixed mazes (AWS Batch)"),
        wr.MarkdownBlock(open(a.related).read()),
        *([wr.H2("Larger models on the procedural setup (live)"),
           wr.PanelGrid(runsets=[rs(g) for g in procgen],
                        panels=[wr.LinePlot(title=t, x="Step", y=[k], max_runs_to_show=30) for k, t in set_keys]
                        + [wr.BarPlot(title="Latest eval success per set", metrics=[k for k, _ in set_keys[1:4]],
                                      orientation="h", max_runs_to_show=30)])] if procgen else []),
        wr.H2("Fixed mazes (live)"),
        wr.PanelGrid(runsets=[rs(g) for g in groups],
                     panels=[wr.LinePlot(title=t, x="Step", y=[k], max_runs_to_show=30) for k, t in keys]
                     + [wr.BarPlot(title="Latest eval success", metrics=["eval/success"], orientation="h",
                                   max_runs_to_show=30)]),
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
        wr.MarkdownBlock(f"_Text written {stamp}; the panels below are live and may have moved on since._\n\n"
                         + takeaways),
        *prompt_blocks(a, wr),
        *related_blocks(a, wr),
        wr.H1("Headline: greedy eval success"),
        bullets("One line per run; run names give prompt and lr (runs named only `lr=...` use `compact`).",
                "Eval = greedy, mean of 3 sets; rollout = T=1 training episodes."),
        grid(lines(["eval/success", "rollout/success"],
                   ["Eval success (greedy, mean of 3 sets)", "Training rollout success (T=1)"])),
        wr.H1("By prompt: absolute vs relative goal"),
        bullets("Runs grouped by `env_kwargs.prompt_style`: line = mean over runs, band = min/max.",
                "`compact` echoes the absolute goal: eval stays < 7%. `compact_rel`: the echo is the straight-line "
                "move, ~20-25% (the band's floor late on is the degenerated lr=3e-5 run)."),
        grid(lines(["eval/success", "rollout/success", "rollout/invalid_frac", "train/mean_logp"],
                   ["Eval success by prompt", "Rollout success by prompt", "Invalid replies by prompt",
                    "Mean log-prob of sampled tokens by prompt"],
                   groupby_aggfunc="mean", groupby_rangefunc="minmax"), by_prompt),
        wr.H1("Generalization: in vs out of distribution"),
        bullets("Memorizing the training layouts would show as `id_layouts` >> the held-out sets.",
                "Similar curves = the skill transfers to unseen (and larger) mazes."),
        grid(lines(set_keys, [f"{s}: {d}" for s, d in SETS.items()])
             + [wr.BarPlot(title="Latest eval success per set", metrics=set_keys, orientation="h",
                           max_runs_to_show=20)]),
        wr.H1("Learning dynamics"),
        bullets("`groups_kept`: groups (of 64) with mixed outcomes, the only ones with a GRPO signal.",
                "`mean_logp` falling steadily = sampling ever-less-likely tokens; it preceded the lr=3e-5 collapse "
                "(-2.3 -> -8.3) and is the early warning to watch.",
                "`grad_norm` spikes = unstable updates (gradients are clipped at 1.0)."),
        grid(lines(["rollout/groups_kept", "train/grad_norm", "train/mean_logp", "rollout/invalid_frac"])),
        wr.H1("Behaviour"),
        bullets("Turns per episode: lower = faster solves; 20 (30 on `ood_larger`) = timed out.",
                "A policy stuck against a wall repeats the same move until max_turns.",
                "Reply length jumps when replies degenerate into free text."),
        grid(lines(["eval/turns", "rollout/turns", "rollout/end_success", "rollout/end_max_turns",
                    "rollout/resp_chars"])),
        wr.H1("Episodes: videos, transcripts and sampled actions"),
        bullets("Per split: episode video (step slider; caption = success / turns / end reason) next to its "
                "transcript (exact prompts and replies).",
                "Below each: the samples table, one row per logged episode (4 per eval) across runs and steps; sort "
                "by success to compare wins and failures.",
                "Train = T=1 samples, eval = greedy. Look for: `compact` repeating the goal's absolute coordinates; "
                "`compact_rel` copying the offset; lr=3e-5 gibberish from step 17."),
        *[blk for name, title in [("train", "Training rollouts (T=1)")] + [(f"eval/{s}", f"Eval {s}: {d}")
                                                                      for s, d in SETS.items()]
          for blk in episode_blocks(name, title)],
        wr.H1("Cost"),
        bullets("Seconds per step for rollouts (incl. DAPO resampling) and the update; sequences trained and "
                "resampling rounds per step."),
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
    ap.add_argument("--takeaways", default=TAKEAWAYS, help="markdown file for the takeaways section")
    ap.add_argument("--url", default=None,
                    help="report to update in place (default: REPORT_URL, or a new report with --draft; '' = new)")
    ap.add_argument("--draft", action="store_true",
                    help="save as a new unpublished draft (never over the live report unless --url is given)")
    ap.add_argument("--related", default=RELATED, help="markdown for the related-runs section ('' = omit)")
    ap.add_argument("--no_prompts", dest="prompts", action="store_false", help="omit the prompts section")
    ap.add_argument("--related_groups", default=RELATED_GROUPS, help="comma-separated wandb groups for its panels")
    ap.add_argument("--related_procgen_groups", default=RELATED_PROCGEN_GROUPS,
                    help="related wandb groups on the procedural setup (per-eval-set panels)")
    a = ap.parse_args()
    if a.url is None:   # a draft is a separate report; only an explicit --url targets an existing one
        a.url = "" if a.draft else REPORT_URL
    print(build(a))


if __name__ == "__main__":
    main()
