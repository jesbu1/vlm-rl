"""Build a wandb Report comparing the runs of one sweep group (e.g. reward fns x models).

    uv pip install -p .venv wandb-workspaces
    python scripts/make_wandb_report.py --group pointmaze-small-v5 --takeaways takeaways.md [--dry_run]

Sections: setup, a final-eval summary table (computed here from run histories, so it also lists
crashed / partial runs), success curves grouped by `config.reward` and by `config.model`, episode
dynamics (turns, invalid actions, DAPO groups kept), wall-clock, and free-form takeaways (markdown file).
`--dry_run` prints the summary table without creating a report.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import wandb

SUCCESS = ["eval/success", "rollout/success"]
DYNAMICS = ["rollout/turns", "eval/turns", "rollout/invalid_frac", "rollout/groups_kept"]
TIMING = ["time/rollout", "time/train"]


def short(model: str) -> str:
    return model.split("/")[-1]


def run_rows(api, entity, project, group):
    """One dict per run: config, state, eval/success at step 0 / last / best, rollout tail, timing."""
    rows = []
    for r in api.runs(f"{entity}/{project}", filters={"group": group}):
        h = list(r.scan_history(keys=["_step", "eval/success"]))
        ev = sorted((x["_step"], x["eval/success"]) for x in h if x.get("eval/success") is not None)
        tr = sorted(r.scan_history(keys=["_step", "rollout/success", "time/rollout", "time/train"]),
                    key=lambda x: x["_step"])
        tail = [x["rollout/success"] for x in tr[-5:]]
        rows.append({
            "name": r.name, "url": r.url, "state": r.state, "model": short(r.config.get("model", "?")),
            "reward": r.config.get("reward", "?"), "steps": max([x["_step"] for x in tr], default=0),
            "planned": r.config.get("steps"),
            "eval0": ev[0][1] if ev and ev[0][0] == 0 else None,
            "eval_last": ev[-1][1] if ev else None, "eval_last_step": ev[-1][0] if ev else None,
            "eval_best": max(v for _, v in ev) if ev else None,
            "rollout_tail": float(np.mean(tail)) if tail else None,
            "min_per_step": float(np.mean([x["time/rollout"] + x["time/train"] for x in tr])) / 60 if tr else None,
        })
    rows.sort(key=lambda x: (x["model"], x["reward"]))
    return rows


def fmt(v, p=3):
    return "–" if v is None else f"{v:.{p}f}"


def summary_markdown(rows) -> str:
    lines = ["| model | reward | state | steps | eval success @0 | final eval success (step) | best eval | "
             "rollout success (last 5 steps) | min/step |", "|" + "---|" * 9]
    for x in rows:
        delta = "" if x["eval0"] is None or x["eval_last"] is None else f" ({x['eval_last'] - x['eval0']:+.3f})"
        lines.append(f"| {x['model']} | `{x['reward']}` | [{x['state']}]({x['url']}) | {x['steps']}/{x['planned']} | "
                     f"{fmt(x['eval0'])} | {fmt(x['eval_last'])}{delta} @{x['eval_last_step']} | "
                     f"{fmt(x['eval_best'])} | {fmt(x['rollout_tail'])} | {fmt(x['min_per_step'], 1)} |")
    return "\n".join(lines)


def build_report(a, rows, cfg):
    import wandb_workspaces.reports.v2 as wr

    flt = f"Group == '{a.group}'"

    def runset(name, groupby=()):
        return wr.Runset(entity=a.entity, project=a.project, name=name, filters=flt, groupby=list(groupby))

    def plots(keys, **kw):
        return [wr.LinePlot(title=k, x="Step", y=[k], **kw) for k in keys]

    env = cfg.get("env_kwargs", {})
    setup = (f"PointMaze **{env.get('maze')}** maze, ≤{env.get('max_turns')} turns, {env.get('image_size')}px top-down "
             f"frames, include_state={env.get('include_state')}. LoRA r={cfg.get('lora_rank')} on the language model "
             f"only, lr {cfg.get('lr')}, {cfg.get('groups_per_step')} groups × {cfg.get('group_size')} rollouts/step, "
             f"adv_mode={cfg.get('adv_mode')}, DAPO group filter, vLLM rollouts (T={cfg.get('temperature')}), "
             f"{cfg.get('steps')} steps. Eval = greedy on a fixed set of {cfg.get('eval_episodes')} start states "
             f"every {cfg.get('eval_every')} steps. One seed per config, so differences of a few points are "
             f"within noise (a 32-episode eval has SE ≈ 0.08 at p=0.5).")
    takeaways = open(a.takeaways).read() if a.takeaways else "_(to be written)_"
    report = wr.Report(
        entity=a.entity, project=a.project, title=a.title or f"PointMaze VLM-RL: rewards × models ({a.group})",
        description=f"wandb group `{a.group}`: grid over config.model × config.reward.",
        blocks=[
            wr.H1("Takeaways"),
            wr.MarkdownBlock(takeaways),
            wr.H1("Setup"),
            wr.MarkdownBlock(setup),
            wr.H1("Final eval success, all runs"),
            wr.MarkdownBlock(summary_markdown(rows)),
            wr.PanelGrid(runsets=[runset("all runs")], panels=plots(["eval/success"], max_runs_to_show=20)),
            wr.H1("Success grouped by reward"),
            wr.P("Mean over models (shaded: min/max) for each reward spec."),
            wr.PanelGrid(runsets=[runset("by reward", ["config.reward"])],
                         panels=plots(SUCCESS, groupby_aggfunc="mean", groupby_rangefunc="minmax")),
            wr.H1("Success grouped by model"),
            wr.P("Mean over reward specs (shaded: min/max) for each model."),
            wr.PanelGrid(runsets=[runset("by model", ["config.model"])],
                         panels=plots(SUCCESS, groupby_aggfunc="mean", groupby_rangefunc="minmax")),
            wr.H1("Per-run success curves"),
            wr.PanelGrid(runsets=[runset("all runs")], panels=plots(SUCCESS, max_runs_to_show=20)),
            wr.H1("Episode dynamics"),
            wr.P("Mean turns per episode (lower = faster solves), fraction of unparseable actions, and how many "
                 "of the groups survived the DAPO zero-variance filter (fewer = saturating or hopeless)."),
            wr.PanelGrid(runsets=[runset("all runs")], panels=plots(DYNAMICS, max_runs_to_show=20)),
            wr.H1("Wall-clock"),
            wr.PanelGrid(runsets=[runset("all runs")], panels=plots(TIMING, max_runs_to_show=20)),
        ])
    report.save(draft=a.draft)
    # Report.url drops the app URL's path prefix (e.g. https://forge.coreweave.com/wandb/), so rebuild it
    from urllib.parse import urlparse
    from wandb_workspaces._graphql import get_app_url
    return get_app_url(wandb.Api()).rstrip("/") + urlparse(report.url).path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="pointmaze-small-v5")
    ap.add_argument("--entity", default="tri")
    ap.add_argument("--project", default="vlm-rl-pointmaze")
    ap.add_argument("--title", default=None)
    ap.add_argument("--takeaways", default=None, help="markdown file for the takeaways section")
    ap.add_argument("--draft", action="store_true", help="save as an unpublished draft")
    ap.add_argument("--dry_run", action="store_true", help="print the summary table only")
    a = ap.parse_args()
    api = wandb.Api()
    rows = run_rows(api, a.entity, a.project, a.group)
    if not rows:
        raise SystemExit(f"no runs in group {a.group}")
    print(summary_markdown(rows))
    if a.dry_run:
        print(json.dumps(rows, indent=1))
        return
    cfg = api.runs(f"{a.entity}/{a.project}", filters={"group": a.group})[0].config
    print(build_report(a, rows, cfg))


if __name__ == "__main__":
    main()
