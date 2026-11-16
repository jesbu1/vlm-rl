"""Synchronous GRPO-style trainer for multi-turn VLM agents (PEAR recipe, single process).

step:
  1. sample N start states, repeat each G times                     (groups share a start state)
  2. batched multi-turn rollout of all N*G episodes                 (rollout.rollout)
  3. reward_fn(groups) -> scores; DAPO filter; resample until N groups have signal
  4. advantages = group-centred scores (trajectory-level, broadcast to every response token)
  5. one optimizer update on the token-mean REINFORCE loss          (algo.policy_loss_sum)
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
import torch

from vlmrl import algo
from vlmrl.envs.base import EnvBundle
from vlmrl.policy import HFPolicy
from vlmrl.rewards import build_reward
from vlmrl.rollout import rollout, to_samples


@dataclass
class TrainConfig:
    # model
    model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor_kwargs: dict = field(default_factory=dict)  # e.g. {"max_pixels": 200704} / {"do_image_splitting": false}
    dtype: str = "auto"
    attn_implementation: str | None = None             # e.g. "flash_attention_2" / "sdpa"
    gradient_checkpointing: bool = True
    lora_rank: int = 0                                  # 0 = full fine-tuning
    lora_alpha: int = 32
    lora_targets: str = "all-linear"                    # PEFT target_modules; a regex string also works
    # rollout backend
    rollout_backend: str = "hf"                         # "hf" (in-process generate) | "vllm" (server, LoRA only)
    vllm_url: str = "http://localhost:8000"
    # task
    env: str = "pointmaze"                              # registry name or "module:factory"
    env_kwargs: dict = field(default_factory=dict)
    reward: str = "success"                             # see rewards.build_reward
    max_images: int | None = None                       # None = full multi-image history (PEAR); k = keep last k
    max_frames: int | None = None                       # k = keep only the last k rounds (frame + text), train & eval
    # RL
    steps: int = 200
    groups_per_step: int = 8                            # PEAR: 128
    group_size: int = 8                                 # PEAR: 8
    filter_groups: bool = True                          # DAPO group filter
    max_resample_rounds: int = 3
    adv_mode: str = "zscore"                            # zscore (GRPO, PEAR Eq. 1) | center | rloo
    policy_loss: str = "reinforce"                      # reinforce (PEAR "bypass", no clipping) | ppo_clip
    ppo_epochs: int = 1                                 # updates per batch; 1 = PEAR
    score_centering: bool = True                        # PEAR: correct rollout/learner mismatch (active when off-policy)
    score_centering_topk: int = 32                      # rollout-policy support used for the correction
    on_invalid: str = "terminate"                       # PEAR: an invalid action ends the episode | "noop"
    max_context_tokens: int | None = None               # PEAR: episode ends when the context budget is reached
    clip_low: float = 0.2
    clip_high: float = 0.28
    # optim
    lr: float = 1e-6                                    # PEAR (27B full FT); LoRA on small models wants ~1e-5..1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    micro_batch_size: int = 1                           # sequences per forward/backward
    # generation
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    gen_batch_size: int = 32
    # eval / io
    eval_every: int = 10
    eval_episodes: int = 32
    eval_temperature: float = 0.0
    save_every: int = 50
    resume: bool = True                                 # continue from out_dir/resume/state.pt if it exists
    resume_every: int = 1                               # steps between resume checkpoints (0 = off)
    log_samples_every: int = 5
    video_format: str = "mp4"                           # episode media: mp4 (H.264, wandb video player) | gif
    video_fps: float = 2.0                              # one frame per round
    out_dir: str = "runs/debug"
    seed: int = 0
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_group: str | None = None                      # e.g. the sweep / comparison name
    wandb_tags: str = ""                                # comma-separated


ENV_REGISTRY = {"pointmaze": "vlmrl.envs.pointmaze:make_pointmaze"}


def make_env_bundle(name: str, kwargs: dict) -> EnvBundle:
    path = ENV_REGISTRY.get(name, name)
    mod, fn = path.split(":")
    return getattr(importlib.import_module(mod), fn)(**kwargs)


def write_video(frames, path: str, fps: float = 2.0) -> None:
    """Encode HxWx3 uint8 frames. ``.mp4``: H.264 / yuv420p (plays in browsers and wandb), frames padded
    to even sizes; the last frame is held for one extra beat so the final state is visible. ``.gif``: looping."""
    import imageio.v2 as imageio
    import numpy as np
    frames = [np.asarray(f, dtype=np.uint8) for f in frames]
    if path.endswith(".gif"):
        imageio.mimsave(path, frames, duration=1.0 / fps, loop=0)
        return
    h, w = frames[0].shape[:2]
    ph, pw = h % 2, w % 2
    with imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                            ffmpeg_log_level="error") as wr:
        for f in frames + frames[-1:]:
            wr.append_data(np.pad(f, ((0, ph), (0, pw), (0, 0)), mode="edge") if (ph or pw) else f)


END_REASONS = ("success", "invalid", "max_turns", "context", "terminated", "truncated")


def end_reason_fracs(trajs, prefix):
    return {f"{prefix}/end_{r}": float(np.mean([t.end_reason == r for t in trajs])) for r in END_REASONS}


class Trainer:
    def __init__(self, cfg: TrainConfig, policy: HFPolicy | None = None, bundle: EnvBundle | None = None,
                 reward_fn=None):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        with open(f"{cfg.out_dir}/config.json", "w") as f:
            json.dump(dataclasses.asdict(cfg), f, indent=2)
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.bundle = bundle or make_env_bundle(cfg.env, cfg.env_kwargs)
        self.env = self.bundle.make_vec_env()
        self.reward_fn = reward_fn or build_reward(cfg.reward, max_turns=self.bundle.max_turns)
        self.policy = policy or HFPolicy(
            cfg.model, dtype=cfg.dtype, attn_implementation=cfg.attn_implementation,
            gradient_checkpointing=cfg.gradient_checkpointing, lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha, lora_targets=cfg.lora_targets, processor_kwargs=cfg.processor_kwargs)
        if cfg.rollout_backend not in ("hf", "vllm"):
            raise ValueError(f"unknown rollout_backend {cfg.rollout_backend!r}")
        params = [p for p in self.policy.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
        self.params = params
        self.eval_starts = self.bundle.eval_sampler.sample(cfg.eval_episodes, np.random.default_rng(10_000 + cfg.seed))
        self.start_step, pending = self.load_resume() if cfg.resume else (0, None)
        if cfg.rollout_backend == "vllm":
            from vlmrl.vllm_client import VLLMServerGenerator
            self.generator = VLLMServerGenerator(cfg.vllm_url, cfg.model, f"{cfg.out_dir}/adapters")
            self.generator.sync(self.policy, self.start_step)  # serve the initial / resumed LoRA policy
        else:
            self.generator = self.policy
        self.wandb = None
        if cfg.wandb_project:
            import wandb
            id_file = f"{cfg.out_dir}/wandb_run_id"
            run_id = open(id_file).read().strip() if cfg.resume and os.path.exists(id_file) else uuid.uuid4().hex[:8]
            with open(id_file, "w") as f:
                f.write(run_id)
            self.wandb = wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, name=cfg.wandb_run_name,
                                    group=cfg.wandb_group, tags=[t for t in cfg.wandb_tags.split(",") if t],
                                    config=dataclasses.asdict(cfg), dir=cfg.out_dir, id=run_id, resume="allow")
            self.backfill_wandb()
        if pending is not None:  # crashed after checkpointing this step but before logging it
            self.log(self.start_step, pending)

    # ------------------------------------------------------------------ rollouts
    def _gen(self, temperature):
        c = self.cfg
        return lambda convs: self.generator.generate(convs, c.max_new_tokens, temperature, c.top_p, c.gen_batch_size)

    def collect(self):
        c, kept, all_trajs, rounds = self.cfg, [], [], 0
        gid = 0
        while len(kept) < c.groups_per_step and rounds < (c.max_resample_rounds if c.filter_groups else 1):
            rounds += 1
            n = c.groups_per_step - len(kept) if rounds > 1 else c.groups_per_step
            starts = self.bundle.train_sampler.sample(n, self.rng)
            rep = [s for s in starts for _ in range(c.group_size)]
            gids = [gid + i for i in range(n) for _ in range(c.group_size)]
            gid += n
            trajs = self._rollout(c.temperature, rep, gids)
            groups = [trajs[i * c.group_size:(i + 1) * c.group_size] for i in range(n)]
            scores = self.reward_fn(groups)
            advs = algo.group_advantages(scores, c.adv_mode)
            for g, s, a in zip(groups, scores, advs):
                for tr, si, ai in zip(g, s, a):
                    tr.score, tr.advantage = float(si), float(ai)
                if not c.filter_groups or algo.keep_group(s):
                    kept.append(g)
            all_trajs += trajs
        return kept[:c.groups_per_step], all_trajs, rounds

    # ------------------------------------------------------------------ update
    def update(self, groups):
        c, pol = self.cfg, self.policy
        trajs = [t for g in groups for t in g if abs(t.advantage) > 1e-8]  # zero-adv => zero gradient
        if not trajs:
            return {"train/skipped": 1.0}
        samples = to_samples(trajs, c.max_images, c.max_frames)
        samples.sort(key=lambda s: len(s.messages))  # length-bucketing -> less padding per micro-batch
        batches = [pol.build_batch(samples[i:i + c.micro_batch_size])
                   for i in range(0, len(samples), c.micro_batch_size)]
        n_tok = sum(float(b["loss_mask"].sum()) for b in batches)  # global token-mean denominator
        # snapshot of the rollout policy mu (needed only when updating more than once per batch)
        sc = c.score_centering and c.policy_loss == "reinforce" and c.ppo_epochs > 1
        old = None
        if c.ppo_epochs > 1:
            with torch.no_grad():
                old = []
                for b in batches:
                    r = pol.token_logprobs(b, topk=c.score_centering_topk) if sc else pol.token_logprobs(b)
                    old.append({"logp": r[0].detach(), **({k: v.detach() for k, v in r[2].items()} if sc else {})})
        stats = {"loss": 0.0, "clipfrac_sum": 0.0, "logp_sum": 0.0}
        for _ in range(c.ppo_epochs):
            self.opt.zero_grad(set_to_none=True)
            for bi, b in enumerate(batches):
                centering, old_logp = None, None
                if sc:
                    logp, mask, st = pol.token_logprobs(b, gather_ids=old[bi]["topk_ids"])
                    centering = algo.score_centering_term(st["logp_at_ids"], old[bi]["topk_probs"])
                else:
                    logp, mask = pol.token_logprobs(b)
                if c.policy_loss == "ppo_clip":
                    old_logp = old[bi]["logp"] if old is not None else logp.detach()
                adv = b["advantages"].to(logp.device)[:, None].expand_as(logp)
                loss_sum, st = algo.policy_loss_sum(logp, adv, mask, old_logp, (c.clip_low, c.clip_high),
                                                    centering=centering)
                loss = loss_sum / n_tok
                if not torch.isfinite(loss):
                    print("[warn] non-finite loss, skipping micro-batch")
                    continue
                loss.backward()
                stats["loss"] += loss.item()
                stats["clipfrac_sum"] += st.get("clipfrac_sum", 0.0)
                stats["logp_sum"] += float((logp.detach() * mask).sum())
            gn = torch.nn.utils.clip_grad_norm_(self.params, c.grad_clip)
            if torch.isfinite(gn):
                self.opt.step()
            else:
                print("[warn] non-finite grad norm, skipping step")
        return {"train/loss": stats["loss"] / c.ppo_epochs, "train/grad_norm": float(gn),
                "train/tokens": n_tok, "train/sequences": len(samples),
                "train/mean_logp": stats["logp_sum"] / max(n_tok, 1) / c.ppo_epochs,
                **({"train/clipfrac": stats["clipfrac_sum"] / n_tok / c.ppo_epochs}
                   if c.policy_loss == "ppo_clip" else {})}

    # ------------------------------------------------------------------ eval / logging
    @torch.no_grad()
    def evaluate(self):
        trajs = self._rollout(self.cfg.eval_temperature, self.eval_starts, list(range(len(self.eval_starts))))
        return {"eval/success": float(np.mean([t.success for t in trajs])),
                "eval/turns": float(np.mean([len(t.turns) for t in trajs])),
                "eval/invalid_frac": float(np.mean([t.num_invalid / max(1, len(t.turns)) for t in trajs])),
                **end_reason_fracs(trajs, "eval")}, trajs

    def _rollout(self, temperature, starts, gids):
        c = self.cfg
        return rollout(self._gen(temperature), self.env, self.bundle.interface, starts, gids, self.bundle.max_turns,
                       c.max_images, on_invalid=c.on_invalid, max_context_tokens=c.max_context_tokens,
                       last_tokens=lambda: getattr(self.generator, "last_num_tokens", []), max_frames=c.max_frames)

    def dump_episode(self, tr, tag, step=None):
        """Write a video (mp4 or gif) + transcript to out_dir/episodes, and to wandb as media if enabled."""
        d = f"{self.cfg.out_dir}/episodes"
        os.makedirs(d, exist_ok=True)
        video = f"{d}/{tag}.{self.cfg.video_format}"
        if tr.frames:
            write_video(tr.frames, video, self.cfg.video_fps)
        lines = [f"success={tr.success} score={tr.score:.3f} adv={tr.advantage:.3f} end={tr.end_reason}", ""]
        for m in tr.messages:
            lines += [f"--- {m['role']}", " ".join(c.get("text", "<image>") for c in m["content"])]
        text = "\n".join(lines)
        with open(f"{d}/{tag}.txt", "w") as f:
            f.write(text)
        if self.wandb:
            import html
            import wandb
            kind = tag.split("_")[0]
            media = {f"{kind}/transcript": wandb.Html(f"<pre>{html.escape(text)}</pre>")}
            if tr.frames:
                caption = f"success={tr.success} score={tr.score:.3f} rounds={len(tr.turns)} end={tr.end_reason}"
                media[f"{kind}/episode"] = wandb.Video(video, caption=caption, format=self.cfg.video_format)
            self.wandb.log(media, step=step, commit=False)

    # ------------------------------------------------------------------ resume
    def save_resume(self, step, metrics):
        """Everything needed to continue after `step`: trainable weights, optimizer, sampler RNG, and the
        step's metrics (so a crash between checkpointing and logging loses nothing). Written atomically."""
        d = f"{self.cfg.out_dir}/resume"
        os.makedirs(d, exist_ok=True)
        state = {"step": step, "metrics": metrics, "opt": self.opt.state_dict(),
                 "trainable": {n: p.detach().cpu() for n, p in self.policy.model.named_parameters() if p.requires_grad},
                 "rng": self.rng.bit_generator.state, "torch_rng": torch.get_rng_state()}
        torch.save(state, f"{d}/state.pt.tmp")
        os.replace(f"{d}/state.pt.tmp", f"{d}/state.pt")

    def load_resume(self):
        """Restore from out_dir/resume/state.pt; returns (step, metrics not yet in metrics.jsonl or None).
        metrics.jsonl is truncated to steps <= the checkpoint, so re-run steps aren't duplicated."""
        path, mpath = f"{self.cfg.out_dir}/resume/state.pt", f"{self.cfg.out_dir}/metrics.jsonl"
        if not os.path.exists(path):
            if os.path.exists(mpath):  # crashed before the first checkpoint: start over cleanly
                os.replace(mpath, f"{mpath}.prev")
            return 0, None
        state = torch.load(path, weights_only=False)
        named = dict(self.policy.model.named_parameters())
        with torch.no_grad():
            for n, v in state["trainable"].items():
                named[n].copy_(v.to(named[n].device, named[n].dtype))
        self.opt.load_state_dict(state["opt"])
        self.rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"])
        step, lines = state["step"], []
        if os.path.exists(mpath):
            with open(mpath) as f:
                lines = [l for l in f if l.strip() and json.loads(l)["step"] <= step]
            with open(mpath, "w") as f:
                f.writelines(lines)
        logged = any(json.loads(l)["step"] == step for l in lines)
        print(f"[resume] continuing after step {step} from {path}", flush=True)
        return step, None if logged else state["metrics"]

    def backfill_wandb(self):
        """A hard kill can lose the last rows before wandb uploads them, so on resume re-send the
        metrics.jsonl rows from the resumed run's step on (scalars only). Some servers report step 0
        for resumed runs (seen on forge.coreweave.com), so this may re-send every row; the server
        keeps one history row per step, so that is harmless (verified: 4 rows for steps 0-3)."""
        path = f"{self.cfg.out_dir}/metrics.jsonl"
        if not os.path.exists(path):
            return
        with open(path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        missing = [r for r in rows if r["step"] >= self.wandb.step]
        for r in missing:
            self.wandb.log({k: v for k, v in r.items() if k != "step"}, step=r["step"], commit=True)
        if missing:
            print(f"[resume] re-sent {len(missing)} metrics rows to wandb (steps {missing[0]['step']}..)", flush=True)

    def log(self, step, metrics):
        line = {"step": step, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}}
        print(json.dumps(line), flush=True)
        with open(f"{self.cfg.out_dir}/metrics.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        if self.wandb:
            self.wandb.log(metrics, step=step, commit=True)  # don't hold the row open until the next step

    # ------------------------------------------------------------------ main loop
    def train(self):
        c = self.cfg
        if c.eval_every and self.start_step == 0:
            m, trajs = self.evaluate()
            self.log(0, m)
            self.dump_episode(trajs[0], "eval_step0", 0)
        for step in range(self.start_step + 1, c.steps + 1):
            t0 = time.time()
            groups, all_trajs, rounds = self.collect()
            t1 = time.time()
            if self.policy.device == "cuda":
                torch.cuda.empty_cache()
            m = self.update(groups) if groups else {"train/skipped": 1.0}
            if self.generator is not self.policy and "train/loss" in m:
                self.generator.sync(self.policy, step)  # push new weights to the rollout engine
            t2 = time.time()
            m.update({
                "rollout/success": float(np.mean([t.success for t in all_trajs])),
                "rollout/score": float(np.mean([t.score for t in all_trajs])),
                "rollout/turns": float(np.mean([len(t.turns) for t in all_trajs])),
                "rollout/invalid_frac": float(np.mean([t.num_invalid / max(1, len(t.turns)) for t in all_trajs])),
                "rollout/resp_chars": float(np.mean([len(x.response) for t in all_trajs for x in t.turns])),
                "rollout/groups_kept": len(groups), "rollout/rounds": rounds,
                **end_reason_fracs(all_trajs, "rollout"),
                "time/rollout": t1 - t0, "time/train": t2 - t1,
            })
            if c.log_samples_every and step % c.log_samples_every == 0:
                self.dump_episode(all_trajs[0], f"train_step{step}", step)
            if c.eval_every and step % c.eval_every == 0:
                em, trajs = self.evaluate()
                m.update(em)
                self.dump_episode(trajs[0], f"eval_step{step}", step)
            if c.resume_every and (step % c.resume_every == 0 or step == c.steps):
                self.save_resume(step, m)
            self.log(step, m)
            if c.save_every and step % c.save_every == 0:
                self.policy.save(f"{c.out_dir}/ckpt_step{step}")
        self.env.close()
        if self.wandb:
            self.wandb.finish()
