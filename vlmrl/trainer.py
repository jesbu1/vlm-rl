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
from vlmrl.rollout import RolloutTask, rollout, rollout_pooled, to_samples


@dataclass
class TrainConfig:
    # model
    model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor_kwargs: dict = field(default_factory=dict)  # e.g. {"max_pixels": 200704} / {"do_image_splitting": false}
    chat_template_kwargs: dict = field(default_factory=dict)  # e.g. {"enable_thinking": true} (HF and vLLM)
    dtype: str = "auto"
    attn_implementation: str | None = None             # e.g. "flash_attention_2" / "sdpa"
    gradient_checkpointing: bool = True
    lora_rank: int = 0                                  # 0 = full fine-tuning
    lora_alpha: int = 32
    lora_targets: str = "all-linear"                    # PEFT target_modules; a regex string also works
    freeze: str = ""                                    # full FT: regex (fullmatch) of parameter names to keep frozen
    fp32_master: bool = True                            # full FT: keep trainable weights in fp32, bf16 autocast forward
    fp32_frozen: bool = False                           # with freeze + fp32_master: keep frozen params in fp32 too
    # rollout backend
    rollout_backend: str = "hf"                         # "hf" (in-process generate) | "vllm" (server)
    vllm_url: str = "http://localhost:8000"
    vllm_sync_dir: str = ""                             # where weights for the server are written ("" = out_dir/adapters)
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
    eval_on_invalid: str | None = None                  # on_invalid for eval episodes; None = same as on_invalid
    max_context_tokens: int | None = None               # PEAR: episode ends when the context budget is reached
    clip_low: float = 0.2
    clip_high: float = 0.28
    kl_coef: float = 0.0                                # per-token k3 KL to the pretrained model in the loss; 0 = off (PEAR)
    # optim
    lr: float = 1e-6                                    # PEAR (27B full FT); LoRA on small models wants ~1e-5..1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    lr_warmup_steps: int = 0                            # linear lr warmup over the first N optimizer steps (0 = off)
    loss_agg: str = "token_mean"                        # token_mean (PEAR: one mean over all tokens) | traj_mean
    init_adapter: str = ""                              # start from a saved LoRA adapter (dir or s3:// URI); a resume wins
    bc_steps: int = 0                                   # behaviour cloning on the env's scripted expert before RL (fresh runs)
    bc_episodes: int = 64                               # expert episodes per BC step
    bc_success_only: bool = True                        # clone only the expert episodes that succeed
    bc_lr: float = 0.0                                  # lr during behaviour cloning (0 = lr)
    bc_dagger_steps: int = 0                            # the last N of bc_steps run the current clone, labelled by the expert (DAgger)
    bc_noise: float = 0.0                               # DART: noise (std, action units: cm on Metaworld) on the expert's executed moves
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
    # named eval environments: {name: {env_kwargs overrides..., "episodes": n}}, e.g.
    # {"id": {"eval_split": "train"}, "ood_goals": {"eval_split": "heldout"}, "ood_medium": {"maze": "medium"}}.
    # Empty = one eval set from env_kwargs (logged as eval/*). All sets run pooled in one rollout.
    eval_sets: dict = field(default_factory=dict)
    save_every: int = 50
    resume: bool = True                                 # continue from out_dir/resume/state.pt if it exists
    resume_every: int = 1                               # steps between resume checkpoints (0 = off)
    log_samples_every: int = 5
    samples_per_log: int = 4                            # episodes (transcript + video) per train/eval sample dump
    video_format: str = "mp4"                           # episode media: mp4 (H.264, wandb video player) | gif
    video_fps: float = 2.0                              # one frame per round
    video_min_size: int = 672                           # mp4: nearest-neighbour upscale until long side >= this (0 = native)
    video_crf: int = 18                                 # mp4: H.264 quality (lower = better, 18 ~ visually lossless)
    out_dir: str = "runs/debug"
    seed: int = 0
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_group: str | None = None                      # e.g. the sweep / comparison name
    wandb_tags: str = ""                                # comma-separated


ENV_REGISTRY = {"pointmaze": "vlmrl.envs.pointmaze:make_pointmaze", "metaworld": "vlmrl.envs.metaworld:make_metaworld"}


def make_env_bundle(name: str, kwargs: dict) -> EnvBundle:
    path = ENV_REGISTRY.get(name, name)
    mod, fn = path.split(":")
    return getattr(importlib.import_module(mod), fn)(**kwargs)


def write_video(frames, path: str, fps: float = 2.0, min_size: int = 672, crf: int = 18) -> None:
    """Encode HxWx3 uint8 frames.

    ``.mp4``: H.264 / yuv420p (plays in browsers and wandb). yuv420p stores colour at half resolution,
    which smears thin lines and small coloured dots in flat renders, so frames are first upscaled with
    nearest-neighbour by an integer factor until the long side is >= ``min_size`` (0 = native size), and
    encoded at quality ``crf`` (lower = better; 18 is visually lossless) with ``-tune animation``.
    Sizes are padded to even; the last frame is held for one extra beat so the final state is visible.
    ``.gif``: looping, native size (palette GIFs are already lossless for flat renders)."""
    import imageio.v2 as imageio
    import numpy as np
    frames = [np.asarray(f, dtype=np.uint8) for f in frames]
    if path.endswith(".gif"):
        imageio.mimsave(path, frames, duration=1.0 / fps, loop=0)
        return
    h, w = frames[0].shape[:2]
    k = max(1, -(-min_size // max(h, w))) if min_size else 1     # ceil(min_size / long side)
    if k > 1:
        frames = [f.repeat(k, 0).repeat(k, 1) for f in frames]
        h, w = h * k, w * k
    ph, pw = h % 2, w % 2
    with imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                            quality=None, ffmpeg_log_level="error",
                            output_params=["-crf", str(crf), "-preset", "slow", "-tune", "animation"]) as wr:
        for f in frames + frames[-1:]:
            wr.append_data(np.pad(f, ((0, ph), (0, pw), (0, 0)), mode="edge") if (ph or pw) else f)


def strip_images(tr) -> None:
    """Free a trajectory's frames and inline images (filtered-out groups are only used for metrics)."""
    for t in tr.turns:
        t.obs.image = None
        t.result.obs.image = None
    for m in tr.messages:
        m["content"] = [{"type": "text", "text": "<image>"} if c.get("type") == "image" else c for c in m["content"]]


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
            lora_alpha=cfg.lora_alpha, lora_targets=cfg.lora_targets, processor_kwargs=cfg.processor_kwargs,
            chat_template_kwargs=cfg.chat_template_kwargs, freeze=cfg.freeze, fp32_master=cfg.fp32_master,
            fp32_frozen=cfg.fp32_frozen)
        if cfg.rollout_backend not in ("hf", "vllm"):
            raise ValueError(f"unknown rollout_backend {cfg.rollout_backend!r}")
        params = [p for p in self.policy.model.parameters() if p.requires_grad]
        # fused AdamW on CUDA: one kernel per tensor, no foreach temporaries. The default foreach step needs extra
        # parameter-sized buffers on top of the two moments, which ran a 48 GB GPU out of memory for Gemma 4 E2B
        fused = all(p.is_cuda for p in params) and len(params) > 0
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999),
                                     **({"fused": True} if fused else {}))
        self.params = params
        self.eval_starts = self.bundle.eval_sampler.sample(cfg.eval_episodes, np.random.default_rng(10_000 + cfg.seed))
        self.eval_sets = self.build_eval_sets()
        if cfg.init_adapter:
            self.policy.load_adapter_weights(self._fetch(cfg.init_adapter))
        self.start_step, pending = self.load_resume() if cfg.resume else (0, None)
        if cfg.rollout_backend == "vllm":
            from vlmrl.vllm_client import VLLMServerGenerator
            self.generator = VLLMServerGenerator(cfg.vllm_url, cfg.model, cfg.vllm_sync_dir or f"{cfg.out_dir}/adapters",
                                                 chat_template_kwargs=cfg.chat_template_kwargs)
            self.generator.sync(self.policy, self.start_step)  # serve the initial / resumed policy
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

    def build_eval_sets(self):
        """-> [{"name": log prefix, "bundle", "env", "starts"}]. Without ``eval_sets`` the single set reuses
        the training bundle (prefix "eval"); named sets get their own bundle / env (prefix "eval/<name>")
        and a fixed set of start states drawn once from their eval sampler."""
        c = self.cfg
        if not c.eval_sets:
            return [{"name": "eval", "bundle": self.bundle, "env": self.env, "starts": self.eval_starts}]
        sets = []
        for name, spec in c.eval_sets.items():
            spec = dict(spec or {})
            n = int(spec.pop("episodes", c.eval_episodes))
            b = make_env_bundle(c.env, {**c.env_kwargs, **spec})
            sets.append({"name": f"eval/{name}", "bundle": b, "env": b.make_vec_env(),
                         "starts": b.eval_sampler.sample(n, np.random.default_rng(10_000 + c.seed))})
        return sets

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
            dropped = 0
            for g, s, a in zip(groups, scores, advs):
                for tr, si, ai in zip(g, s, a):
                    tr.score, tr.advantage = float(si), float(ai)
                if not c.filter_groups or algo.keep_group(s):
                    kept.append(g)
                elif (dropped := dropped + 1) > 1 or rounds > 1:
                    for tr in g:  # filtered out: only metrics need it (one group keeps its media for logging)
                        strip_images(tr)
            all_trajs += trajs
        return kept[:c.groups_per_step], all_trajs, rounds

    # ------------------------------------------------------------------ update
    def _batches(self, samples):
        """Micro-batches built lazily (one ahead, on a worker thread), so a large step never holds every
        sample's pixel values in memory at once and CPU preprocessing overlaps the GPU forward/backward."""
        from concurrent.futures import ThreadPoolExecutor
        mb = self.cfg.micro_batch_size
        chunks = [samples[i:i + mb] for i in range(0, len(samples), mb)]
        with ThreadPoolExecutor(1) as ex:
            nxt = ex.submit(self.policy.build_batch, chunks[0]) if chunks else None
            for i in range(len(chunks)):
                b = nxt.result()
                nxt = ex.submit(self.policy.build_batch, chunks[i + 1]) if i + 1 < len(chunks) else None
                yield b

    def _weight_by_trajectory(self, samples):
        """loss_agg=traj_mean: every trajectory carries the same total weight, its tokens averaged within it.
        With the token mean, a trajectory's weight grows with its token count; failures run all their turns while
        successes stop early, so z-scored advantages that balance per trajectory are net-negative per token. That
        pushes down whatever was sampled and moves mass to what was not (SmolVLM2-2.2B: replies shrank to minimal
        JSON, then to bare words). Implemented by rescaling advantages: the global token mean below then gives
        sum_tokens A * logp / (n_traj * T_traj)."""
        if self.cfg.loss_agg not in ("token_mean", "traj_mean"):
            raise ValueError(f"unknown loss_agg {self.cfg.loss_agg!r}")
        tok = getattr(self.policy, "tokenizer", None)

        def n_tokens(s):
            if s.gen_ids:
                return len(s.gen_ids)
            text = "".join(c.get("text", "") for k in s.train_msg_idx for c in s.messages[k]["content"]
                           if isinstance(c, dict))
            return max(1, len(tok(text, add_special_tokens=False)["input_ids"]) + len(s.train_msg_idx)
                       if tok is not None else len(text.split()) + 1)
        nt = [n_tokens(s) for s in samples]
        per_traj: dict = {}
        for s, n in zip(samples, nt):
            per_traj[s.meta.get("traj")] = per_traj.get(s.meta.get("traj"), 0) + n
        total, n_traj = sum(nt), len(per_traj)
        for s in samples:
            s.advantage *= total / (n_traj * per_traj[s.meta.get("traj")])

    def _fetch(self, uri: str) -> str:
        """A local dir for ``uri``: s3:// prefixes are copied to out_dir/init_adapter with the AWS CLI."""
        if not uri.startswith("s3://"):
            return uri
        import subprocess
        dst = os.path.join(self.cfg.out_dir, "init_adapter")
        subprocess.run(["aws", "s3", "cp", "--recursive", "--only-show-errors", uri.rstrip("/") + "/", dst], check=True)
        return dst

    def _opt_steps(self) -> int:
        """Optimizer steps taken so far, from AdamW's own state (saved with resume checkpoints)."""
        for st in self.opt.state.values():
            if "step" in st:
                return int(st["step"])
        return 0

    def _set_lr(self):
        """Linear warmup: the n-th optimizer step uses lr * min(1, n / lr_warmup_steps). Adam's first steps move every
        parameter by about lr whatever the gradient, which flipped SmolVLM2-2.2B from JSON moves to bare direction
        words in one update; warmup shrinks those steps. Stateless, so resumes need nothing extra."""
        w = self.cfg.lr_warmup_steps
        f = min(1.0, (self._opt_steps() + 1) / w) if w > 0 else 1.0
        base = getattr(self, "_lr_base", None) or self.cfg.lr      # bc_lr during behaviour cloning
        for g in self.opt.param_groups:
            g["lr"] = base * f

    def update(self, groups):
        c, pol = self.cfg, self.policy
        trajs = [t for g in groups for t in g if abs(t.advantage) > 1e-8]  # zero-adv => zero gradient
        if not trajs:
            return {"train/skipped": 1.0}
        samples = to_samples(trajs, c.max_images, c.max_frames)
        samples.sort(key=lambda s: len(s.messages))  # length-bucketing -> less padding per micro-batch
        if c.loss_agg not in ("token_mean", "traj_mean"):
            raise ValueError(f"unknown loss_agg {c.loss_agg!r}")
        if c.loss_agg == "traj_mean":
            self._weight_by_trajectory(samples)
        # snapshot of the rollout policy mu (needed only when updating more than once per batch)
        sc = c.score_centering and c.policy_loss == "reinforce" and c.ppo_epochs > 1
        old = None
        if c.ppo_epochs > 1:
            with torch.no_grad():
                old = []
                for b in self._batches(samples):
                    r = pol.token_logprobs(b, topk=c.score_centering_topk) if sc else pol.token_logprobs(b)
                    old.append({"logp": r[0].detach(), **({k: v.detach() for k, v in r[2].items()} if sc else {})})
        # Loss = sum over response tokens / global token count. Batches are built on the fly, so each one is
        # back-propagated as loss_sum / len(samples) and the accumulated gradient is rescaled by
        # len(samples) / n_tok once n_tok is known: exactly the gradient of the global token mean.
        stats = {"loss": 0.0, "clipfrac_sum": 0.0, "logp_sum": 0.0, "kl_sum": 0.0}
        scale = float(len(samples))
        for _ in range(c.ppo_epochs):
            self.opt.zero_grad(set_to_none=True)
            n_tok, loss_sum_total = 0.0, 0.0
            for bi, b in enumerate(self._batches(samples)):
                n_tok += float(b["loss_mask"].sum())
                centering, old_logp = None, None
                ref_logp = pol.ref_token_logprobs(b) if c.kl_coef > 0 else None
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
                if ref_logp is not None:
                    kl_sum = algo.kl_k3_sum(logp, ref_logp.to(logp.device), mask)
                    loss_sum = loss_sum + c.kl_coef * kl_sum
                    stats["kl_sum"] += float(kl_sum.detach())
                if not torch.isfinite(loss_sum):
                    print("[warn] non-finite loss, skipping micro-batch")
                    continue
                (loss_sum / scale).backward()
                loss_sum_total += loss_sum.item()
                stats["clipfrac_sum"] += st.get("clipfrac_sum", 0.0)
                stats["logp_sum"] += float((logp.detach() * mask).sum())
            for p in self.params:
                if p.grad is not None:
                    p.grad.mul_(scale / max(n_tok, 1.0))
            stats["loss"] += loss_sum_total / max(n_tok, 1.0)
            gn = torch.nn.utils.clip_grad_norm_(self.params, c.grad_clip)
            if torch.isfinite(gn):
                self._set_lr()
                self.opt.step()
            else:
                print("[warn] non-finite grad norm, skipping step")
        # rollout engine vs trainer on the same trained replies. On-policy (the forward pass above runs before the
        # optimizer step) the two mean log-probs should match: a growing gap means the engine is not sampling from
        # these weights (sync) or the re-tokenized replies are not the sampled tokens (tok_ratio far from 1)
        gen = [(tu.gen_logp, tu.gen_ntok) for t in trajs for tu in t.turns if tu.gen_logp is not None]
        n_gen = sum(n for _, n in gen)
        diag = {}
        if n_gen:
            vlogp = sum(x for x, _ in gen) / n_gen
            diag = {"rollout/vllm_logp": vlogp,
                    "train/logp_gap": stats["logp_sum"] / max(n_tok, 1) / c.ppo_epochs - vlogp,
                    "train/tok_ratio": n_tok / n_gen}
        # how often re-tokenizing a reply's text would not give the sampled ids (training uses the sampled ids
        # whenever the engine reports them; this measures what the old text path would have trained instead)
        with_ids = [s for s in samples if s.gen_ids and s.train_msg_idx == [len(s.messages) - 1]]
        if with_ids and hasattr(pol, "retok_differs"):
            diag["rollout/retok_mismatch"] = float(np.mean([pol.retok_differs(s) for s in with_ids]))
        return {**diag, "train/lr": float(self.opt.param_groups[0]["lr"]),
                "train/loss": stats["loss"] / c.ppo_epochs, "train/grad_norm": float(gn),
                "train/tokens": n_tok, "train/sequences": len(samples),
                "train/mean_logp": stats["logp_sum"] / max(n_tok, 1) / c.ppo_epochs,
                **({"train/clipfrac": stats["clipfrac_sum"] / n_tok / c.ppo_epochs}
                   if c.policy_loss == "ppo_clip" else {}),
                **({"train/kl": stats["kl_sum"] / max(n_tok, 1) / c.ppo_epochs} if c.kl_coef > 0 else {})}

    # ------------------------------------------------------------------ eval / logging
    @staticmethod
    def eval_metrics(trajs, prefix):
        return {f"{prefix}/success": float(np.mean([t.success for t in trajs])),
                f"{prefix}/turns": float(np.mean([len(t.turns) for t in trajs])),
                f"{prefix}/invalid_frac": float(np.mean([t.num_invalid / max(1, len(t.turns)) for t in trajs])),
                **end_reason_fracs(trajs, prefix)}

    @torch.no_grad()
    def evaluate(self):
        """Greedy episodes on every eval set, pooled: all sets share each round's generation batch.
        Returns (metrics, {log prefix: trajectories}). With named sets, each gets ``eval/<name>/*`` and
        ``eval/*`` is the unweighted mean over sets (overall success = mean of the per-set rates)."""
        c = self.cfg
        tasks = [RolloutTask(s["env"], s["bundle"].interface, s["starts"], list(range(len(s["starts"]))),
                             s["bundle"].max_turns) for s in self.eval_sets]
        out = rollout_pooled(self._gen(c.eval_temperature), tasks, c.max_images, on_invalid=c.eval_on_invalid or c.on_invalid,
                             max_context_tokens=c.max_context_tokens,
                             last_tokens=lambda: getattr(self.generator, "last_num_tokens", []),
                             max_frames=c.max_frames, thinking=self._thinking)
        by_set = {s["name"]: trajs for s, trajs in zip(self.eval_sets, out)}
        m = {}
        for name, trajs in by_set.items():
            m.update(self.eval_metrics(trajs, name))
        if self.cfg.eval_sets:
            per = [self.eval_metrics(trajs, "eval") for trajs in by_set.values()]
            m.update({k: float(np.mean([p[k] for p in per])) for k in per[0]})
        return m, by_set

    @property
    def _thinking(self) -> bool:
        """The chat template was explicitly asked to think: a reply must close its reasoning to count."""
        return self.cfg.chat_template_kwargs.get("enable_thinking") is True

    def _rollout(self, temperature, starts, gids):
        c = self.cfg
        return rollout(self._gen(temperature), self.env, self.bundle.interface, starts, gids, self.bundle.max_turns,
                       c.max_images, on_invalid=c.on_invalid, max_context_tokens=c.max_context_tokens,
                       last_tokens=lambda: getattr(self.generator, "last_num_tokens", []), max_frames=c.max_frames,
                       last_logprobs=lambda: getattr(self.generator, "last_logprobs", []), thinking=self._thinking,
                       last_gen_ids=lambda: getattr(self.generator, "last_gen_ids", []))

    def dump_episode(self, tr, tag, step=None):
        """Write a video (mp4 or gif) + transcript to out_dir/episodes; returns (transcript, video path or None)."""
        d = f"{self.cfg.out_dir}/episodes"
        os.makedirs(d, exist_ok=True)
        video = f"{d}/{tag}.{self.cfg.video_format}"
        if tr.frames:
            write_video(tr.frames, video, self.cfg.video_fps, self.cfg.video_min_size, self.cfg.video_crf)
        lines = [f"success={tr.success} score={tr.score:.3f} adv={tr.advantage:.3f} end={tr.end_reason}", ""]
        for m in tr.messages:
            lines += [f"--- {m['role']}", " ".join(c.get("text", "<image>") for c in m["content"])]
        text = "\n".join(lines)
        with open(f"{d}/{tag}.txt", "w") as f:
            f.write(text)
        return text, (video if tr.frames else None)

    def log_episodes(self, split, step, trajs):
        """Dump up to `samples_per_log` episodes (full reasoning chains + videos) to out_dir/episodes and
        wandb: `{split}/transcript` + `{split}/episode` for the first, and one row per episode in the
        incremental `{split}/samples` table (filterable across steps); `split` is "train", "eval" or
        "eval/<set name>". Train picks a success/failure mix;
        eval uses the first fixed eval start states, so the same mazes can be compared across steps."""
        n = self.cfg.samples_per_log
        if split == "train":
            wins, losses = [t for t in trajs if t.success], [t for t in trajs if not t.success]
            picked = wins[:n // 2]
            picked += losses[:n - len(picked)]
            trajs = picked + wins[n // 2:][:n - len(picked)]  # top up with more wins if few failures
        trajs = trajs[:n]
        dumped = [self.dump_episode(tr, f"{split.replace('/', '_')}_step{step}" + (f"_{i}" if i else ""), step)
                  for i, tr in enumerate(trajs)]
        if not self.wandb or not trajs:
            return
        import html
        import wandb
        if not hasattr(self, "_tables"):
            self._tables = {}
        table = self._tables.setdefault(split, wandb.Table(
            columns=["step", "episode", "success", "score", "advantage", "turns", "invalid_turns", "end_reason",
                     "mean_reply_chars", "transcript", "video"], log_mode="INCREMENTAL"))
        fmt = self.cfg.video_format
        for i, (tr, (text, video)) in enumerate(zip(trajs, dumped)):
            table.add_data(step, i, bool(tr.success), tr.score, tr.advantage, len(tr.turns), tr.num_invalid,
                           tr.end_reason, float(np.mean([len(x.response) for x in tr.turns])) if tr.turns else 0.0,
                           text, wandb.Video(video, format=fmt) if video else None)
        tr, (text, video) = trajs[0], dumped[0]
        media = {f"{split}/transcript": wandb.Html(f"<pre>{html.escape(text)}</pre>"), f"{split}/samples": table}
        if video:
            caption = f"success={tr.success} score={tr.score:.3f} rounds={len(tr.turns)} end={tr.end_reason}"
            media[f"{split}/episode"] = wandb.Video(video, caption=caption, format=fmt)
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
        # 6 significant digits, not 4 decimals: learning rates (1e-6) and small gaps must survive
        line = {"step": step, **{k: (float(f"{v:.6g}") if isinstance(v, float) else v) for k, v in metrics.items()}}
        print(json.dumps(line), flush=True)
        with open(f"{self.cfg.out_dir}/metrics.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        if self.wandb:
            self.wandb.log(metrics, step=step, commit=True)  # don't hold the row open until the next step

    # ------------------------------------------------------------------ main loop
    def behaviour_clone(self) -> dict:
        """bc_steps updates on the scripted expert's transcripts (advantage +1 on every turn: with the token-mean loss
        this is maximum likelihood on the expert's replies). Runs before the step-0 eval, which then measures the
        cloned policy; the rollout engine gets the cloned weights before RL starts. The last ``bc_dagger_steps`` steps
        are DAgger: the current clone drives the episodes and the expert labels every state it reaches, so the clone
        also learns what to do after its own mistakes (pure cloning only shows the expert's states)."""
        from vlmrl.rollout import expert_rollout
        c = self.cfg
        if self.bundle.expert is None:
            raise ValueError("bc_steps > 0 needs an env with a scripted expert (EnvBundle.expert)")
        rng = np.random.default_rng(20_000 + c.seed)
        succ, last = [], {}
        self._lr_base = c.bc_lr or c.lr
        for i in range(c.bc_steps):
            dagger = i >= c.bc_steps - c.bc_dagger_steps
            if dagger and i > 0 and self.generator is not self.policy:
                self.generator.sync(self.policy, 0)          # DAgger rolls out the current clone
            starts = self.bundle.train_sampler.sample(c.bc_episodes, rng)
            trajs = expert_rollout(self.env, self.bundle.interface, self.bundle.expert, starts,
                                   list(range(len(starts))), self.bundle.max_turns,
                                   generate=self._gen(c.temperature) if dagger else None, max_images=c.max_images,
                                   max_frames=c.max_frames, thinking=self._thinking, noise=c.bc_noise, rng=rng)
            succ.append(float(np.mean([t.success for t in trajs])))
            if dagger:   # every turn has the expert's label for the state the clone reached, failed episodes included
                keep = trajs
            else:
                keep = [t for t in trajs if t.success] if c.bc_success_only else trajs
            for t in keep:
                t.advantage = 1.0
            last = self.update([keep]) if keep else {"train/skipped": 1.0}
            print(f"[bc] step {i + 1}/{c.bc_steps}: {'clone (DAgger)' if dagger else 'expert'} success {succ[-1]:.2f}, "
                  f"{len(keep)} episodes, "
                  f"mean log-prob {last.get('train/mean_logp', float('nan')):.3f}", flush=True)
        self._lr_base = None
        self.policy.save(f"{c.out_dir}/ckpt_bc")   # the clone alone, reusable via init_adapter (LoRA)
        if self.generator is not self.policy:
            self.generator.sync(self.policy, 0)
        n_exp = c.bc_steps - min(c.bc_dagger_steps, c.bc_steps)
        out = {"bc/expert_success": float(np.mean(succ[:n_exp])) if n_exp else float("nan"),
               "bc/mean_logp": last.get("train/mean_logp", 0.0), "bc/steps": c.bc_steps}
        if c.bc_dagger_steps:
            out["bc/dagger_success"] = float(np.mean(succ[n_exp:]))     # the clone's own episodes during DAgger
            out["bc/dagger_last_success"] = succ[-1]
        return out

    def train(self):
        c = self.cfg
        bc = self.behaviour_clone() if c.bc_steps and self.start_step == 0 else {}
        if c.eval_every and self.start_step == 0:
            m, by_set = self.evaluate()
            self.log(0, {**bc, **m})
            for name, trajs in by_set.items():
                self.log_episodes(name, 0, trajs)
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
                self.log_episodes("train", step, all_trajs)
            if c.eval_every and step % c.eval_every == 0:
                em, by_set = self.evaluate()
                m.update(em)
                for name, trajs in by_set.items():
                    self.log_episodes(name, step, trajs)
            if c.resume_every and (step % c.resume_every == 0 or step == c.steps):
                self.save_resume(step, m)
            self.log(step, m)
            if c.save_every and step % c.save_every == 0:
                self.policy.save(f"{c.out_dir}/ckpt_step{step}")
        self.env.close()
        for s in self.eval_sets:
            if s["env"] is not self.env:
                s["env"].close()
        if self.wandb:
            self.wandb.finish()
