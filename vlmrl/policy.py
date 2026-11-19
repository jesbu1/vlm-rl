"""Model-agnostic Hugging Face VLM policy: sampling + per-token log-probs for multi-turn chats.

Everything goes through ``processor.apply_chat_template`` on HF-format messages, so any
``AutoModelForImageTextToText`` checkpoint whose processor has a chat template works (Qwen2.5-VL,
Qwen3-VL, Gemma-3, SmolVLM/Idefics3, LLaVA-OneVision, InternVL-HF, ...). The model-specific bits
(image token expansion, pixel tensor layout) are left entirely to the processor.

Multi-turn loss masks are derived from *prefix lengths*: for assistant message k,
    start = len(tokens(messages[:k], add_generation_prompt=True))
    end   = len(tokens(messages[:k+1]))
which is correct whenever the chat template is prefix-stable (verified on the first batch; we raise
a clear error otherwise -- e.g. templates that strip past reasoning; use ``context=per_turn`` then).
"""
from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class Sample:
    """One training sequence: a chat + which assistant messages to train on + their advantage.

    ``gen_ids``: the token ids the rollout engine generated for the last message (the trained reply of a per-turn
    sample). When set, training uses exactly these ids instead of re-tokenizing the reply's text."""
    messages: list[dict]
    train_msg_idx: list[int]
    advantage: float
    meta: dict = field(default_factory=dict)
    gen_ids: list[int] | None = None


def chunked_token_logprobs(logits: torch.Tensor, targets: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """log p(target) = logit[target] - logsumexp(logits), in fp32, chunked over the sequence.

    Avoids materialising a full [B, T, V] fp32 log_softmax (V ~ 150k for Qwen => GBs).
    """
    out = []
    for s in range(0, logits.shape[1], chunk):
        lg = logits[:, s:s + chunk].float()
        tg = targets[:, s:s + chunk]
        out.append(lg.gather(-1, tg.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(lg, dim=-1))
    return torch.cat(out, dim=1)


def chunked_token_stats(logits: torch.Tensor, targets: torch.Tensor, chunk: int = 1024, topk: int = 0,
                        gather_ids: torch.Tensor | None = None):
    """Like ``chunked_token_logprobs`` but can also return, per position,
    * ``topk`` > 0: the top-k token ids and their (true, unnormalised) probabilities -- no grad;
    * ``gather_ids`` [B, T, k]: log-probs of those ids (with grad) -- used by score centering."""
    lp, tk_ids, tk_p, g = [], [], [], []
    for s in range(0, logits.shape[1], chunk):
        lg = logits[:, s:s + chunk].float()
        lse = torch.logsumexp(lg, dim=-1, keepdim=True)
        lp.append(lg.gather(-1, targets[:, s:s + chunk].unsqueeze(-1)).squeeze(-1) - lse.squeeze(-1))
        if topk:
            with torch.no_grad():
                v, i = lg.topk(topk, dim=-1)
                tk_ids.append(i)
                tk_p.append((v - lse).exp())
        if gather_ids is not None:
            g.append(lg.gather(-1, gather_ids[:, s:s + chunk]) - lse)
    out = {"logp": torch.cat(lp, 1)}
    if topk:
        out["topk_ids"], out["topk_probs"] = torch.cat(tk_ids, 1), torch.cat(tk_p, 1)
    if gather_ids is not None:
        out["logp_at_ids"] = torch.cat(g, 1)
    return out


def _to(v, device):
    """Move tensors to ``device``; leave other processor outputs (lists, e.g. SmolVLM2's per-image metadata) as is."""
    return v.to(device) if torch.is_tensor(v) else v


class HFPolicy:
    def __init__(self, model_name: str, device: str | None = None, dtype: str = "auto",
                 attn_implementation: str | None = None, gradient_checkpointing: bool = True,
                 lora_rank: int = 0, lora_alpha: int = 32, lora_targets: str = "all-linear",
                 processor_kwargs: dict | None = None, logprob_chunk: int = 1024,
                 chat_template_kwargs: dict | None = None, freeze: str = "", fp32_master: bool = True):
        """Full fine-tuning (``lora_rank=0``): parameters whose name fully matches the regex ``freeze`` stay
        frozen; with ``fp32_master`` the trainable ones are kept in fp32 while the forward pass runs under
        bf16/fp16 autocast. (Plain bf16 weights silently drop most updates: at lr 1e-6 an Adam step is
        ~1e-4 of a typical weight, far below bf16's ~4e-3 relative resolution.) LoRA adapters are fp32
        already (PEFT)."""
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available() else "cpu")
        if dtype == "auto":
            dtype = "bfloat16" if self.device == "cuda" else "float32"
        self.dtype = getattr(torch, dtype)
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name, **(processor_kwargs or {}))
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        kw = {"dtype": self.dtype}
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, **kw).to(self.device)
        self.model.config.use_cache = False
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if lora_rank > 0:
            from peft import LoraConfig, get_peft_model
            if gradient_checkpointing:
                self.model.enable_input_require_grads()
            self.model = get_peft_model(self.model, LoraConfig(
                r=lora_rank, lora_alpha=lora_alpha, target_modules=lora_targets, lora_dropout=0.0))
            self.model.print_trainable_parameters()
        self._autocast = False
        if lora_rank == 0:
            if freeze:
                for n, p in self.model.named_parameters():
                    if re.fullmatch(freeze, n):
                        p.requires_grad_(False)
            if fp32_master and self.dtype in (torch.bfloat16, torch.float16):
                for p in self.model.parameters():
                    if p.requires_grad:
                        p.data = p.data.float()
                self._autocast = True
            n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            n_all = sum(p.numel() for p in self.model.parameters())
            print(f"[policy] full fine-tuning: {n_train / 1e9:.3f}B / {n_all / 1e9:.3f}B params trainable"
                  + (" (fp32 master weights, autocast forward)" if self._autocast else ""), flush=True)
        self.logprob_chunk = logprob_chunk
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self._prefix_checked = False
        self.stop_ids = self._stop_token_ids()

    def _stop_token_ids(self) -> set[int]:
        """Token ids generation stops at (generation_config eos ids + the tokenizer's eos). The model's reply ends
        with one of them; the chat template may add more tokens after it (e.g. '<|im_end|>' + '\n')."""
        ids = set()
        gc = getattr(self.model, "generation_config", None)
        eos = getattr(gc, "eos_token_id", None)
        for e in (eos if isinstance(eos, (list, tuple)) else [eos]):
            if e is not None:
                ids.add(int(e))
        if getattr(self.tokenizer, "eos_token_id", None) is not None:
            ids.add(int(self.tokenizer.eos_token_id))
        return ids

    def _cut_at_stop(self, ids: list[int]) -> list[int]:
        """Generated ids up to and including the first stop token (generate() pads after it); all of them if the
        reply hit max_new_tokens."""
        for i, t in enumerate(ids):
            if t in self.stop_ids:
                return ids[:i + 1]
        return ids

    def retok_differs(self, s: Sample) -> bool:
        """Would the text path (re-render the chat, re-tokenize the reply after the prompt, trim at the stop token)
        train other ids than the ones generated? Sampling can produce a non-canonical split of a string, and the
        stored reply text is stripped."""
        tpl, kw = self.processor.apply_chat_template, self.chat_template_kwargs
        full = tpl(s.messages, add_generation_prompt=False, tokenize=False, **kw)
        prompt = tpl(s.messages[:-1], add_generation_prompt=True, tokenize=False, **kw)
        if not (isinstance(full, str) and full.startswith(prompt)):
            return True
        sfx = self.tokenizer(full[len(prompt):], add_special_tokens=False)["input_ids"]
        return self._cut_at_stop(sfx) != list(s.gen_ids)

    def _trim_to_stop(self, ids: torch.Tensor, start: int, end: int) -> int:
        """End of the trained span: just after the first stop token in [start, end). Template tokens after the
        model's end-of-turn token (every chat template here appends a '\n') are never generated, so they must not
        be trained: with net-negative advantages their log-prob runs away (-108 seen) with nothing in the rollouts
        to pull it back, its gradient dominates the update and the policy collapses. (veRL / PEAR train exactly
        the generated ids, so they never see this.)"""
        if not self.stop_ids:
            return end
        for i in range(start, end):
            if int(ids[i]) in self.stop_ids:
                return i + 1
        return end

    def _amp(self):
        dev = self.device.split(":")[0]
        return torch.autocast(dev, dtype=self.dtype) if self._autocast else contextlib.nullcontext()

    # ------------------------------------------------------------------ tokenisation helpers
    def _apply(self, convs, add_generation_prompt: bool, padding_side: str = "right", tensors: bool = True):
        pk = {"padding": True, "padding_side": padding_side}
        top = {}
        if tensors:
            # some processors (e.g. SmolVLM2's) only honour return_tensors at the top level and would otherwise
            # return input_ids / attention_mask as lists; others read it from processor_kwargs: pass both
            pk["return_tensors"] = top["return_tensors"] = "pt"
        return self.processor.apply_chat_template(
            convs, add_generation_prompt=add_generation_prompt, tokenize=True, return_dict=True,
            processor_kwargs=pk, **top, **self.chat_template_kwargs)

    def _len(self, messages, add_generation_prompt: bool) -> list[int]:
        return self._apply([messages], add_generation_prompt, tensors=False)["input_ids"][0]

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def generate(self, convs: list[list[dict]], max_new_tokens: int = 256, temperature: float = 1.0,
                 top_p: float = 1.0, batch_size: int = 16) -> list[str]:
        """Sample one reply per conversation. Adaptive batching: on CUDA OOM the batch is halved and
        retried, and the smaller size is remembered (multi-image KV caches grow every turn)."""
        self.model.eval()
        self._gen_bs = min(getattr(self, "_gen_bs", batch_size), batch_size)
        texts, s = [], 0
        self.last_num_tokens: list[int] = []   # context length (prompt + reply) per item, for token budgets
        self.last_gen_ids: list[list[int]] = []  # generated token ids per item, up to and including the stop token
        while s < len(convs):
            bs = self._gen_bs
            try:
                texts += self._generate_chunk(convs[s:s + bs], max_new_tokens, temperature, top_p)
                s += bs
            except torch.OutOfMemoryError:
                if bs == 1:
                    raise
                self._gen_bs = max(1, bs // 2)
                torch.cuda.empty_cache()
                print(f"[policy] OOM in generate; generation batch size -> {self._gen_bs}")
        return texts

    def _generate_chunk(self, convs, max_new_tokens, temperature, top_p):
        enc = self._apply(convs, add_generation_prompt=True, padding_side="left")
        enc = {k: _to(v, self.device) for k, v in enc.items()}
        if "pixel_values" in enc:
            enc["pixel_values"] = enc["pixel_values"].to(self.dtype)
        greedy = temperature <= 0
        with self._amp():
            out = self._sample(enc, max_new_tokens, temperature, top_p, greedy)
        new = out[:, enc["input_ids"].shape[1]:]
        n_new = (new != self.tokenizer.pad_token_id).sum(1)
        self.last_num_tokens += (enc["attention_mask"].sum(1) + n_new).tolist()
        self.last_gen_ids += [self._cut_at_stop(row.tolist()) for row in new]
        return [t.strip() for t in self.tokenizer.batch_decode(new, skip_special_tokens=True)]

    def _sample(self, enc, max_new_tokens, temperature, top_p, greedy):
        return self.model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=not greedy, use_cache=True,
            # override checkpoint generation_config (Qwen2.5-VL ships temperature=1e-6 + repetition_penalty=1.05,
            # Qwen3-VL temperature=0.7/top_p=0.8/top_k=20): sample the full distribution, as PEAR does
            **({} if greedy else {"temperature": temperature, "top_p": top_p, "top_k": 0}),
            repetition_penalty=1.0, pad_token_id=self.tokenizer.pad_token_id)

    # ------------------------------------------------------------------ training
    def _last_reply_span(self, s: Sample, ids: torch.Tensor, length: int) -> tuple[int, int] | None:
        """[start, end) of the trained reply when it is the sample's last message, found from the *text*
        templates (no image processing) and verified against the batch token ids. None -> use the slow path
        (re-tokenising the prefixes with their images), e.g. for full-history samples."""
        if s.train_msg_idx != [len(s.messages) - 1]:
            return None
        tpl, kw = self.processor.apply_chat_template, self.chat_template_kwargs
        try:
            full = tpl(s.messages, add_generation_prompt=False, tokenize=False, **kw)
            prompt = tpl(s.messages[:-1], add_generation_prompt=True, tokenize=False, **kw)
        except Exception:  # noqa: BLE001 - processors without text-only rendering fall back
            return None
        if not (isinstance(full, str) and isinstance(prompt, str) and full.startswith(prompt)):
            return None
        sfx = self.tokenizer(full[len(prompt):], add_special_tokens=False)["input_ids"]
        start = length - len(sfx)
        if not sfx or start <= 0 or ids[start:length].tolist() != list(sfx):
            return None
        return start, length

    def _build_exact(self, samples: list[Sample]) -> dict[str, Any]:
        """Per-turn samples with the engine's generated ids: render only the prompt (with its images, the way the
        engine did), append the generated ids, and train exactly those (veRL-style). Nothing is re-tokenized, and
        nothing after the stop token, or after a reply cut off at max_new_tokens, is trained."""
        enc = dict(self._apply([s.messages[:-1] for s in samples], add_generation_prompt=True, padding_side="right"))
        B, Tp = enc["input_ids"].shape
        lens = enc["attention_mask"].sum(1).tolist()
        T = max(int(l) + len(s.gen_ids) for l, s in zip(lens, samples))
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        per_token = [k for k, v in enc.items()
                     if torch.is_tensor(v) and v.dim() == 2 and tuple(v.shape) == (B, Tp) and k != "input_ids"]
        ids = torch.full((B, T), pad, dtype=enc["input_ids"].dtype)
        out = {k: torch.zeros((B, T), dtype=enc[k].dtype) for k in per_token}
        loss_mask = torch.zeros(B, T)
        for b, (l, s) in enumerate(zip(lens, samples)):
            l, g = int(l), torch.tensor(s.gen_ids, dtype=ids.dtype)
            ids[b, :l], ids[b, l:l + len(g)] = enc["input_ids"][b, :l], g
            for k in per_token:          # attention_mask -> 1 on the reply; other per-token ids (token types) -> 0
                out[k][b, :l] = enc[k][b, :l]
                if k == "attention_mask":
                    out[k][b, l:l + len(g)] = 1
            loss_mask[b, l:l + len(g)] = 1
        enc.update(out, input_ids=ids)
        enc["loss_mask"] = loss_mask
        enc["advantages"] = torch.tensor([s.advantage for s in samples], dtype=torch.float32)
        return enc

    def build_batch(self, samples: list[Sample]) -> dict[str, Any]:
        if samples and all(s.gen_ids and s.train_msg_idx == [len(s.messages) - 1] for s in samples):
            return self._build_exact(samples)
        enc = self._apply([s.messages for s in samples], add_generation_prompt=False, padding_side="right")
        B, T = enc["input_ids"].shape
        loss_mask = torch.zeros(B, T)
        for b, s in enumerate(samples):
            span = self._last_reply_span(s, enc["input_ids"][b], int(enc["attention_mask"][b].sum()))
            if span is not None:  # fast path: per-turn samples (the trained reply is the last message)
                loss_mask[b, span[0]:self._trim_to_stop(enc["input_ids"][b], *span)] = 1
                continue
            for k in s.train_msg_idx:
                assert s.messages[k]["role"] == "assistant"
                start = len(self._len(s.messages[:k], True))
                end_ids = self._len(s.messages[:k + 1], False)
                if not self._prefix_checked:
                    if enc["input_ids"][b, :len(end_ids)].tolist() != list(end_ids):
                        raise RuntimeError(
                            "Chat template is not prefix-stable for this model (re-rendering the full chat "
                            "changes earlier turns), so multi-turn masks are ambiguous. Set max_images=<int> (per-turn samples).")
                loss_mask[b, start:self._trim_to_stop(enc["input_ids"][b], start, len(end_ids))] = 1
        self._prefix_checked = True
        enc = dict(enc)
        enc["loss_mask"] = loss_mask * enc["attention_mask"]
        enc["advantages"] = torch.tensor([s.advantage for s in samples], dtype=torch.float32)
        return enc

    def token_logprobs(self, batch: dict[str, Any], topk: int = 0, gather_ids: torch.Tensor | None = None):
        """Return (logp, mask) over positions [first_response_token, T); grad flows through logp.
        With ``topk``/``gather_ids`` returns (logp, mask, stats) -- see ``chunked_token_stats``."""
        self.model.train()
        return self._logprobs(self.model, batch, topk, gather_ids)

    @torch.no_grad()
    def ref_token_logprobs(self, batch: dict[str, Any]) -> torch.Tensor:
        """Token log-probs under the frozen reference policy (the pretrained model), aligned with
        ``token_logprobs``: the base model with adapters disabled (LoRA), or a frozen copy of the pretrained
        checkpoint in the model dtype (full fine-tuning; loaded on first use, so it costs memory only when a KL
        penalty is on, and it stays the pretrained model after a resume)."""
        if hasattr(self.model, "disable_adapter"):
            self.model.eval()
            with self.model.disable_adapter():
                return self._logprobs(self.model, batch)[0]
        if getattr(self, "_ref", None) is None:
            from transformers import AutoModelForImageTextToText
            self._ref = AutoModelForImageTextToText.from_pretrained(self.model_name, dtype=self.dtype).to(self.device)
            self._ref.eval().requires_grad_(False)
        return self._logprobs(self._ref, batch)[0]

    def _logprobs(self, model, batch, topk: int = 0, gather_ids: torch.Tensor | None = None):
        mask = batch["loss_mask"]
        first = int(mask.any(0).nonzero().min())
        T = mask.shape[1]
        inputs = {k: _to(v, self.device) for k, v in batch.items() if k not in ("loss_mask", "advantages")}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        # only compute LM-head logits from the first response token on (prompt logits are wasted work)
        with self._amp():
            out = model(**inputs, use_cache=False, logits_to_keep=T - first + 1)
        # some models ignore logits_to_keep (e.g. MiniCPM-V 4.6 returns all T positions): keep the last ones
        logits = out.logits[:, -(T - first + 1):][:, :-1]
        targets = inputs["input_ids"][:, first:]
        mask = mask[:, first:].to(self.device)
        if not topk and gather_ids is None:
            return chunked_token_logprobs(logits, targets, self.logprob_chunk), mask
        st = chunked_token_stats(logits, targets, self.logprob_chunk, topk,
                                 None if gather_ids is None else gather_ids.to(self.device))
        return st.pop("logp"), mask, st

    def save(self, path: str, processor: bool = True):
        """HF-format checkpoint. fp32 master weights are written in the model dtype, so the files look like
        the original checkpoint (this is also what the vLLM full-weight sync reloads)."""
        if self._autocast:
            full = self.model.state_dict()
            # tied weights (e.g. lm_head = embed_tokens) are one tensor under two names; HF writes it once, under
            # the non-lm_head name, but casting makes separate copies its de-duplication can't see: drop them here
            by_ptr: dict[int, list[str]] = {}
            for k, v in full.items():
                by_ptr.setdefault(v.data_ptr(), []).append(k)
            drop = {k for names in by_ptr.values() if len(names) > 1
                    for k in (n for n in names if "lm_head" in n) if len(names) > sum("lm_head" in n for n in names)}
            sd = {k: (v.to(self.dtype) if v.is_floating_point() else v) for k, v in full.items() if k not in drop}
            self.model.save_pretrained(path, state_dict=sd)
        else:
            self.model.save_pretrained(path)
        if processor:
            self.processor.save_pretrained(path)
