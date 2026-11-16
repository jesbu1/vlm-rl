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

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class Sample:
    """One training sequence: a chat + which assistant messages to train on + their advantage."""
    messages: list[dict]
    train_msg_idx: list[int]
    advantage: float
    meta: dict = field(default_factory=dict)


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


class HFPolicy:
    def __init__(self, model_name: str, device: str | None = None, dtype: str = "auto",
                 attn_implementation: str | None = None, gradient_checkpointing: bool = True,
                 lora_rank: int = 0, lora_alpha: int = 32, lora_targets: str = "all-linear",
                 processor_kwargs: dict | None = None, logprob_chunk: int = 1024):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available() else "cpu")
        if dtype == "auto":
            dtype = "bfloat16" if self.device == "cuda" else "float32"
        self.dtype = getattr(torch, dtype)
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
        self.logprob_chunk = logprob_chunk
        self._prefix_checked = False

    # ------------------------------------------------------------------ tokenisation helpers
    def _apply(self, convs, add_generation_prompt: bool, padding_side: str = "right", tensors: bool = True):
        pk = {"padding": True, "padding_side": padding_side}
        if tensors:
            pk["return_tensors"] = "pt"
        return self.processor.apply_chat_template(
            convs, add_generation_prompt=add_generation_prompt, tokenize=True, return_dict=True,
            processor_kwargs=pk)

    def _len(self, messages, add_generation_prompt: bool) -> list[int]:
        return self._apply([messages], add_generation_prompt, tensors=False)["input_ids"][0]

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def generate(self, convs: list[list[dict]], max_new_tokens: int = 256, temperature: float = 1.0,
                 top_p: float = 1.0, batch_size: int = 16) -> list[str]:
        self.model.eval()
        texts = []
        for s in range(0, len(convs), batch_size):
            enc = self._apply(convs[s:s + batch_size], add_generation_prompt=True, padding_side="left")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            if "pixel_values" in enc:
                enc["pixel_values"] = enc["pixel_values"].to(self.dtype)
            greedy = temperature <= 0
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=not greedy, use_cache=True,
                # override checkpoint generation_config (e.g. Qwen ships top_k=1): sample the full distribution
                **({} if greedy else {"temperature": temperature, "top_p": top_p, "top_k": 0}),
                repetition_penalty=1.0, pad_token_id=self.tokenizer.pad_token_id)
            new = out[:, enc["input_ids"].shape[1]:]
            texts += self.tokenizer.batch_decode(new, skip_special_tokens=True)
        return [t.strip() for t in texts]

    # ------------------------------------------------------------------ training
    def build_batch(self, samples: list[Sample]) -> dict[str, Any]:
        enc = self._apply([s.messages for s in samples], add_generation_prompt=False, padding_side="right")
        B, T = enc["input_ids"].shape
        loss_mask = torch.zeros(B, T)
        for b, s in enumerate(samples):
            for k in s.train_msg_idx:
                assert s.messages[k]["role"] == "assistant"
                start = len(self._len(s.messages[:k], True))
                end_ids = self._len(s.messages[:k + 1], False)
                if not self._prefix_checked:
                    if enc["input_ids"][b, :len(end_ids)].tolist() != list(end_ids):
                        raise RuntimeError(
                            "Chat template is not prefix-stable for this model (re-rendering the full chat "
                            "changes earlier turns), so multi-turn masks are ambiguous. Use context='per_turn'.")
                loss_mask[b, start:len(end_ids)] = 1
        self._prefix_checked = True
        enc = dict(enc)
        enc["loss_mask"] = loss_mask * enc["attention_mask"]
        enc["advantages"] = torch.tensor([s.advantage for s in samples], dtype=torch.float32)
        return enc

    def token_logprobs(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logp, mask) over positions [first_response_token, T). Grad flows through logp."""
        self.model.train()
        mask = batch["loss_mask"]
        first = int(mask.any(0).nonzero().min())
        T = mask.shape[1]
        inputs = {k: v.to(self.device) for k, v in batch.items() if k not in ("loss_mask", "advantages")}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        # only compute LM-head logits from the first response token on (prompt logits are wasted work)
        out = self.model(**inputs, use_cache=False, logits_to_keep=T - first + 1)
        logits = out.logits[:, :-1]
        targets = inputs["input_ids"][:, first:]
        logp = chunked_token_logprobs(logits, targets, self.logprob_chunk)
        return logp, mask[:, first:].to(self.device)

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
