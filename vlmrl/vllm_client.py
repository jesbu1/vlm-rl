"""Optional fast rollout backend: a vLLM OpenAI-compatible server + LoRA hot-swapping.

Why: multi-turn VLM episodes re-send the whole (multi-image) context every turn. HF ``generate``
re-prefills it from scratch; vLLM's automatic prefix caching reuses the KV cache of the shared prefix
(PEAR uses veRL + LMCache for the same reason) and continuous batching keeps the GPU busy.

Why a *server*: it keeps vLLM's pinned torch/CUDA stack in its own venv/GPU, works for any model vLLM
serves, and makes weight sync trivial for LoRA: after each update we save the adapter and call
``/v1/load_lora_adapter``. Full fine-tuning saves the whole checkpoint and has the server reload it in
place (``/collective_rpc`` -> ``reload_weights``, dev mode). That costs a disk round trip per step (seconds
for a few-B model on local NVMe); an NCCL broadcast (FeynRL / veRL) would avoid it.

Launch (separate venv, separate GPU), then train with ``--rollout_backend vllm --vllm_url ...``:

    VLLM_ALLOW_RUNTIME_LORA_UPDATING=True CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-VL-4B-Instruct \
        --enable-lora --max-lora-rank 64 --enable-prefix-caching --max-model-len 32768 \
        --limit-mm-per-prompt '{"image": 32}' --port 8000
"""
from __future__ import annotations

import base64
import io
import json
import os
import shutil
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor


def _post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    return json.loads(body) if body.strip().startswith("{") else {"text": body}


def _image_url(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)  # lossless; level 1 is several times faster to encode
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def to_openai_messages(messages: list[dict]) -> list[dict]:
    """HF multimodal chat (PIL images inline) -> OpenAI chat format (data-URL images)."""
    out = []
    for m in messages:
        parts = []
        for c in m["content"]:
            if c["type"] == "image":
                parts.append({"type": "image_url", "image_url": {"url": _image_url(c["image"])}})
            else:
                parts.append({"type": "text", "text": c["text"]})
        if m["role"] != "user" and all(p["type"] == "text" for p in parts):
            out.append({"role": m["role"], "content": "".join(p["text"] for p in parts)})
        else:
            out.append({"role": m["role"], "content": parts})
    return out


class VLLMServerGenerator:
    def __init__(self, url: str, base_model: str, adapter_dir: str, max_workers: int = 256,
                 chat_template_kwargs: dict | None = None):
        self.url = url.rstrip("/")
        self.base_model = base_model
        self.adapter_dir = adapter_dir
        self.current = base_model       # model name to request; becomes the LoRA name after first sync
        self._prev_adapter: tuple[str, str] | None = None
        # adapter names must be unique per server: a sweep reuses one server across runs (and resumed runs
        # restart at an old step), and vLLM refuses to load a name that is still loaded from an earlier run
        self.tag = uuid.uuid4().hex[:8]
        self.pool = ThreadPoolExecutor(max_workers)
        self.chat_template_kwargs = dict(chat_template_kwargs or {})

    def _one(self, msgs, max_new_tokens, temperature, top_p):
        greedy = temperature <= 0
        payload = {"model": self.current, "messages": to_openai_messages(msgs), "max_tokens": max_new_tokens,
                   "temperature": 0.0 if greedy else temperature, "top_p": 1.0 if greedy else top_p,
                   "top_k": -1, "n": 1}
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        r = _post(f"{self.url}/v1/chat/completions", payload)
        u = r.get("usage") or {}
        ntok = int(u.get("prompt_tokens", 0)) + int(u.get("completion_tokens", 0))
        return (r["choices"][0]["message"].get("content") or "").strip(), ntok

    def generate(self, convs, max_new_tokens=256, temperature=1.0, top_p=1.0, batch_size=None):
        futs = [self.pool.submit(self._one, c, max_new_tokens, temperature, top_p) for c in convs]
        res = [f.result() for f in futs]
        self.last_num_tokens = [n for _, n in res]  # context length per item, for token budgets
        return [t for t, _ in res]

    def sync(self, policy, step: int):
        """Push the current weights to the server. LoRA: save adapter -> hot-load -> unload previous.
        Full fine-tuning: save the full checkpoint -> in-place ``reload_weights`` -> reset the prefix cache."""
        if not hasattr(policy.model, "peft_config"):
            return self._sync_full(policy, step)
        name = f"policy_{self.tag}_step{step}"
        path = os.path.abspath(os.path.join(self.adapter_dir, name))
        policy.model.save_pretrained(path)
        _post(f"{self.url}/v1/load_lora_adapter", {"lora_name": name, "lora_path": path})
        self.current = name
        if self._prev_adapter:
            old_name, old_path = self._prev_adapter
            try:
                _post(f"{self.url}/v1/unload_lora_adapter", {"lora_name": old_name})
            except Exception as e:  # noqa: BLE001 - best effort
                print(f"[vllm] unload {old_name} failed: {e}")
            shutil.rmtree(old_path, ignore_errors=True)
        self._prev_adapter = (name, path)

    def _sync_full(self, policy, step: int):
        """Full-weight sync through vLLM's dev endpoints (server needs ``VLLM_SERVER_DEV_MODE=1``;
        ``scripts/launch_vllm.sh`` sets it). The checkpoint is written in HF format, i.e. exactly what vLLM
        loads at startup, then reloaded in place on every worker. Cached prefixes were computed with the old
        weights, so the prefix cache is reset too. Step 0 is the base model the server already has."""
        if step == 0:
            return
        path = os.path.abspath(os.path.join(self.adapter_dir, f"full_{self.tag}_step{step}"))
        if hasattr(policy, "save"):
            policy.save(path, processor=False)
        else:
            policy.model.save_pretrained(path)
        _post(f"{self.url}/collective_rpc", {"method": "reload_weights", "kwargs": {"weights_path": path}},
              timeout=1800)
        _post(f"{self.url}/reset_prefix_cache", {})
        if self._prev_adapter:
            shutil.rmtree(self._prev_adapter[1], ignore_errors=True)
        self._prev_adapter = (self.base_model, path)
