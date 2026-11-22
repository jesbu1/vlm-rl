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


def _check_checkpoint(path: str) -> None:
    """The server loads ``path`` from disk: fail here, with a listing, if the checkpoint is not there."""
    import glob
    files = glob.glob(os.path.join(path, "*.safetensors"))
    if not os.path.isdir(path) or not files:
        parent = os.path.dirname(path)
        du = shutil.disk_usage(parent) if os.path.isdir(parent) else None
        raise RuntimeError(f"checkpoint for vLLM missing or empty: {path} (isdir={os.path.isdir(path)}, "
                           f"safetensors={len(files)}, parent={os.listdir(parent) if os.path.isdir(parent) else None}, "
                           f"free={du.free / 1e9 if du else '?'} GB)")


class ServerError(RuntimeError):
    """An HTTP error from the vLLM server, with its response body (the server log often has no details)."""


def _post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    import urllib.error
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode()
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:2000]
        raise ServerError(f"POST {url} -> HTTP {e.code}: {detail}") from e
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
        self._n_sync = 0        # sync counter: names stay unique when a step is synced twice (e.g. after BC at step 0)
        # adapter names must be unique per server: a sweep reuses one server across runs (and resumed runs
        # restart at an old step), and vLLM refuses to load a name that is still loaded from an earlier run
        self.tag = uuid.uuid4().hex[:8]
        self.pool = ThreadPoolExecutor(max_workers)
        self.chat_template_kwargs = dict(chat_template_kwargs or {})

    def _one(self, msgs, max_new_tokens, temperature, top_p):
        greedy = temperature <= 0
        payload = {"model": self.current, "messages": to_openai_messages(msgs), "max_tokens": max_new_tokens,
                   "temperature": 0.0 if greedy else temperature, "top_p": 1.0 if greedy else top_p,
                   "top_k": -1, "n": 1, "logprobs": True, "return_tokens_as_token_ids": True}
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        r = _post(f"{self.url}/v1/chat/completions", payload)
        u = r.get("usage") or {}
        ntok = int(u.get("prompt_tokens", 0)) + int(u.get("completion_tokens", 0))
        # log-probs of the sampled tokens, from the raw model output (vLLM's default logprobs mode, i.e. before
        # temperature / top-p): the trainer's log-probs of the same tokens should match them on-policy
        content = (r["choices"][0].get("logprobs") or {}).get("content") or []
        lps = [c.get("logprob") for c in content]
        lps = [x for x in lps if x is not None and x > -1e30]
        lp = (float(sum(lps)), len(lps)) if lps else (None, 0)
        # the sampled token ids ("token_id:<n>" with return_tokens_as_token_ids), so training uses exactly them
        ids = [int(c["token"].split(":", 1)[1]) for c in content
               if isinstance(c.get("token"), str) and c["token"].startswith("token_id:")]
        gen_ids = ids if content and len(ids) == len(content) else None
        return (r["choices"][0]["message"].get("content") or "").strip(), ntok, (*lp, gen_ids)

    def generate(self, convs, max_new_tokens=256, temperature=1.0, top_p=1.0, batch_size=None):
        futs = [self.pool.submit(self._one, c, max_new_tokens, temperature, top_p) for c in convs]
        res = [f.result() for f in futs]
        self.last_num_tokens = [n for _, n, _ in res]  # context length per item, for token budgets
        self.last_logprobs = [lp[:2] for _, _, lp in res]  # (sum of sampled-token log-probs, n tokens) per item
        self.last_gen_ids = [lp[2] for _, _, lp in res]     # sampled token ids per item (None if not reported)
        return [t for t, _, _ in res]

    def sync(self, policy, step: int):
        """Push the current weights to the server. LoRA: save adapter -> hot-load -> unload previous.
        Full fine-tuning: save the full checkpoint -> in-place ``reload_weights`` -> reset the prefix cache."""
        if not hasattr(policy.model, "peft_config"):
            return self._sync_full(policy, step)
        self._n_sync += 1
        name = f"policy_{self.tag}_step{step}_{self._n_sync}"
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
        loads at startup, then reloaded in place on every worker. Cached prefixes and image embeddings were computed
        with the old weights, so the prefix and encoder caches are reset too. The first sync at step 0 is skipped: the
        server already serves those base weights. Any later step-0 sync (after behaviour cloning) reloads."""
        self._n_sync += 1
        if step == 0 and self._n_sync == 1:
            return
        path = os.path.abspath(os.path.join(self.adapter_dir, f"full_{self.tag}_step{step}_{self._n_sync}"))
        if hasattr(policy, "save"):
            policy.save(path, processor=False)
        else:
            policy.model.save_pretrained(path)
        _check_checkpoint(path)
        for attempt in (1, 2):  # one retry: a reload failed transiently on one machine while identical ones worked
            try:
                _post(f"{self.url}/collective_rpc", {"method": "reload_weights", "kwargs": {"weights_path": path}},
                      timeout=1800)
                break
            except ServerError as e:
                print(f"[vllm] reload_weights failed (attempt {attempt}): {e}", flush=True)
                if attempt == 2:
                    raise
                import time
                time.sleep(30)
        _post(f"{self.url}/reset_prefix_cache", {})
        # the worker drops its cached image embeddings on reload, but the scheduler would still count them as
        # cached ("Encoder cache miss" crash on the next image request); this resets both sides
        _post(f"{self.url}/reset_encoder_cache", {})
        if self._prev_adapter:
            shutil.rmtree(self._prev_adapter[1], ignore_errors=True)
        self._prev_adapter = (self.base_model, path)
