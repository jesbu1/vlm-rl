"""vLLM server backend against a fake OpenAI-compatible HTTP server (no vLLM/GPU needed)."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
import torch
from PIL import Image

from vlmrl.vllm_client import VLLMServerGenerator, to_openai_messages


class _Handler(BaseHTTPRequestHandler):
    log = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.log.append((self.path, body))
        if self.path == "/v1/chat/completions":
            n_img = sum(1 for m in body["messages"] if isinstance(m["content"], list)
                        for c in m["content"] if c["type"] == "image_url")
            out = {"choices": [{"message": {"content": f" {body['model']}|{n_img}|{body['temperature']} "}}]}
            data = json.dumps(out).encode()
        else:
            data = b"Success"
        self.send_response(200)
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    _Handler.log = []
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _conv():
    img = Image.fromarray(np.zeros((4, 4, 3), np.uint8))
    return [{"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "again"}]}]


def test_message_conversion():
    m = to_openai_messages(_conv())
    assert m[0] == {"role": "system", "content": "sys"}
    assert m[2] == {"role": "assistant", "content": "ok"}
    assert m[1]["content"][0]["type"] == "image_url"
    assert m[1]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert m[1]["content"][1] == {"type": "text", "text": "hi"}


def test_generate_preserves_order_and_params(server):
    g = VLLMServerGenerator(server, "base-model", "/tmp/unused", max_workers=4)
    convs = [_conv(), _conv()[:2], _conv()]
    out = g.generate(convs, temperature=0.7)
    assert out == ["base-model|2|0.7", "base-model|1|0.7", "base-model|2|0.7"]
    assert all(b["top_k"] == -1 for p, b in _Handler.log)
    assert g.generate([_conv()[:2]], temperature=0)[0].endswith("|0.0")


class _FakePeft:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1)
        self.model.peft_config = {}

        def save_pretrained(path):
            import os
            os.makedirs(path, exist_ok=True)
            open(f"{path}/adapter_model.safetensors", "w").write("x")
        self.model.save_pretrained = save_pretrained


def test_sync_hot_swaps_lora(server, tmp_path):
    g = VLLMServerGenerator(server, "base", str(tmp_path))
    pol = _FakePeft()
    g.sync(pol, 0)
    g.sync(pol, 1)
    paths = [p for p, _ in _Handler.log]
    assert paths == ["/v1/load_lora_adapter", "/v1/load_lora_adapter", "/v1/unload_lora_adapter"]
    s0, s1 = f"policy_{g.tag}_step0", f"policy_{g.tag}_step1"
    assert _Handler.log[1][1]["lora_name"] == s1 and _Handler.log[2][1]["lora_name"] == s0
    assert g.current == s1
    assert not (tmp_path / s0).exists() and (tmp_path / s1).exists()
    assert g.generate([_conv()[:2]])[0].startswith(f"{s1}|")


def test_adapter_names_unique_across_generators(server, tmp_path):
    """Sweep runs share one server and each starts at step 0 (or a resumed step): names must not collide."""
    pol = _FakePeft()
    a, b = VLLMServerGenerator(server, "base", str(tmp_path / "a")), VLLMServerGenerator(server, "base", str(tmp_path / "b"))
    a.sync(pol, 30)
    b.sync(pol, 30)
    loaded = [body["lora_name"] for p, body in _Handler.log if p == "/v1/load_lora_adapter"]
    assert len(loaded) == 2 and loaded[0] != loaded[1]


class _FullPolicy:
    """Full fine-tuning policy (no peft_config): sync saves a whole checkpoint via policy.save."""
    model = torch.nn.Linear(1, 1)

    def __init__(self):
        self.saved = []

    def save(self, path, processor=True):
        import os
        os.makedirs(path, exist_ok=True)
        open(f"{path}/model.safetensors", "w").write("x")
        self.saved.append((path, processor))


def test_full_weight_sync_reloads_in_place(server, tmp_path):
    g = VLLMServerGenerator(server, "base", str(tmp_path))
    pol = _FullPolicy()
    g.sync(pol, 0)                      # step 0 = the base weights the server already serves
    assert _Handler.log == [] and pol.saved == []
    g.sync(pol, 1)
    g.sync(pol, 2)
    assert [p for p, _ in _Handler.log] == ["/collective_rpc", "/reset_prefix_cache", "/reset_encoder_cache"] * 2
    p1, p2 = (f"{tmp_path}/full_{g.tag}_step{s}" for s in (1, 2))
    assert _Handler.log[0][1] == {"method": "reload_weights", "kwargs": {"weights_path": p1}}
    assert _Handler.log[3][1]["kwargs"]["weights_path"] == p2
    assert pol.saved == [(p1, False), (p2, False)]       # weights only, no processor files
    assert not os.path.exists(p1) and os.path.exists(p2)  # only the served checkpoint is kept
    assert g.current == "base" and g.generate([_conv()[:2]])[0].startswith("base|")


def test_chat_template_kwargs_forwarded_only_when_set(server):
    VLLMServerGenerator(server, "base", "/tmp/unused").generate([_conv()], max_new_tokens=4)
    VLLMServerGenerator(server, "base", "/tmp/unused", chat_template_kwargs={"enable_thinking": True}).generate(
        [_conv()], max_new_tokens=4)
    bodies = [b for p, b in _Handler.log if p == "/v1/chat/completions"]
    assert "chat_template_kwargs" not in bodies[0] and bodies[1]["chat_template_kwargs"] == {"enable_thinking": True}


def test_server_errors_carry_the_response_body():
    """A 500 from the server must surface its body (vLLM's log often has nothing)."""
    from vlmrl.vllm_client import ServerError, _post

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error": "reload_weights: weights not found"}')

        def log_message(self, *a):
            pass
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pytest.raises(ServerError, match="HTTP 500.*weights not found"):
            _post(f"http://127.0.0.1:{srv.server_port}/collective_rpc", {})
    finally:
        srv.shutdown()


def test_full_sync_fails_clearly_when_the_checkpoint_is_missing(server, tmp_path):
    class NoFiles(_FullPolicy):
        def save(self, path, processor=True):
            os.makedirs(path, exist_ok=True)          # directory but no *.safetensors
    g = VLLMServerGenerator(server, "base", str(tmp_path))
    with pytest.raises(RuntimeError, match="checkpoint for vLLM missing or empty"):
        g.sync(NoFiles(), 1)
    assert _Handler.log == []                         # the server is never asked to load it
