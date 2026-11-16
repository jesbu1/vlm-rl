"""vLLM server backend against a fake OpenAI-compatible HTTP server (no vLLM/GPU needed)."""
import json
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
    assert _Handler.log[1][1]["lora_name"] == "policy_step1" and _Handler.log[2][1]["lora_name"] == "policy_step0"
    assert g.current == "policy_step1"
    assert not (tmp_path / "policy_step0").exists() and (tmp_path / "policy_step1").exists()
    assert g.generate([_conv()[:2]])[0].startswith("policy_step1|")


def test_sync_requires_lora(server, tmp_path):
    g = VLLMServerGenerator(server, "base", str(tmp_path))

    class P:
        model = torch.nn.Linear(1, 1)
    with pytest.raises(NotImplementedError):
        g.sync(P(), 0)
