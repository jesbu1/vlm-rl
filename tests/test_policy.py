"""HFPolicy tests. Pure-tensor helpers run by default; real-model tests need --run-slow."""
import numpy as np
import pytest
import torch

from conftest import random_generate
from vlmrl.policy import Sample, chunked_token_logprobs
from vlmrl.rollout import rollout, to_samples


@pytest.mark.parametrize("chunk", [1, 3, 1024])
def test_chunked_logprobs_matches_log_softmax(chunk):
    torch.manual_seed(0)
    logits = torch.randn(2, 7, 11, dtype=torch.bfloat16)
    tgt = torch.randint(0, 11, (2, 7))
    ref = logits.float().log_softmax(-1).gather(-1, tgt[..., None]).squeeze(-1)
    out = chunked_token_logprobs(logits, tgt, chunk)
    assert out.dtype == torch.float32 and torch.allclose(out, ref, atol=1e-5)


def test_chunked_logprobs_gradients():
    logits = torch.randn(1, 5, 6, requires_grad=True)
    tgt = torch.randint(0, 6, (1, 5))
    chunked_token_logprobs(logits, tgt, 2).sum().backward()
    ref = logits.detach().clone().requires_grad_(True)
    ref.log_softmax(-1).gather(-1, tgt[..., None]).sum().backward()
    assert torch.allclose(logits.grad, ref.grad, atol=1e-6)


def test_adaptive_generation_batch_on_oom(monkeypatch):
    """generate() halves the batch on OOM, retries, and remembers the smaller size."""
    from vlmrl.policy import HFPolicy
    pol = HFPolicy.__new__(HFPolicy)
    pol.model = torch.nn.Linear(1, 1)
    calls = []

    def chunk(convs, *a):
        calls.append(len(convs))
        if len(convs) > 2:
            raise torch.OutOfMemoryError("fake")
        return [f"r{c}" for c in convs]
    pol._generate_chunk = chunk
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    out = pol.generate(list(range(5)), batch_size=8)
    assert out == ["r0", "r1", "r2", "r3", "r4"]
    assert calls == [5, 4, 2, 2, 1]
    assert pol._gen_bs == 2
    pol.generate([0, 1, 2], batch_size=8)
    assert calls[-2:] == [2, 1]  # starts from the remembered size


# ----------------------------------------------------------------------------- real tiny VLM

def _traj_with_texts(umaze, rng, texts):
    env = umaze.make_vec_env()
    it = iter(texts)
    return rollout(lambda convs: [next(it) for _ in convs], env, umaze.interface,
                   umaze.train_sampler.sample(1, rng), [0], len(texts), on_invalid="noop")[0]


@pytest.mark.slow
def test_generate_returns_strings(tiny_vlm, umaze, rng):
    env = umaze.make_vec_env()
    trajs = rollout(lambda c: tiny_vlm.generate(c, max_new_tokens=8, batch_size=2), env, umaze.interface,
                    umaze.train_sampler.sample(3, rng), [0, 0, 0], 2)
    assert all(isinstance(t.turns[0].response, str) for t in trajs)
    greedy = tiny_vlm.generate([trajs[0].messages[:2]] * 2, max_new_tokens=8, temperature=0)
    assert greedy[0] == greedy[1]


@pytest.mark.slow
def test_mask_covers_exactly_assistant_turns(tiny_vlm, umaze, rng):
    texts = ['first <action>{"dx": 1, "dy": 0}</action>', "second reply", "third <action>0,1</action>"]
    t = _traj_with_texts(umaze, rng, texts)
    t.advantage = 1.0
    batch = tiny_vlm.build_batch(to_samples([t], None))
    m = batch["loss_mask"][0].bool()
    dec = tiny_vlm.tokenizer.decode(batch["input_ids"][0][m], skip_special_tokens=True)
    for want in texts:
        assert want in dec
    assert "Round" not in dec and "control a small ball" not in dec
    # three contiguous spans
    edges = (m[1:].int() - m[:-1].int())
    assert int((edges == 1).sum()) == 3
    assert batch["advantages"].tolist() == [1.0]


@pytest.mark.slow
def test_logits_to_keep_alignment(tiny_vlm, umaze, rng):
    """token_logprobs (which skips prompt logits) must equal a naive full forward."""
    t = _traj_with_texts(umaze, rng, ["a <action>0,0</action>", "b <action>1,0</action>"])
    batch = tiny_vlm.build_batch([Sample(t.messages, [2, 4], 1.0)])
    logp, mask = tiny_vlm.token_logprobs(batch)
    with torch.no_grad():
        inp = {k: v for k, v in batch.items() if k not in ("loss_mask", "advantages")}
        full = tiny_vlm.model(**inp, use_cache=False).logits.float().log_softmax(-1)
        ids = inp["input_ids"]
        ref = full[:, :-1].gather(-1, ids[:, 1:, None]).squeeze(-1)
        ref_mask = batch["loss_mask"][:, 1:]
    first = int(batch["loss_mask"].any(0).nonzero().min())
    assert torch.allclose(mask.cpu(), ref_mask[:, first - 1:])
    sel = mask.bool().cpu()
    assert torch.allclose(logp.detach().cpu()[sel], ref[:, first - 1:][sel], atol=1e-3)


@pytest.mark.slow
def test_batch_padding_does_not_change_logprobs(tiny_vlm, umaze, rng):
    short = _traj_with_texts(umaze, rng, ["x <action>0,0</action>"])
    long = _traj_with_texts(umaze, np.random.default_rng(1), ["y <action>0,0</action>", "z", "w"])
    s1 = Sample(short.messages, [2], 1.0)
    s2 = Sample(long.messages, [2, 4, 6], 1.0)
    with torch.no_grad():
        alone, m_alone = tiny_vlm.token_logprobs(tiny_vlm.build_batch([s1]))
        both, m_both = tiny_vlm.token_logprobs(tiny_vlm.build_batch([s1, s2]))
    a = alone[0][m_alone[0].bool()]
    b = both[0][m_both[0].bool()]
    assert a.shape == b.shape and torch.allclose(a, b, atol=2e-3)


@pytest.mark.slow
def test_full_ft_freeze_fp32_master_and_save_format(umaze, rng, tmp_path):
    """Full FT: frozen params stay bf16, trainable ones become fp32 masters (bf16 autocast forward), grads are
    fp32, and `save` writes a bf16 checkpoint with exactly the original checkpoint's tensor names (what the
    vLLM full-weight sync reloads)."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    from conftest import TINY_VLM
    from vlmrl.policy import HFPolicy
    pol = HFPolicy(TINY_VLM, device="cpu", dtype="bfloat16", processor_kwargs={"do_image_splitting": False},
                   gradient_checkpointing=False, freeze=r".*embed_tokens.*")
    named = dict(pol.model.named_parameters())
    frozen = [n for n in named if "embed_tokens" in n]
    assert frozen and all(not named[n].requires_grad and named[n].dtype == torch.bfloat16 for n in frozen)
    train = [p for p in named.values() if p.requires_grad]
    assert train and all(p.dtype == torch.float32 for p in train)
    t = _traj_with_texts(umaze, rng, ["a <action>0,0</action>"])
    logp, mask = pol.token_logprobs(pol.build_batch([Sample(t.messages, [2], 1.0)]))
    assert torch.isfinite(logp[mask.bool()]).all()
    (-(logp * mask).sum()).backward()
    grads = [p.grad for p in train if p.grad is not None]
    assert grads and all(g.dtype == torch.float32 for g in grads)

    pol.save(str(tmp_path / "ck"), processor=False)
    saved = {}
    for f in (tmp_path / "ck").glob("*.safetensors"):
        with safe_open(str(f), "pt") as fh:
            saved.update({k: fh.get_slice(k).get_dtype() for k in fh.keys()})
    with safe_open(hf_hub_download(TINY_VLM, "model.safetensors"), "pt") as fh:
        original = set(fh.keys())
    assert set(saved) == original and set(saved.values()) == {"BF16"}


@pytest.mark.slow
@pytest.mark.parametrize("style", ["pear", "compact"])
def test_fast_reply_span_matches_slow_path(tiny_vlm, rng, style):
    """Per-turn samples locate the reply from text templates only; the mask must equal the slow path's
    (re-tokenising the prefix with its images)."""
    from vlmrl.envs.pointmaze import PointMazeConfig, make_pointmaze
    b = make_pointmaze(PointMazeConfig(maze="umaze", max_turns=3, image_size=64, prompt_style=style))
    fmt = random_generate(0) if style == "pear" else (lambda convs: ['{"x": 0.4, "y": -0.2}'] * len(convs))
    trajs = rollout(fmt, b.make_vec_env(), b.interface, b.train_sampler.sample(2, rng), [0, 1], 3, max_frames=1)
    samples = to_samples(trajs, None, max_frames=1)
    fast = tiny_vlm.build_batch(samples)["loss_mask"]
    orig = tiny_vlm._last_reply_span
    tiny_vlm._last_reply_span = lambda *a: None
    try:
        slow = tiny_vlm.build_batch(samples)["loss_mask"]
    finally:
        tiny_vlm._last_reply_span = orig
    assert torch.equal(fast, slow) and fast.sum() > 0
    enc = tiny_vlm.build_batch(samples[:1])
    assert orig(samples[0], enc["input_ids"][0], int(enc["attention_mask"][0].sum())) is not None  # fast path used

@pytest.mark.slow
def test_reference_logprobs_match_the_untrained_policy(tiny_vlm, rng):
    """The KL reference is the pretrained checkpoint: before any update it scores tokens like the policy itself."""
    from vlmrl.envs.pointmaze import PointMazeConfig, make_pointmaze
    b = make_pointmaze(PointMazeConfig(maze="umaze", max_turns=2, image_size=64, prompt_style="compact_rel"))
    gen = lambda convs: ['{"x": 0.4, "y": -0.2}'] * len(convs)
    trajs = rollout(gen, b.make_vec_env(), b.interface, b.train_sampler.sample(2, rng), [0, 1], 2)
    batch = tiny_vlm.build_batch(to_samples(trajs, None))
    with torch.no_grad():
        logp, mask = tiny_vlm.token_logprobs(batch)
    ref = tiny_vlm.ref_token_logprobs(batch)
    assert ref.shape == logp.shape and not ref.requires_grad
    assert torch.allclose(ref[mask > 0], logp[mask > 0], atol=1e-3)


@pytest.mark.slow
def test_reply_spelling_image_token_still_builds_a_batch(tiny_vlm, rng):
    """Regression: SmolVLM wrote a literal "<image>" in a reply and build_batch raised "Found total 65 <image>
    tokens and 64 images"; rollout scrubs placeholders before the reply enters the conversation."""
    from vlmrl.envs.pointmaze import PointMazeConfig, make_pointmaze
    b = make_pointmaze(PointMazeConfig(maze="umaze", max_turns=2, image_size=64, prompt_style="compact"))
    gen = lambda convs: ['<image> {"x": 0.4, "y": -0.2}'] * len(convs)
    trajs = rollout(gen, b.make_vec_env(), b.interface, b.train_sampler.sample(2, rng), [0, 1], 2)
    assert tiny_vlm.build_batch(to_samples(trajs, None))["loss_mask"].sum() > 0


@pytest.mark.parametrize("honors_keep", [True, False])
def test_token_logprobs_whether_or_not_the_model_honors_logits_to_keep(honors_keep):
    """MiniCPM-V 4.6 ignores logits_to_keep and returns logits for every position; both must give the same
    log-probs of the response tokens."""
    from vlmrl.policy import HFPolicy
    V, T, first = 11, 9, 5
    torch.manual_seed(0)
    table = torch.randn(T, V)

    class M(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=0):
            full = table.unsqueeze(0).expand(input_ids.shape[0], T, V)
            return type("O", (), {"logits": full[:, -logits_to_keep:] if honors_keep and logits_to_keep else full})()
    pol = HFPolicy.__new__(HFPolicy)
    pol.model, pol.device, pol.logprob_chunk, pol._autocast, pol.dtype = M(), "cpu", 3, False, torch.float32
    ids = torch.randint(0, V, (1, T))
    mask = torch.zeros(1, T)
    mask[0, first:] = 1
    logp, m = pol.token_logprobs({"input_ids": ids, "attention_mask": torch.ones(1, T), "loss_mask": mask,
                                  "advantages": torch.ones(1)})
    ref = table.log_softmax(-1)[first - 1:T - 1].gather(-1, ids[0, first:, None]).squeeze(-1)
    assert logp.shape == (1, T - first) and torch.allclose(logp[0], ref, atol=1e-6) and m.sum() == T - first
