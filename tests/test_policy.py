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
