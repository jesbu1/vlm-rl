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


@pytest.mark.slow
def test_mask_ends_at_the_stop_token_not_the_template_newline(tiny_vlm, umaze, rng):
    """Chat templates append e.g. '<end_of_utterance>' + '\\n' after a reply; the model only generates up to the stop
    token, so the trailing '\\n' must not be trained (its log-prob ran away to -108 and collapsed full-FT runs)."""
    t = _traj_with_texts(umaze, rng, ['one <action>{"dx": 1, "dy": 0}</action>', "two"])
    t.advantage = 1.0
    for s in to_samples([t], None) + to_samples([t], 1):
        batch = tiny_vlm.build_batch([s])
        ids, m = batch["input_ids"][0], batch["loss_mask"][0].bool()
        ends = [i for i in range(1, len(m)) if m[i - 1] and not m[i]] + ([len(m)] if m[-1] else [])
        assert ends and all(int(ids[e - 1]) in tiny_vlm.stop_ids for e in ends)   # every span ends on a stop token
        assert all(tiny_vlm.tokenizer.decode([int(ids[e])]) == "\n" for e in ends if e < len(ids))


@pytest.mark.slow
def test_exact_generated_ids_are_trained_as_generated(tiny_vlm, umaze, rng):
    """With the engine's ids, the batch is prompt + exactly those ids: a canonical reply gives the same trained tokens
    as the text path; a non-canonical split is trained as generated; a reply cut off at max_new_tokens gets no
    end-of-turn token it never generated."""
    t = _traj_with_texts(umaze, rng, ['{"dx": 1, "dy": 0}'])
    t.advantage = 1.0
    s_text = to_samples([t], 1)[0]
    tok, eos = tiny_vlm.tokenizer, sorted(tiny_vlm.stop_ids)[0]
    body = tok('{"dx": 1, "dy": 0}', add_special_tokens=False)["input_ids"]
    text_b = tiny_vlm.build_batch([s_text])
    text_ids = text_b["input_ids"][0][text_b["loss_mask"][0].bool()].tolist()
    s_exact = Sample(s_text.messages, s_text.train_msg_idx, 1.0, gen_ids=text_ids)
    ex = tiny_vlm.build_batch([s_exact])
    assert ex["input_ids"][0][ex["loss_mask"][0].bool()].tolist() == text_ids        # same tokens as the text path
    assert not tiny_vlm.retok_differs(s_exact)
    # non-canonical: split the first token into its characters (same string, other ids)
    first = tok.decode([body[0]])
    pieces = [tok.convert_tokens_to_ids(c) for c in first] if len(first) > 1 else None
    if pieces and all(isinstance(p, int) and p != tok.unk_token_id for p in pieces):
        odd = pieces + body[1:] + [eos]
        b = tiny_vlm.build_batch([Sample(s_text.messages, s_text.train_msg_idx, 1.0, gen_ids=odd)])
        assert b["input_ids"][0][b["loss_mask"][0].bool()].tolist() == odd
        assert tiny_vlm.retok_differs(Sample(s_text.messages, s_text.train_msg_idx, 1.0, gen_ids=odd))
    cut = body[:3]                                                                   # no stop token generated
    b = tiny_vlm.build_batch([Sample(s_text.messages, s_text.train_msg_idx, 1.0, gen_ids=cut)])
    trained = b["input_ids"][0][b["loss_mask"][0].bool()].tolist()
    assert trained == cut and not set(trained) & tiny_vlm.stop_ids
    # mixed lengths in one batch: prompt + ids per row, padding untrained, log-probs finite
    b = tiny_vlm.build_batch([s_exact, Sample(s_text.messages, s_text.train_msg_idx, -1.0, gen_ids=cut)])
    logp, m = tiny_vlm.token_logprobs(b)
    assert int(b["loss_mask"].sum()) == len(text_ids) + len(cut) and torch.isfinite(logp[m.bool()]).all()


@pytest.mark.slow
def test_generated_image_token_id_is_not_counted_as_an_image(tiny_vlm, umaze, rng):
    """Regression: SmolVLM2 at T=1.2 sampled the <image> placeholder as a raw token id. Spliced into the batch, the
    forward pass counted 1025 image tokens for 1024 image features and raised. The id is replaced by padding and left
    untrained; the rest of the reply is trained as generated."""
    assert tiny_vlm.media_ids
    t = _traj_with_texts(umaze, rng, ['{"dx": 1, "dy": 0}'])
    t.advantage = 1.0
    s = to_samples([t], 1)[0]
    body = tiny_vlm.tokenizer('{"dx": 1, "dy": 0}', add_special_tokens=False)["input_ids"]
    img = next(iter(tiny_vlm.media_ids))
    gen = body[:2] + [img] + body[2:] + [sorted(tiny_vlm.stop_ids)[0]]
    b = tiny_vlm.build_batch([Sample(s.messages, s.train_msg_idx, 1.0, gen_ids=gen)])
    trained = b["input_ids"][0][b["loss_mask"][0].bool()].tolist()
    assert img not in trained and trained == [i for i in gen if i != img]
    logp, m = tiny_vlm.token_logprobs(b)                                   # raised before the fix
    assert torch.isfinite(logp[m.bool()]).all()


@pytest.mark.slow
def test_fp32_frozen_keeps_frozen_params_fp32():
    from conftest import TINY_VLM
    from vlmrl.policy import HFPolicy
    kw = dict(device="cpu", dtype="bfloat16", processor_kwargs={"do_image_splitting": False}, lora_rank=0,
              gradient_checkpointing=False, freeze=r"model[.]vision_model[.].*", fp32_master=True)
    for fp32_frozen, frozen_dtype in ((False, torch.bfloat16), (True, torch.float32)):
        pol = HFPolicy(TINY_VLM, fp32_frozen=fp32_frozen, **kw)
        frozen = {p.dtype for n, p in pol.model.named_parameters() if not p.requires_grad}
        trained = {p.dtype for n, p in pol.model.named_parameters() if p.requires_grad}
        assert frozen == {frozen_dtype} and trained == {torch.float32}


@pytest.mark.slow
def test_init_adapter_round_trip_and_mismatch(tmp_path):
    """init_adapter: a saved LoRA adapter loads into a fresh policy exactly; a different rank is rejected."""
    from conftest import TINY_VLM
    from vlmrl.policy import HFPolicy
    kw = dict(device="cpu", processor_kwargs={"do_image_splitting": False}, gradient_checkpointing=False)
    a = HFPolicy(TINY_VLM, lora_rank=4, **kw)
    with torch.no_grad():
        for n, p in a.model.named_parameters():
            if "lora_B" in n:
                p.add_(0.01)                                  # make the adapter non-trivial (B starts at 0)
    a.save(str(tmp_path / "ad"), processor=False)
    b = HFPolicy(TINY_VLM, lora_rank=4, **kw)
    b.load_adapter_weights(str(tmp_path / "ad"))
    pa = {n: p for n, p in a.model.named_parameters() if "lora_" in n}
    pb = {n: p for n, p in b.model.named_parameters() if "lora_" in n}
    assert pa.keys() == pb.keys() and all(torch.equal(pa[n], pb[n]) for n in pa)
    with pytest.raises(Exception):
        HFPolicy(TINY_VLM, lora_rank=8, **kw).load_adapter_weights(str(tmp_path / "ad"))
