import copy

import pytest
import torch

from conftest import layout
from ofai.model.config import ModelConfig
from ofai.model.planes import to_tensors
from ofai.model.policy import Policy, parameter_count

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def setup(synth_dump):
    obs, _, _, rows = synth_dump
    torch.manual_seed(0)
    lay = layout()
    pol = Policy(lay, ModelConfig(amp=False)).to(DEV).eval()
    t = to_tensors(obs, DEV)
    with torch.no_grad():
        h, ctx, _ = pol.core(t, pol.initial_state(rows, DEV))
    return pol, lay, t, h, ctx, rows


def test_param_count_between_10m_and_15m():
    n = parameter_count(Policy(layout(), ModelConfig()))
    assert 10_000_000 <= n <= 15_000_000, n


def _repeat(h, ctx, n):
    idx = torch.arange(n, device=h.device) % h.shape[0]
    return h[idx], ctx.rows(idx)


def test_masked_fields_never_sampled(setup):
    pol, lay, _, h0, ctx0, _ = setup
    h, ctx = _repeat(h0, ctx0, 250)
    attack = lay.at("ATTACK")
    ctx.masks["type"] = torch.zeros_like(ctx.masks["type"])
    ctx.masks["type"][:, attack] = True
    pm = torch.zeros_like(ctx.masks["player"])
    pm[:, attack, 5] = True
    ctx.masks["player"] = pm
    gen = torch.Generator(DEV).manual_seed(1)
    for _ in range(40):  # 10k samples
        acts, _, _ = pol.sample(h, ctx, generator=gen)
        assert (acts[:, :, lay.af("type")] == attack).all()
        assert (acts[:, :, lay.af("player")] == 5).all()


def test_log_prob_of_sampled_actions_equals_returned_log_prob(setup):
    pol, _, _, h, ctx, _ = setup
    h, ctx = _repeat(h, ctx, 256)
    acts, lp, _ = pol.sample(h, ctx, generator=torch.Generator(DEV).manual_seed(2))
    with torch.no_grad():
        lp2, ent, info = pol.log_prob(h, ctx, acts)
    assert torch.allclose(lp, lp2, atol=1e-4), (lp - lp2).abs().max()
    assert int(info["rescued"]) == 0
    assert torch.isfinite(ent).all()


def test_fields_unused_by_a_type_do_not_enter_log_prob(setup):
    pol, lay, _, h, ctx, _ = setup
    n = h.shape[0]
    acts = torch.full((n, lay.K, lay.A), -1, dtype=torch.long, device=DEV)
    acts[:, 0, lay.af("type")] = lay.at("ALLIANCE_REQUEST")
    acts[:, 0, lay.af("player")] = 33
    acts[:, 1, lay.af("type")] = lay.at("STOP")
    with torch.no_grad():
        base, _, _ = pol.log_prob(h, ctx, acts)
        noisy = acts.clone()
        for name, v in (("troops", 7), ("build", 2), ("mmCell", 999), ("fine", 3), ("entity", 4)):
            noisy[:, 0, lay.af(name)] = v
        noisy[:, 2:, lay.af("type")] = lay.at("ATTACK")  # after STOP nothing counts
        other, _, _ = pol.log_prob(h, ctx, noisy)
    assert torch.allclose(base, other)


def test_fresh_resets_lstm_state(setup):
    pol, _, _, _, _, _ = setup
    emb = torch.randn(3, pol.cfg.width, device=DEV)
    state = (torch.randn(3, pol.cfg.lstm, device=DEV), torch.randn(3, pol.cfg.lstm, device=DEV))
    with torch.no_grad():
        h_fresh, _ = pol.lstm_step(emb, state, torch.tensor([1, 0, 1], device=DEV))
        h_zero, _ = pol.lstm_step(emb, pol.initial_state(3, DEV))
        h_kept, _ = pol.lstm_step(emb, state)
    assert torch.allclose(h_fresh[0], h_zero[0]) and torch.allclose(h_fresh[2], h_zero[2])
    assert torch.allclose(h_fresh[1], h_kept[1])


def test_no_nan_with_all_but_stop_masked(setup):
    pol, lay, _, h, ctx, n = setup
    ctx = copy.copy(ctx)
    ctx.masks = dict(ctx.masks)
    ctx.masks["type"] = torch.zeros_like(ctx.masks["type"])
    ctx.masks["type"][:, lay.at("STOP")] = True
    acts, lp, _ = pol.sample(h, ctx)
    assert (acts[:, 0, lay.af("type")] == lay.at("STOP")).all()
    assert (acts[:, 1:] == -1).all()
    assert torch.isfinite(lp).all()
    # A teacher label the mask forbids is rescued, not -inf.
    bad = torch.full_like(acts, -1)
    bad[:, 0, lay.af("type")] = lay.at("ATTACK")
    bad[:, 0, lay.af("player")] = 0
    bad[:, 0, lay.af("troops")] = 3
    bad[:, 1, lay.af("type")] = lay.at("STOP")
    with torch.no_grad():
        lp2, _, info = pol.log_prob(h, ctx, bad)
    assert torch.isfinite(lp2).all()
    assert int(info["rescued"]) >= n


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cpu_and_cuda_give_the_same_log_prob(synth_dump):
    obs, _, _, rows = synth_dump
    lay = layout()
    torch.manual_seed(3)
    cpu = Policy(lay, ModelConfig(amp=False)).eval()
    gpu = copy.deepcopy(cpu).cuda()
    with torch.no_grad():
        h, ctx, _ = cpu.core(to_tensors(obs, "cpu"), cpu.initial_state(rows, "cpu"))
        acts, _, _ = cpu.sample(h, ctx, generator=torch.Generator("cpu").manual_seed(4))
        lp_cpu, _, _ = cpu.log_prob(h, ctx, acts)
        hg, ctxg, _ = gpu.core(to_tensors(obs, "cuda"), gpu.initial_state(rows, "cuda"))
        lp_gpu, _, _ = gpu.log_prob(hg, ctxg, acts.cuda())
    assert torch.allclose(lp_cpu, lp_gpu.cpu(), atol=1e-4), (lp_cpu - lp_gpu.cpu()).abs().max()
