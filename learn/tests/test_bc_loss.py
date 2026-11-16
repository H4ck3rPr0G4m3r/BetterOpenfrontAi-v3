import pytest
import torch

from conftest import layout, tiny_config
from ofai.bc.loss import bc_loss
from ofai.bc.weights import placement_value, teacher_weight
from ofai.model.planes import to_tensors
from ofai.model.policy import Policy

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_weights_are_in_quarter_to_one():
    for elo in (None, 0, 1500, 2000, 9999):
        for v in (0.0, 0.3, 1.0, 2.0, -1.0):
            w = teacher_weight(elo, v, 1500, 2000)
            assert 0.25 <= w <= 1.0
    assert teacher_weight(2000, 1.0, 1500, 2000) == 1.0
    assert teacher_weight(1000, 0.0, 1500, 2000) == 0.25
    assert placement_value(1, 50) == 1.0 and placement_value(50, 50) == 0.0 and placement_value(None, 3) == 0.5


@pytest.fixture(scope="module")
def batch(synth_dump):
    obs, _, _, rows = synth_dump
    torch.manual_seed(0)
    lay = layout()
    pol = Policy(lay, tiny_config()).to(DEV)
    t = to_tensors(obs, DEV)
    h, ctx, _ = pol.core(t, pol.initial_state(rows, DEV))
    return pol, lay, t, h, ctx, rows


def _labels(lay, rows, stop_rows):
    a = torch.full((rows, lay.K, lay.A), -1, dtype=torch.long, device=DEV)
    a[:, 0, lay.af("type")] = lay.at("ALLIANCE_REQUEST")
    a[:, 0, lay.af("player")] = 33
    a[:, 1, lay.af("type")] = lay.at("STOP")
    for r in stop_rows:
        a[r, 0, lay.af("type")] = lay.at("STOP")
        a[r, 0, lay.af("player")] = -1
        a[r, 1, lay.af("type")] = -1
    return a


def test_stop_only_rows_are_down_weighted(batch):
    pol, lay, t, h, ctx, rows = batch
    labels = _labels(lay, rows, stop_rows=[0])
    w = torch.ones(rows, device=DEV)
    loss, m = bc_loss(pol, h, ctx, t, labels, w, torch.full((rows,), 0.5, device=DEV), torch.ones(rows, dtype=torch.bool, device=DEV),
                      stop_weight=0.3, value_coef=0.0)
    lp, _, _ = pol.log_prob(h, ctx, labels)
    rw = torch.ones(rows, device=DEV)
    rw[0] = 0.3
    assert torch.allclose(loss, (-(lp) * rw).sum() / rw.sum(), atol=1e-5)
    assert m["rows"] == rows and 0 < m["stop_rows"] < 1


def test_all_invalid_batch_gives_zero_loss_and_no_grad(batch):
    pol, lay, t, h, ctx, rows = batch
    loss, m = bc_loss(pol, h, ctx, t, _labels(lay, rows, []), torch.ones(rows, device=DEV), torch.zeros(rows, device=DEV),
                      torch.zeros(rows, dtype=torch.bool, device=DEV))
    assert float(loss) == 0.0 and not loss.requires_grad and m["skipped"]
    assert torch.isfinite(loss)


def test_perfect_logits_give_near_zero_loss(batch, monkeypatch):
    pol, lay, t, h, ctx, rows = batch
    real = pol.log_prob

    def perfect(hh, cc, aa):
        lp, ent, info = real(hh, cc, aa)
        return torch.zeros_like(lp), ent, info

    monkeypatch.setattr(pol, "log_prob", perfect)
    target = pol.value(h).detach()
    loss, m = bc_loss(pol, h, ctx, t, _labels(lay, rows, []), torch.ones(rows, device=DEV), target,
                      torch.ones(rows, dtype=torch.bool, device=DEV))
    assert float(loss) < 1e-6
    assert "type_top1" in m and "value_mae" in m
