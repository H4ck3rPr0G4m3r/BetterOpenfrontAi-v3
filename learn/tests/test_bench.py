import gzip
import json

import pytest
import torch

from conftest import SMALL_MAP, layout, pool_kwargs, tiny_config
from ofai import checkpoints as ck
from ofai.bench import suites as S
from ofai.bench.mind import read_mind
from ofai.bench.runner import aggregate, land_curve_gap, run_bench
from ofai.data.index import Index
from ofai.model.policy import Policy


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(S.paths, "DATA", tmp_path / "data")
    monkeypatch.setattr(S.paths, "CHECKPOINTS", tmp_path / "checkpoints")
    return tmp_path


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    torch.manual_seed(0)
    cfg = tiny_config()
    lay = layout()
    pol = Policy(lay, cfg)
    p = tmp_path_factory.mktemp("ck") / "tiny.pt"
    ck.save_checkpoint(p, {"model": pol.state_dict(), "config": cfg.to_dict(), "layout": lay.raw, "step": 0})
    return p


def test_bench_file_is_frozen(dirs, tiny_ckpt):
    ix = Index(dirs / "i.sqlite")
    out = S.freeze_bench_v1(ix, tiny_ckpt, commit="SYNTHETIC")
    d = json.loads(out.read_text())
    assert d["proxy"]["sha256"] and S.proxy_path().exists()
    with pytest.raises(FileExistsError):
        S.freeze_bench_v1(ix, tiny_ckpt, commit="SYNTHETIC")


def test_seeds_and_maps_are_fixed(dirs):
    ix = Index(dirs / "i.sqlite")
    a = S.bench_definition(ix, "c")
    b = S.bench_definition(ix, "c")
    assert a == b
    maps = [m["map"] for m in a["maps"]]
    assert maps[:4] == ["World", "Onion", "Sol", "Las Vegas Strip"] and len(maps) == 12 and len(set(maps)) == 12
    assert all(m["seeds"] == [f"bench-v1:{m['map']}:{i}" for i in range(5)] for m in a["maps"])


def test_results_aggregate_rates():
    rs = [
        {"terminal": "death", "ticks": 600 * 10, "placement": 9, "landCurve": [0.1, 0.2]},
        {"terminal": "tick_cap", "ticks": 600 * 30, "placement": 2, "landCurve": [0.1, 0.3, 0.5]},
        {"terminal": "win", "ticks": 600 * 25, "placement": 1, "landCurve": [0.2]},
        {"terminal": "sim_error", "ticks": 0, "placement": 0, "landCurve": [], "map": "X", "seed": "s", "error": "boom"},
    ]
    a = aggregate(rs)
    assert a["games"] == 4 and len(a["simErrors"]) == 1
    assert a["survived20"] == pytest.approx(2 / 3) and a["wins"] == pytest.approx(1 / 3)
    assert a["top3"] == pytest.approx(2 / 3) and a["placementMedian"] == 2
    assert a["landCurve"] == pytest.approx([0.1, 0.25, 0.5])
    assert land_curve_gap([0.1, 0.2], [0.2, 0.2]) == pytest.approx(0.05)


def test_sim_error_is_recorded_and_excluded_and_ai_seats_carry_the_tag(dirs, tiny_ckpt):
    bench = {
        "bench": "bench-test", "commit": "SYNTHETIC", "latency": [1, 3], "suites": S.SUITES,
        "maps": [{"map": SMALL_MAP, "seeds": ["t:0"], "bots": 10, "nations": "disabled"},
                 {"map": "NoSuchMap", "seeds": ["t:1"], "bots": 0, "nations": "disabled"}],
    }
    out = run_bench(tiny_ckpt, ["solo_medium"], workers=1, games_per_worker=2, bench=bench, out_dir=dirs / "run",
                    minutes=0.5, pool_kwargs=pool_kwargs(), device="cpu")
    s = out["suites"]["solo_medium"]
    assert s["games"] == 2
    assert [e["map"] for e in s["simErrors"]] == ["NoSuchMap"]
    assert "survived20" in s
    assert json.loads((dirs / "run" / "results.json").read_text())["suites"]["solo_medium"]["games"] == 2
    recs = list((dirs / "run" / "replays").glob("*.json.gz"))
    recs = [r for r in recs if not r.name.endswith(".mind.jsonl.gz")]
    assert len(recs) == 1
    with gzip.open(recs[0], "rt") as f:
        rec = json.load(f)
    assert [p["clanTag"] for p in rec["info"]["players"]] == ["AI"]
    assert rec["info"]["players"][0]["username"] == "ofai"
    mind = read_mind(next((dirs / "run" / "replays").glob("*.mind.jsonl.gz")))
    assert mind and {"turn", "clientID", "value", "actions", "top"} <= set(mind[0])
