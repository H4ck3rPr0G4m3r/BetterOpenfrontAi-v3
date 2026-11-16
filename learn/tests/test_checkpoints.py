import json
import os
import time

import pytest
import torch

from ofai import checkpoints as ck


def test_roundtrip(tmp_path):
    p = tmp_path / "a.pt"
    ck.save_checkpoint(p, {"model": {"w": torch.arange(5)}, "step": 7})
    d = ck.load_checkpoint(p)
    assert d["step"] == 7 and torch.equal(d["model"]["w"], torch.arange(5))
    assert not (tmp_path / "a.pt.tmp").exists()


def test_truncated_checkpoint_is_skipped(tmp_path):
    old, new = tmp_path / "old.pt", tmp_path / "new.pt"
    ck.save_checkpoint(old, {"step": 1})
    time.sleep(0.05)
    ck.save_checkpoint(new, {"step": 2, "blob": torch.zeros(1000)})
    assert ck.latest_checkpoint(tmp_path) == new
    data = new.read_bytes()
    new.write_bytes(data[: len(data) // 2])  # power cut mid-write
    os.utime(new)
    assert ck.latest_checkpoint(tmp_path) == old
    with pytest.raises(ValueError):
        ck.load_checkpoint(new)


def test_missing_sha_is_skipped(tmp_path):
    ck.save_checkpoint(tmp_path / "a.pt", {"step": 1})
    time.sleep(0.05)
    torch.save({"step": 2}, tmp_path / "b.pt")  # written without the sha (crash before it)
    assert ck.latest_checkpoint(tmp_path) == tmp_path / "a.pt"


def test_register_is_atomic(tmp_path, monkeypatch):
    idx = tmp_path / "index.json"
    ck.register("bc-1", tmp_path / "x.pt", "bc", 10, index_path=idx)
    before = idx.read_text()

    def crash(src, dst):
        raise OSError("power cut")

    monkeypatch.setattr(ck.os, "replace", crash)
    with pytest.raises(OSError):
        ck.register("bc-2", tmp_path / "y.pt", "bc", 20, index_path=idx)
    assert idx.read_text() == before
    assert [e["name"] for e in json.loads(before)] == ["bc-1"]
