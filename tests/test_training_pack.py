"""The training pack (training.pack): what is in it, which one fits the hardware, its state, loading it."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from modalytics.training import pack

ROOT = Path(__file__).resolve().parents[1]


def test_pack_json_is_the_lock_files():
    """resources/training/pack.json is what uv.lock says (packaging/build.py writes it; a stale one fails here)."""
    assert pack.load_spec() == pack.from_lock(ROOT / "uv.lock")


def test_contents():
    spec = pack.from_lock(ROOT / "uv.lock")
    assert spec["cuda"].startswith("cu") and spec["torch"][0].isdigit() and "+" not in spec["torch"]
    install = spec["install"]
    for name in ("torch", "torchvision", "segmentation-models-pytorch", "timm", "onnx"):
        assert name in install
    for name in ("numpy", "pillow", "safetensors"):              # the program carries them: never installed again
        assert name not in install and name in spec["provided"]
    reqs = pack.requirements(pack.variants(spec)["cuda"], spec)
    assert reqs[0] == f"torch=={spec['torch']}+{spec['cuda']}"
    assert pack.requirements(pack.variants(spec)["cpu"], spec)[0] == f"torch=={spec['torch']}+cpu"
    assert all("==" in r for r in reqs)                           # every version pinned


@pytest.mark.parametrize("hw, variant", [({"nvidia": ["NVIDIA RTX A2000 12GB"], "intel": []}, "cuda"),
                                         ({"nvidia": [], "intel": ["Intel(R) Arc(TM) A770"]}, "xpu"),
                                         ({"nvidia": [], "intel": []}, "cpu")])
def test_hardware_decides(hw, variant):
    assert pack.recommended(hw) == variant


@pytest.fixture
def no_torch(monkeypatch, tmp_path):
    """As in the installed program: PyTorch not in the environment, packs kept in a temporary folder."""
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "torch" else real(name, *a))
    monkeypatch.setenv("MODALYTICS_TRAINING", str(tmp_path))
    monkeypatch.delattr(sys, "_modalytics_pack", raising=False)
    path = list(sys.path)
    yield tmp_path
    sys.path[:] = path
    if hasattr(sys, "_modalytics_pack"):
        del sys._modalytics_pack


def _installed(folder: Path, variant: str, hw: dict) -> None:
    (folder / f"{variant}-2.11.0").mkdir()
    (folder / "current.json").write_text(json.dumps({"folder": f"{variant}-2.11.0", "variant": variant,
                                                     "torch": "2.11.0", "hardware": hw}), encoding="utf-8")


def test_states(no_torch):
    nvidia = {"nvidia": ["NVIDIA RTX A2000 12GB"], "intel": []}
    none = pack.status(nvidia)
    assert none["state"] == "none" and none["recommended"] == "cuda" and "2.6 GB" in none["message"]
    _installed(no_torch, "cpu", {"nvidia": [], "intel": []})
    changed = pack.status(nvidia)
    assert changed["state"] == "changed" and "NVIDIA RTX A2000" in changed["message"] and "Switch" in changed["message"]
    _installed(no_torch, "cuda", nvidia)
    assert pack.status(nvidia)["state"] == "ok"
    assert pack.status({"nvidia": ["NVIDIA RTX 4090"], "intel": []})["state"] == "ok"   # another NVIDIA: same pack


def test_activate(no_torch):
    assert pack.activate() == ""                                  # nothing installed
    _installed(no_torch, "cuda", {"nvidia": ["x"], "intel": []})
    folder = pack.activate()
    assert folder == str(no_torch / "cuda-2.11.0") and sys.path[-1] == folder
    assert pack.activate() == folder and sys.path.count(folder) == 1   # once


def test_from_the_source():
    """With the training extra, PyTorch is in the environment: no pack, nothing to offer."""
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch not installed here")
    assert pack.status()["state"] == "environment" and pack.activate() == ""
