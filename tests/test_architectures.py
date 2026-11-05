"""Segmentation architectures and encoder weights as libraries: built-ins, your own scripts, the ONNX check
(remembered per file), weights packages fitted to their encoder, downloads checked, a model trained from your own
weights recording them, and the licence warning when sharing it."""

from __future__ import annotations

import hashlib
import json
import zipfile

import pytest

torch = pytest.importorskip("torch")
smp = pytest.importorskip("segmentation_models_pytorch")

from modalytics.training import architectures as ar                  # noqa: E402
from modalytics.training import encoders as en                       # noqa: E402

OWN = '''
KEY = 'own_net'
NAME = 'My network'
ENCODER = 'resnet18'

def build(in_channels, classes, encoder_weights):
    import segmentation_models_pytorch as smp
    return smp.Unet('resnet18', encoder_weights=encoder_weights, in_channels=in_channels, classes=classes,
                    decoder_attention_type='scse')
'''
HALF = '''
KEY = 'half_size'
NAME = 'Half-size output'
ENCODER = 'resnet18'

def build(in_channels, classes, encoder_weights):
    import torch
    return torch.nn.Conv2d(in_channels, classes, 3, stride=2, padding=1)     # (N, C, H/2, W/2): breaks the contract
'''


def test_builtin_library_user_scripts_and_errors(tmp_path):
    user = tmp_path / "architectures"
    user.mkdir()
    lib = ar.ArchitectureLibrary(user=user)
    assert list(lib.specs)[:4] == lib.recommended() == ["unet_resnet34", "deeplabv3p_resnet50", "unet_convnext_tiny",
                                                         "unet_efficientnet_b0"]
    assert len(lib.specs) == 6 and not lib.errors
    assert all(s.references for s in lib.specs.values())                     # every built-in cites its parts
    (user / "own.py").write_text(OWN, encoding="utf-8")
    (user / "bad.py").write_text("KEY = 'bad'\nDECODER = 'NoSuchNet'\nENCODER = 'resnet18'\n", encoding="utf-8")
    (user / "unet_resnet18.py").write_text("KEY = 'unet_resnet18'\nNAME = 'Mine'\nDECODER = 'FPN'\n"
                                           "ENCODER = 'resnet18'\n", encoding="utf-8")
    lib.reload()
    assert lib.get("own_net").custom and lib.get("own_net").encoder == "resnet18"
    assert lib.get("unet_resnet18").decoder == "FPN" and lib.overridden == ["unet_resnet18"]   # yours replaces it
    assert "DECODER must be one of" in lib.problems["bad.py"]


def test_check_passes_fails_with_a_reason_and_is_remembered(tmp_path):
    user = tmp_path / "architectures"
    user.mkdir()
    (user / "own.py").write_text(OWN, encoding="utf-8")
    (user / "half.py").write_text(HALF, encoding="utf-8")
    lib = ar.ArchitectureLibrary(user=user)
    own = lib.get("own_net")
    assert lib.checked(own) is None
    ok = lib.check(own)
    assert ok["ok"] and ok["max_diff"] < ar.TOLERANCE and ok["params_m"] > 10 and ok["onnx_mb"] > 10
    assert lib.checked(own) == ok and lib.checked(own, in_channels=3) is None   # per channel count
    bad = lib.check(lib.get("half_size"))
    assert not bad["ok"] and "shape" in bad["why"]
    (user / "own.py").write_text(OWN + "\n# changed\n", encoding="utf-8")     # another file: checked again
    lib.reload()
    assert lib.checked(lib.get("own_net")) is None


def test_package_record_rebuilds_library_parts_but_not_an_own_network(tmp_path):
    lib = ar.ArchitectureLibrary(user=tmp_path / "architectures")

    class Pkg:
        manifest = {"source": {"architecture_id": "gone", "architecture": {
            "key": "gone", "name": "Gone", "decoder": "FPN", "encoder": "resnet18", "decoder_options": {},
            "custom": False}}}
    spec = lib.for_package(Pkg)
    assert spec.decoder == "FPN" and spec.source == "package"
    assert ar.build(spec, 1)(torch.zeros(1, 1, 64, 64)).shape == (1, 1, 64, 64)
    Pkg.manifest["source"]["architecture"]["custom"] = True
    with pytest.raises(ValueError, match="network of its own"):
        lib.for_package(Pkg)


def _resnet18_weights(path, seed=0):
    torch.manual_seed(seed)
    enc = smp.encoders.get_encoder("resnet18", in_channels=3, weights=None)
    state = {"state_dict": {"module." + k: v for k, v in enc.state_dict().items()}}   # wrapped, as files often are
    torch.save(state, path)
    return enc.state_dict()


def test_encoder_library_import_fit_and_channels(tmp_path):
    lib = en.EncoderLibrary(user=tmp_path / "encoders")
    assert list(lib.specs)[0] == "imagenet" and {"micronet", "imagenet_micronet"} <= set(lib.specs)
    assert [s.key for s in lib.for_encoder("tu-convnext_tiny")] == ["imagenet"]             # MicroNet has none
    assert {s.key for s in lib.for_encoder("resnet34", "phase")} == {"imagenet", "micronet", "imagenet_micronet"}
    original = _resnet18_weights(tmp_path / "mine.pth")
    spec = en.import_weights(tmp_path / "mine.pth", "resnet18", "Phase SSL", image_types=["phase"],
                             licence="research only", redistributable=False, library=lib)
    assert spec.kind == "package" and spec.fits("resnet18", "phase") and not spec.fits("resnet18", "wsi")
    assert not spec.fits("resnet34")
    # Loaded into a one-channel U-Net: everything matched, the first convolution summed over colour
    model = ar.build(ar.ArchitectureLibrary(user=tmp_path / "a").get("unet_resnet18"), 1)
    report = en.load_into(model, en.state_dict(spec, "resnet18", lib))
    assert report["matched"] == report["total"] and report["adapted"] == ["conv1.weight"]
    assert torch.allclose(model.encoder.conv1.weight, original["conv1.weight"].sum(1, keepdim=True))
    assert torch.allclose(model.encoder.layer4[1].conv2.weight, original["layer4.1.conv2.weight"])
    # Weights of another encoder are refused, and a key is not taken twice
    with pytest.raises(ValueError, match="another encoder"):
        en.import_weights(tmp_path / "mine.pth", "efficientnet-b0", "Wrong", library=lib)
    with pytest.raises(ValueError, match="already"):
        en.import_weights(tmp_path / "mine.pth", "resnet18", "Phase SSL", library=lib)


def test_download_is_checked_and_kept(tmp_path):
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    _resnet18_weights(tmp_path / "remote.pth")
    data = (tmp_path / "remote.pth").read_bytes()
    entry = {"key": "far", "name": "Far away", "kind": "download", "licence": "MIT",
             "files": {"resnet18": {"url": (tmp_path / "remote.pth").as_uri(), "size": len(data),
                                    "sha256": hashlib.sha256(data).hexdigest()}}}
    (builtin / "far.json").write_text(json.dumps(entry), encoding="utf-8")
    lib = en.EncoderLibrary(builtin=builtin, user=tmp_path / "encoders")
    seen = []
    state = en.state_dict(lib.get("far"), "resnet18", lib, progress=seen.append)
    assert "conv1.weight" in state and seen[-1] == 1.0
    assert (lib.cache_dir() / "far" / "remote.pth").exists()                  # fetched once, then kept
    entry["key"], entry["files"]["resnet18"]["sha256"] = "tampered", "0" * 64
    (builtin / "tampered.json").write_text(json.dumps(entry), encoding="utf-8")
    lib.reload()
    with pytest.raises(ValueError, match="SHA-256"):
        en.state_dict(lib.get("tampered"), "resnet18", lib)


def test_trained_from_your_weights_records_them_and_sharing_warns(tmp_path):
    from synthetic_projects import PX, graded_project
    from modalytics import paths
    from modalytics.profiles import bundle
    from modalytics.profiles.library import ProfileLibrary
    from modalytics.services.segmentation import load_package
    from modalytics.training import seg_train as st
    _resnet18_weights(tmp_path / "mine.pth")
    en.import_weights(tmp_path / "mine.pth", "resnet18", "Phase SSL", key="phase_ssl", licence="research only",
                      redistributable=False)
    projects = [graded_project(str(tmp_path / f"p{i}"), seed=i, n_low=3, n_high=2) for i in range(2)]
    settings = st.Settings(start="phase_ssl", arch_id="unet_resnet18", modality="phase", um_per_px=PX, epochs=1,
                           steps=1, batch_size=2, device="cpu")
    res = st.train(settings, projects[:1], projects[1:], "Gland", str(paths.user_dir("models") / "mine.mmodel"))
    src = load_package(res.path).manifest["source"]
    assert src["encoder_weights"]["key"] == "phase_ssl" and src["encoder_weights"]["redistributable"] is False
    assert src["architecture"] == {"key": "unet_resnet18", "name": "U-Net (ResNet-18)", "decoder": "Unet",
                                   "encoder": "resnet18", "decoder_options": {}, "custom": False}
    with pytest.raises(ValueError, match="has no weights"):              # MicroNet has none for ConvNeXt
        st.train(st.Settings(start="micronet", arch_id="unet_convnext_tiny", device="cpu"), projects, [], "Gland",
                 str(tmp_path / "x.mmodel"))
    # Sharing a profile that uses it says the licence travels with it
    profile = ProfileLibrary().get("colonic_tissue")
    profile.segmentation = {"Gland": {"phase": "mine"}}
    warnings = bundle.export_profile(profile, tmp_path / "shared", {"mine": res.path}, {})
    assert any("restricted licence" in w and "research only" in w for w in warnings)
    with zipfile.ZipFile(tmp_path / "shared.mprofile") as z:
        assert "models/mine.mmodel" in z.namelist()


def test_encoder_catalogue_says_which_have_imagenet_and_orders_the_choices(tmp_path):
    cat = en.catalogue()
    assert cat["resnet34"]["imagenet"] and cat["tu-convnext_tiny"]["imagenet"]
    assert not cat["resnext101_32x48d"]["imagenet"]                         # Instagram weights only
    lib = en.EncoderLibrary(user=tmp_path / "encoders")
    assert not lib.get("imagenet").fits("resnext101_32x48d") and lib.get("imagenet").fits("resnet34")
    names = [n for n, _s in en.choices(lib)]
    with_micronet = {"resnet18", "resnet34", "resnet50", "efficientnet-b0", "efficientnet-b3"}
    assert set(names[:5]) == with_micronet                                  # beyond ImageNet first
    assert names.index("resnet101") < names.index("tu-convnext_tiny") < names.index("resnext101_32x48d")
    assert dict(en.choices(lib))["resnext101_32x48d"] == []                 # random only
    known, close = en.known_encoder("resnet43")
    assert not known and "resnet34" in close and en.known_encoder("tu-convnext_tiny") == (True, [])


def test_training_refuses_imagenet_where_there_is_none(tmp_path):
    from modalytics import paths
    from modalytics.training import seg_train as st
    (paths.user_dir("architectures") / "dpn.py").write_text(
        "KEY = 'unet_dpn68b'\nNAME = 'U-Net (DPN-68b)'\nDECODER = 'Unet'\nENCODER = 'dpn68b'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no ImageNet weights"):
        st.train(st.Settings(start="imagenet", arch_id="unet_dpn68b", device="cpu"), [], [], "Gland",
                 str(tmp_path / "x.mmodel"))
