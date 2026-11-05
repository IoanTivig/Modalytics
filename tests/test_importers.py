"""Importers: the image contracts (phase → radians), the importer library, the built-in readers
(on synthetic files, and on HoloPath's real files when present) and importers in profile bundles."""
import math
import os

import numpy as np
import pytest

import testdata
from modalytics.image_import import Image, import_file
from modalytics.image_import.contracts import ContractError, check, describe_source
from modalytics.image_import.registry import ImporterLibrary, load_script
from modalytics.image_import.run import ImportFailed
from modalytics.parameters.script_file import ImporterScript

QPI = testdata.QPI


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _lyncee(tmp_path, values, unit="rad"):
    h, w = values.shape
    body = "\n".join(" ".join(f"{v:.6f}" for v in row) for row in values)
    return _write(tmp_path / "area" / "1" / "Phase.txt",
                  f"h={h} w={w}\npixel size=3.14006e-007 m\ndata unit={unit}\n"
                  f"height conversion factor (-> m)=2.11517e-007\n{body}\n")


def _phasestudio(tmp_path, values, sep="\t", name="PHA_x.txt"):
    header = ("Soft : PhaseStudio v2502.0.4, Project name : P, Measurement name : m0, SNP : SID4-1791, "
              "Phase Unit : wv, Pixel Size (µm) : 24.00, Filters Used : None, Wavelength (nm) : 1000.00, "
              "WaveFront Scale : -1.00, Rc (mm) : 1.0")
    body = "\n".join(sep.join(f"{v:.4f}" for v in row) for row in values)
    return _write(tmp_path / name, header + "\n" + body + "\n")


# ---------------------------------------------------------------- contracts
@pytest.mark.parametrize("unit, factor", [("rad", 1.0), ("mrad", 1e-3), ("deg", math.pi / 180),
                                          ("wv", 2 * math.pi), ("waves", 2 * math.pi)])
def test_phase_contract_converts_to_radians(unit, factor):
    v = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = check(Image(v, 0.5, unit=unit, height_factor_nm=100.0), "phase")
    assert np.allclose(out.pixels, v * factor, rtol=1e-6) and out.pixels.dtype == np.float32
    assert out.unit == "rad" and out.metadata["unit_read"] == unit
    # The optical height (value × factor) is the same before and after
    assert np.isclose(out.height_factor_nm * out.pixels[1, 1], 100.0 * v[1, 1], rtol=1e-6)
    assert out.pixel_size_um == (0.5, 0.5)


@pytest.mark.parametrize("image, why", [
    (Image(np.zeros((4, 4)), 1.0, unit="furlongs"), "unknown phase unit"),
    (Image(np.zeros((4, 4)), 1.0, unit=""), "unknown phase unit"),
    (Image(np.zeros((4, 4, 3)), 1.0, unit="rad"), "one value per pixel"),
    (Image(np.zeros(16), 1.0, unit="rad"), "2D image"),
    (Image(np.zeros((4, 4)), 0.0, unit="rad"), "pixel size"),
    (Image(np.zeros((4, 4)), float("nan"), unit="rad"), "pixel size"),
])
def test_phase_contract_refuses(image, why):
    with pytest.raises(ContractError, match=why):
        check(image, "phase")


def test_brightfield_contract():
    rgba = np.full((4, 5, 4), 200, np.uint8)
    out = check(Image(rgba, (0.2, 0.3)), "brightfield")
    assert out.pixels.shape == (4, 5, 3) and out.pixel_size_um == (0.2, 0.3)          # alpha dropped
    out = check(Image(np.full((4, 4), 0.5, np.float32), 1.0), "brightfield")
    assert out.pixels.dtype == np.uint16 and out.pixels[0, 0] == 32768                 # 0–1 floats → 16 bit
    with pytest.raises(ContractError, match="8- or 16-bit"):
        check(Image(np.zeros((4, 4), np.int32), 1.0), "brightfield")
    with pytest.raises(ContractError, match="0–1"):
        check(Image(np.full((4, 4), 7.0), 1.0), "brightfield")
    with pytest.raises(ContractError, match="own readers"):
        check(Image(np.zeros((4, 4), np.uint8), 1.0), "wsi")


# ---------------------------------------------------------------- the library
def test_builtin_importers_load():
    lib = ImporterLibrary()
    assert set(lib.specs) == {"phase:lyncee_phase_txt", "phase:phasestudio_qpi", "brightfield:image_file"}
    assert not lib.errors
    assert lib.extensions() == [".bmp", ".csv", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".txt"]
    for spec in lib.specs.values():                  # built-ins are in the editor's canonical form
        text = open(spec.path, encoding="utf-8").read()
        assert ImporterScript.parse(text).render() == text


def test_each_file_goes_to_its_importer(tmp_path):
    lib = ImporterLibrary()
    lyn = _lyncee(tmp_path, np.zeros((3, 3)))
    qpi = _phasestudio(tmp_path, np.zeros((3, 3)))
    other = _write(tmp_path / "notes.txt", "hello\n")
    assert lib.candidates(str(lyn))[0][0].key == "lyncee_phase_txt"
    assert lib.candidates(str(qpi))[0][0].key == "phasestudio_qpi"
    assert lib.candidates(str(other)) == []
    with pytest.raises(ImportFailed, match="notes.txt: not recognised by the importers for .txt files"):
        import_file(str(other), lib)
    with pytest.raises(ImportFailed, match="no importer reads .xyz files"):
        import_file(str(_write(tmp_path / "a.xyz", "")), lib)


def test_lyncee_unit_line_is_honoured(tmp_path):
    v = np.array([[0, 90], [180, 270]], np.float32)
    node = import_file(str(_lyncee(tmp_path, v, unit="deg")))[0]
    assert np.allclose(node.source.read_tile(0, 0, 0), np.deg2rad(v), atol=1e-6)
    assert node.metadata["unit_read"] == "deg" and node.name == "area/1 Phase"
    assert np.isclose(node.metadata["phase_to_height_nm"], 211.517 / (math.pi / 180))
    assert describe_source(node.metadata) == ("Values in rad, converted from deg × 0.01745 · pixel size from the "
                                              "file · read by Lyncée Tec DHM (Phase.txt)")
    assert node.pixel_size_um == (0.314006, 0.314006) and node.metadata["pixel_size_source"] == "file"


def test_phasestudio_waves_and_pixel_size(tmp_path):
    v = np.array([[0.0, 0.25], [0.5, -1.0]], np.float32)
    for sep, name in (("\t", "a.txt"), (";", "a.csv")):
        node = import_file(str(_phasestudio(tmp_path, v, sep, name)), pixel_size_um=0.6)[0]
        assert np.allclose(node.source.read_tile(0, 0, 0), v * 2 * np.pi, atol=1e-5)
        assert np.allclose(node.pixel_size_um, (0.6, 0.6)) and node.metadata["pixel_size_source"] == "given"
        assert node.metadata["camera_pitch_um"] == 24.0
        assert np.isclose(node.metadata["phase_to_height_nm"], 1000 / (2 * np.pi))   # nm per rad
    spec = ImporterLibrary().get("phase:phasestudio_qpi")
    assert spec.to_ask(str(tmp_path / "a.txt")) == []                     # the pixel size is not an option
    assert "camera pitch: 24.00 µm" in spec.hint_for(str(tmp_path / "a.txt"))


def test_pixel_size_rules(tmp_path):
    """The file's value, else yours; a value you changed replaces the file's; unknown is refused."""
    from modalytics.image_import import defaults
    qpi = str(_phasestudio(tmp_path, np.zeros((3, 3))))
    lyn = str(_lyncee(tmp_path, np.zeros((3, 3))))
    # 1. PhaseStudio files do not give it: refused until it is known
    assert defaults.asks("phase:phasestudio_qpi")                          # the first import asks
    with pytest.raises(ImportFailed, match="does not give the pixel size"):
        import_file(qpi)
    defaults.save_pixel("phase:phasestudio_qpi", 0.6)
    defaults.set_asks("phase:phasestudio_qpi", False)                      # "don't ask again"
    node = import_file(qpi)[0]
    assert node.pixel_size_um == (0.6, 0.6) and node.metadata["pixel_size_source"] == "your default"
    assert not defaults.asks("phase:phasestudio_qpi")
    # 2. Lyncée files give it: the file's value, unless you replaced it
    defaults.save_pixel("phase:lyncee_phase_txt", 0.5, prefer_file=True)
    assert import_file(lyn)[0].pixel_size_um == (0.314006, 0.314006)
    defaults.save_pixel("phase:lyncee_phase_txt", prefer_file=False)
    node = import_file(lyn)[0]
    assert node.pixel_size_um == (0.5, 0.5) and node.metadata["pixel_size_source"] == "your value"
    assert np.isclose(node.metadata["pixel_size_in_file"], 0.314006)
    assert defaults.resolve_pixel("wsi:zarr", 0.25) == (0.25, "file") and defaults.resolve_pixel("x:y", None) == (None, "")


def test_user_importer_replaces_builtin_and_errors_are_listed(tmp_path):
    user = tmp_path / "user"
    builtin = ImporterLibrary().get("phase:phasestudio_qpi")
    text = open(builtin.path, encoding="utf-8").read().replace("NAME = 'Phasics", "NAME = 'My Phasics")
    _write(user / "phase" / "phasestudio_qpi.py", text)
    _write(user / "phase" / "broken.py", "KEY = 'broken'\nIMAGE_TYPE = 'phase'\nEXTENSIONS = ['.x']\n")
    _write(user / "wsi" / "slides.py", "KEY = 'slides'\nIMAGE_TYPE = 'wsi'\nEXTENSIONS = ['.svs']\n"
                                       "def can_read(p): return 1\ndef read(p, o): pass\n")
    lib = ImporterLibrary(user=user)
    assert lib.get("phase:phasestudio_qpi").name.startswith("My Phasics")
    assert lib.overridden == ["phase:phasestudio_qpi"]
    errors = {os.path.basename(p): why for p, why in lib.errors.items()}
    assert "can_read" in errors["broken.py"] and "Modalytics itself" in errors["slides.py"]


def test_a_script_returning_a_bad_image_is_reported(tmp_path):
    f = _write(tmp_path / "bad.py", "from modalytics.image_import import Image\nimport numpy as np\n"
               "KEY = 'bad'\nIMAGE_TYPE = 'phase'\nEXTENSIONS = ['.bad']\n"
               "def can_read(p): return 1.0\n"
               "def read(p, o): return Image(np.zeros((4, 4)), 1.0, unit='parsecs')\n")
    spec = load_script(f)
    data = _write(tmp_path / "x.bad", "")
    with pytest.raises(ImportFailed, match="unknown phase unit 'parsecs'"):
        import_file(str(data), importer=spec)


def test_new_importer_templates_load_and_read(tmp_path):
    import cv2

    from modalytics.ui.settings.importers import TEMPLATE, TEMPLATES
    table = _write(tmp_path / "t.dat", "1 2 3\n4 5 6\n")
    png = tmp_path / "t.png"
    cv2.imwrite(str(png), np.full((5, 6, 3), (10, 20, 30), np.uint8))           # BGR on disk
    for image_type, sample in (("phase", table), ("brightfield", png)):
        f = _write(tmp_path / f"{image_type}.py", TEMPLATE.format(
            name="Mine", key="mine", image_type=image_type, extensions=[sample.suffix], description="test",
            type_name=image_type, read=TEMPLATES[image_type]))
        spec = load_script(f)
        node = import_file(str(sample), importer=spec, pixel_size_um=0.5)[0]
        assert node.modality == image_type and node.pixel_size_um == (0.5, 0.5)
        if image_type == "brightfield":
            assert tuple(node.source.read_tile(0, 0, 0)[0, 0]) == (30, 20, 10)       # RGB


# ---------------------------------------------------------------- HoloPath's real files
def test_real_phasestudio_txt_and_csv_agree():
    txt = os.path.join(QPI, "txt", "PHA_ep_capture_0000.txt")
    csv = os.path.join(QPI, "csv", "PHA_ep_capture_0000.csv")
    if not (os.path.exists(txt) and os.path.exists(csv)):
        pytest.skip("PhaseStudio test data not found")
    a = import_file(txt, pixel_size_um=0.6)[0]
    b = import_file(csv, pixel_size_um=0.6)[0]
    va, vb = a.source.read_tile(0, 0, 0), b.source.read_tile(0, 0, 0)
    assert va.shape == (360, 416) and np.array_equal(va, vb)
    raw = np.loadtxt(txt, skiprows=1, dtype=np.float32)
    assert np.allclose(va, raw * 2 * np.pi, atol=1e-5)                           # waves → radians


# ---------------------------------------------------------------- profile bundles
def test_bundle_carries_importers(tmp_path):
    from modalytics.profiles import bundle
    from modalytics.profiles.library import ProfileLibrary, default_profile
    profile, _pid = default_profile()
    mine = _write(tmp_path / "mine" / "phase" / "my_qpi.py",
                  open(ImporterLibrary().get("phase:phasestudio_qpi").path, encoding="utf-8").read()
                  .replace("KEY = 'phasestudio_qpi'", "KEY = 'my_qpi'"))
    out = tmp_path / "p.mprofile"
    bundle.export_profile(profile, out, {}, {}, {"phase:my_qpi": mine})
    importers_dir = tmp_path / "theirs"
    r = bundle.import_profile(out, ProfileLibrary(), tmp_path / "m", tmp_path / "par", {}, {},
                              importers_dir, {})
    assert "phase:my_qpi" in r.added and (importers_dir / "phase" / "my_qpi.py").read_bytes() == mine.read_bytes()
    r2 = bundle.import_profile(out, ProfileLibrary(), tmp_path / "m", tmp_path / "par", {}, {},
                               importers_dir, {"phase:my_qpi": importers_dir / "phase" / "my_qpi.py"})
    assert "phase:my_qpi" in r2.reused
    assert "phase:my_qpi" in ImporterLibrary(user=importers_dir).specs


def test_importers_may_produce_map_types(tmp_path):
    """Maps another program computed (OPD, OD, height…) can be imported directly: checked against their type's
    contract (unit converted to the standard one, value rules), drawn like the same type made by processing,
    and measured by that type's parameters."""
    import numpy as np
    from modalytics.image_import.contracts import ContractError, Image, check, to_node
    from modalytics.parameters import engine
    from modalytics.parameters.registry import ParameterLibrary
    from shapely.geometry import box
    opd_um = np.full((40, 40), 0.25, np.float32)                       # 0.25 µm = 250 nm
    ok = check(Image(opd_um, 0.5, unit="µm"), "opd")
    assert ok.unit == "nm" and np.allclose(ok.pixels, 250.0) and ok.metadata["unit_read"] == "µm"
    node = to_node(ok, "opd", str(tmp_path / "opd.txt"), "Test")
    assert node.modality == "opd" and node.display.colormap == "viridis"
    lib = ParameterLibrary()
    values, problems = engine.measure_object(box(2, 2, 12, 12), node, engine.select(
        lib, {"object:opd:mean": {}, "object:phase:mean": {}}, "object")[0])
    assert values["object:opd:mean"] == pytest.approx(250.0)
    assert problems["object:phase:mean"] == "not for opd images"          # no original under it, no "Process" hint
    with pytest.raises(ContractError, match="≥ 0"):
        check(Image(np.full((10, 10), -0.5, np.float32), 1.0, unit="OD"), "od")
    with pytest.raises(ContractError, match="unknown height unit"):
        check(Image(np.ones((10, 10), np.float32), 1.0, unit="mm"), "height")
    with pytest.raises(ContractError, match="Modalytics' own readers"):
        check(Image(np.ones((10, 10, 3), np.uint8), 1.0), "wsi")
