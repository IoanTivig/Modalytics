"""External scripts: the library, running one (the DHM phase extractor), cancelling, profiles and bundles."""

from __future__ import annotations

import os
import zipfile

import pytest

from modalytics.core.settings import Setting
from modalytics.scripts import registry as sr

MINE = '''KEY = 'count_files'
NAME = 'Count files'
CATEGORY = 'Checks'
OPTIONS = [{'key': 'folder', 'label': 'Folder', 'kind': 'folder', 'default': ''}]


def run(options, ctx):
    names = os.listdir(options['folder'])
    for i, n in enumerate(names, start=1):
        ctx.check()
        ctx.log(n)
        ctx.progress(i / len(names))
    return f"{len(names)} files"

import os
'''


def _dhm_tree(root):
    """sample01/z1/1/Phase.txt, sample01/z1/2/Phase.txt, sample01/z2/1/PHASE.TXT, and a stray file."""
    for area, image, name in (("z1", "1", "Phase.txt"), ("z1", "2", "Phase.txt"), ("z2", "1", "PHASE.TXT")):
        d = root / "sample01" / area / image
        d.mkdir(parents=True)
        (d / name).write_text(f"{area} {image}", encoding="utf-8")
        (d / "Amplitude.txt").write_text("not this", encoding="utf-8")
    return root / "sample01"


def test_library_builtin_user_and_broken(tmp_path):
    lib = sr.ScriptLibrary(user=tmp_path)
    assert "dhm_phase_extractor" in lib.specs and lib.get("dhm_phase_extractor").source == "builtin"
    (tmp_path / "count_files.py").write_text(MINE, encoding="utf-8")
    (tmp_path / "broken.py").write_text("KEY = 'broken'\n", encoding="utf-8")          # no run()
    lib.reload()
    mine = lib.get("count_files")
    assert mine.source == "user" and mine.group == "Checks" and mine.options[0].kind == "folder"
    assert any("run(options, ctx)" in why for why in lib.errors.values())
    assert list(lib.user_files()) == ["count_files"]
    assert [c for c, _s in lib.grouped()] == ["Checks", "Data preparation"]


def test_phase_extractor_renames_copies_and_offers_import(tmp_path):
    src = _dhm_tree(tmp_path / "db")
    out = tmp_path / "flat"
    out.mkdir()
    (out / "sample01-1-1.txt").write_text("already here", encoding="utf-8")        # a name taken: not overwritten
    spec = sr.ScriptLibrary(user=tmp_path / "none").get("dhm_phase_extractor")
    lines, steps = [], []
    ctx = sr.ScriptContext(lines.append, steps.append)
    summary = sr.run(spec, {"input": str(src), "output": str(out)}, ctx)
    assert summary.startswith("3 file(s) copied")
    names = sorted(os.listdir(out))
    assert names == ["sample01-1-1.txt", "sample01-1-1_2.txt", "sample01-1-2.txt", "sample01-2-1.TXT"]
    assert (out / "sample01-1-1.txt").read_text(encoding="utf-8") == "already here"
    assert sorted(os.path.basename(p) for p in ctx.to_import) == ["sample01-1-1_2.txt", "sample01-1-2.txt",
                                                                 "sample01-2-1.TXT"]
    assert steps[-1] == 1.0 and len(lines) == 3 and all(x.startswith("✅") for x in lines)
    # not offered for import when switched off; a missing input folder is said plainly
    ctx2 = sr.ScriptContext()
    sr.run(spec, {"input": str(src), "output": str(tmp_path / "flat2"), "offer_import": False}, ctx2)
    assert ctx2.to_import == []
    with pytest.raises(ValueError, match="input folder"):
        sr.run(spec, {"input": str(tmp_path / "nowhere"), "output": str(out)}, sr.ScriptContext())


def test_stop_and_import_a_folder(tmp_path):
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "count_files.py").write_text(MINE, encoding="utf-8")
    spec = sr.load_script(tmp_path / "s" / "count_files.py")
    with pytest.raises(sr.Cancelled):
        sr.run(spec, {"folder": str(tmp_path)}, sr.ScriptContext(cancelled=lambda: True))
    ctx = sr.ScriptContext()
    ctx.import_files(tmp_path / "s")
    assert [os.path.basename(p) for p in ctx.to_import] == ["count_files.py"]


def test_folder_and_text_settings():
    f = Setting.from_dict({"key": "out", "label": "Output", "kind": "folder", "default": ""})
    t = Setting.from_dict({"key": "name", "label": "Name", "kind": "text", "default": "Phase.txt"})
    assert f.clean(None) == "" and f.clean("C:/x") == "C:/x" and t.clean(5) == "5"


def test_profile_offers_scripts_and_the_bundle_carries_yours(tmp_path):
    from modalytics.profiles import Profile, bundle
    from modalytics.profiles.library import ProfileLibrary
    p = Profile("Lab", images=["phase"], scripts=["count_files", "dhm_phase_extractor"])
    assert Profile.from_yaml(p.to_yaml()).scripts == ["count_files", "dhm_phase_extractor"]
    mine = tmp_path / "count_files.py"
    mine.write_text(MINE, encoding="utf-8")
    out = tmp_path / "lab.mprofile"
    bundle.export_profile(p, out, {}, {}, scripts={"count_files": mine})
    assert "scripts/count_files.py" in zipfile.ZipFile(out).namelist()
    # elsewhere: a different script of the same id → added as _2, the profile offers that one
    here = tmp_path / "here"
    here.mkdir()
    theirs = here / "count_files.py"
    theirs.write_text(MINE.replace("files", "things"), encoding="utf-8")
    lib = ProfileLibrary()
    plan = bundle.inspect(out, lib, {"scripts": {"count_files": theirs}}, {"scripts": {"dhm_phase_extractor"}})
    assert ("scripts", "count_files", "different") in plan.items and plan.missing == []
    r = bundle.import_profile(out, lib, tmp_path / "m", tmp_path / "p", {}, {}, scripts_dir=here,
                              known_scripts={"count_files": theirs})
    got = lib.get(r.profile_id)
    assert got.scripts == ["count_files_2", "dhm_phase_extractor"]
    assert "KEY = 'count_files_2'" in (here / "count_files_2.py").read_text(encoding="utf-8")
