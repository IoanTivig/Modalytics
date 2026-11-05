"""
Build the Modalytics installer (Windows): ``Modalytics-<version>-setup.exe`` in ``build/installer``.

1. **Environment** — ``build/venv`` made from ``uv.lock`` (``uv sync --frozen``) with the ``build`` group
   (PyInstaller) and *without* the training libraries: PyTorch comes as the training pack, downloaded for each
   computer's hardware (``training.pack``); the pack's contents are written from ``uv.lock`` into
   ``resources/training/pack.json``. The development ``.venv`` is never touched.
2. **Program** — PyInstaller (``packaging/modalytics.spec``) → ``build/dist/Modalytics``: ``Modalytics.exe`` and
   its console twin ``modalytics-cli.exe``.
3. **Smoke test** — the built program, not the source: ``--version``, ``--gpu-info`` (GPU probes in child
   processes, ONNX Runtime's devices), ``--pip --version`` (Settings › Libraries), ``--self-test`` (measuring,
   segmentation on the GPU, the canvas, PDF, training with PyTorch: ``modalytics.selftest``), the help's pages
   and the shipped models in the bundle. A failure stops the build.
4. **Installer** — Inno Setup (``packaging/modalytics.iss``; ``ISCC.exe`` found on the PATH, in Program Files or
   in the per-user programs folder). When the program is larger than one setup file can hold, the setup is split
   into ``setup.exe`` + ``setup-N.bin`` files, which go together.

Usage::

    python packaging/build.py              # all four steps
    python packaging/build.py --skip-sync  # the environment is up to date
    python packaging/build.py --no-installer
    python packaging/build.py --only-installer   # step 4 on the program already in build/dist
    python packaging/build.py --test-pack        # also: the built program installs the training pack, trains
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
VENV = BUILD / "venv"
DIST = BUILD / "dist" / "Modalytics"
ONE_FILE_LIMIT = 3.5e9          # bytes of program a single setup file is trusted with (Inno Setup's limit is ~4 GB)


def version() -> str:
    """The version in pyproject.toml."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return re.search(r'^version\s*=\s*"([^"]+)"', text, re.M).group(1)


def numeric(v: str) -> str:
    """A version Windows accepts in a file's version info (four numbers): 0.1.0.dev0 → 0.1.0.0."""
    nums = [int(x) for x in re.findall(r"\d+", v.split("dev")[0].split("rc")[0])][:3]
    return ".".join(str(n) for n in (nums + [0, 0, 0])[:3]) + ".0"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("ℹ️ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=ROOT, **kw)


def iscc() -> Path | None:
    """Inno Setup's compiler."""
    found = shutil.which("ISCC") or shutil.which("iscc")
    if found:
        return Path(found)
    for base in (os.environ.get("ProgramFiles(x86)", ""), os.environ.get("ProgramFiles", ""),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")):
        p = Path(base) / "Inno Setup 6" / "ISCC.exe"
        if base and p.is_file():
            return p
    return None


def folder_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def smoke_test() -> list[str]:
    """The built program's own checks; returns the problems (empty: fine)."""
    problems = []
    cli = DIST / "modalytics-cli.exe"
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop("MODALYTICS_HOME", None)

    def out(*args, timeout=180):
        res = subprocess.run([str(cli), *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                             timeout=timeout, env=env)
        return res.returncode, (res.stdout + res.stderr).strip()

    code, text = out("--version")
    print(f"   --version: {text}")
    if code != 0 or version() not in text:
        problems.append(f"--version: {text[-300:]}")
    code, text = out("--gpu-info")
    print("   --gpu-info:\n      " + "\n      ".join(text.splitlines()[:14]))
    if code != 0 or "Traceback" in text:
        problems.append(f"--gpu-info: {text[-500:]}")
    code, text = out("--pip", "--version")
    print(f"   --pip --version: {text}")
    if code != 0 or not text.startswith("pip "):
        problems.append(f"--pip: {text[-300:]}")
    code, text = out("--self-test", timeout=900)          # the real work, in the built program
    print("   --self-test:\n      " + "\n      ".join(x for x in text.splitlines() if x[:1] in "✅❌"))
    if code != 0:
        problems.append("--self-test: " + "; ".join(x for x in text.splitlines() if x.startswith("❌")))
    internal = DIST / "_internal"
    models = list((internal / "models").glob("*.mmodel"))
    pages = list((internal / "modalytics" / "help").rglob("*.md"))
    scripts = list((internal / "modalytics" / "resources" / "parameters").rglob("*.py"))
    print(f"   bundled: {len(models)} models, {len(pages)} help pages, {len(scripts)} parameter scripts")
    if not models or not pages or not scripts:
        problems.append("models, help pages or parameter scripts missing from the bundle")
    if (internal / ".." / "docs").exists() or list(internal.rglob("docs/dev")):
        problems.append("the developers' guide (docs/dev) is in the bundle")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--skip-sync", action="store_true", help="use build/venv as it is")
    ap.add_argument("--no-installer", action="store_true", help="stop after the smoke test")
    ap.add_argument("--only-installer", action="store_true", help="only step 4, on the program in build/dist")
    ap.add_argument("--test-pack", action="store_true",
                    help="also download and install the training pack with the built program (a temporary folder) "
                         "and train with it")
    args = ap.parse_args()
    v = version()
    t0 = time.perf_counter()
    py = VENV / "Scripts" / "python.exe"

    # 1. The build environment, from the lock file
    if args.only_installer:
        if not (DIST / "Modalytics.exe").is_file():
            print(f"❌ No program in {DIST}: build it first")
            return 1
        return installer(v, folder_size(DIST), t0)
    if not args.skip_sync:
        env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(VENV))
        if run(["uv", "sync", "--frozen", "--python", "3.12", "--group", "build", "--no-editable"],
               env=env).returncode != 0:
            print("❌ uv sync failed")
            return 1

    # The training pack's contents, from the lock file (the program installs it per computer)
    sys.path.insert(0, str(ROOT / "src"))
    from modalytics.training import pack
    spec = pack.from_lock(ROOT / "uv.lock")
    pack.PACK_SPEC.parent.mkdir(parents=True, exist_ok=True)
    pack.PACK_SPEC.write_text(json.dumps(spec, indent=1) + "\n", encoding="utf-8")
    print(f"✅ Training pack: PyTorch {spec['torch']} and {len(spec['install']) - 1} libraries ({pack.PACK_SPEC.name})")

    # 2. The program
    log = BUILD / "pyinstaller.log"
    with open(log, "w", encoding="utf-8") as f:
        res = run([py, "-m", "PyInstaller", ROOT / "packaging" / "modalytics.spec", "--noconfirm",
                   "--distpath", BUILD / "dist", "--workpath", BUILD / "work"], stdout=f, stderr=subprocess.STDOUT,
                  env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    if res.returncode != 0:
        print(f"❌ PyInstaller failed — see {log}")
        return 1
    size = folder_size(DIST)
    print(f"✅ Program: {DIST} ({size / 1e9:.2f} GB)")

    # 3. The built program checks itself
    problems = smoke_test()
    for p in problems:
        print(f"❌ {p}")
    if problems:
        return 1
    print("✅ Smoke test passed")
    if args.test_pack and test_pack():
        return 1
    if args.no_installer:
        return 0

    return installer(v, size, t0)


def test_pack() -> list[str]:
    """``--test-pack``: the built program downloads and installs the training pack for this computer into a
    temporary folder, then its self-test must train with it. Returns the problems."""
    import tempfile
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MODALYTICS_TRAINING=tempfile.mkdtemp(prefix="pack_test_"))
    env.pop("MODALYTICS_HOME", None)
    cli = str(DIST / "modalytics-cli.exe")
    t0 = time.perf_counter()
    res = subprocess.run([cli, "--install-training"], env=env, capture_output=True, text=True, encoding="utf-8",
                         errors="replace", timeout=7200)
    lines = [x for x in res.stdout.splitlines() if x[:1] in "✅❌ℹ"]
    print("   --install-training:\n      " + "\n      ".join(lines[-6:])
          + f"\n      ({(time.perf_counter() - t0) / 60:.1f} min)")
    if res.returncode != 0:
        print("❌ the training pack did not install:\n" + res.stdout[-1500:] + res.stderr[-800:])
        return ["training pack"]
    res = subprocess.run([cli, "--self-test"], env=env, capture_output=True, text=True, encoding="utf-8",
                         errors="replace", timeout=900)
    training = [x for x in res.stdout.splitlines() if "Training" in x]
    print("   --self-test with the pack: " + " ".join(training))
    if res.returncode != 0 or not training or not training[0].startswith("✅"):
        print("❌ the self-test did not train with the pack:\n" + res.stdout[-1500:])
        return ["training pack self-test"]
    print("✅ Training pack installed and trained with")
    shutil.rmtree(env["MODALYTICS_TRAINING"], ignore_errors=True)
    return []


def installer(v: str, size: int, t0: float) -> int:
    """Step 4: Inno Setup on the program folder."""
    compiler = iscc()
    if compiler is None:
        print("❌ Inno Setup's ISCC.exe not found (winget install JRSoftware.InnoSetup)")
        return 1
    out = BUILD / "installer"
    shutil.rmtree(out, ignore_errors=True)
    defines = [f"/DAppVersion={v}", f"/DNumericVersion={numeric(v)}", f"/DSource={DIST}", f"/DOutput={out}"]
    if size > ONE_FILE_LIMIT:
        defines.append("/DSpanning=1")
    if run([compiler, "/Qp", *defines, ROOT / "packaging" / "modalytics.iss"]).returncode != 0:
        print("❌ Inno Setup failed")
        return 1
    files = sorted(out.iterdir())
    total = sum(f.stat().st_size for f in files)
    for f in files:
        if f.stat().st_size >= 2 ** 31:
            print(f"⚠️ {f.name} is {f.stat().st_size / 2 ** 30:.2f} GiB: GitHub takes release files under 2 GiB")
    print(f"✅ Installer ({total / 1e9:.2f} GB, {(time.perf_counter() - t0) / 60:.0f} min): "
          + ", ".join(f.name for f in files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
