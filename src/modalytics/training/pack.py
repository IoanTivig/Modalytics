"""
The training pack: PyTorch and the libraries that train segmentation models, installed beside the program for this
computer's hardware (Qt-free).

The installed program carries everything except training: PyTorch alone is 4 GB with NVIDIA's libraries, and most
people only analyse. Training › Segmentation models › Train (or the installer's *Download training support now*, or
``modalytics-cli --install-training``) installs the pack that fits the hardware:

=========  ==============================================  =================================================
Variant    For                                             PyTorch from
=========  ==============================================  =================================================
``cuda``   an NVIDIA GPU                                   download.pytorch.org/whl/cu128 (about 2.6 GB)
``xpu``    an Intel Arc GPU                                download.pytorch.org/whl/xpu (about 1.2 GB)
``cpu``    anything else — AMD GPUs on Windows included    download.pytorch.org/whl/cpu (about 0.3 GB; slow)
=========  ==============================================  =================================================

1. **What is in it** — exactly the training-only part of ``uv.lock`` (``from_lock``): PyTorch, torchvision,
   segmentation-models-pytorch, timm, onnx and what they need, at the versions Modalytics is tested with; only
   PyTorch's build differs per variant. ``packaging/build.py`` writes it into ``resources/training/pack.json``.
   What the pack needs that the program already carries (numpy, sympy, networkx…) is not installed again.
2. **Where** — per user, ``%LOCALAPPDATA%\\Modalytics\\training\\<variant>-<torch version>`` (no administrator
   rights; not copied with roaming profiles), ``current.json`` naming the one in use. ``activate`` puts it on
   ``sys.path`` when the program starts, before anything imports PyTorch.
3. **Installing** (``install``) — pip, run by the program itself (``parameters.libraries.pip``), resolves the
   pack with the program's own libraries pinned (they are never changed), then installs it into a staging folder;
   a separate process imports PyTorch from it and trains one step (``verify``). Only then does it replace the pack
   in use — a failed download or a broken pack leaves the old one working.
4. **Hardware changes** (``status``) — the pack records the hardware it was chosen for; when the GPUs found now ask
   for another variant (an NVIDIA card added, say), Training offers to switch, with the download's size.

From the source (``uv sync --extra training``) PyTorch is in the environment and no pack is used.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from modalytics import paths

PACK_SPEC = paths.RESOURCES / "training" / "pack.json"
PYPI = "https://pypi.org/simple"


@dataclass(frozen=True)
class Variant:
    key: str
    label: str           # what it runs on, for messages
    index: str           # PyTorch's package index for this build
    local: str           # PyTorch's local version label ("cu128", "xpu", "cpu")
    approx_gb: float     # download size, roughly


def variants(spec: dict | None = None) -> dict[str, Variant]:
    """The packs there are (CUDA's build is the one ``uv.lock`` names)."""
    cuda = (spec or load_spec()).get("cuda", "cu128")
    return {"cuda": Variant("cuda", "an NVIDIA GPU (CUDA)", f"https://download.pytorch.org/whl/{cuda}", cuda, 2.6),
            "xpu": Variant("xpu", "an Intel GPU (XPU)", "https://download.pytorch.org/whl/xpu", "xpu", 1.2),
            "cpu": Variant("cpu", "the processor (CPU — slow)", "https://download.pytorch.org/whl/cpu", "cpu", 0.3)}


# ---------------------------------------------------------------- 1. what is in it
def from_lock(lock_path: str | os.PathLike) -> dict:
    """The pack from ``uv.lock``: the packages only the ``training`` extra needs, for this platform.

    Returns:
        {"torch": "2.11.0", "cuda": "cu128", "install": {name: version} (PyTorch's own version without its build
        label), "provided": [names the pack needs that the program carries]}.
    """
    import tomllib
    from packaging.markers import Marker
    lock = tomllib.loads(Path(lock_path).read_text(encoding="utf-8"))
    entries: dict[str, list[dict]] = {}
    for p in lock["package"]:
        entries.setdefault(p["name"], []).append(p)
    env = {"extra": ""}

    def applies(dep: dict) -> bool:
        return "marker" not in dep or Marker(dep["marker"]).evaluate(env)

    def pick(dep: dict) -> dict:
        found = entries[dep["name"]]
        if "version" in dep:
            found = [p for p in found if p["version"] == dep["version"]] or found
        return found[0]

    def closure(deps: list[dict]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        todo = [d for d in deps if applies(d)]
        while todo:
            d = todo.pop()
            p = pick(d)
            if p["name"] in out:
                continue
            out[p["name"]] = p
            todo += [x for x in p.get("dependencies", []) if applies(x)]
            for extra in d.get("extra", []):
                todo += [x for x in p.get("optional-dependencies", {}).get(extra, []) if applies(x)]
        return out

    me = entries["modalytics"][0]
    base = closure(me.get("dependencies", []))
    training = closure(me.get("optional-dependencies", {}).get("training", []))
    torch = training["torch"]
    version, _, label = torch["version"].partition("+")
    install = {n: p["version"] for n, p in sorted(training.items()) if n not in base}
    install["torch"] = version
    return {"torch": version, "cuda": label or "cu128", "install": install,
            "provided": sorted(n for n in training if n in base)}


def load_spec() -> dict:
    """The pack this program was built with (``resources/training/pack.json``)."""
    try:
        return json.loads(PACK_SPEC.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def requirements(variant: Variant, spec: dict | None = None) -> list[str]:
    """pip requirements of a pack: every package at its locked version, PyTorch's build for the variant."""
    spec = spec or load_spec()
    reqs = [f"{n}=={v}" for n, v in spec.get("install", {}).items() if n != "torch"]
    return [f"torch=={spec['torch']}+{variant.local}", *reqs]


# ---------------------------------------------------------------- 2. where
def root() -> Path:
    """Where packs are kept (``MODALYTICS_TRAINING`` moves it, e.g. for tests)."""
    if os.environ.get("MODALYTICS_TRAINING"):
        return Path(os.environ["MODALYTICS_TRAINING"])
    base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
    return Path(base) / "Modalytics" / "training"


def installed() -> dict | None:
    """The pack in use: {"folder", "variant", "torch", "hardware", "installed"}, or None."""
    try:
        meta = json.loads((root() / "current.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta if (root() / meta.get("folder", "")).is_dir() else None


def in_environment() -> bool:
    """PyTorch is part of the running Python (from the source with the training extra): no pack needed."""
    from importlib.util import find_spec
    return find_spec("torch") is not None and not getattr(sys, "_modalytics_pack", None)


def activate() -> str:
    """Put the pack in use on ``sys.path`` (once, before PyTorch is imported); returns its folder or ""."""
    if getattr(sys, "_modalytics_pack", None):
        return sys._modalytics_pack
    from importlib.util import find_spec
    if find_spec("torch") is not None:                     # PyTorch in the environment already (the source)
        return ""
    meta = installed()
    if meta is None:
        return ""
    folder = str(root() / meta["folder"])
    sys.path.append(folder)                                # after the program's own libraries: they win
    sys._modalytics_pack = folder
    return folder


# ---------------------------------------------------------------- 4. the hardware
def hardware() -> dict:
    """The GPUs that decide the variant: {"nvidia": [names], "intel": [names]}."""
    from modalytics.services import compute
    nvidia = [name for _i, name, _mb in compute.nvidia_gpus()]
    intel = [name for _i, name, _mb in compute.dxgi_adapters() if "intel" in name.lower() and "arc" in name.lower()]
    return {"nvidia": nvidia, "intel": intel}


def recommended(hw: dict | None = None) -> str:
    """The variant for this hardware: NVIDIA → cuda, Intel Arc → xpu, else cpu (AMD has no Windows PyTorch yet)."""
    hw = hw if hw is not None else hardware()
    return "cuda" if hw.get("nvidia") else "xpu" if hw.get("intel") else "cpu"


def describe(hw: dict) -> str:
    gpus = hw.get("nvidia", []) + hw.get("intel", [])
    return ", ".join(gpus) if gpus else "no NVIDIA or Intel GPU"


def status(hw: dict | None = None) -> dict:
    """What Training should say.

    Returns:
        {"state": "environment" (PyTorch in the running Python) | "none" | "ok" | "changed", "installed": meta or
        None, "recommended": variant key, "hardware": hw, "message": a sentence for the user}.
    """
    if in_environment():
        return {"state": "environment", "installed": None, "recommended": "", "hardware": {},
                "message": "PyTorch is part of this Python environment."}
    hw = hw if hw is not None else hardware()
    want = variants()[recommended(hw)]
    meta = installed()
    if meta is None:
        return {"state": "none", "installed": None, "recommended": want.key, "hardware": hw,
                "message": f"Training needs the training pack (PyTorch and its libraries). For this computer "
                           f"({describe(hw)}): the pack for {want.label}, about {want.approx_gb:.1f} GB to download."}
    have = variants()[meta["variant"]]
    if meta["variant"] != want.key:
        return {"state": "changed", "installed": meta, "recommended": want.key, "hardware": hw,
                "message": f"The hardware changed: {describe(hw)} found, the training pack installed is for "
                           f"{have.label} (chosen for {describe(meta.get('hardware', {}))}). Switch to the pack for "
                           f"{want.label} (about {want.approx_gb:.1f} GB to download)? The current one stays until "
                           f"the new one works."}
    return {"state": "ok", "installed": meta, "recommended": want.key, "hardware": hw,
            "message": f"Training pack: PyTorch {meta['torch']} for {have.label}."}


# ---------------------------------------------------------------- 3. installing
def _run(cmd: list[str], say) -> tuple[int, str]:
    """Run a command, passing its output on line by line; returns (exit code, the output's end)."""
    say("ℹ️ " + " ".join(cmd[:4]) + " …")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            say(line)
            tail = (tail + [line])[-15:]
    return proc.wait(), "\n".join(tail)


def install(key: str | None = None, say=print) -> int:
    """Install (or switch to) the pack for ``key`` (default: the one this hardware asks for); 0 when it worked.

    The pack in use is replaced only once the new one has passed ``verify`` in a separate process.
    """
    from modalytics.parameters import libraries as libs
    spec = load_spec()
    if not spec:
        say(f"❌ The training pack's description is missing ({PACK_SPEC})")
        return 1
    hw = hardware()
    variant = variants(spec)[key or recommended(hw)]
    say(f"ℹ️ Training pack for {variant.label}: PyTorch {spec['torch']}+{variant.local} and "
        f"{len(spec['install']) - 1} libraries (about {variant.approx_gb:.1f} GB to download) — {describe(hw)}")
    base = root()
    base.mkdir(parents=True, exist_ok=True)
    stage = base / ".staging"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir()
    work = Path(tempfile.mkdtemp())
    indexes = ["--index-url", variant.index, "--extra-index-url", PYPI]
    reqs = requirements(variant, spec)
    # 1. What to download: the program's own libraries pinned, never installed again
    report, constraints = work / "plan.json", libs.constraints_file(work / "constraints.txt")
    code, tail = _run([*libs.pip(), "install", "--dry-run", "--quiet", "--ignore-installed", "--only-binary=:all:",
                       *indexes, "--constraint", str(constraints), "--report", str(report), *reqs], say)
    if code != 0:
        say(f"❌ The pack could not be resolved:\n{tail}")
        return 1
    todo, problems = libs.plan_from_report(json.loads(report.read_text(encoding="utf-8")))
    if problems:
        say("❌ The pack would change a library of the program: " + "; ".join(problems))
        return 1
    # 2. Download and install into the staging folder
    code, tail = _run([*libs.pip(), "install", "--no-deps", "--only-binary=:all:", *indexes, "--target", str(stage),
                       "--progress-bar", "off", *todo], say)
    if code != 0:
        say(f"❌ Installing the pack failed:\n{tail}")
        shutil.rmtree(stage, ignore_errors=True)
        return 1
    # 3. Checked in a separate process before it is used
    say("ℹ️ Checking the pack: PyTorch imported, one training step …")
    ok, detail = verify(stage)
    if not ok:
        say(f"❌ The new pack does not work — the current one is kept:\n{detail}")
        shutil.rmtree(stage, ignore_errors=True)
        return 1
    say(f"✅ {detail}")
    # 4. In use from now on (the next start; Training loads it at once)
    folder = f"{variant.key}-{spec['torch']}"
    old = installed()
    shutil.rmtree(base / folder, ignore_errors=True)
    os.replace(stage, base / folder)
    meta = {"folder": folder, "variant": variant.key, "torch": spec["torch"], "hardware": hw,
            "installed": time.strftime("%Y-%m-%d %H:%M")}
    (base / "current.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    if old and old["folder"] != folder:
        shutil.rmtree(base / old["folder"], ignore_errors=True)
    say(f"✅ Training pack installed: {base / folder}")
    return 0


def verify(folder: str | os.PathLike) -> tuple[bool, str]:
    """Import PyTorch from a pack in a fresh process and train one step: (ok, what it found or why not)."""
    cmd = [*paths.helper_command(), "--check-training-pack", str(folder)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    lines = [x for x in res.stdout.splitlines() if x.strip()]
    if res.returncode != 0 or not lines:
        return False, (res.stdout + res.stderr).strip()[-800:]
    return True, lines[-1]


def check_main(folder: str) -> int:
    """``--check-training-pack FOLDER`` (a separate process): PyTorch from that pack, one training step."""
    sys.path.append(folder)
    sys._modalytics_pack = folder
    import torch
    from modalytics.training import seg_train as st
    dev = st.devices()[0]
    model = st.build("unet_resnet18", 1).to(dev.key)
    x = torch.randn(1, 1, 64, 64, device=dev.key)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), (x > 0).float())
    loss.backward()
    print(f"PyTorch {torch.__version__} works on {dev.label} (one training step, loss {float(loss.detach()):.2f})")
    return 0
