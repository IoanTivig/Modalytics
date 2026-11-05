"""
Rendering GPU: picks the best adapter wgpu can use on this machine.

Modalytics has to run on whatever it is given: a workstation with two
discrete GPUs, a laptop with an integrated one, a VM or remote session with
only a software renderer. The choice is made in three steps:

1. **Probe** every graphics API this OS offers (Windows: Vulkan, Direct3D 12,
   OpenGL; Linux: Vulkan, OpenGL; macOS: Metal), each in its own short-lived
   process (``python -m modalytics --gpu-probe <API>``). A driver that crashes
   while being enumerated then only takes its probe down, never the app. (On
   the dev machine, wgpu crashes when Vulkan and OpenGL share one instance.)
   The probes start as early as possible (``start_probe`` in ``__main__``) and
   run in parallel while Qt and the main window are built. Start-up only waits
   for them in order of preference until one finds a discrete GPU (usually
   Vulkan, ~1 s); the slower ones (Direct3D 12: 1.5–4.5 s on the dev machine)
   finish in the background and only feed the report.
2. **Rank** the adapters found (``rank``): a hardware GPU before a software
   renderer, discrete before integrated before virtual, then the API this OS
   draws best with, then the order the OS lists them in (the GPU driving the
   display comes first).
3. **Select**: wgpu is restricted to the winning API (one instance per process)
   and pygfx is given the winning adapter.

Environment overrides:
    MODALYTICS_WGPU_BACKENDS  skip probing, use these wgpu backends, e.g. "DX12", "Vulkan", "GL".
    MODALYTICS_GPU            prefer the adapter whose name contains this, e.g. "NVIDIA";
                              "software" (or "cpu") forces the software renderer.

``python -m modalytics --gpu-info`` prints what was found and chosen.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

# Graphics APIs to probe, best first, per OS (names are wgpu instance backends)
APIS = {"win32": ("Vulkan", "DX12", "GL"), "linux": ("Vulkan", "GL"), "darwin": ("Metal",)}
# wgpu's adapter ``backend_type`` → instance backend name
BACKEND_OF = {"Vulkan": "Vulkan", "D3D12": "DX12", "Metal": "Metal", "OpenGL": "GL"}
# Adapter kinds, best first ("Unknown" is what OpenGL reports for any GPU)
KIND_ORDER = ("DiscreteGPU", "IntegratedGPU", "Unknown", "VirtualGPU", "CPU")
# Names of software renderers (OpenGL does not report them as "CPU")
SOFTWARE_NAMES = ("llvmpipe", "softpipe", "lavapipe", "swiftshader", "microsoft basic render", "gdi generic")
PROBE_TIMEOUT_S = 20.0


@dataclass(frozen=True)
class Adapter:
    """One adapter as reported by a probe.

    Attributes:
        name: Device name, e.g. "AMD Radeon RX 7900 XT".
        api: wgpu instance backend: "Vulkan", "DX12", "Metal" or "GL".
        kind: wgpu adapter type: "DiscreteGPU", "IntegratedGPU", "VirtualGPU", "CPU" or "Unknown".
        driver: Driver description/version (may be empty).
        index: Position in its API's adapter list (tells identical GPUs apart).
    """

    name: str
    api: str
    kind: str
    driver: str = ""
    index: int = 0

    @property
    def software(self) -> bool:
        return self.kind == "CPU" or any(s in self.name.lower() for s in SOFTWARE_NAMES)

    def label(self) -> str:
        api = {"DX12": "Direct3D 12", "GL": "OpenGL"}.get(self.api, self.api)
        return f"{self.name} ({api}{', software' if self.software else ''})"


## Probing (each API in its own process) ##

def probe_main(api: str) -> int:
    """Entry point of a probe process: print the adapters of one API as JSON.

    Args:
        api: wgpu instance backend name.

    Returns:
        int: Exit code.
    """
    import wgpu
    from wgpu.backends.wgpu_native.extras import set_instance_extras
    set_instance_extras(backends=[api])
    out = []
    for i, a in enumerate(wgpu.gpu.enumerate_adapters_sync()):
        info = a.info
        out.append({"name": info.get("device", ""), "api": BACKEND_OF.get(info.get("backend_type"), api),
                    "kind": info.get("adapter_type", "Unknown"),
                    "driver": info.get("description") or info.get("vendor") or "", "index": i})
    print(json.dumps(out))
    return 0


def _probe_command(api: str) -> tuple[list[str], dict]:
    """Command line and environment that run one probe (also from a frozen app)."""
    env = dict(os.environ)
    from modalytics import paths
    if paths.installed():                                       # the console twin: its output can be read
        return [*paths.helper_command(), "--gpu-probe", api], env
    # Make ``-m modalytics`` importable even when running from a source checkout
    pkg_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = os.pathsep.join(p for p in (pkg_parent, env.get("PYTHONPATH", "")) if p)
    return [sys.executable, "-m", "modalytics", "--gpu-probe", api], env


def _run_probe(api: str) -> tuple[list[Adapter], str]:
    """Run one probe process.

    Returns:
        tuple: (adapters, problem): problem is "" on success, else what went wrong.
    """
    cmd, env = _probe_command(api)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)          # no console flash on Windows
    try:
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=PROBE_TIMEOUT_S, creationflags=flags)
    except subprocess.TimeoutExpired:
        return [], f"timed out after {PROBE_TIMEOUT_S:.0f} s"
    except OSError as exc:
        return [], str(exc)
    if res.returncode != 0:
        last = (res.stderr.strip().splitlines() or [""])[-1]
        return [], f"crashed (exit code {res.returncode}){': ' + last if last else ''}"
    try:
        line = res.stdout.strip().splitlines()[-1]
        return [Adapter(**a) for a in json.loads(line)], ""
    except (IndexError, ValueError, TypeError) as exc:
        return [], f"unreadable output ({exc})"


_jobs: dict[str, Future] = {}


def start_probe() -> None:
    """Start probing all APIs in the background (call as early as possible)."""
    if not _jobs and not os.environ.get("MODALYTICS_WGPU_BACKENDS"):
        apis = APIS.get(sys.platform, APIS["linux"])
        pool = ThreadPoolExecutor(max_workers=len(apis), thread_name_prefix="gpu-probe")
        _jobs.update({api: pool.submit(_run_probe, api) for api in apis})
        pool.shutdown(wait=False)


def probe_result(api: str) -> tuple[list[Adapter], str]:
    """Adapters of one API (waits for its probe; starts the probes if needed)."""
    start_probe()
    return _jobs[api].result() if api in _jobs else ([], "not probed")


def probe_results() -> dict[str, tuple[list[Adapter], str]]:
    """Adapters per API, for every API of this OS (waits for all probes)."""
    return {api: probe_result(api) for api in APIS.get(sys.platform, APIS["linux"])}


def probes_done() -> bool:
    """True when every probe has finished (the full report is then available without waiting)."""
    return all(job.done() for job in _jobs.values())


## Ranking ##

def rank(adapters: list[Adapter], platform: str = sys.platform, wanted: str = "") -> list[Adapter]:
    """Order adapters best first.

    Args:
        adapters: Adapters in the order the probes listed them (API by API).
        platform: ``sys.platform`` value (selects the API preference).
        wanted: Name substring to put first, or "software"/"cpu" to put software renderers first.

    Returns:
        list[Adapter]: Best first.
    """
    apis = APIS.get(platform, APIS["linux"])
    wanted = wanted.lower()

    def key(item):
        index, a = item
        if wanted in ("software", "cpu"):
            preferred = 0 if a.software else 1
        else:
            preferred = 0 if wanted and wanted in a.name.lower() else 1
        kind = KIND_ORDER.index(a.kind) if a.kind in KIND_ORDER else KIND_ORDER.index("Unknown")
        api = apis.index(a.api) if a.api in apis else len(apis)
        return preferred, a.software, kind, api, index

    return [a for _i, a in sorted(enumerate(adapters), key=key)]


## Selection ##

@dataclass
class GpuChoice:
    """What the canvas renders with, and what else was found (for the status bar and --gpu-info)."""

    adapter: Adapter | None
    found: list[Adapter]
    problems: dict[str, str]
    forced_backends: str = ""

    def label(self) -> str:
        return self.adapter.label() if self.adapter is not None else "no GPU"


choice: GpuChoice | None = None
adapter_summary = "not initialised"


def init_gpu() -> str:
    """Pick the rendering adapter once (see the module docstring) and hand it to pygfx.

    Returns:
        str: Human-readable choice, e.g. "AMD Radeon RX 7900 XT (Vulkan)".
    """
    global choice, adapter_summary
    if choice is not None:
        return adapter_summary
    import wgpu
    from wgpu.backends.wgpu_native.extras import set_instance_extras
    wanted = os.environ.get("MODALYTICS_GPU", "")
    forced = os.environ.get("MODALYTICS_WGPU_BACKENDS", "")

    # 1. Candidates: from the probes, or (override) from the backends asked for
    if forced:
        found, problems = [], {}
        best = None
        backends = [b.strip() for b in forced.split(",") if b.strip()]
    else:
        # Wait for the probes in order of preference, until one finds a discrete GPU (the one asked
        # for, if any); the others keep running in the background for the report
        found, problems = [], {}
        for api in APIS.get(sys.platform, APIS["linux"]):
            adapters, msg = probe_result(api)
            found += adapters
            if msg:
                problems[api] = msg
            if wanted.lower() not in ("software", "cpu") and \
                    any(a.kind == "DiscreteGPU" and wanted.lower() in a.name.lower() for a in adapters):
                break
        ranked = rank(found, wanted=wanted)
        best = ranked[0] if ranked else None
        # Last resort (every probe failed): let wgpu decide among the primary APIs
        backends = [best.api] if best is not None else ["Primary"]

    # 2. Restrict wgpu to that API and find the same adapter again in this process
    set_instance_extras(backends=backends)
    adapters = wgpu.gpu.enumerate_adapters_sync()
    if best is not None:
        # Same API, same enumeration order as in the probe: check the name to be safe
        ok = best.index < len(adapters) and adapters[best.index].info.get("device") == best.name
        adapter = adapters[best.index] if ok else None
    else:
        local = [Adapter(a.info.get("device", ""), BACKEND_OF.get(a.info.get("backend_type"), "?"),
                         a.info.get("adapter_type", "Unknown"), index=i) for i, a in enumerate(adapters)]
        order = rank(local, wanted=wanted)
        adapter = adapters[local.index(order[0])] if order else None
        best = order[0] if order else None
    if adapter is None:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        best = Adapter(adapter.info.get("device", "?"), BACKEND_OF.get(adapter.info.get("backend_type"), "?"),
                       adapter.info.get("adapter_type", "Unknown"))
    if wanted and wanted.lower() not in ("software", "cpu") and wanted.lower() not in best.name.lower():
        print(f"⚠️ No GPU matching '{wanted}', using {best.name}")

    # 3. Hand it to pygfx and report
    import pygfx as gfx
    gfx.renderers.wgpu.select_adapter(adapter)
    choice = GpuChoice(best, found, problems, forced)
    adapter_summary = choice.label()
    for api, msg in problems.items():
        print(f"ℹ️ GPU probe {api}: {msg}")
    if best.software and wanted.lower() in ("software", "cpu"):
        print(f"ℹ️ Rendering in software ({best.name}), as MODALYTICS_GPU asks")
    elif best.software:
        print(f"⚠️ Rendering in software ({best.name}): no usable GPU was found, the canvas will be slow")
    else:
        print(f"✅ Rendering on {adapter_summary}")
    return adapter_summary


def report() -> str:
    """Multi-line description of the rendering adapters (for --gpu-info and the status-bar tooltip).

    Lists what every probe found, so it waits for the slow ones: check ``probes_done()`` first
    where waiting matters.
    """
    init_gpu()
    lines = [f"Rendering: {choice.label()}"]
    if choice.forced_backends:
        lines.append(f"  (MODALYTICS_WGPU_BACKENDS={choice.forced_backends}: not probed)")
    results = probe_results() if not choice.forced_backends else {}
    found = [a for api in results for a in results[api][0]]
    problems = {api: msg for api, (_a, msg) in results.items() if msg}
    shown = set()
    for a in rank(found, wanted=os.environ.get("MODALYTICS_GPU", "")):
        if (a.name, a.api, a.kind) in shown:          # the same card listed twice (virtual displays)
            continue
        shown.add((a.name, a.api, a.kind))
        mark = "→" if a == choice.adapter else " "
        lines.append(f"  {mark} {a.label()}" + (f"  [{a.driver}]" if a.driver else ""))
    for api, msg in problems.items():
        lines.append(f"    {api}: {msg}")
    return "\n".join(lines)
