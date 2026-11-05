"""
Where the models run: picks the best ONNX Runtime execution provider and device.

Model inference must work on any machine, from a CUDA workstation to a laptop
without a usable GPU. The installed ONNX Runtime package decides which
*execution providers* exist (``onnxruntime-directml``: DirectML on any Windows
GPU; ``onnxruntime-gpu``: CUDA; AMD/Intel builds: MIGraphX, ROCm, OpenVINO;
plain ``onnxruntime``: CPU only). This module:

1. **Lists devices** (``devices``): one entry per GPU each provider can use
   (DirectML adapters come from DXGI, CUDA GPUs from ``nvidia-smi``), plus the CPU.
2. **Ranks** them (``rank``): the dedicated vendor stacks first (CUDA, MIGraphX,
   ROCm), then DirectML, CoreML, OpenVINO, then the CPU; within one provider the
   GPU with the most dedicated memory first.
3. **Builds sessions' provider lists** (``providers``) and checks afterwards that
   the session really runs where it was asked to (``check_session``): ONNX
   Runtime silently falls back to the CPU when e.g. the CUDA libraries are missing.
4. **Hints** (``missing_runtime_hint``) when a GPU is present but the installed
   ONNX Runtime cannot use it, with the package to install.

Device keys (the ``device`` argument of the segmentation and registration
services): "auto" (the best), "cpu", or "<provider>:<index>", e.g. "dml:1".

Environment overrides:
    MODALYTICS_INFERENCE   what "auto" means: "cpu", a device key ("dml:1"), or a
                           substring of a device name ("NVIDIA").
    MODALYTICS_DML_DEVICE  DirectML adapter index (older name for MODALYTICS_INFERENCE=dml:<n>).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass

# Provider name → (short key, name shown to users), best first
PROVIDERS = {
    "CUDAExecutionProvider": ("cuda", "CUDA"),
    "MIGraphXExecutionProvider": ("migraphx", "MIGraphX"),
    "ROCMExecutionProvider": ("rocm", "ROCm"),
    "DmlExecutionProvider": ("dml", "DirectML"),
    "CoreMLExecutionProvider": ("coreml", "Core ML"),
    "OpenVINOExecutionProvider": ("openvino", "OpenVINO"),
}
CPU = "CPUExecutionProvider"


@dataclass(frozen=True)
class Device:
    """One place a model can run.

    Attributes:
        key: "cpu" or "<provider key>:<index>", e.g. "dml:1".
        name: Hardware name, e.g. "NVIDIA RTX A2000 12GB".
        provider: ONNX Runtime execution provider name.
        memory_mb: Dedicated memory (0 if unknown or the CPU).
    """

    key: str
    name: str
    provider: str
    memory_mb: int = 0

    @property
    def is_gpu(self) -> bool:
        return self.provider != CPU

    def label(self) -> str:
        if not self.is_gpu:
            return "CPU"
        return f"{self.name} ({PROVIDERS[self.provider][1]})"

    def providers(self) -> list:
        """Provider list for ``onnxruntime.InferenceSession`` (always ending with the CPU)."""
        if not self.is_gpu:
            return [CPU]
        index = self.key.split(":", 1)[1]
        if self.provider == "OpenVINOExecutionProvider":
            options = {"device_type": index}
        elif self.provider == "CoreMLExecutionProvider":
            options = {}
        else:
            options = {"device_id": int(index)}
        return [(self.provider, options), CPU]


## Hardware listing (no ONNX Runtime needed) ##

def dxgi_adapters() -> list[tuple[int, str, int]]:
    """Hardware DirectX adapters as DirectML numbers them (Windows only).

    DirectML's ``device_id`` is the adapter's index in DXGI's ``EnumAdapters1``
    order, so the index is kept even when entries are skipped. Software
    adapters are skipped, and so are repeated entries of the same GPU: virtual
    display drivers (VR headsets, streaming) can make one card appear twice
    under different LUIDs. Two truly identical cards therefore show as one
    here; the second stays reachable with ``MODALYTICS_INFERENCE=dml:<index>``.

    Returns:
        list[tuple[int, str, int]]: (DXGI index, name, dedicated memory in MB).
    """
    if sys.platform != "win32":
        return []
    import ctypes

    class GUID(ctypes.Structure):
        _fields_ = [("a", ctypes.c_uint32), ("b", ctypes.c_uint16), ("c", ctypes.c_uint16),
                    ("d", ctypes.c_ubyte * 8)]

    class DESC1(ctypes.Structure):                  # DXGI_ADAPTER_DESC1
        _fields_ = [("description", ctypes.c_wchar * 128), ("vendor", ctypes.c_uint),
                    ("device", ctypes.c_uint), ("subsys", ctypes.c_uint), ("revision", ctypes.c_uint),
                    ("dedicated_video", ctypes.c_size_t), ("dedicated_system", ctypes.c_size_t),
                    ("shared_system", ctypes.c_size_t), ("luid_low", ctypes.c_uint32),
                    ("luid_high", ctypes.c_int32), ("flags", ctypes.c_uint)]

    def method(obj, slot, *argtypes):
        """COM method ``slot`` of the vtable of ``obj`` (IUnknown: 0-2, IDXGIObject: 3-6, …)."""
        vtable = ctypes.cast(ctypes.cast(obj, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
        return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtable[slot])

    iid_factory1 = GUID(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
    factory = ctypes.c_void_p()
    try:
        if ctypes.windll.dxgi.CreateDXGIFactory1(ctypes.byref(iid_factory1), ctypes.byref(factory)) != 0:
            return []
    except OSError:
        return []
    out, seen = [], set()
    index = 0
    while True:
        adapter = ctypes.c_void_p()
        # IDXGIFactory1::EnumAdapters1 is vtable slot 12; DXGI_ERROR_NOT_FOUND ends the list
        if method(factory, 12, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(factory, index, ctypes.byref(adapter)):
            break
        desc = DESC1()
        method(adapter, 10, ctypes.POINTER(DESC1))(adapter, ctypes.byref(desc))   # IDXGIAdapter1::GetDesc1
        method(adapter, 2)(adapter)                                              # Release
        same_card = (desc.description, desc.vendor, desc.device, desc.subsys, desc.revision, desc.dedicated_video)
        if not desc.flags & 2 and same_card not in seen:                         # 2 = DXGI_ADAPTER_FLAG_SOFTWARE
            seen.add(same_card)
            out.append((index, desc.description, desc.dedicated_video >> 20))
        index += 1
    method(factory, 2)(factory)
    return out


def nvidia_gpus() -> list[tuple[int, str, int]]:
    """NVIDIA GPUs in CUDA order, from ``nvidia-smi`` (empty if it is not installed).

    Returns:
        list[tuple[int, str, int]]: (CUDA index, name, memory in MB).
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []
    try:
        res = subprocess.run([exe, "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired):
        return []
    out = []
    for line in res.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            out.append((int(parts[0]), parts[1], int(float(parts[2])) if parts[2].replace(".", "").isdigit() else 0))
    return out


## Devices ##

def list_devices(available: list[str]) -> list[Device]:
    """Devices the given ONNX Runtime providers can use (unranked, CPU last).

    Args:
        available: ``onnxruntime.get_available_providers()``.

    Returns:
        list[Device]
    """
    out = []
    if "CUDAExecutionProvider" in available:
        gpus = nvidia_gpus() or [(0, "NVIDIA GPU", 0)]
        out += [Device(f"cuda:{i}", n, "CUDAExecutionProvider", m) for i, n, m in gpus]
    for prov, name in (("MIGraphXExecutionProvider", "AMD GPU"), ("ROCMExecutionProvider", "AMD GPU")):
        if prov in available:
            out.append(Device(f"{PROVIDERS[prov][0]}:0", name, prov))
    if "DmlExecutionProvider" in available:
        adapters = dxgi_adapters() or [(0, "Default GPU", 0)]
        out += [Device(f"dml:{i}", n, "DmlExecutionProvider", m) for i, n, m in adapters]
    if "CoreMLExecutionProvider" in available:
        out.append(Device("coreml:0", "Apple Neural Engine / GPU", "CoreMLExecutionProvider"))
    if "OpenVINOExecutionProvider" in available:
        out.append(Device("openvino:GPU", "Intel GPU", "OpenVINOExecutionProvider"))
    out.append(Device("cpu", "CPU", CPU))
    return out


def rank(devices: list[Device], wanted: str = "") -> list[Device]:
    """Order devices best first.

    Args:
        devices: From ``list_devices``.
        wanted: "cpu", a device key, or a name substring to put first.

    Returns:
        list[Device]: Best first; the CPU last unless wanted.
    """
    order = list(PROVIDERS) + [CPU]
    wanted = wanted.lower()

    def key(d: Device):
        preferred = 0 if wanted and (wanted == d.key or (wanted != "cpu" and wanted in d.name.lower())) else 1
        return preferred, order.index(d.provider), -d.memory_mb

    return sorted(devices, key=key)


_lock = threading.Lock()
_devices: list[Device] | None = None


def devices() -> list[Device]:
    """All devices on this machine, best first (cached; imports ONNX Runtime, ~1 s the first time)."""
    global _devices
    with _lock:
        if _devices is None:
            import onnxruntime as ort
            ort.set_default_logger_severity(3)       # errors only (DirectML logs node placement warnings)
            _devices = rank(list_devices(ort.get_available_providers()), wanted=_wanted())
        return _devices


def warm_up() -> None:
    """List the devices in a background thread, so the first tool that needs them does not wait."""
    threading.Thread(target=devices, name="compute-devices", daemon=True).start()


def known_devices() -> list[Device] | None:
    """The device list if it is already known, without waiting (None while it is being built)."""
    return _devices


def _wanted() -> str:
    wanted = os.environ.get("MODALYTICS_INFERENCE", "")
    if not wanted and os.environ.get("MODALYTICS_DML_DEVICE"):
        wanted = f"dml:{os.environ['MODALYTICS_DML_DEVICE']}"
    return wanted


def resolve(device: str = "auto") -> Device:
    """The device for a ``device`` argument.

    Args:
        device: "auto", "cpu" or a device key.

    Returns:
        Device: The device (the best one if the key is unknown on this machine).
    """
    devs = devices()                                 # best first, always includes "cpu"
    return next((d for d in devs if d.key == device), devs[0])


def best() -> Device:
    return resolve("auto")


## Sessions ##

def providers(device: str = "auto") -> list:
    """Provider list for ``onnxruntime.InferenceSession`` for a ``device`` argument."""
    return resolve(device).providers()


_warned: set[str] = set()


def check_session(sess, device: str = "auto") -> str:
    """Warn (once per device) if a session fell back to the CPU, and return where it runs.

    Args:
        sess: ``onnxruntime.InferenceSession``.
        device: The ``device`` argument it was created for.

    Returns:
        str: The provider the session actually uses first.
    """
    dev = resolve(device)
    actual = sess.get_providers()[0]
    if actual != dev.provider and dev.key not in _warned:
        _warned.add(dev.key)
        print(f"⚠️ {dev.label()} could not be used ({actual} instead): check the GPU driver"
              + (" and the CUDA/cuDNN libraries" if dev.provider == "CUDAExecutionProvider" else ""))
    return actual


def missing_runtime_hint(available: list[str] | None = None) -> str:
    """A hint when this machine has a GPU the installed ONNX Runtime cannot use ("" if none).

    Args:
        available: ``onnxruntime.get_available_providers()`` (queried if None).

    Returns:
        str: Advice, e.g. which package to install.
    """
    if available is None:
        import onnxruntime as ort
        available = ort.get_available_providers()
    if any(p in available for p in PROVIDERS):
        return ""
    from modalytics.paths import installed
    fix = (lambda _pkg: "this installation's ONNX Runtime has no GPU support: reinstall Modalytics") if installed() \
        else (lambda pkg: f"pip install {pkg}")
    if sys.platform == "win32" and dxgi_adapters():
        return f"a GPU is present but ONNX Runtime is CPU-only: {fix('onnxruntime-directml')}"
    if shutil.which("nvidia-smi") or os.path.exists("/proc/driver/nvidia"):
        return f"an NVIDIA GPU is present but ONNX Runtime is CPU-only: {fix('onnxruntime-gpu')}"
    if os.path.exists("/dev/kfd"):
        return ("an AMD GPU is present but ONNX Runtime is CPU-only: install AMD's onnxruntime-migraphx "
                "build (see the ROCm documentation)")
    return ""


def report() -> str:
    """Multi-line description of the inference devices (for --gpu-info and the status-bar tooltip)."""
    import onnxruntime as ort
    devs = devices()
    lines = [f"Models: {devs[0].label()}  (ONNX Runtime {ort.__version__})"]
    for d in devs:
        mark = "→" if d == devs[0] else " "
        lines.append(f"  {mark} {d.label()}" + (f"  [{d.key}, {d.memory_mb} MB]" if d.memory_mb else f"  [{d.key}]"))
    hint = missing_runtime_hint()
    if hint:
        lines.append(f"  ℹ️ {hint}")
    return "\n".join(lines)
