"""Hardware selection: ranking of rendering adapters and model devices (no GPU needed)."""

from modalytics.render import gpu
from modalytics.render.gpu import Adapter
from modalytics.services import compute
from modalytics.services.compute import Device


def _adapters():
    # As the probes list them on a Windows desktop with a display GPU and a second card
    return [Adapter("AMD Radeon RX 7900 XT", "Vulkan", "DiscreteGPU", index=0),
            Adapter("NVIDIA RTX A2000 12GB", "Vulkan", "DiscreteGPU", index=1),
            Adapter("AMD Radeon RX 7900 XT", "DX12", "DiscreteGPU", index=0),
            Adapter("Microsoft Basic Render Driver", "DX12", "CPU", index=3),
            Adapter("AMD Radeon RX 7900 XT", "GL", "Unknown", index=0)]


def test_render_rank_prefers_hardware_discrete_then_api_then_os_order():
    ranked = gpu.rank(_adapters(), platform="win32")
    assert (ranked[0].name, ranked[0].api) == ("AMD Radeon RX 7900 XT", "Vulkan")   # first listed = display GPU
    assert ranked[-1].software


def test_render_rank_laptop_and_vm():
    laptop = [Adapter("Intel UHD", "Vulkan", "IntegratedGPU"), Adapter("RTX 4060 Laptop", "Vulkan", "DiscreteGPU", index=1)]
    assert gpu.rank(laptop, platform="linux")[0].name == "RTX 4060 Laptop"
    # A VM: a virtual GPU beats a software renderer; OpenGL's "Unknown" type counts as a real GPU
    vm = [Adapter("llvmpipe (LLVM 17)", "GL", "Unknown"), Adapter("SVGA3D", "GL", "VirtualGPU", index=1)]
    assert gpu.rank(vm, platform="linux")[0].name == "SVGA3D"
    assert Adapter("llvmpipe (LLVM 17)", "GL", "Unknown").software
    # Only the Direct3D 12 software renderer and OpenGL survived the probes: OpenGL on the GPU wins
    rest = [Adapter("Microsoft Basic Render Driver", "DX12", "CPU"), Adapter("Radeon", "GL", "Unknown")]
    assert gpu.rank(rest, platform="win32")[0].api == "GL"


def test_render_rank_overrides():
    assert gpu.rank(_adapters(), platform="win32", wanted="nvidia")[0].name == "NVIDIA RTX A2000 12GB"
    assert gpu.rank(_adapters(), platform="win32", wanted="software")[0].software


def test_render_probe_failure_is_reported_not_raised():
    adapters, problem = gpu._run_probe("NoSuchApi")
    assert adapters == [] and "crashed" in problem


def _fake_hardware(monkeypatch):
    monkeypatch.setattr(compute, "dxgi_adapters", lambda: [(0, "AMD Radeon RX 7900 XT", 20437),
                                                            (1, "NVIDIA RTX A2000 12GB", 12110)])
    monkeypatch.setattr(compute, "nvidia_gpus", lambda: [(0, "NVIDIA RTX A2000 12GB", 12282)])


def test_devices_directml_ranked_by_memory(monkeypatch):
    _fake_hardware(monkeypatch)
    devs = compute.rank(compute.list_devices(["DmlExecutionProvider", "CPUExecutionProvider"]))
    assert [d.key for d in devs] == ["dml:0", "dml:1", "cpu"]
    assert devs[1].providers() == [("DmlExecutionProvider", {"device_id": 1}), "CPUExecutionProvider"]
    assert devs[0].label() == "AMD Radeon RX 7900 XT (DirectML)"


def test_devices_cuda_before_directml_and_overrides(monkeypatch):
    _fake_hardware(monkeypatch)
    found = compute.list_devices(["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"])
    assert compute.rank(found)[0].key == "cuda:0"
    assert compute.rank(found, wanted="cpu")[0].key == "cpu"
    assert compute.rank(found, wanted="dml:0")[0].key == "dml:0"
    assert compute.rank(found, wanted="amd")[0].key == "dml:0"


def test_cpu_only_machine():
    devs = compute.rank(compute.list_devices(["CPUExecutionProvider"]))
    assert devs == [Device("cpu", "CPU", "CPUExecutionProvider")]
    assert devs[0].providers() == ["CPUExecutionProvider"]


def test_device_setting_maps_labels_to_keys(monkeypatch):
    from modalytics.tools.base import _device_choices, device_key
    devs = [Device("dml:0", "AMD", "DmlExecutionProvider", 20000), Device("dml:1", "NVIDIA", "DmlExecutionProvider", 12000),
            Device("cpu", "CPU", "CPUExecutionProvider")]
    monkeypatch.setattr(compute, "_devices", devs)
    assert _device_choices() == ["Auto", "AMD (DirectML)", "NVIDIA (DirectML)", "CPU"]
    assert [device_key(c) for c in _device_choices()] == ["auto", "dml:0", "dml:1", "cpu"]
    assert device_key("A GPU from another machine (DirectML)") == "auto"
    monkeypatch.setattr(compute, "_devices", None)             # still being listed
    assert _device_choices() == ["Auto", "CPU"]
