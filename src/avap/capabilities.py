"""Device capability probe (architecture §2).

Load-bearing, not cosmetic: the pipeline cannot assume a decode engine
exists on every device (some Instinct SKUs have no VCN), and per-GFX-gen
backends key off gfx_generation.

Probes sysfs (amdgpu IP discovery) so it works without ROCm userspace;
fields are refined from HIP device properties once the extension is up.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

AMD_VENDOR = "0x1002"

# GC (graphics core) IP major version -> gfx generation label
_GC_TO_GFX = {9: "gfx9", 10: "gfx10", 11: "gfx11", 12: "gfx12"}


@dataclass
class DeviceCapabilities:
    device_ordinal: int          # index among AMD render nodes (HIP ordinal candidate)
    drm_render_node: str         # /dev/dri/renderD*
    has_decode_engine: bool      # VCN present?
    gfx_generation: str          # coarse: "gfx9" .. "gfx12" / "unknown" (backend selection)
    gcn_arch: str                # exact: e.g. "gfx1201" (kernel tuning / ISA)
    total_vram: int              # bytes; 0 if unreadable
    pci_device_id: str


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _gfx_versions(dev_sysfs: str) -> tuple[str, str]:
    """(coarse generation, exact arch) from amdgpu IP discovery GC version."""
    gc = os.path.join(dev_sysfs, "ip_discovery/die/0/GC/0")
    major, minor, rev = (_read(os.path.join(gc, f)) for f in ("major", "minor", "revision"))
    if major is None:
        return "unknown", "unknown"
    gen = _GC_TO_GFX.get(int(major), "unknown")
    if gen == "gfx10" and minor is not None and int(minor) >= 3:
        gen = "gfx10.3"
    # HIP arch naming: gfx<major><minor><revision-hex>, e.g. 12.0.1 -> gfx1201
    arch = (f"gfx{int(major)}{int(minor)}{int(rev):x}"
            if minor is not None and rev is not None else "unknown")
    return gen, arch


def _has_decode_engine(dev_sysfs: str) -> bool:
    # ip_discovery keeps historical hardware-ID names: VCN blocks appear as
    # "UVD" (verified on RDNA4); check both.
    die = os.path.join(dev_sysfs, "ip_discovery/die/0")
    return any(os.path.isdir(os.path.join(die, name)) for name in ("VCN", "UVD"))


def probe_devices() -> list[DeviceCapabilities]:
    devices: list[DeviceCapabilities] = []
    ordinal = 0
    for node in sorted(glob.glob("/dev/dri/renderD*")):
        sysfs = f"/sys/class/drm/{os.path.basename(node)}/device"
        if _read(os.path.join(sysfs, "vendor")) != AMD_VENDOR:
            continue
        vram = _read(os.path.join(sysfs, "mem_info_vram_total"))
        gen, arch = _gfx_versions(sysfs)
        devices.append(
            DeviceCapabilities(
                device_ordinal=ordinal,
                drm_render_node=node,
                has_decode_engine=_has_decode_engine(sysfs),
                gfx_generation=gen,
                gcn_arch=arch,
                total_vram=int(vram) if vram else 0,
                pci_device_id=_read(os.path.join(sysfs, "device")) or "unknown",
            )
        )
        ordinal += 1
    return devices


def require_decode_device(devices: list[DeviceCapabilities]) -> DeviceCapabilities:
    for d in devices:
        if d.has_decode_engine:
            return d
    raise RuntimeError(
        "No AMD device with a decode engine found. On decode-less hardware "
        "(some Instinct SKUs) use a deployment profile that feeds frames from elsewhere."
    )
