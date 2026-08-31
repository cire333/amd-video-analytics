"""Bridge smoke test: decode N frames via VAAPI, import each dmabuf into
HIP, run the fused NV12->RGB kernel, dump PNGs for visual inspection.

    python scripts/smoke_bridge.py <video> <out_dir> [n_frames]

Reports the DRM modifier (linear vs tiled — bring-up risk #1) and whether
hipImportExternalMemory accepts the dmabuf (risk #2). Correct output looks
like the source video; classic failure signatures:
  - diagonal shear        -> pitch treated as width
  - garbled blocks        -> tiled surface imported as linear
  - luma fine, chroma off -> wrong UV plane offset
  - washed out / oversat  -> wrong color range/matrix
"""
import sys

import numpy as np

from avap import _core
from avap.capabilities import probe_devices, require_decode_device


def save_png(path: str, chw: np.ndarray) -> None:
    rgb = (np.clip(chw, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
    try:
        from PIL import Image
        Image.fromarray(rgb).save(path)
    except ImportError:  # no pillow: fall back to ppm, same visual check
        path = path.rsplit(".", 1)[0] + ".ppm"
        with open(path, "wb") as f:
            f.write(f"P6 {rgb.shape[1]} {rgb.shape[0]} 255\n".encode())
            f.write(rgb.tobytes())
    print(f"  wrote {path}")


def main():
    video, out_dir = sys.argv[1], sys.argv[2]
    n_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    dev = require_decode_device(probe_devices())
    print(f"device: {dev.drm_render_node} {dev.gcn_arch} "
          f"(hip sees {_core.hip_device_count()} device(s))")

    dec = _core.Decoder(video, dev.drm_render_node)
    for i in range(n_frames):
        f = dec.next_frame()
        if f is None:
            print("EOF")
            break
        path = "dmabuf" if f.dmabuf_fd >= 0 else "host-copy"
        mod = "LINEAR" if f.drm_modifier == 0 else f"0x{f.drm_modifier:016x}"
        print(f"frame {i}: {f.width}x{f.height} crop={f.crop_w}x{f.crop_h} "
              f"planes={f.planes} modifier={mod} path={path} "
              f"range={'full' if f.full_range else 'limited'} matrix={f.color_matrix}")
        args = (list(f.planes), (f.crop_x, f.crop_y, f.crop_w, f.crop_h),
                (960, 540),  # downscale exercises the bilinear path
                f.full_range, f.color_matrix != "bt601", dev.device_ordinal)
        if f.dmabuf_fd >= 0:
            tensor = _core.nv12_dmabuf_to_rgb(
                f.dmabuf_fd, f.width, f.height, args[0], f.drm_modifier, *args[1:])
        else:
            tensor = _core.nv12_host_to_rgb(f.host_data, *args)
        print(f"  tensor {tensor.shape} min={tensor.min():.3f} "
              f"max={tensor.max():.3f} mean={tensor.mean():.3f}")
        save_png(f"{out_dir}/frame_{i:03d}.png", tensor)
    dec.close()
    print("smoke test complete")


if __name__ == "__main__":
    main()
