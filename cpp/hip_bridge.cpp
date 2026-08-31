#include "hip_bridge.h"

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>
#include <unistd.h>

namespace avap {

namespace {

void check(hipError_t err, const char* what) {
    if (err != hipSuccess)
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(err));
}

// DRM_FORMAT_MOD_LINEAR
constexpr uint64_t kLinearModifier = 0;

}  // namespace

int hip_device_count() {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess) return 0;
    return n;
}

void nv12_dmabuf_to_rgb(const ConvertRequest& req, float* out_host) {
    if (req.drm_modifier != kLinearModifier) {
        ::close(req.dmabuf_fd);
        throw std::runtime_error(
            "tiled dmabuf (modifier=" + std::to_string(req.drm_modifier) +
            "): v1 handles linear only — force linear export or add the "
            "per-GFX-gen detile kernel (architecture §5 edge case 1)");
    }
    if (req.planes.size() < 2) {
        ::close(req.dmabuf_fd);
        throw std::runtime_error("NV12 needs 2 planes, got " +
                                 std::to_string(req.planes.size()));
    }

    check(hipSetDevice(req.device_ordinal), "hipSetDevice");

    // dmabufs support lseek(SEEK_END) to report their size.
    off_t size = ::lseek(req.dmabuf_fd, 0, SEEK_END);
    if (size <= 0) {
        ::close(req.dmabuf_fd);
        throw std::runtime_error("lseek on dmabuf failed — cannot size the import");
    }

    hipExternalMemoryHandleDesc mem_desc{};
    mem_desc.type = hipExternalMemoryHandleTypeOpaqueFd;
    mem_desc.handle.fd = req.dmabuf_fd;
    mem_desc.size = static_cast<unsigned long long>(size);

    hipExternalMemory_t ext_mem{};
    hipError_t err = hipImportExternalMemory(&ext_mem, &mem_desc);
    if (err != hipSuccess) {
        // fd ownership transfers to the runtime only on success
        ::close(req.dmabuf_fd);
        check(err, "hipImportExternalMemory (dmabuf)");
    }

    void* base = nullptr;
    float* d_out = nullptr;
    try {
        hipExternalMemoryBufferDesc buf_desc{};
        buf_desc.offset = 0;
        buf_desc.size = mem_desc.size;
        check(hipExternalMemoryGetMappedBuffer(&base, ext_mem, &buf_desc),
              "hipExternalMemoryGetMappedBuffer");

        // honor per-plane offsets/pitches (edge cases #2, #3)
        const auto* y_plane = static_cast<const uint8_t*>(base) + req.planes[0].first;
        const auto* uv_plane = static_cast<const uint8_t*>(base) + req.planes[1].first;

        const size_t out_bytes = 3ull * req.dst_w * req.dst_h * sizeof(float);
        check(hipMalloc(&d_out, out_bytes), "hipMalloc(out)");

        launch_nv12_to_rgb(y_plane, uv_plane,
                           static_cast<int>(req.planes[0].second),
                           static_cast<int>(req.planes[1].second),
                           req.src_x, req.src_y, req.src_w, req.src_h,
                           d_out, req.dst_w, req.dst_h,
                           req.full_range, req.bt709, /*stream=*/nullptr);
        check(hipGetLastError(), "nv12_to_rgb kernel launch");

        // V1: sync copy back to host; zero-copy handoff to ORT is the
        // planned optimization once e2e is verified.
        check(hipMemcpy(out_host, d_out, out_bytes, hipMemcpyDeviceToHost),
              "hipMemcpy D2H");
    } catch (...) {
        if (d_out) hipFree(d_out);
        if (base) hipFree(base);
        hipDestroyExternalMemory(ext_mem);
        throw;
    }

    hipFree(d_out);
    hipFree(base);
    hipDestroyExternalMemory(ext_mem);
}

}  // namespace avap
