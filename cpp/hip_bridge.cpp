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

void nv12_host_to_rgb(const ConvertRequest& req, const uint8_t* nv12,
                      size_t nv12_size, float* out_host) {
    if (req.planes.size() < 2)
        throw std::runtime_error("NV12 needs 2 planes, got " +
                                 std::to_string(req.planes.size()));
    check(hipSetDevice(req.device_ordinal), "hipSetDevice");

    uint8_t* d_nv12 = nullptr;
    float* d_out = nullptr;
    try {
        check(hipMalloc(&d_nv12, nv12_size), "hipMalloc(nv12)");
        check(hipMemcpy(d_nv12, nv12, nv12_size, hipMemcpyHostToDevice),
              "hipMemcpy H2D");

        const size_t out_bytes = 3ull * req.dst_w * req.dst_h * sizeof(float);
        check(hipMalloc(&d_out, out_bytes), "hipMalloc(out)");

        launch_nv12_to_rgb(d_nv12 + req.planes[0].first,
                           d_nv12 + req.planes[1].first,
                           static_cast<int>(req.planes[0].second),
                           static_cast<int>(req.planes[1].second),
                           req.src_x, req.src_y, req.src_w, req.src_h,
                           d_out, req.dst_w, req.dst_h,
                           req.full_range, req.bt709, /*stream=*/nullptr);
        check(hipGetLastError(), "nv12_to_rgb kernel launch");
        check(hipMemcpy(out_host, d_out, out_bytes, hipMemcpyDeviceToHost),
              "hipMemcpy D2H");
    } catch (...) {
        if (d_out) hipFree(d_out);
        if (d_nv12) hipFree(d_nv12);
        throw;
    }
    hipFree(d_out);
    hipFree(d_nv12);
}

// --- canvas / device helpers ---------------------------------------------------

uintptr_t device_alloc(size_t n_bytes, int device_ordinal) {
    check(hipSetDevice(device_ordinal), "hipSetDevice");
    void* p = nullptr;
    check(hipMalloc(&p, n_bytes), "hipMalloc(batch)");
    return reinterpret_cast<uintptr_t>(p);
}

void device_free(uintptr_t ptr) {
    if (ptr) hipFree(reinterpret_cast<void*>(ptr));
}

void device_memset(uintptr_t ptr, size_t n_bytes, int value) {
    check(hipMemset(reinterpret_cast<void*>(ptr), value, n_bytes), "hipMemset");
}

void device_to_host(uintptr_t ptr, size_t n_bytes, void* out_host) {
    check(hipMemcpy(out_host, reinterpret_cast<void*>(ptr), n_bytes, hipMemcpyDeviceToHost),
          "hipMemcpy D2H (canvas)");
}

namespace {

// Import a linear NV12 dmabuf; on success the caller owns ext_mem/base.
void import_dmabuf(const ConvertRequest& req, hipExternalMemory_t& ext_mem, void*& base) {
    if (req.drm_modifier != kLinearModifier) {
        ::close(req.dmabuf_fd);
        throw std::runtime_error(
            "tiled dmabuf (modifier=" + std::to_string(req.drm_modifier) +
            "): linear export required (architecture §5 edge case 1)");
    }
    if (req.planes.size() < 2) {
        ::close(req.dmabuf_fd);
        throw std::runtime_error("NV12 needs 2 planes, got " +
                                 std::to_string(req.planes.size()));
    }
    check(hipSetDevice(req.device_ordinal), "hipSetDevice");
    off_t size = ::lseek(req.dmabuf_fd, 0, SEEK_END);
    if (size <= 0) {
        ::close(req.dmabuf_fd);
        throw std::runtime_error("lseek on dmabuf failed — cannot size the import");
    }
    hipExternalMemoryHandleDesc mem_desc{};
    mem_desc.type = hipExternalMemoryHandleTypeOpaqueFd;
    mem_desc.handle.fd = req.dmabuf_fd;
    mem_desc.size = static_cast<unsigned long long>(size);
    hipError_t err = hipImportExternalMemory(&ext_mem, &mem_desc);
    if (err != hipSuccess) {
        ::close(req.dmabuf_fd);
        check(err, "hipImportExternalMemory (dmabuf)");
    }
    hipExternalMemoryBufferDesc buf_desc{};
    buf_desc.offset = 0;
    buf_desc.size = mem_desc.size;
    err = hipExternalMemoryGetMappedBuffer(&base, ext_mem, &buf_desc);
    if (err != hipSuccess) {
        hipDestroyExternalMemory(ext_mem);
        check(err, "hipExternalMemoryGetMappedBuffer");
    }
}

void run_canvas(const ConvertRequest& req, const CanvasPlacement& place,
                const uint8_t* y_plane, const uint8_t* uv_plane, uintptr_t d_slot) {
    launch_nv12_to_rgb_canvas(y_plane, uv_plane,
                              static_cast<int>(req.planes[0].second),
                              static_cast<int>(req.planes[1].second),
                              req.src_x, req.src_y, req.src_w, req.src_h,
                              reinterpret_cast<float*>(d_slot),
                              place.canvas_w, place.canvas_h,
                              place.dst_x, place.dst_y, place.dst_w, place.dst_h,
                              req.full_range, req.bt709, place.nearest, /*stream=*/nullptr);
    check(hipGetLastError(), "nv12_to_rgb_canvas kernel launch");
    check(hipDeviceSynchronize(), "hipDeviceSynchronize (canvas)");
}

}  // namespace

void nv12_dmabuf_to_canvas(const ConvertRequest& req, const CanvasPlacement& place,
                           uintptr_t d_slot) {
    hipExternalMemory_t ext_mem{};
    void* base = nullptr;
    import_dmabuf(req, ext_mem, base);
    try {
        const auto* y_plane = static_cast<const uint8_t*>(base) + req.planes[0].first;
        const auto* uv_plane = static_cast<const uint8_t*>(base) + req.planes[1].first;
        run_canvas(req, place, y_plane, uv_plane, d_slot);
    } catch (...) {
        hipFree(base);
        hipDestroyExternalMemory(ext_mem);
        throw;
    }
    hipFree(base);
    hipDestroyExternalMemory(ext_mem);
}

void nv12_host_to_canvas(const ConvertRequest& req, const CanvasPlacement& place,
                         const uint8_t* nv12, size_t nv12_size, uintptr_t d_slot) {
    if (req.planes.size() < 2)
        throw std::runtime_error("NV12 needs 2 planes, got " +
                                 std::to_string(req.planes.size()));
    check(hipSetDevice(req.device_ordinal), "hipSetDevice");
    uint8_t* d_nv12 = nullptr;
    check(hipMalloc(&d_nv12, nv12_size), "hipMalloc(nv12)");
    try {
        check(hipMemcpy(d_nv12, nv12, nv12_size, hipMemcpyHostToDevice), "hipMemcpy H2D");
        run_canvas(req, place, d_nv12 + req.planes[0].first, d_nv12 + req.planes[1].first,
                   d_slot);
    } catch (...) {
        hipFree(d_nv12);
        throw;
    }
    hipFree(d_nv12);
}

}  // namespace avap
