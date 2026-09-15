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

// Runs the fused NV12->RGB kernel against device plane pointers, returning
// a freshly hipMalloc'd CHW float output the caller owns.
float* convert_to_device(const ConvertRequest& req, const uint8_t* y_plane,
                         const uint8_t* uv_plane) {
    float* d_out = nullptr;
    const size_t out_bytes = 3ull * req.dst_w * req.dst_h * sizeof(float);
    check(hipMalloc(&d_out, out_bytes), "hipMalloc(out)");
    launch_nv12_to_rgb(y_plane, uv_plane,
                       static_cast<int>(req.planes[0].second),
                       static_cast<int>(req.planes[1].second),
                       req.src_x, req.src_y, req.src_w, req.src_h,
                       d_out, req.dst_w, req.dst_h,
                       req.full_range, req.bt709, /*stream=*/nullptr);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        hipFree(d_out);
        check(err, "nv12_to_rgb kernel launch");
    }
    return d_out;
}

// dmabuf import + convert; on success the fd is owned by the runtime.
float* dmabuf_to_device(const ConvertRequest& req) {
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
        ::close(req.dmabuf_fd);   // ownership transfers only on success
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
        const auto* y_plane = static_cast<const uint8_t*>(base) + req.planes[0].first;
        const auto* uv_plane = static_cast<const uint8_t*>(base) + req.planes[1].first;
        d_out = convert_to_device(req, y_plane, uv_plane);
        // the kernel reads from the imported surface: finish before unmapping
        check(hipDeviceSynchronize(), "hipDeviceSynchronize");
    } catch (...) {
        if (base) hipFree(base);
        hipDestroyExternalMemory(ext_mem);
        throw;
    }
    hipFree(base);
    hipDestroyExternalMemory(ext_mem);
    return d_out;
}

// host NV12 upload + convert
float* host_to_device(const ConvertRequest& req, const uint8_t* nv12,
                      size_t nv12_size) {
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
        d_out = convert_to_device(req, d_nv12 + req.planes[0].first,
                                  d_nv12 + req.planes[1].first);
        check(hipDeviceSynchronize(), "hipDeviceSynchronize");
    } catch (...) {
        if (d_out) hipFree(d_out);
        if (d_nv12) hipFree(d_nv12);
        throw;
    }
    hipFree(d_nv12);
    return d_out;
}

void copy_out_and_free(float* d_out, const ConvertRequest& req, float* out_host) {
    const size_t out_bytes = 3ull * req.dst_w * req.dst_h * sizeof(float);
    hipError_t err = hipMemcpy(out_host, d_out, out_bytes, hipMemcpyDeviceToHost);
    hipFree(d_out);
    check(err, "hipMemcpy D2H");
}

}  // namespace

int hip_device_count() {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess) return 0;
    return n;
}

void nv12_dmabuf_to_rgb(const ConvertRequest& req, float* out_host) {
    copy_out_and_free(dmabuf_to_device(req), req, out_host);
}

void nv12_host_to_rgb(const ConvertRequest& req, const uint8_t* nv12,
                      size_t nv12_size, float* out_host) {
    copy_out_and_free(host_to_device(req, nv12, nv12_size), req, out_host);
}

uintptr_t nv12_dmabuf_to_device_rgb(const ConvertRequest& req) {
    return reinterpret_cast<uintptr_t>(dmabuf_to_device(req));
}

uintptr_t nv12_host_to_device_rgb(const ConvertRequest& req,
                                  const uint8_t* nv12, size_t nv12_size) {
    return reinterpret_cast<uintptr_t>(host_to_device(req, nv12, nv12_size));
}

void rgb_crop_resize_device(uintptr_t src, int src_w, int src_h,
                            int cx, int cy, int cw, int ch,
                            uintptr_t dst, int dst_w, int dst_h,
                            int device_ordinal) {
    check(hipSetDevice(device_ordinal), "hipSetDevice");
    launch_rgb_crop_resize(reinterpret_cast<const float*>(src), src_w, src_h,
                           cx, cy, cw, ch,
                           reinterpret_cast<float*>(dst), dst_w, dst_h,
                           /*stream=*/nullptr);
    check(hipGetLastError(), "rgb_crop_resize kernel launch");
    // the destination is typically another runtime's buffer (MIGraphX input):
    // make the write visible before returning
    check(hipDeviceSynchronize(), "hipDeviceSynchronize");
}

void device_rgb_to_host(uintptr_t src, size_t n_floats, float* out_host) {
    check(hipMemcpy(out_host, reinterpret_cast<const void*>(src),
                    n_floats * sizeof(float), hipMemcpyDeviceToHost),
          "hipMemcpy D2H");
}

void free_device_buffer(uintptr_t ptr) {
    hipFree(reinterpret_cast<void*>(ptr));
}

}  // namespace avap
