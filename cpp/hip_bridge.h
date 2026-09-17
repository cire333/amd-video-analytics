#pragma once
// dmabuf -> HIP import + fused NV12->RGB/crop/resize (architecture §5).

#include <cstdint>
#include <utility>
#include <vector>

namespace avap {

struct ConvertRequest {
    int dmabuf_fd = -1;         // ownership transfers here (closed/consumed)
    int surf_width = 0;         // exported surface dims (padded)
    int surf_height = 0;
    std::vector<std::pair<uint32_t, uint32_t>> planes;  // (offset, pitch) Y, UV
    uint64_t drm_modifier = 0;
    int src_x = 0, src_y = 0, src_w = 0, src_h = 0;  // region to convert (ROI ∩ crop)
    int dst_w = 0, dst_h = 0;   // model input size
    bool full_range = false;
    bool bt709 = true;
    int device_ordinal = 0;
};

// Writes float32 CHW RGB (normalized 0-1) into out_host (3*dst_w*dst_h floats).
// Throws std::runtime_error on failure; always consumes/closes the fd.
void nv12_dmabuf_to_rgb(const ConvertRequest& req, float* out_host);

// Same conversion from host NV12 (driver-detiled fallback for tiled
// surfaces): uploads the planes, runs the same fused kernel. req.dmabuf_fd
// is ignored.
void nv12_host_to_rgb(const ConvertRequest& req, const uint8_t* nv12,
                      size_t nv12_size, float* out_host);

// --- Device-resident frames (GPU SGIE crop path) ---------------------------
// Same conversions, but the RGB tensor STAYS on the GPU; the returned
// pointer (as uintptr) must be released with free_device_buffer.
uintptr_t nv12_dmabuf_to_device_rgb(const ConvertRequest& req);
uintptr_t nv12_host_to_device_rgb(const ConvertRequest& req,
                                  const uint8_t* nv12, size_t nv12_size);

// Crop + bilinear-resize from a device CHW RGB frame directly into a
// destination device buffer (e.g. a MIGraphX input argument's data_ptr).
// Synchronizes before returning so the consumer can read immediately.
void rgb_crop_resize_device(uintptr_t src, int src_w, int src_h,
                            int cx, int cy, int cw, int ch,
                            uintptr_t dst, int dst_w, int dst_h,
                            int device_ordinal);

void device_rgb_to_host(uintptr_t src, size_t n_floats, float* out_host);
void free_device_buffer(uintptr_t ptr);

// Kernel launcher (defined in kernels.hip).
void launch_rgb_crop_resize(const float* src, int src_w, int src_h,
                            int cx, int cy, int cw, int ch,
                            float* dst, int dst_w, int dst_h, void* stream);

int hip_device_count();

// Kernel launcher (defined in kernels.hip).
void launch_nv12_to_rgb(const uint8_t* y_plane, const uint8_t* uv_plane,
                        int y_pitch, int uv_pitch,
                        int src_x, int src_y, int src_w, int src_h,
                        float* out, int dst_w, int dst_h,
                        bool full_range, bool bt709, void* stream);

}  // namespace avap
