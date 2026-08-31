// Fused NV12 -> RGB + crop + bilinear resize (architecture §5, §8).
// Cropping happens at conversion, so the full RGB frame is never
// materialized. Output: float32 CHW, normalized 0-1.

#include <hip/hip_runtime.h>

#include "hip_bridge.h"

namespace avap {

namespace {

__device__ __forceinline__ float sample_plane(const uint8_t* p, int pitch,
                                              float x, float y,
                                              int max_x, int max_y, int stride_px) {
    // bilinear sample; coords in plane pixel units, stride_px = bytes per sample step
    int x0 = static_cast<int>(floorf(x)), y0 = static_cast<int>(floorf(y));
    float fx = x - x0, fy = y - y0;
    int x1 = min(x0 + 1, max_x), y1 = min(y0 + 1, max_y);
    x0 = max(x0, 0); y0 = max(y0, 0);
    float v00 = p[y0 * pitch + x0 * stride_px];
    float v01 = p[y0 * pitch + x1 * stride_px];
    float v10 = p[y1 * pitch + x0 * stride_px];
    float v11 = p[y1 * pitch + x1 * stride_px];
    return (v00 * (1 - fx) + v01 * fx) * (1 - fy) + (v10 * (1 - fx) + v11 * fx) * fy;
}

__global__ void nv12_to_rgb_kernel(const uint8_t* __restrict__ y_plane,
                                   const uint8_t* __restrict__ uv_plane,
                                   int y_pitch, int uv_pitch,
                                   int src_x, int src_y, int src_w, int src_h,
                                   float* __restrict__ out, int dst_w, int dst_h,
                                   bool full_range, bool bt709) {
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dst_w || oy >= dst_h) return;

    // map output pixel center into the source crop region
    const float sx = src_x + (ox + 0.5f) * src_w / dst_w - 0.5f;
    const float sy = src_y + (oy + 0.5f) * src_h / dst_h - 0.5f;

    const float y_raw = sample_plane(y_plane, y_pitch, sx, sy,
                                     src_x + src_w - 1, src_y + src_h - 1, 1);
    // chroma at half resolution, interleaved U,V
    const float cx = sx * 0.5f, cy = sy * 0.5f;
    const int cmax_x = (src_x + src_w) / 2 - 1, cmax_y = (src_y + src_h) / 2 - 1;
    const float u_raw = sample_plane(uv_plane, uv_pitch, cx, cy, cmax_x, cmax_y, 2);
    const float v_raw = sample_plane(uv_plane + 1, uv_pitch, cx, cy, cmax_x, cmax_y, 2);

    // range normalization (edge case #6): limited 16-235 vs full 0-255
    const float Y = full_range ? y_raw / 255.0f : (y_raw - 16.0f) / 219.0f;
    const float U = full_range ? (u_raw - 128.0f) / 255.0f : (u_raw - 128.0f) / 224.0f;
    const float V = full_range ? (v_raw - 128.0f) / 255.0f : (v_raw - 128.0f) / 224.0f;

    float r, g, b;
    if (bt709) {
        r = Y + 1.5748f * V;
        g = Y - 0.1873f * U - 0.4681f * V;
        b = Y + 1.8556f * U;
    } else {  // BT.601
        r = Y + 1.4020f * V;
        g = Y - 0.3441f * U - 0.7141f * V;
        b = Y + 1.7720f * U;
    }

    const int plane_sz = dst_w * dst_h;
    const int idx = oy * dst_w + ox;
    out[idx] = fminf(fmaxf(r, 0.0f), 1.0f);
    out[plane_sz + idx] = fminf(fmaxf(g, 0.0f), 1.0f);
    out[2 * plane_sz + idx] = fminf(fmaxf(b, 0.0f), 1.0f);
}

}  // namespace

void launch_nv12_to_rgb(const uint8_t* y_plane, const uint8_t* uv_plane,
                        int y_pitch, int uv_pitch,
                        int src_x, int src_y, int src_w, int src_h,
                        float* out, int dst_w, int dst_h,
                        bool full_range, bool bt709, void* stream) {
    const dim3 block(16, 16);
    const dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
    hipLaunchKernelGGL(nv12_to_rgb_kernel, grid, block, 0,
                       static_cast<hipStream_t>(stream),
                       y_plane, uv_plane, y_pitch, uv_pitch,
                       src_x, src_y, src_w, src_h,
                       out, dst_w, dst_h, full_range, bt709);
}

}  // namespace avap
