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

__device__ __forceinline__ void yuv_to_rgb(float y_raw, float u_raw, float v_raw,
                                           bool full_range, bool bt709,
                                           float& r, float& g, float& b) {
    // range normalization (edge case #6): limited 16-235 vs full 0-255
    const float Y = full_range ? y_raw / 255.0f : (y_raw - 16.0f) / 219.0f;
    const float U = full_range ? (u_raw - 128.0f) / 255.0f : (u_raw - 128.0f) / 224.0f;
    const float V = full_range ? (v_raw - 128.0f) / 255.0f : (v_raw - 128.0f) / 224.0f;
    if (bt709) {
        r = Y + 1.5748f * V;
        g = Y - 0.1873f * U - 0.4681f * V;
        b = Y + 1.8556f * U;
    } else {  // BT.601
        r = Y + 1.4020f * V;
        g = Y - 0.3441f * U - 0.7141f * V;
        b = Y + 1.7720f * U;
    }
    r = fminf(fmaxf(r, 0.0f), 1.0f);
    g = fminf(fmaxf(g, 0.0f), 1.0f);
    b = fminf(fmaxf(b, 0.0f), 1.0f);
}

// nvstreammux canvas analog (docs/nvstreammux_reverse_engineering.md §2):
// the whole canvas_w x canvas_h slot is written — the source crop is scaled
// into [dst_x, dst_x+dst_w) x [dst_y, dst_y+dst_h) and everything else is
// black (letterbox fill). Bilinear (half-pixel centres) or nearest.
__global__ void nv12_to_rgb_canvas_kernel(const uint8_t* __restrict__ y_plane,
                                          const uint8_t* __restrict__ uv_plane,
                                          int y_pitch, int uv_pitch,
                                          int src_x, int src_y, int src_w, int src_h,
                                          float* __restrict__ out,
                                          int canvas_w, int canvas_h,
                                          int dst_x, int dst_y, int dst_w, int dst_h,
                                          bool full_range, bool bt709, bool nearest) {
    const int cx_ = blockIdx.x * blockDim.x + threadIdx.x;
    const int cy_ = blockIdx.y * blockDim.y + threadIdx.y;
    if (cx_ >= canvas_w || cy_ >= canvas_h) return;
    const int plane_sz = canvas_w * canvas_h;
    const int idx = cy_ * canvas_w + cx_;
    const int ox = cx_ - dst_x, oy = cy_ - dst_y;
    if (ox < 0 || oy < 0 || ox >= dst_w || oy >= dst_h) {
        out[idx] = 0.0f; out[plane_sz + idx] = 0.0f; out[2 * plane_sz + idx] = 0.0f;
        return;
    }
    float y_raw, u_raw, v_raw;
    if (nearest) {
        const int sx = src_x + min(static_cast<int>((ox + 0.5f) * src_w / dst_w), src_w - 1);
        const int sy = src_y + min(static_cast<int>((oy + 0.5f) * src_h / dst_h), src_h - 1);
        y_raw = y_plane[sy * y_pitch + sx];
        const int cx = sx >> 1, cy = sy >> 1;
        u_raw = uv_plane[cy * uv_pitch + cx * 2];
        v_raw = uv_plane[cy * uv_pitch + cx * 2 + 1];
    } else {
        const float sx = src_x + (ox + 0.5f) * src_w / dst_w - 0.5f;
        const float sy = src_y + (oy + 0.5f) * src_h / dst_h - 0.5f;
        y_raw = sample_plane(y_plane, y_pitch, sx, sy,
                             src_x + src_w - 1, src_y + src_h - 1, 1);
        const float cx = sx * 0.5f, cy = sy * 0.5f;
        const int cmax_x = (src_x + src_w) / 2 - 1, cmax_y = (src_y + src_h) / 2 - 1;
        u_raw = sample_plane(uv_plane, uv_pitch, cx, cy, cmax_x, cmax_y, 2);
        v_raw = sample_plane(uv_plane + 1, uv_pitch, cx, cy, cmax_x, cmax_y, 2);
    }
    float r, g, b;
    yuv_to_rgb(y_raw, u_raw, v_raw, full_range, bt709, r, g, b);
    out[idx] = r;
    out[plane_sz + idx] = g;
    out[2 * plane_sz + idx] = b;
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

__global__ void rgb_crop_resize_kernel(const float* __restrict__ src,
                                       int src_w, int src_h,
                                       int cx, int cy, int cw, int ch,
                                       float* __restrict__ dst,
                                       int dst_w, int dst_h) {
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dst_w || oy >= dst_h) return;

    const float sx = cx + (ox + 0.5f) * cw / dst_w - 0.5f;
    const float sy = cy + (oy + 0.5f) * ch / dst_h - 0.5f;
    int x0 = static_cast<int>(floorf(sx)), y0 = static_cast<int>(floorf(sy));
    const float fx = sx - x0, fy = sy - y0;
    const int x1 = min(x0 + 1, src_w - 1), y1 = min(y0 + 1, src_h - 1);
    x0 = max(x0, 0); y0 = max(y0, 0);

    const int src_plane = src_w * src_h;
    const int dst_plane = dst_w * dst_h;
    for (int c = 0; c < 3; ++c) {
        const float* p = src + c * src_plane;
        const float v =
            (p[y0 * src_w + x0] * (1 - fx) + p[y0 * src_w + x1] * fx) * (1 - fy) +
            (p[y1 * src_w + x0] * (1 - fx) + p[y1 * src_w + x1] * fx) * fy;
        dst[c * dst_plane + oy * dst_w + ox] = v;
    }
}

}  // namespace

void launch_nv12_to_rgb_canvas(const uint8_t* y_plane, const uint8_t* uv_plane,
                               int y_pitch, int uv_pitch,
                               int src_x, int src_y, int src_w, int src_h,
                               float* out, int canvas_w, int canvas_h,
                               int dst_x, int dst_y, int dst_w, int dst_h,
                               bool full_range, bool bt709, bool nearest, void* stream) {
    const dim3 block(16, 16);
    const dim3 grid((canvas_w + block.x - 1) / block.x, (canvas_h + block.y - 1) / block.y);
    hipLaunchKernelGGL(nv12_to_rgb_canvas_kernel, grid, block, 0,
                       static_cast<hipStream_t>(stream),
                       y_plane, uv_plane, y_pitch, uv_pitch,
                       src_x, src_y, src_w, src_h,
                       out, canvas_w, canvas_h, dst_x, dst_y, dst_w, dst_h,
                       full_range, bt709, nearest);
}

void launch_rgb_crop_resize(const float* src, int src_w, int src_h,
                            int cx, int cy, int cw, int ch,
                            float* dst, int dst_w, int dst_h, void* stream) {
    const dim3 block(16, 16);
    const dim3 grid((dst_w + block.x - 1) / block.x,
                    (dst_h + block.y - 1) / block.y);
    hipLaunchKernelGGL(rgb_crop_resize_kernel, grid, block, 0,
                       static_cast<hipStream_t>(stream),
                       src, src_w, src_h, cx, cy, cw, ch, dst, dst_w, dst_h);
}

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

namespace {

// Inverse of nv12_to_rgb: CHW float RGB [0,1] -> NV12 (Y plane + interleaved
// UV at half resolution). One thread per 2x2 block: writes 4 Y and 1 UV pair;
// chroma from the 2x2 RGB average (center siting, matching common encoders).
__global__ void rgb_to_nv12_kernel(const float* __restrict__ rgb,
                                   int w, int h,
                                   uint8_t* __restrict__ y_out, int y_pitch,
                                   uint8_t* __restrict__ uv_out, int uv_pitch,
                                   bool full_range, bool bt709) {
    const int bx = blockIdx.x * blockDim.x + threadIdx.x;  // 2x2 block coords
    const int by = blockIdx.y * blockDim.y + threadIdx.y;
    if (2 * bx >= w || 2 * by >= h) return;

    const int plane = w * h;
    const float kr = bt709 ? 0.2126f : 0.2990f;
    const float kg = bt709 ? 0.7152f : 0.5870f;
    const float kb = bt709 ? 0.0722f : 0.1140f;

    float r_sum = 0, g_sum = 0, b_sum = 0;
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            const int x = min(2 * bx + dx, w - 1);
            const int y = min(2 * by + dy, h - 1);
            const int i = y * w + x;
            const float r = rgb[i], g = rgb[plane + i], b = rgb[2 * plane + i];
            const float lum = kr * r + kg * g + kb * b;
            const float yv = full_range ? 255.0f * lum : 16.0f + 219.0f * lum;
            if (2 * bx + dx < w && 2 * by + dy < h)
                y_out[y * y_pitch + x] = static_cast<uint8_t>(
                    fminf(fmaxf(yv + 0.5f, 0.0f), 255.0f));
            r_sum += r; g_sum += g; b_sum += b;
        }
    }
    const float r = r_sum * 0.25f, g = g_sum * 0.25f, b = b_sum * 0.25f;
    const float lum = kr * r + kg * g + kb * b;
    const float u = (b - lum) / (2.0f * (1.0f - kb));
    const float v = (r - lum) / (2.0f * (1.0f - kr));
    const float scale = full_range ? 255.0f : 224.0f;
    uv_out[by * uv_pitch + 2 * bx] = static_cast<uint8_t>(
        fminf(fmaxf(128.0f + scale * u + 0.5f, 0.0f), 255.0f));
    uv_out[by * uv_pitch + 2 * bx + 1] = static_cast<uint8_t>(
        fminf(fmaxf(128.0f + scale * v + 0.5f, 0.0f), 255.0f));
}

}  // namespace

void launch_rgb_to_nv12(const float* rgb, int w, int h,
                        uint8_t* y_out, int y_pitch,
                        uint8_t* uv_out, int uv_pitch,
                        bool full_range, bool bt709, void* stream) {
    const dim3 block(16, 16);
    const dim3 grid(((w + 1) / 2 + block.x - 1) / block.x,
                    ((h + 1) / 2 + block.y - 1) / block.y);
    hipLaunchKernelGGL(rgb_to_nv12_kernel, grid, block, 0,
                       static_cast<hipStream_t>(stream),
                       rgb, w, h, y_out, y_pitch, uv_out, uv_pitch,
                       full_range, bt709);
}

}  // namespace avap
