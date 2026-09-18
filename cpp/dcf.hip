// NvDCF-class correlation-filter engine (see dcf.h).

#include "dcf.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace avap {

namespace {

void check(hipError_t err, const char* what) {
    if (err != hipSuccess)
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(err));
}

void check_fft(hipfftResult r, const char* what) {
    if (r != HIPFFT_SUCCESS)
        throw std::runtime_error(std::string(what) + ": hipfft error " + std::to_string(int(r)));
}

constexpr int kCN = 11;   // colour-name prototypes
constexpr int kHOG = 9;   // orientation bins over [0, pi)

// Prototype RGB (0-1) for the 11 English colour names (black, blue, brown,
// grey, green, orange, pink, purple, red, white, yellow). A soft assignment
// to these stands in for the learned ColorNames table NvDCF uses.
__constant__ float kProto[kCN][3] = {
    {0.05f, 0.05f, 0.05f}, {0.10f, 0.20f, 0.85f}, {0.45f, 0.28f, 0.12f},
    {0.50f, 0.50f, 0.50f}, {0.15f, 0.65f, 0.20f}, {0.95f, 0.55f, 0.10f},
    {0.95f, 0.55f, 0.75f}, {0.55f, 0.20f, 0.70f}, {0.85f, 0.10f, 0.10f},
    {0.95f, 0.95f, 0.95f}, {0.95f, 0.90f, 0.15f},
};

__device__ __forceinline__ float sample_rgb(const float* __restrict__ frame, int W, int H,
                                            float fx, float fy, int c) {
    fx = fminf(fmaxf(fx, 0.0f), W - 1.0f);
    fy = fminf(fmaxf(fy, 0.0f), H - 1.0f);
    int x0 = static_cast<int>(fx), y0 = static_cast<int>(fy);
    int x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1);
    float ax = fx - x0, ay = fy - y0;
    const float* p = frame + static_cast<size_t>(c) * W * H;
    float v00 = p[y0 * W + x0], v01 = p[y0 * W + x1];
    float v10 = p[y1 * W + x0], v11 = p[y1 * W + x1];
    return (v00 * (1 - ax) + v01 * ax) * (1 - ay) + (v10 * (1 - ax) + v11 * ax) * ay;
}

__device__ __forceinline__ float gray_at(const float* frame, int W, int H, float fx, float fy) {
    return 0.299f * sample_rgb(frame, W, H, fx, fy, 0) + 0.587f * sample_rgb(frame, W, H, fx, fy, 1) +
           0.114f * sample_rgb(frame, W, H, fx, fy, 2);
}

// One thread per (target, y, x) feature pixel; writes all C channels.
__global__ void extract_kernel(const float* __restrict__ frame, int W, int H,
                               float sx, float sy, float ox, float oy,
                               const float* __restrict__ windows, int n, int S, int C,
                               bool use_gray, bool use_cn, bool use_hog,
                               const float* __restrict__ hann,
                               float* __restrict__ out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int t = blockIdx.z;
    if (x >= S || y >= S || t >= n) return;
    const float cx = windows[t * 4 + 0], cy = windows[t * 4 + 1];
    const float w = fmaxf(windows[t * 4 + 2], 1.0f), h = fmaxf(windows[t * 4 + 3], 1.0f);
    const float u = cx - w * 0.5f + (x + 0.5f) * w / S;    // source px
    const float v = cy - h * 0.5f + (y + 0.5f) * h / S;
    const float fx = ox + u * sx, fy = oy + v * sy;         // frame px
    float* o = out + (static_cast<size_t>(t) * C) * S * S + y * S + x;
    const size_t plane = static_cast<size_t>(S) * S;
    int c = 0;
    const float r = sample_rgb(frame, W, H, fx, fy, 0);
    const float g = sample_rgb(frame, W, H, fx, fy, 1);
    const float b = sample_rgb(frame, W, H, fx, fy, 2);
    if (use_gray) {
        o[c * plane] = 0.299f * r + 0.587f * g + 0.114f * b;
        ++c;
    }
    if (use_cn) {
        float p[kCN];
        float sum = 0.0f;
        const float inv2tau2 = 1.0f / (2.0f * 0.15f * 0.15f);
        for (int k = 0; k < kCN; ++k) {
            const float dr = r - kProto[k][0], dg = g - kProto[k][1], db = b - kProto[k][2];
            p[k] = __expf(-(dr * dr + dg * dg + db * db) * inv2tau2);
            sum += p[k];
        }
        for (int k = 0; k < kCN; ++k) {
            o[c * plane] = p[k] / (sum + 1e-6f);
            ++c;
        }
    }
    if (use_hog) {
        const float du = w / S, dv = h / S;   // one feature pixel in source units
        const float gx = gray_at(frame, W, H, ox + (u + du) * sx, fy) -
                         gray_at(frame, W, H, ox + (u - du) * sx, fy);
        const float gy = gray_at(frame, W, H, fx, oy + (v + dv) * sy) -
                         gray_at(frame, W, H, fx, oy + (v - dv) * sy);
        const float mag = sqrtf(gx * gx + gy * gy);
        float ang = atan2f(gy, gx);            // [-pi, pi]
        if (ang < 0) ang += 3.14159265f;       // unsigned orientation [0, pi)
        const float bin = ang / 3.14159265f * kHOG;
        int b0 = static_cast<int>(bin) % kHOG;
        int b1 = (b0 + 1) % kHOG;
        const float f1 = bin - floorf(bin);
        for (int k = 0; k < kHOG; ++k) {
            float val = 0.0f;
            if (k == b0) val += mag * (1.0f - f1);
            if (k == b1) val += mag * f1;
            o[c * plane] = val;
            ++c;
        }
    }
}

// Per (target, channel) plane: subtract the plane mean, multiply by the Hann
// window. One block per plane, blockDim.x threads strided over S*S.
__global__ void normalize_kernel(float* __restrict__ feat, int S, const float* __restrict__ hann) {
    const int plane_idx = blockIdx.x;           // t*C + c
    const int n = S * S;
    float* f = feat + static_cast<size_t>(plane_idx) * n;
    __shared__ float partial[256];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += f[i];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    const float mean = partial[0] / n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) f[i] = (f[i] - mean) * hann[i];
}

// R_hat[t] = sum_c A[slot,c] * F_hat[t,c] / (B[slot] + lambda)
__global__ void response_kernel(const hipfftComplex* __restrict__ fhat,
                                const hipfftComplex* __restrict__ A,
                                const float* __restrict__ B,
                                const int* __restrict__ slots, int n, int C, int S, int K,
                                float lambda, hipfftComplex* __restrict__ rhat) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;   // 0..S*K-1
    const int t = blockIdx.y;
    if (k >= S * K || t >= n) return;
    const int slot = slots[t];
    float re = 0.0f, im = 0.0f;
    for (int c = 0; c < C; ++c) {
        const hipfftComplex a = A[(static_cast<size_t>(slot) * C + c) * S * K + k];
        const hipfftComplex f = fhat[(static_cast<size_t>(t) * C + c) * S * K + k];
        re += a.x * f.x - a.y * f.y;
        im += a.x * f.y + a.y * f.x;
    }
    const float denom = B[static_cast<size_t>(slot) * S * K + k] + lambda;
    rhat[static_cast<size_t>(t) * S * K + k] = make_float2(re / denom, im / denom);
}

// A[slot,c] = (1-lr) A + lr * G_hat * conj(F_hat[t,c]);  B[slot] = (1-lr) B + lr * sum_c |F_hat|^2
__global__ void update_kernel(const hipfftComplex* __restrict__ fhat,
                              const hipfftComplex* __restrict__ ghat,
                              const int* __restrict__ slots, const uint8_t* __restrict__ init,
                              int n, int C, int S, int K, float lr,
                              hipfftComplex* __restrict__ A, float* __restrict__ B) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (k >= S * K || t >= n) return;
    const int slot = slots[t];
    const float a_lr = init[t] ? 1.0f : lr, keep = 1.0f - a_lr;
    const hipfftComplex g = ghat[k];
    float energy = 0.0f;
    for (int c = 0; c < C; ++c) {
        const hipfftComplex f = fhat[(static_cast<size_t>(t) * C + c) * S * K + k];
        // g * conj(f)
        const float re = g.x * f.x + g.y * f.y;
        const float im = g.y * f.x - g.x * f.y;
        hipfftComplex& a = A[(static_cast<size_t>(slot) * C + c) * S * K + k];
        a.x = keep * a.x + a_lr * re;
        a.y = keep * a.y + a_lr * im;
        energy += f.x * f.x + f.y * f.y;
    }
    float& b = B[static_cast<size_t>(slot) * S * K + k];
    b = keep * b + a_lr * energy;
}

__global__ void scale_kernel(float* __restrict__ resp, int total, float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) resp[i] *= scale;
}

}  // namespace

DcfEngine::DcfEngine(const DcfParams& p) : p_(p) {
    N_ = p.max_targets;
    S_ = p.feature_size;
    K_ = S_ / 2 + 1;
    C_ = (p.use_gray ? 1 : 0) + (p.use_colornames ? kCN : 0) + (p.use_hog ? kHOG : 0);
    if (C_ == 0) throw std::runtime_error("DcfEngine: no feature channel enabled");
    if (N_ <= 0 || S_ < 8) throw std::runtime_error("DcfEngine: bad max_targets/feature_size");
    check(hipSetDevice(p.device_ordinal), "hipSetDevice");

    const size_t nplanes = static_cast<size_t>(N_) * C_;
    check(hipMalloc(&d_windows_, N_ * 4 * sizeof(float)), "hipMalloc windows");
    check(hipMalloc(&d_slots_, N_ * sizeof(int)), "hipMalloc slots");
    check(hipMalloc(&d_init_, N_ * sizeof(uint8_t)), "hipMalloc init");
    check(hipMalloc(&d_feat_, nplanes * S_ * S_ * sizeof(float)), "hipMalloc feat");
    check(hipMalloc(&d_fhat_, nplanes * S_ * K_ * sizeof(hipfftComplex)), "hipMalloc fhat");
    check(hipMalloc(&d_A_, nplanes * S_ * K_ * sizeof(hipfftComplex)), "hipMalloc A");
    check(hipMalloc(&d_B_, static_cast<size_t>(N_) * S_ * K_ * sizeof(float)), "hipMalloc B");
    check(hipMalloc(&d_rhat_, static_cast<size_t>(N_) * S_ * K_ * sizeof(hipfftComplex)), "hipMalloc rhat");
    check(hipMalloc(&d_resp_, static_cast<size_t>(N_) * S_ * S_ * sizeof(float)), "hipMalloc resp");
    check(hipMalloc(&d_ghat_, S_ * K_ * sizeof(hipfftComplex)), "hipMalloc ghat");
    check(hipMalloc(&d_hann_, S_ * S_ * sizeof(float)), "hipMalloc hann");
    check(hipMemset(d_A_, 0, nplanes * S_ * K_ * sizeof(hipfftComplex)), "memset A");
    check(hipMemset(d_B_, 0, static_cast<size_t>(N_) * S_ * K_ * sizeof(float)), "memset B");
    check(hipMemset(d_feat_, 0, nplanes * S_ * S_ * sizeof(float)), "memset feat");

    int dims[2] = {S_, S_};
    int inembed[2] = {S_, S_}, onembed[2] = {S_, K_};
    check_fft(hipfftPlanMany(&plan_fwd_, 2, dims, inembed, 1, S_ * S_, onembed, 1, S_ * K_,
                             HIPFFT_R2C, static_cast<int>(nplanes)), "hipfftPlanMany R2C");
    int inembed2[2] = {S_, K_}, onembed2[2] = {S_, S_};
    check_fft(hipfftPlanMany(&plan_inv_, 2, dims, inembed2, 1, S_ * K_, onembed2, 1, S_ * S_,
                             HIPFFT_C2R, N_), "hipfftPlanMany C2R");
    build_window();
    build_gaussian();
}

DcfEngine::~DcfEngine() {
    hipfftDestroy(plan_fwd_);
    hipfftDestroy(plan_inv_);
    for (void* p : {static_cast<void*>(d_windows_), static_cast<void*>(d_slots_),
                    static_cast<void*>(d_init_), static_cast<void*>(d_feat_),
                    static_cast<void*>(d_fhat_), static_cast<void*>(d_A_),
                    static_cast<void*>(d_B_), static_cast<void*>(d_rhat_),
                    static_cast<void*>(d_resp_), static_cast<void*>(d_ghat_),
                    static_cast<void*>(d_hann_)})
        if (p) hipFree(p);
}

void DcfEngine::build_window() {
    std::vector<float> h(static_cast<size_t>(S_) * S_);
    const float cy = 0.5f * (S_ - 1) + p_.focus_offset_y * S_;
    for (int y = 0; y < S_; ++y) {
        // Hann centred at cy with the same half-width as the standard window
        float ry = (y - cy) / (0.5f * (S_ - 1));
        float wy = ry <= -1.0f || ry >= 1.0f ? 0.0f : 0.5f * (1.0f + std::cos(3.14159265f * ry));
        for (int x = 0; x < S_; ++x) {
            float wx = 0.5f * (1.0f - std::cos(2.0f * 3.14159265f * (x + 0.5f) / S_));
            h[y * S_ + x] = wx * wy;
        }
    }
    check(hipMemcpy(d_hann_, h.data(), h.size() * sizeof(float), hipMemcpyHostToDevice), "H2D hann");
}

void DcfEngine::set_gaussian_sigma(float sigma) {
    p_.gaussian_sigma = sigma;
    build_gaussian();
}

void DcfEngine::build_gaussian() {
    // Desired response: Gaussian peaked at (0,0) (circularly shifted), FFT'd once
    // with the same (unnormalised) forward transform used for features.
    std::vector<float> g(static_cast<size_t>(S_) * S_);
    const float s2 = 2.0f * p_.gaussian_sigma * p_.gaussian_sigma;
    for (int y = 0; y < S_; ++y) {
        int dy = y <= S_ / 2 ? y : y - S_;
        for (int x = 0; x < S_; ++x) {
            int dx = x <= S_ / 2 ? x : x - S_;
            g[y * S_ + x] = std::exp(-(dx * dx + dy * dy) / s2);
        }
    }
    float* d_g = nullptr;
    check(hipMalloc(&d_g, g.size() * sizeof(float)), "hipMalloc g");
    check(hipMemcpy(d_g, g.data(), g.size() * sizeof(float), hipMemcpyHostToDevice), "H2D g");
    hipfftHandle plan;
    int dims[2] = {S_, S_};
    check_fft(hipfftPlan2d(&plan, S_, S_, HIPFFT_R2C), "hipfftPlan2d G");
    check_fft(hipfftExecR2C(plan, d_g, d_ghat_), "hipfftExecR2C G");
    check(hipDeviceSynchronize(), "sync G");
    hipfftDestroy(plan);
    hipFree(d_g);
    (void)dims;
}

void DcfEngine::clear_slot(int slot) {
    if (slot < 0 || slot >= N_) return;
    check(hipMemset(d_A_ + static_cast<size_t>(slot) * C_ * S_ * K_, 0,
                    static_cast<size_t>(C_) * S_ * K_ * sizeof(hipfftComplex)), "memset A slot");
    check(hipMemset(d_B_ + static_cast<size_t>(slot) * S_ * K_, 0, S_ * K_ * sizeof(float)),
          "memset B slot");
}

void DcfEngine::extract(uintptr_t frame, int W, int H, float sx, float sy, float ox, float oy,
                        const float* windows, int n) {
    if (n <= 0 || n > N_) throw std::runtime_error("DcfEngine: n out of range");
    check(hipSetDevice(p_.device_ordinal), "hipSetDevice");
    check(hipMemcpy(d_windows_, windows, n * 4 * sizeof(float), hipMemcpyHostToDevice), "H2D windows");
    const dim3 block(16, 16, 1);
    const dim3 grid((S_ + 15) / 16, (S_ + 15) / 16, n);
    hipLaunchKernelGGL(extract_kernel, grid, block, 0, nullptr,
                       reinterpret_cast<const float*>(frame), W, H, sx, sy, ox, oy,
                       d_windows_, n, S_, C_, p_.use_gray, p_.use_colornames, p_.use_hog,
                       d_hann_, d_feat_);
    check(hipGetLastError(), "extract_kernel");
    hipLaunchKernelGGL(normalize_kernel, dim3(n * C_), dim3(256), 0, nullptr, d_feat_, S_, d_hann_);
    check(hipGetLastError(), "normalize_kernel");
    // batched forward FFT over the full pool (unused planes are stale but unread)
    check_fft(hipfftExecR2C(plan_fwd_, d_feat_, d_fhat_), "hipfftExecR2C features");
}

void DcfEngine::extract_features(uintptr_t frame, int W, int H, float sx, float sy, float ox,
                                 float oy, const float* windows, int n, float* features_out) {
    extract(frame, W, H, sx, sy, ox, oy, windows, n);
    check(hipDeviceSynchronize(), "sync extract");
    check(hipMemcpy(features_out, d_feat_, static_cast<size_t>(n) * C_ * S_ * S_ * sizeof(float),
                    hipMemcpyDeviceToHost), "D2H features");
}

void DcfEngine::localize(uintptr_t frame, int W, int H, float sx, float sy, float ox, float oy,
                         const float* windows, const int* slots, int n, float* responses_out) {
    extract(frame, W, H, sx, sy, ox, oy, windows, n);
    check(hipMemcpy(d_slots_, slots, n * sizeof(int), hipMemcpyHostToDevice), "H2D slots");
    const int SK = S_ * K_;
    const dim3 block(256, 1, 1), grid((SK + 255) / 256, n, 1);
    hipLaunchKernelGGL(response_kernel, grid, block, 0, nullptr,
                       d_fhat_, d_A_, d_B_, d_slots_, n, C_, S_, K_, p_.lambda, d_rhat_);
    check(hipGetLastError(), "response_kernel");
    check_fft(hipfftExecC2R(plan_inv_, d_rhat_, d_resp_), "hipfftExecC2R response");
    // both transforms unnormalised: forward on features (S*S) and inverse (S*S);
    // G_hat carries one forward factor, so R = IFFT(G_hat F F*/(FF*)) ~ S*S * G.
    const int total = n * S_ * S_;
    hipLaunchKernelGGL(scale_kernel, dim3((total + 255) / 256), dim3(256), 0, nullptr,
                       d_resp_, total, 1.0f / (static_cast<float>(S_) * S_));
    check(hipGetLastError(), "scale_kernel");
    check(hipDeviceSynchronize(), "sync localize");
    check(hipMemcpy(responses_out, d_resp_, static_cast<size_t>(total) * sizeof(float),
                    hipMemcpyDeviceToHost), "D2H responses");
}

void DcfEngine::update(uintptr_t frame, int W, int H, float sx, float sy, float ox, float oy,
                       const float* windows, const int* slots, const uint8_t* init, int n,
                       float lr) {
    extract(frame, W, H, sx, sy, ox, oy, windows, n);
    check(hipMemcpy(d_slots_, slots, n * sizeof(int), hipMemcpyHostToDevice), "H2D slots");
    check(hipMemcpy(d_init_, init, n * sizeof(uint8_t), hipMemcpyHostToDevice), "H2D init");
    const int SK = S_ * K_;
    const dim3 block(256, 1, 1), grid((SK + 255) / 256, n, 1);
    hipLaunchKernelGGL(update_kernel, grid, block, 0, nullptr,
                       d_fhat_, d_ghat_, d_slots_, d_init_, n, C_, S_, K_, lr, d_A_, d_B_);
    check(hipGetLastError(), "update_kernel");
    check(hipDeviceSynchronize(), "sync update");
}

}  // namespace avap
