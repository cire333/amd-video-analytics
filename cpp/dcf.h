#pragma once
// NvDCF-class visual tracker core: multi-channel discriminative correlation
// filters (MOSSE/DCF formulation) on HIP + hipFFT.
//
// One DcfEngine serves one stream: it owns a pool of `max_targets` filter
// slots (numerator A[c], denominator B in the Fourier domain) and batched
// FFT plans for a fixed S x S feature template with C channels
// (gray, 11 ColorNames-style soft colour channels, 9 gradient-orientation
// channels; each optional).
//
// Per frame the Python tracker calls
//   localize(): extract windowed features around every target's predicted
//               search window, correlate with its filter, return the S x S
//               response maps (host) — peak = displacement, peak value =
//               tracker confidence, value at a detection = visual similarity.
//   update():   extract features at the final target box and EMA the filter
//               (or initialise it for new targets).
// Frames are device CHW float32 RGB (a BatchCanvas slot or any uploaded
// tensor); an affine (sx, sy, ox, oy) maps source pixels to frame pixels so
// the tracker can work in source coordinates.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <hipfft/hipfft.h>

namespace avap {

struct DcfParams {
    int max_targets = 150;
    int feature_size = 32;         // S
    bool use_gray = true;
    bool use_colornames = true;
    bool use_hog = false;
    float lambda = 1e-2f;          // filter regularisation
    float gaussian_sigma = 2.0f;   // desired-response sigma, feature pixels
    float focus_offset_y = 0.0f;   // Hann window centre offset (fraction of S)
    int device_ordinal = 0;
};

class DcfEngine {
public:
    explicit DcfEngine(const DcfParams& p);
    ~DcfEngine();
    DcfEngine(const DcfEngine&) = delete;
    DcfEngine& operator=(const DcfEngine&) = delete;

    int channels() const { return C_; }
    int feature_size() const { return S_; }
    int max_targets() const { return N_; }

    // windows: n x 4 (cx, cy, w, h) in SOURCE pixels; slots: n filter slots.
    // responses_out: n * S * S floats, normalised so a perfect match peaks ~1.
    void localize(uintptr_t frame, int W, int H, float sx, float sy, float ox, float oy,
                  const float* windows, const int* slots, int n, float* responses_out);

    // Same extraction at the final boxes; init[i] != 0 replaces the filter,
    // otherwise A = (1-lr) A + lr * G conj(F), B = (1-lr) B + lr * sum_c |F_c|^2.
    void update(uintptr_t frame, int W, int H, float sx, float sy, float ox, float oy,
                const float* windows, const int* slots, const uint8_t* init, int n, float lr);

    // Debug/parity: extracted windowed features for the given windows, n*C*S*S floats.
    void extract_features(uintptr_t frame, int W, int H, float sx, float sy, float ox, float oy,
                          const float* windows, int n, float* features_out);

    void clear_slot(int slot);
    void set_gaussian_sigma(float sigma);

private:
    void extract(uintptr_t frame, int W, int H, float sx, float sy, float ox, float oy,
                 const float* windows, int n);
    void build_gaussian();
    void build_window();

    int N_, S_, C_, K_;            // K_ = S/2+1 (R2C width)
    DcfParams p_;
    hipfftHandle plan_fwd_{};      // R2C, batch N*C
    hipfftHandle plan_inv_{};      // C2R, batch N
    float* d_windows_ = nullptr;   // N x 4
    int* d_slots_ = nullptr;       // N
    uint8_t* d_init_ = nullptr;    // N
    float* d_feat_ = nullptr;      // N x C x S x S (windowed features)
    hipfftComplex* d_fhat_ = nullptr;   // N x C x S x K
    hipfftComplex* d_A_ = nullptr;      // N x C x S x K  (per slot)
    float* d_B_ = nullptr;              // N x S x K      (per slot)
    hipfftComplex* d_rhat_ = nullptr;   // N x S x K
    float* d_resp_ = nullptr;           // N x S x S
    hipfftComplex* d_ghat_ = nullptr;   // S x K  (FFT of desired response)
    float* d_hann_ = nullptr;           // S x S
};

}  // namespace avap
