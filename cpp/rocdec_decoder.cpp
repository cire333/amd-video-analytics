#include "rocdec_decoder.h"

#include <hip/hip_runtime.h>

#include <deque>
#include <stdexcept>

#include "hip_bridge.h"
#include "thirdparty/rocdecode/roc_video_dec.h"
#include "thirdparty/rocdecode/video_demuxer.h"

extern "C" {
#include <libavformat/avformat.h>
}

namespace avap {

namespace {
void check(hipError_t err, const char* what) {
    if (err != hipSuccess)
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(err));
}
}  // namespace

struct RocDecoder::Impl {
    std::unique_ptr<VideoDemuxer> demuxer;
    std::unique_ptr<RocVideoDecoder> dec;
    OutputSurfaceInfo* surf = nullptr;
    int device_ordinal = 0;
    bool eof = false;
    int pending = 0;   // frames decoded but not yet fetched
    bool bt709 = true;        // untagged -> BT.709 limited (VAAPI-backend parity)
    bool full_range = false;
};

// One cheap FFmpeg probe for the stream's colorimetry tags (the rocDecode
// reference decoder does not expose video_signal_description).
static void probe_colorimetry(const std::string& uri, bool* bt709,
                              bool* full_range) {
    AVFormatContext* fmt = nullptr;
    if (avformat_open_input(&fmt, uri.c_str(), nullptr, nullptr) != 0) return;
    if (avformat_find_stream_info(fmt, nullptr) >= 0) {
        int idx = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (idx >= 0) {
            const AVCodecParameters* par = fmt->streams[idx]->codecpar;
            if (par->color_space == AVCOL_SPC_BT470BG ||
                par->color_space == AVCOL_SPC_SMPTE170M)
                *bt709 = false;
            if (par->color_range == AVCOL_RANGE_JPEG)
                *full_range = true;
        }
    }
    avformat_close_input(&fmt);
}

RocDecoder::RocDecoder(const std::string& uri, int device_ordinal)
    : impl_(new Impl) {
    impl_->device_ordinal = device_ordinal;
    probe_colorimetry(uri, &impl_->bt709, &impl_->full_range);
    impl_->demuxer.reset(new VideoDemuxer(uri.c_str()));
    rocDecVideoCodec codec =
        AVCodec2RocDecVideoCodec(impl_->demuxer->GetCodecID());
    impl_->dec.reset(new RocVideoDecoder(device_ordinal,
                                         OUT_SURFACE_MEM_DEV_INTERNAL, codec,
                                         /*force_zero_latency=*/false));
}

RocDecoder::~RocDecoder() { close(); }

void RocDecoder::close() {
    impl_->dec.reset();
    impl_->demuxer.reset();
}

int RocDecoder::width() const {
    return impl_->surf ? static_cast<int>(impl_->surf->output_width) : 0;
}

int RocDecoder::height() const {
    return impl_->surf
        ? static_cast<int>(impl_->surf->disp_rect.bottom - impl_->surf->disp_rect.top)
        : 0;
}

// Pump demux+decode until one decoded frame is available (device NV12).
// Returns false at EOF (after draining the decoder).
bool RocDecoder::advance(uint8_t** dev_nv12, int64_t* pts) {
    if (!impl_->dec) return false;
    while (impl_->pending == 0) {
        if (impl_->eof) return false;
        uint8_t* pkt = nullptr;
        int n_bytes = 0;
        int64_t pkt_pts = 0;
        impl_->demuxer->Demux(&pkt, &n_bytes, &pkt_pts);
        int flags = 0;
        if (n_bytes == 0) {
            impl_->eof = true;
            flags = ROCDEC_PKT_ENDOFSTREAM;   // drain
        }
        impl_->pending += impl_->dec->DecodeFrame(pkt, n_bytes, flags, pkt_pts);
        if (!impl_->surf && impl_->pending > 0) {
            if (!impl_->dec->GetOutputSurfaceInfo(&impl_->surf))
                throw std::runtime_error("rocDecode: no output surface info");
            if (impl_->surf->bit_depth != 8 || impl_->surf->num_chroma_planes != 1)
                throw std::runtime_error("rocDecode backend supports 8-bit 4:2:0 only");
        }
    }
    *dev_nv12 = impl_->dec->GetFrame(pts);
    impl_->pending -= 1;
    return *dev_nv12 != nullptr;
}

bool RocDecoder::stream_bt709() const { return impl_->bt709; }
bool RocDecoder::stream_full_range() const { return impl_->full_range; }

bool RocDecoder::next_frame_device_rgb(uintptr_t* rgb_out, int* width_out,
                                       int* height_out, int64_t* pts_us) {
    const bool bt709 = impl_->bt709;
    const bool full_range = impl_->full_range;
    uint8_t* nv12 = nullptr;
    int64_t pts = 0;
    if (!advance(&nv12, &pts)) return false;

    const auto* s = impl_->surf;
    const int w = static_cast<int>(s->disp_rect.right - s->disp_rect.left);
    const int h = static_cast<int>(s->disp_rect.bottom - s->disp_rect.top);
    const int pitch = static_cast<int>(s->output_pitch);
    const uint8_t* y_plane = nv12;
    const uint8_t* uv_plane = nv12 + static_cast<size_t>(pitch) * s->output_vstride;

    check(hipSetDevice(impl_->device_ordinal), "hipSetDevice");
    float* d_rgb = nullptr;
    check(hipMalloc(&d_rgb, 3ull * w * h * sizeof(float)), "hipMalloc(rgb)");
    launch_nv12_to_rgb(y_plane, uv_plane, pitch, pitch,
                       s->disp_rect.left, s->disp_rect.top, w, h,
                       d_rgb, w, h, full_range, bt709, /*stream=*/nullptr);
    hipError_t err = hipGetLastError();
    if (err == hipSuccess)
        err = hipDeviceSynchronize();   // finish before releasing the surface
    impl_->dec->ReleaseFrame(pts);
    if (err != hipSuccess) {
        hipFree(d_rgb);
        check(err, "nv12_to_rgb (rocdecode surface)");
    }
    *rgb_out = reinterpret_cast<uintptr_t>(d_rgb);
    *width_out = w;
    *height_out = h;
    *pts_us = pts;
    return true;
}

bool RocDecoder::next_frame_host_nv12(uint8_t* nv12_out, int* width_out,
                                      int* height_out, int64_t* pts_us) {
    uint8_t* nv12 = nullptr;
    int64_t pts = 0;
    if (!advance(&nv12, &pts)) return false;

    const auto* s = impl_->surf;
    const int w = static_cast<int>(s->disp_rect.right - s->disp_rect.left);
    const int h = static_cast<int>(s->disp_rect.bottom - s->disp_rect.top);
    const int pitch = static_cast<int>(s->output_pitch);
    check(hipSetDevice(impl_->device_ordinal), "hipSetDevice");
    // strided D2H copies: crop pitch/vstride padding away
    hipError_t err = hipMemcpy2D(nv12_out, w, nv12, pitch, w, h,
                                 hipMemcpyDeviceToHost);
    if (err == hipSuccess) {
        const uint8_t* uv = nv12 + static_cast<size_t>(pitch) * s->output_vstride;
        err = hipMemcpy2D(nv12_out + static_cast<size_t>(w) * h, w, uv, pitch,
                          w, h / 2, hipMemcpyDeviceToHost);
    }
    impl_->dec->ReleaseFrame(pts);
    check(err, "hipMemcpy2D (rocdecode nv12)");
    *width_out = w;
    *height_out = h;
    *pts_us = pts;
    return true;
}

}  // namespace avap
