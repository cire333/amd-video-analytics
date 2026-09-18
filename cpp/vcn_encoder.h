#pragma once
// VCN hardware video encoder (the NVENC analog): FFmpeg h264/hevc/av1_vaapi
// encode + containers (mp4/mkv by extension, rtsp:// push) on the AMD VCN.
// Input: host NV12 (Y then interleaved UV, pitch = width); the Python layer
// converts device RGB frames via the rgb_to_nv12 HIP kernel.

#include <cstdint>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
}

namespace avap {

class VcnEncoder {
public:
    VcnEncoder(const std::string& output, const std::string& codec,
               int width, int height, double fps, int64_t bitrate,
               const std::string& render_node);
    ~VcnEncoder();

    VcnEncoder(const VcnEncoder&) = delete;
    VcnEncoder& operator=(const VcnEncoder&) = delete;

    // nv12 buffer: width*height Y followed by width*height/2 interleaved UV.
    void write_nv12(const uint8_t* nv12, size_t size);
    void close();   // flush encoder, write trailer; idempotent

private:
    void drain(bool flush);

    int width_, height_;
    int64_t pts_ = 0;
    bool closed_ = false;
    AVBufferRef* hw_device_ = nullptr;
    AVBufferRef* hw_frames_ = nullptr;
    AVCodecContext* enc_ = nullptr;
    AVFormatContext* fmt_ = nullptr;
    AVStream* stream_ = nullptr;
};

}  // namespace avap
