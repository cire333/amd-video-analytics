#pragma once
// VAAPI decoder: FFmpeg demux/parse -> VCN decode -> dmabuf export.
// One instance per stream; owns its AVFormatContext/AVCodecContext and the
// VAAPI hw device bound to a specific DRM render node.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
}

namespace avap {

struct DecodedFrame {
    int64_t pts_us = 0;
    int width = 0;            // exported surface dims (may be padded)
    int height = 0;
    int crop_x = 0, crop_y = 0, crop_w = 0, crop_h = 0;  // real content rect
    int dmabuf_fd = -1;       // ownership passes to Python (RawFrame); -1 => host path
    std::vector<std::pair<uint32_t, uint32_t>> planes;   // (offset, pitch), Y then UV
    uint64_t drm_modifier = 0;  // 0 == DRM_FORMAT_MOD_LINEAR
    // Fallback when the surface is tiled: driver-detiled NV12 in host memory
    // (av_hwframe_transfer_data). Same planes/offset/pitch semantics.
    std::vector<uint8_t> host_data;
    bool full_range = false;
    std::string color_matrix;   // "bt601" / "bt709" / "unknown"
};

class VaapiDecoder {
public:
    // uri: file path or rtsp:// url. render_node: e.g. /dev/dri/renderD128 —
    // must be the AMD node (this box also has an NVIDIA card).
    VaapiDecoder(const std::string& uri, const std::string& render_node);
    ~VaapiDecoder();

    VaapiDecoder(const VaapiDecoder&) = delete;
    VaapiDecoder& operator=(const VaapiDecoder&) = delete;

    // Blocking. nullopt on EOF; throws std::runtime_error on decode/export error.
    std::optional<DecodedFrame> next_frame();
    void close();

private:
    std::optional<DecodedFrame> export_frame(AVFrame* frame);
    void transfer_to_host(AVFrame* frame, DecodedFrame& out);

    AVFormatContext* fmt_ = nullptr;
    AVCodecContext* codec_ = nullptr;
    AVBufferRef* hw_device_ = nullptr;
    int video_stream_ = -1;
    AVRational time_base_{};
    bool draining_ = false;
};

}  // namespace avap
