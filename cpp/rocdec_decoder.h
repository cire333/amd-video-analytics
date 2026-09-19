#pragma once
// Zero-copy decode backend on AMD's rocDecode SDK: bitstream -> VCN decode
// -> device NV12 -> (our fused kernel) -> device RGB. Decoded frames never
// touch host memory, and rocDecode owns the tiled-surface problem that
// forces the VAAPI backend's host detile fallback.
//
// Wraps the vendored AMD reference utilities (cpp/thirdparty/rocdecode,
// MIT): VideoDemuxer (FFmpeg) + RocVideoDecoder (parser/decoder/queue).

#include <cstdint>
#include <memory>
#include <string>

namespace avap {

class RocDecoder {
public:
    // uri: file path or network stream FFmpeg can demux.
    RocDecoder(const std::string& uri, int device_ordinal);
    ~RocDecoder();

    RocDecoder(const RocDecoder&) = delete;
    RocDecoder& operator=(const RocDecoder&) = delete;

    // Decode the next frame and convert to CHW float RGB entirely on the
    // GPU, honoring the stream's tagged colorimetry (probed at open;
    // untagged defaults to BT.709 limited, matching the VAAPI backend).
    // Returns false on EOF. *rgb_out is a device buffer the caller owns
    // (free with free_device_buffer).
    bool next_frame_device_rgb(uintptr_t* rgb_out, int* width, int* height,
                               int64_t* pts_us);

    bool stream_bt709() const;
    bool stream_full_range() const;

    // Compat path: same decode, NV12 copied to host (annotation, or code
    // written against the VAAPI decoder's host fallback). Fills nv12_out
    // (w*h*3/2 bytes, pitch = width). Returns false on EOF.
    bool next_frame_host_nv12(uint8_t* nv12_out, int* width, int* height,
                              int64_t* pts_us);

    int width() const;
    int height() const;
    void close();

private:
    bool advance(uint8_t** dev_nv12, int64_t* pts);   // demux+decode pump

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace avap
