#include "vcn_encoder.h"

#include <stdexcept>

extern "C" {
#include <libavutil/hwcontext_vaapi.h>
#include <libavutil/opt.h>
}

namespace avap {

namespace {

[[noreturn]] void fail(const std::string& msg, int err = 0) {
    if (err != 0) {
        char buf[AV_ERROR_MAX_STRING_SIZE] = {};
        av_strerror(err, buf, sizeof(buf));
        throw std::runtime_error(msg + ": " + buf);
    }
    throw std::runtime_error(msg);
}

}  // namespace

VcnEncoder::VcnEncoder(const std::string& output, const std::string& codec,
                       int width, int height, double fps, int64_t bitrate,
                       const std::string& render_node)
    : width_(width), height_(height) {
    if (width % 2 || height % 2)
        fail("encoder needs even dimensions");
    const char* enc_name = codec == "hevc" ? "hevc_vaapi"
                         : codec == "av1" ? "av1_vaapi"
                         : codec == "h264" ? "h264_vaapi" : nullptr;
    if (!enc_name) fail("codec must be h264 | hevc | av1");

    int err = av_hwdevice_ctx_create(&hw_device_, AV_HWDEVICE_TYPE_VAAPI,
                                     render_node.c_str(), nullptr, 0);
    if (err < 0) fail("av_hwdevice_ctx_create(" + render_node + ")", err);

    hw_frames_ = av_hwframe_ctx_alloc(hw_device_);
    if (!hw_frames_) fail("av_hwframe_ctx_alloc");
    auto* fctx = reinterpret_cast<AVHWFramesContext*>(hw_frames_->data);
    fctx->format = AV_PIX_FMT_VAAPI;
    fctx->sw_format = AV_PIX_FMT_NV12;
    fctx->width = width;
    fctx->height = height;
    fctx->initial_pool_size = 8;
    if ((err = av_hwframe_ctx_init(hw_frames_)) < 0)
        fail("av_hwframe_ctx_init", err);

    const AVCodec* avcodec = avcodec_find_encoder_by_name(enc_name);
    if (!avcodec) fail(std::string("encoder not available: ") + enc_name);
    enc_ = avcodec_alloc_context3(avcodec);
    if (!enc_) fail("avcodec_alloc_context3");
    const AVRational fr = av_d2q(fps, 1 << 16);
    enc_->width = width;
    enc_->height = height;
    enc_->pix_fmt = AV_PIX_FMT_VAAPI;
    enc_->framerate = fr;
    enc_->time_base = av_inv_q(fr);
    enc_->bit_rate = bitrate;
    enc_->gop_size = static_cast<int>(fps * 2);
    enc_->max_b_frames = 0;   // low latency; also keeps pts == dts
    enc_->hw_frames_ctx = av_buffer_ref(hw_frames_);

    const bool is_rtsp = output.rfind("rtsp://", 0) == 0;
    err = avformat_alloc_output_context2(&fmt_, nullptr,
                                         is_rtsp ? "rtsp" : nullptr,
                                         output.c_str());
    if (err < 0 || !fmt_) fail("avformat_alloc_output_context2(" + output + ")", err);
    if (fmt_->oformat->flags & AVFMT_GLOBALHEADER)
        enc_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    if ((err = avcodec_open2(enc_, avcodec, nullptr)) < 0)
        fail(std::string("avcodec_open2(") + enc_name + ")", err);

    stream_ = avformat_new_stream(fmt_, nullptr);
    if (!stream_) fail("avformat_new_stream");
    stream_->time_base = enc_->time_base;
    if ((err = avcodec_parameters_from_context(stream_->codecpar, enc_)) < 0)
        fail("avcodec_parameters_from_context", err);

    if (!(fmt_->oformat->flags & AVFMT_NOFILE)) {
        if ((err = avio_open(&fmt_->pb, output.c_str(), AVIO_FLAG_WRITE)) < 0)
            fail("avio_open(" + output + ")", err);
    }
    if ((err = avformat_write_header(fmt_, nullptr)) < 0)
        fail("avformat_write_header", err);
}

VcnEncoder::~VcnEncoder() {
    try { close(); } catch (...) {}
}

void VcnEncoder::write_nv12(const uint8_t* nv12, size_t size) {
    if (closed_) fail("encoder is closed");
    const size_t need = static_cast<size_t>(width_) * height_ * 3 / 2;
    if (size != need)
        fail("nv12 size mismatch: got " + std::to_string(size) +
             ", need " + std::to_string(need));

    AVFrame* sw = av_frame_alloc();
    AVFrame* hw = av_frame_alloc();
    if (!sw || !hw) fail("av_frame_alloc");
    int err = 0;
    try {
        sw->format = AV_PIX_FMT_NV12;
        sw->width = width_;
        sw->height = height_;
        sw->data[0] = const_cast<uint8_t*>(nv12);
        sw->data[1] = const_cast<uint8_t*>(nv12) +
                      static_cast<size_t>(width_) * height_;
        sw->linesize[0] = width_;
        sw->linesize[1] = width_;

        if ((err = av_hwframe_get_buffer(hw_frames_, hw, 0)) < 0)
            fail("av_hwframe_get_buffer", err);
        if ((err = av_hwframe_transfer_data(hw, sw, 0)) < 0)
            fail("av_hwframe_transfer_data (upload)", err);
        hw->pts = pts_++;

        if ((err = avcodec_send_frame(enc_, hw)) < 0)
            fail("avcodec_send_frame", err);
        drain(false);
    } catch (...) {
        av_frame_free(&sw);
        av_frame_free(&hw);
        throw;
    }
    av_frame_free(&sw);
    av_frame_free(&hw);
}

void VcnEncoder::drain(bool flush) {
    AVPacket* pkt = av_packet_alloc();
    if (!pkt) fail("av_packet_alloc");
    while (true) {
        int err = avcodec_receive_packet(enc_, pkt);
        if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) break;
        if (err < 0) { av_packet_free(&pkt); fail("avcodec_receive_packet", err); }
        // VAAPI encoders emit zero-duration packets; without a duration the
        // mp4 edit list ends at the last packet's START and demuxers discard
        // the final frame. Stamp every packet with one frame's duration.
        if (pkt->duration == 0)
            pkt->duration = 1;   // one frame in enc time_base
        av_packet_rescale_ts(pkt, enc_->time_base, stream_->time_base);
        pkt->stream_index = stream_->index;
        err = av_interleaved_write_frame(fmt_, pkt);
        if (err < 0) { av_packet_free(&pkt); fail("write_frame", err); }
    }
    av_packet_free(&pkt);
    (void)flush;
}

void VcnEncoder::close() {
    if (closed_) return;
    closed_ = true;
    if (enc_ && fmt_) {
        avcodec_send_frame(enc_, nullptr);   // enter drain mode
        drain(true);
        av_write_trailer(fmt_);
    }
    if (fmt_) {
        if (!(fmt_->oformat->flags & AVFMT_NOFILE) && fmt_->pb)
            avio_closep(&fmt_->pb);
        avformat_free_context(fmt_);
        fmt_ = nullptr;
    }
    if (enc_) avcodec_free_context(&enc_);
    if (hw_frames_) av_buffer_unref(&hw_frames_);
    if (hw_device_) av_buffer_unref(&hw_device_);
}

}  // namespace avap
