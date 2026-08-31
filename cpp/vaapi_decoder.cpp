#include "vaapi_decoder.h"

#include <stdexcept>
#include <unistd.h>

#include <va/va.h>
#include <va/va_drmcommon.h>

extern "C" {
#include <libavutil/hwcontext_vaapi.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
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

enum AVPixelFormat pick_vaapi(AVCodecContext*, const enum AVPixelFormat* fmts) {
    for (const enum AVPixelFormat* p = fmts; *p != AV_PIX_FMT_NONE; ++p)
        if (*p == AV_PIX_FMT_VAAPI) return *p;
    return AV_PIX_FMT_NONE;  // no software fallback: fail loud, VCN is the point
}

VADisplay va_display_of(AVBufferRef* hw_device) {
    auto* dev = reinterpret_cast<AVHWDeviceContext*>(hw_device->data);
    return static_cast<AVVAAPIDeviceContext*>(dev->hwctx)->display;
}

}  // namespace

VaapiDecoder::VaapiDecoder(const std::string& uri, const std::string& render_node) {
    int err = av_hwdevice_ctx_create(&hw_device_, AV_HWDEVICE_TYPE_VAAPI,
                                     render_node.c_str(), nullptr, 0);
    if (err < 0) fail("av_hwdevice_ctx_create(" + render_node + ")", err);

    AVDictionary* opts = nullptr;
    if (uri.rfind("rtsp://", 0) == 0) {
        av_dict_set(&opts, "rtsp_transport", "tcp", 0);
        av_dict_set(&opts, "stimeout", "5000000", 0);  // 5s socket timeout, us
    }
    err = avformat_open_input(&fmt_, uri.c_str(), nullptr, &opts);
    av_dict_free(&opts);
    if (err < 0) fail("avformat_open_input(" + uri + ")", err);
    if ((err = avformat_find_stream_info(fmt_, nullptr)) < 0)
        fail("avformat_find_stream_info", err);

    const AVCodec* codec = nullptr;
    video_stream_ = av_find_best_stream(fmt_, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (video_stream_ < 0) fail("no video stream in " + uri);
    time_base_ = fmt_->streams[video_stream_]->time_base;

    codec_ = avcodec_alloc_context3(codec);
    if (!codec_) fail("avcodec_alloc_context3");
    if ((err = avcodec_parameters_to_context(codec_, fmt_->streams[video_stream_]->codecpar)) < 0)
        fail("avcodec_parameters_to_context", err);
    codec_->hw_device_ctx = av_buffer_ref(hw_device_);
    codec_->get_format = pick_vaapi;
    if ((err = avcodec_open2(codec_, codec, nullptr)) < 0)
        fail("avcodec_open2 (VAAPI decode unsupported for this codec?)", err);
}

VaapiDecoder::~VaapiDecoder() { close(); }

void VaapiDecoder::close() {
    if (codec_) avcodec_free_context(&codec_);
    if (fmt_) avformat_close_input(&fmt_);
    if (hw_device_) av_buffer_unref(&hw_device_);
}

std::optional<DecodedFrame> VaapiDecoder::next_frame() {
    AVFrame* frame = av_frame_alloc();
    AVPacket* pkt = av_packet_alloc();
    if (!frame || !pkt) fail("alloc frame/packet");

    std::optional<DecodedFrame> out;
    try {
        while (true) {
            int err = avcodec_receive_frame(codec_, frame);
            if (err == 0) {
                out = export_frame(frame);
                break;
            }
            if (err == AVERROR_EOF) break;  // drained: EOF
            if (err != AVERROR(EAGAIN)) fail("avcodec_receive_frame", err);
            if (draining_) continue;

            // feed the next video packet
            while (true) {
                err = av_read_frame(fmt_, pkt);
                if (err == AVERROR_EOF) {
                    avcodec_send_packet(codec_, nullptr);  // enter drain mode
                    draining_ = true;
                    break;
                }
                if (err < 0) fail("av_read_frame", err);
                if (pkt->stream_index == video_stream_) {
                    err = avcodec_send_packet(codec_, pkt);
                    av_packet_unref(pkt);
                    if (err < 0 && err != AVERROR(EAGAIN)) fail("avcodec_send_packet", err);
                    break;
                }
                av_packet_unref(pkt);
            }
        }
    } catch (...) {
        av_frame_free(&frame);
        av_packet_free(&pkt);
        throw;
    }
    av_frame_free(&frame);
    av_packet_free(&pkt);
    return out;
}

std::optional<DecodedFrame> VaapiDecoder::export_frame(AVFrame* frame) {
    if (frame->format != AV_PIX_FMT_VAAPI)
        fail("decoder produced a non-VAAPI frame (software fallback not allowed)");

    VADisplay display = va_display_of(hw_device_);
    auto surface = static_cast<VASurfaceID>(reinterpret_cast<uintptr_t>(frame->data[3]));

    // FENCE (edge case #4): the decode write must complete before anyone
    // reads the dmabuf. Load-dependent bug — passes single-stream, fails
    // under multi-stream throughput if skipped.
    VAStatus st = vaSyncSurface(display, surface);
    if (st != VA_STATUS_SUCCESS) fail("vaSyncSurface: " + std::string(vaErrorStr(st)));

    VADRMPRIMESurfaceDescriptor desc{};
    st = vaExportSurfaceHandle(display, surface,
                               VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                               VA_EXPORT_SURFACE_READ_ONLY |
                               VA_EXPORT_SURFACE_COMPOSED_LAYERS,
                               &desc);
    if (st != VA_STATUS_SUCCESS)
        fail("vaExportSurfaceHandle: " + std::string(vaErrorStr(st)));

    // NV12 from radeonsi: expect a single DRM object; multi-object export
    // is unhandled for now — close everything and say so.
    if (desc.num_objects != 1) {
        for (uint32_t i = 0; i < desc.num_objects; ++i) ::close(desc.objects[i].fd);
        fail("expected 1 DRM object for NV12, got " + std::to_string(desc.num_objects));
    }

    DecodedFrame out;
    out.dmabuf_fd = desc.objects[0].fd;      // ownership -> Python
    out.drm_modifier = desc.objects[0].drm_format_modifier;
    out.width = static_cast<int>(desc.width);
    out.height = static_cast<int>(desc.height);
    for (uint32_t l = 0; l < desc.num_layers; ++l)
        for (uint32_t p = 0; p < desc.layers[l].num_planes; ++p)
            out.planes.emplace_back(desc.layers[l].offset[p], desc.layers[l].pitch[p]);

    // Real content rect (edge case #7): decoded surface may be padded;
    // frame->width/height are the display dims.
    out.crop_x = 0;
    out.crop_y = 0;
    out.crop_w = frame->width;
    out.crop_h = frame->height;

    out.pts_us = (frame->pts == AV_NOPTS_VALUE)
        ? 0 : av_rescale_q(frame->pts, time_base_, AVRational{1, 1000000});
    out.full_range = (frame->color_range == AVCOL_RANGE_JPEG);
    switch (frame->colorspace) {
        case AVCOL_SPC_BT470BG:
        case AVCOL_SPC_SMPTE170M: out.color_matrix = "bt601"; break;
        case AVCOL_SPC_BT709:     out.color_matrix = "bt709"; break;
        default:                  out.color_matrix = "unknown"; break;
    }
    return out;
}

}  // namespace avap
