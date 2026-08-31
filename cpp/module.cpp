// pybind11 bindings for avap._core: VAAPI decode + dmabuf->HIP bridge.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "vaapi_decoder.h"
#ifdef AVAP_WITH_HIP
#include "hip_bridge.h"
#endif

namespace py = pybind11;
using namespace avap;

PYBIND11_MODULE(_core, m) {
    m.doc() = "avap native core: VAAPI decode, dmabuf->HIP bridge";

    py::class_<DecodedFrame>(m, "DecodedFrame")
        .def_readonly("pts_us", &DecodedFrame::pts_us)
        .def_readonly("width", &DecodedFrame::width)
        .def_readonly("height", &DecodedFrame::height)
        .def_readonly("crop_x", &DecodedFrame::crop_x)
        .def_readonly("crop_y", &DecodedFrame::crop_y)
        .def_readonly("crop_w", &DecodedFrame::crop_w)
        .def_readonly("crop_h", &DecodedFrame::crop_h)
        .def_readonly("dmabuf_fd", &DecodedFrame::dmabuf_fd)
        .def_readonly("planes", &DecodedFrame::planes)
        .def_readonly("drm_modifier", &DecodedFrame::drm_modifier)
        .def_readonly("full_range", &DecodedFrame::full_range)
        .def_readonly("color_matrix", &DecodedFrame::color_matrix)
        .def_property_readonly("host_data", [](const DecodedFrame& f) -> py::object {
            if (f.host_data.empty()) return py::none();
            return py::bytes(reinterpret_cast<const char*>(f.host_data.data()),
                             f.host_data.size());
        });

    py::class_<VaapiDecoder>(m, "Decoder")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("uri"), py::arg("render_node"))
        .def("next_frame",
             [](VaapiDecoder& self) -> py::object {
                 std::optional<DecodedFrame> f;
                 {
                     py::gil_scoped_release release;  // decode blocks on I/O + VCN
                     f = self.next_frame();
                 }
                 if (!f) return py::none();
                 return py::cast(*f);
             })
        .def("close", &VaapiDecoder::close);

    m.def("nv12_dmabuf_to_rgb",
          [](int fd, int surf_w, int surf_h,
             std::vector<std::pair<uint32_t, uint32_t>> planes, uint64_t modifier,
             std::tuple<int, int, int, int> src_rect, std::tuple<int, int> dst_wh,
             bool full_range, bool bt709, int device_ordinal) -> py::array_t<float> {
#ifdef AVAP_WITH_HIP
              ConvertRequest req;
              req.dmabuf_fd = fd;
              req.surf_width = surf_w;
              req.surf_height = surf_h;
              req.planes = std::move(planes);
              req.drm_modifier = modifier;
              std::tie(req.src_x, req.src_y, req.src_w, req.src_h) = src_rect;
              std::tie(req.dst_w, req.dst_h) = dst_wh;
              req.full_range = full_range;
              req.bt709 = bt709;
              req.device_ordinal = device_ordinal;

              auto out = py::array_t<float>({3, req.dst_h, req.dst_w});
              {
                  py::gil_scoped_release release;
                  nv12_dmabuf_to_rgb(req, out.mutable_data());
              }
              return out;
#else
              throw std::runtime_error(
                  "avap._core was built without HIP (AVAP_WITH_HIP=OFF); "
                  "rebuild with ROCm installed");
#endif
          },
          py::arg("fd"), py::arg("surf_w"), py::arg("surf_h"), py::arg("planes"),
          py::arg("modifier"), py::arg("src_rect"), py::arg("dst_wh"),
          py::arg("full_range"), py::arg("bt709"), py::arg("device_ordinal"));

    m.def("nv12_host_to_rgb",
          [](py::bytes nv12, std::vector<std::pair<uint32_t, uint32_t>> planes,
             std::tuple<int, int, int, int> src_rect, std::tuple<int, int> dst_wh,
             bool full_range, bool bt709, int device_ordinal) -> py::array_t<float> {
#ifdef AVAP_WITH_HIP
              ConvertRequest req;
              req.planes = std::move(planes);
              std::tie(req.src_x, req.src_y, req.src_w, req.src_h) = src_rect;
              std::tie(req.dst_w, req.dst_h) = dst_wh;
              req.full_range = full_range;
              req.bt709 = bt709;
              req.device_ordinal = device_ordinal;

              auto data = nv12.cast<std::string_view>();  // no copy; nv12 outlives use
              auto out = py::array_t<float>({3, req.dst_h, req.dst_w});
              {
                  py::gil_scoped_release release;
                  nv12_host_to_rgb(req,
                                   reinterpret_cast<const uint8_t*>(data.data()),
                                   data.size(), out.mutable_data());
              }
              return out;
#else
              throw std::runtime_error(
                  "avap._core was built without HIP (AVAP_WITH_HIP=OFF); "
                  "rebuild with ROCm installed");
#endif
          },
          py::arg("nv12"), py::arg("planes"), py::arg("src_rect"), py::arg("dst_wh"),
          py::arg("full_range"), py::arg("bt709"), py::arg("device_ordinal"));

    m.def("hip_device_count", []() -> int {
#ifdef AVAP_WITH_HIP
        return hip_device_count();
#else
        return 0;
#endif
    });
}
