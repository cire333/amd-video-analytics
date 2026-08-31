// Custom nvinfer output parser for Ultralytics YOLO26 end-to-end ONNX export.
// Output tensor: (1, 300, 6) = x1, y1, x2, y2, score, class_id in
// network-input pixel coordinates (nvinfer rescales to frame coords).
//
// Build (inside the DeepStream container):
//   g++ -shared -fPIC -std=c++14 yolo26_parser.cpp -o libyolo26parser.so \
//       -I/opt/nvidia/deepstream/deepstream/sources/includes \
//       -I/usr/local/cuda/include

#include <algorithm>
#include <cstring>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseYolo26(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList) {
    if (outputLayersInfo.empty()) return false;

    const NvDsInferLayerInfo& layer = outputLayersInfo[0];
    // dims: (300, 6) after the implicit batch dim
    const int rows = layer.inferDims.d[0];
    const int cols = layer.inferDims.d[1];
    if (cols != 6) return false;
    const float* data = static_cast<const float*>(layer.buffer);

    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);

    for (int i = 0; i < rows; ++i) {
        const float* row = data + i * cols;
        const float score = row[4];
        const int cls = static_cast<int>(row[5]);
        if (cls < 0 || cls >= static_cast<int>(detectionParams.numClassesConfigured))
            continue;
        if (score < detectionParams.perClassPreclusterThreshold[cls]) continue;

        float x1 = std::max(0.0f, std::min(row[0], netW - 1));
        float y1 = std::max(0.0f, std::min(row[1], netH - 1));
        float x2 = std::max(0.0f, std::min(row[2], netW - 1));
        float y2 = std::max(0.0f, std::min(row[3], netH - 1));
        if (x2 <= x1 || y2 <= y1) continue;

        NvDsInferParseObjectInfo obj{};
        obj.classId = static_cast<unsigned int>(cls);
        obj.detectionConfidence = score;
        obj.left = x1;
        obj.top = y1;
        obj.width = x2 - x1;
        obj.height = y2 - y1;
        objectList.push_back(obj);
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo26);
