"""Advanced API: the integrated pipeline layer.

Where the light API (avap.AMDStream) is one detector + one tracker + one
sink per stream, the advanced API reproduces the DeepStream-style
composed pipeline used by GridMatrix's ingestion system: primary
detection -> tracking -> secondary inference stages operating on object
crops (child detections, classifiers, embedding extractors) -> user
probes -> published DataRecords — including the LPR cascade (plate
detect -> plate OCR -> DINOv2 re-ID) with plate voting and vehicle
re-identification.
"""
from .meta import AdvObjectMeta, AdvFrameMeta, Classification
from .stages import (ClassifierStage, DetectionStage, EmbeddingStage,
                     InferConfig, PrimaryStage)
from .device import DeviceFrame, DeviceModel, device_api_available
from .pipeline import AdvancedPipeline, SourceContext
from .publisher import DataPublisher, FileTransport
from .metrics import PerfData
from .lpr import LPRConfig, LPRResult, LPRStage, PlateVoter, VehicleReID
from .records import ObjectRecord, DataRecord, ClassMapper

__all__ = [
    "AdvFrameMeta", "AdvObjectMeta", "AdvancedPipeline", "Classification",
    "ClassMapper", "ClassifierStage", "DataPublisher", "DataRecord",
    "DetectionStage", "DeviceFrame", "DeviceModel", "EmbeddingStage",
    "FileTransport", "InferConfig", "device_api_available",
    "LPRConfig", "LPRResult", "LPRStage", "ObjectRecord", "PerfData",
    "PlateVoter", "PrimaryStage", "SourceContext", "VehicleReID",
]
