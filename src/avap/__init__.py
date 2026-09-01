"""AMD Video Analytics Pipeline — DeepStream-equivalent for AMD hardware.

Pure-Python modules import without the compiled extension; the extension
(avap._core: VAAPI decode, dmabuf->HIP bridge) is required to run streams.
"""
from .batcher import DynamicBatcher
from .capabilities import DeviceCapabilities, probe_devices
from .frame import (BatchMeta, ColorMatrix, ColorRange, CropRect, FrameMeta,
                    ObjectMeta, PlaneLayout, RawFrame)
from .graph import GraphExecutor, ModelGraph, OnnxModel
from .kalman_tracker import SortTracker
from .pipeline import Pipeline
from .registry import StreamRegistry
from .roi import RoiConfig, RoiTransform
from .tracker import IouTracker, TrackerBank

__version__ = "0.1.0"

__all__ = [
    "BatchMeta", "ColorMatrix", "ColorRange", "CropRect", "DeviceCapabilities",
    "DynamicBatcher", "FrameMeta", "GraphExecutor", "IouTracker", "ModelGraph",
    "ObjectMeta", "OnnxModel", "Pipeline", "PlaneLayout", "RawFrame",
    "RoiConfig", "RoiTransform", "SortTracker", "StreamRegistry", "TrackerBank",
    "probe_devices",
]
