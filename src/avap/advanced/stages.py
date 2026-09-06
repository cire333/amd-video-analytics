"""Inference stages — the nvinfer PGIE/SGIE analog on MIGraphX.

A PrimaryStage detects on the full frame. Secondary stages operate on
object crops (DeepStream process-mode=2), selected by the producing
stage's component id (operate-on-gie-id) and class ids
(operate-on-class-ids):

- DetectionStage  (network-type 0):  creates CHILD objects (plate under
  vehicle), bboxes mapped back to full-frame coordinates.
- ClassifierStage (network-type 1):  attaches a Classification to the
  object (plate OCR). Output parsing is pluggable, like
  parse-classifier-func-name.
- EmbeddingStage  (network-type 100, output-tensor-meta=1): attaches the
  raw output tensor to the object (DINOv2 re-ID embedding).

Stages consume/extend AdvFrameMeta against the full-res RGB frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .meta import AdvFrameMeta, AdvObjectMeta, Classification


@dataclass
class InferConfig:
    model: str                       # zoo name or path to .onnx
    component_id: int                # gie-unique-id analog; must be unique
    quant: str = "fp16"
    operate_on_component: int | None = None   # SGIE: parent stage id
    operate_on_class_ids: frozenset[int] | None = None
    conf_threshold: float = 0.3
    labels: list[str] = field(default_factory=list)
    min_object_width: int = 16       # skip tiny crops (input-object-min-* analog)
    min_object_height: int = 16
    normalize_output: bool = False   # embeddings: L2-normalize


def _resize_chw(rgb_hwc: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2
    r = cv2.resize(rgb_hwc, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(r.transpose(2, 0, 1)[None])


def _crop(frame_rgb: np.ndarray, bbox, fw: int, fh: int) -> np.ndarray | None:
    x1 = max(0, int(bbox[0])); y1 = max(0, int(bbox[1]))
    x2 = min(fw, int(bbox[2])); y2 = min(fh, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_rgb[y1:y2, x1:x2]


def default_detection_parser(outputs: np.ndarray, conf: float):
    """Ultralytics end-to-end (N, 300, 6): x1,y1,x2,y2,score,class in
    model-input pixels. Yields (bbox, score, class_id)."""
    for x1, y1, x2, y2, s, c in outputs[0]:
        if s >= conf:
            yield (float(x1), float(y1), float(x2), float(y2)), float(s), int(c)


def default_classifier_parser(outputs: np.ndarray, labels: list[str]
                              ) -> tuple[str, float] | None:
    """Argmax over a (1, C) logits/probs output."""
    probs = outputs.reshape(-1).astype(np.float64)
    if probs.size == 0:
        return None
    e = np.exp(probs - probs.max())
    probs = e / e.sum()
    idx = int(np.argmax(probs))
    label = labels[idx] if idx < len(labels) else str(idx)
    return label, float(probs[idx])


class _Stage:
    def __init__(self, config: InferConfig, device_ordinal: int = 0):
        self.config = config
        self.device_ordinal = device_ordinal
        self._model = None
        self._input_hw: tuple[int, int] | None = None

    def prepare(self) -> None:
        """Compile the model (call once, on the pipeline's device)."""
        from ..model_zoo import MigraphxModel, resolve_model
        onnx = resolve_model(self.config.model) \
            if not self.config.model.endswith(".onnx") else self.config.model
        self._model = MigraphxModel(onnx, self.config.quant, self.device_ordinal)
        shape = self._model.input_shape          # (1, 3, H, W)
        self._input_hw = (int(shape[2]), int(shape[3]))

    def _infer(self, rgb_hwc: np.ndarray) -> np.ndarray:
        h, w = self._input_hw
        return self._model(_resize_chw(rgb_hwc, w, h))

    def _targets(self, frame: AdvFrameMeta) -> list[AdvObjectMeta]:
        cfg = self.config
        out = []
        for obj in frame.objects:
            if obj.component_id != cfg.operate_on_component:
                continue
            if (cfg.operate_on_class_ids is not None
                    and obj.class_id not in cfg.operate_on_class_ids):
                continue
            if (obj.bbox[2] - obj.bbox[0] < cfg.min_object_width
                    or obj.bbox[3] - obj.bbox[1] < cfg.min_object_height):
                continue
            out.append(obj)
        return out

    def _label(self, class_id: int) -> str:
        labels = self.config.labels
        return labels[class_id] if class_id < len(labels) else str(class_id)


class PrimaryStage(_Stage):
    """Full-frame detector (PGIE). parse: (outputs, conf) -> iter of
    ((x1,y1,x2,y2) in model-input px, score, class_id)."""

    def __init__(self, config: InferConfig, device_ordinal: int = 0,
                 parse: Callable = default_detection_parser):
        super().__init__(config, device_ordinal)
        self.parse = parse

    def run(self, frame_rgb: np.ndarray, frame: AdvFrameMeta) -> None:
        out = self._infer(frame_rgb)
        ih, iw = self._input_hw
        sx, sy = frame.frame_width / iw, frame.frame_height / ih
        for bbox, score, cls in self.parse(out, self.config.conf_threshold):
            frame.objects.append(AdvObjectMeta(
                object_id=None, component_id=self.config.component_id,
                class_id=cls, label=self._label(cls), confidence=score,
                bbox=(bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy)))


class DetectionStage(_Stage):
    """SGIE detector: creates child objects inside parent crops."""

    def __init__(self, config: InferConfig, device_ordinal: int = 0,
                 parse: Callable = default_detection_parser):
        super().__init__(config, device_ordinal)
        self.parse = parse

    def run(self, frame_rgb: np.ndarray, frame: AdvFrameMeta) -> None:
        new_children = []
        for parent in self._targets(frame):
            crop = _crop(frame_rgb, parent.bbox,
                         frame.frame_width, frame.frame_height)
            if crop is None:
                continue
            out = self._infer(crop)
            ih, iw = self._input_hw
            ch, cw = crop.shape[:2]
            sx, sy = cw / iw, ch / ih
            ox, oy = max(0, parent.bbox[0]), max(0, parent.bbox[1])
            for bbox, score, cls in self.parse(out, self.config.conf_threshold):
                child = AdvObjectMeta(
                    object_id=None, component_id=self.config.component_id,
                    class_id=cls, label=self._label(cls), confidence=score,
                    bbox=(bbox[0] * sx + ox, bbox[1] * sy + oy,
                          bbox[2] * sx + ox, bbox[3] * sy + oy),
                    parent=parent)
                parent.children.append(child)
                new_children.append(child)
        frame.objects.extend(new_children)


class ClassifierStage(_Stage):
    """SGIE classifier: attaches (label, confidence) to matching objects.
    parse: (outputs, labels) -> (label, confidence) | None — the
    parse-classifier-func-name analog (e.g. CTC decode for plate OCR)."""

    def __init__(self, config: InferConfig, device_ordinal: int = 0,
                 parse: Callable = default_classifier_parser):
        super().__init__(config, device_ordinal)
        self.parse = parse

    def run(self, frame_rgb: np.ndarray, frame: AdvFrameMeta) -> None:
        for obj in self._targets(frame):
            crop = _crop(frame_rgb, obj.bbox,
                         frame.frame_width, frame.frame_height)
            if crop is None:
                continue
            result = self.parse(self._infer(crop), self.config.labels)
            if result is not None:
                obj.classifications.append(Classification(
                    component_id=self.config.component_id,
                    label=result[0], confidence=result[1]))


class EmbeddingStage(_Stage):
    """SGIE tensor stage: attaches the raw output vector to the object
    (output-tensor-meta=1 analog)."""

    def run(self, frame_rgb: np.ndarray, frame: AdvFrameMeta) -> None:
        for obj in self._targets(frame):
            crop = _crop(frame_rgb, obj.bbox,
                         frame.frame_width, frame.frame_height)
            if crop is None:
                continue
            vec = self._infer(crop).reshape(-1).astype(np.float32)
            if self.config.normalize_output:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            obj.tensors[self.config.component_id] = vec
