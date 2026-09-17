"""Published record schema — wire-compatible with the ingestion system's
DataRecord/ObjectRecord JSON (keys, casing, and conditional LPR fields),
so downstream consumers cannot tell which stack produced a message."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .lpr import LPRResult
from .meta import AdvFrameMeta, AdvObjectMeta

DEFAULT_CLASS_MAPPING = {
    "person": "person",
    "car": "light automobile",
    "truck": "heavy automobile",
    "bus": "heavy automobile",
    "motorcycle": "motorbike",
    "bicycle": "bicycle",
}


class ClassMapper:
    """Model label -> customer taxonomy. Unmapped labels pass through
    unless drop_unmapped is set."""

    def __init__(self, mapping: dict[str, str] | None = None,
                 drop_unmapped: bool = False):
        self.mapping = dict(DEFAULT_CLASS_MAPPING if mapping is None else mapping)
        self.drop_unmapped = drop_unmapped

    def map(self, label: str) -> str | None:
        mapped = self.mapping.get(label)
        if mapped is not None:
            return mapped
        return None if self.drop_unmapped else label


@dataclass
class ObjectRecord:
    class_name: str
    object_id: int
    x1: float
    x2: float
    y1: float
    y2: float
    confidence_score: float
    frame_timestamp: float
    vehicle_id: str | None = None
    plate_text: str | None = None
    plate_confidence: float | None = None
    plate_read_count: int | None = None
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "class": self.class_name,
            "id": self.object_id,
            "x1": self.x1, "x2": self.x2, "y1": self.y1, "y2": self.y2,
            "confidence_score": str(self.confidence_score),
            "frame_timestamp": self.frame_timestamp,
        }
        if self.vehicle_id is not None:
            d["vehicle_id"] = self.vehicle_id
        if self.plate_text is not None:
            d["plate_text"] = self.plate_text
            d["plate_confidence"] = self.plate_confidence
            d["plate_read_count"] = self.plate_read_count
        if self.embedding is not None:
            d["embedding"] = self.embedding
        return d


@dataclass
class DataRecord:
    data: list[ObjectRecord]
    frame: int
    stream: int | str
    time: str
    fps: float
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "Data": [o.to_dict() for o in self.data],
            "Frame": self.frame,
            "Stream": self.stream,
            "Time": self.time,
            "FPS": self.fps,
            "Metadata": self.metadata,
        }


def build_object_record(obj: AdvObjectMeta, frame_ts: float,
                        mapper: ClassMapper,
                        lpr: LPRResult | None = None,
                        include_embedding: bool = False
                        ) -> ObjectRecord | None:
    class_name = mapper.map(obj.label)
    if class_name is None or obj.object_id is None:
        return None
    rec = ObjectRecord(
        class_name=class_name, object_id=obj.object_id,
        x1=obj.bbox[0], x2=obj.bbox[2], y1=obj.bbox[1], y2=obj.bbox[3],
        confidence_score=obj.confidence, frame_timestamp=frame_ts)
    if lpr is not None:
        rec.vehicle_id = lpr.vehicle_id
        rec.plate_text = lpr.plate_text
        rec.plate_confidence = lpr.plate_confidence
        rec.plate_read_count = lpr.plate_read_count
        if include_embedding and lpr.embedding is not None:
            rec.embedding = [float(v) for v in lpr.embedding]
    return rec
