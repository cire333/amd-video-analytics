"""Hierarchical metadata for the advanced pipeline — the NvDsObjectMeta
analog: objects carry a component id (which stage produced them), a parent
link (a plate is a child of a vehicle), classifier results, and attached
tensors (embeddings)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Classification:
    """One classifier stage's result on an object (e.g. plate OCR)."""
    component_id: int
    label: str
    confidence: float


@dataclass
class AdvObjectMeta:
    object_id: int | None        # tracker id (primary objects) or None
    component_id: int            # id of the stage that created this object
    class_id: int
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]   # full-frame pixels x1,y1,x2,y2
    parent: "AdvObjectMeta | None" = None
    children: list["AdvObjectMeta"] = field(default_factory=list)
    classifications: list[Classification] = field(default_factory=list)
    tensors: dict[int, np.ndarray] = field(default_factory=dict)  # by component_id

    # alias so avap trackers (which read/write .track_id) work directly on
    # primary objects
    @property
    def track_id(self) -> int | None:
        return self.object_id

    @track_id.setter
    def track_id(self, value: int | None) -> None:
        self.object_id = value

    def classification_from(self, component_id: int) -> Classification | None:
        for c in self.classifications:
            if c.component_id == component_id:
                return c
        return None

    def best_child_classification(self, child_component: int,
                                  classifier_component: int
                                  ) -> Classification | None:
        """Highest-confidence classifier result over this object's children
        from a given stage — e.g. best OCR read across plate crops."""
        best = None
        for child in self.children:
            if child.component_id != child_component:
                continue
            c = child.classification_from(classifier_component)
            if c is not None and (best is None or c.confidence > best.confidence):
                best = c
        return best


@dataclass
class AdvFrameMeta:
    source_id: str
    frame: int
    pts_us: int
    frame_width: int
    frame_height: int
    objects: list[AdvObjectMeta] = field(default_factory=list)

    def primary_objects(self) -> list[AdvObjectMeta]:
        return [o for o in self.objects if o.parent is None]
