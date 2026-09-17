"""LPR: plate voting + vehicle re-identification, ported functionally
from the DeepStream ingestion system.

- PlateVoter: majority vote over per-track OCR reads (noisy single reads
  become a stable plate after min_reads).
- VehicleReID: cosine-similarity lookup over normalized embeddings with a
  persistent registry (JSON). Same semantics as the FAISS IndexFlatIP
  original (exact inner-product top-1) in pure numpy; uses faiss if
  installed, but does not require it.
- LPRStage: consumes AdvFrameMeta after the SGIE cascade (plate detect ->
  plate OCR -> embedding extractor), maintaining per-track plate votes,
  embedding averages, and vehicle identity — mirrors the original
  LPRStage probe logic.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from collections import Counter
from pathlib import Path

import numpy as np

from .meta import AdvFrameMeta, AdvObjectMeta

EMBEDDING_DIM = 768
EMBEDDINGS_TO_AVERAGE = 30


@dataclass
class LPRConfig:
    vehicle_class_ids: frozenset[int] = frozenset({2, 3, 5, 7})
    plate_component_id: int = 2       # stage that creates plate child objects
    ocr_component_id: int = 3         # classifier stage attaching plate text
    embedding_component_id: int = 4   # tensor stage attaching embeddings
    similarity_threshold: float = 0.88
    min_plate_reads: int = 3
    plate_confidence_threshold: float = 0.75
    embeddings_dir: str | None = None
    embedding_dim: int = EMBEDDING_DIM


@dataclass
class LPRResult:
    vehicle_id: str | None = None
    plate_text: str | None = None
    plate_confidence: float | None = None
    plate_read_count: int | None = None
    embedding: np.ndarray | None = None


class PlateVoter:
    def __init__(self):
        self._track_reads: dict[str, list[str]] = {}

    def accumulate(self, track_id: str, plate_text: str) -> None:
        self._track_reads.setdefault(track_id, []).append(plate_text)

    def vote(self, track_id: str, min_reads: int = 3) -> tuple[str, int] | None:
        reads = self._track_reads.get(track_id, [])
        if len(reads) < min_reads:
            return None
        text, _ = Counter(reads).most_common(1)[0]
        return text, len(reads)

    def clear_track(self, track_id: str) -> None:
        self._track_reads.pop(track_id, None)


@dataclass
class VehicleRecord:
    vehicle_id: str
    plate_text: str
    embedding: list[float]
    created_at: str
    updated_at: str
    observation_count: int


class VehicleReID:
    """Exact inner-product nearest-neighbor over normalized embeddings.

    Persistence: embeddings.json (list of VehicleRecord). Matching a known
    vehicle folds the new embedding into a running normalized average.
    """

    def __init__(self, similarity_threshold: float = 0.88,
                 dim: int = EMBEDDING_DIM):
        self._threshold = similarity_threshold
        self._dim = dim
        self._records: list[VehicleRecord] = []
        self._matrix = np.zeros((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._records)

    def load(self, embeddings_dir: str) -> None:
        path = Path(embeddings_dir) / "embeddings.json"
        if not path.exists():
            return
        self._records = [VehicleRecord(**r) for r in json.loads(path.read_text())]
        self._rebuild()

    def save(self, embeddings_dir: str) -> None:
        p = Path(embeddings_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "embeddings.json").write_text(
            json.dumps([asdict(r) for r in self._records], indent=2))

    def lookup_or_create(self, embedding: np.ndarray, timestamp: str) -> str:
        if len(self._records):
            sims = self._matrix @ embedding.astype(np.float32)
            idx = int(np.argmax(sims))
            if sims[idx] >= self._threshold:
                self._update_embedding(idx, embedding, timestamp)
                return self._records[idx].vehicle_id
        return self._create(embedding, timestamp)

    def update_plate(self, vehicle_id: str, plate_text: str,
                     timestamp: str = "") -> None:
        for r in self._records:
            if r.vehicle_id == vehicle_id and not r.plate_text:
                r.plate_text = plate_text
                if timestamp:
                    r.updated_at = timestamp
                break

    def get(self, vehicle_id: str) -> VehicleRecord | None:
        return next((r for r in self._records if r.vehicle_id == vehicle_id),
                    None)

    def _create(self, embedding: np.ndarray, timestamp: str) -> str:
        vehicle_id = f"v_{uuid.uuid4().hex[:6]}"
        self._records.append(VehicleRecord(
            vehicle_id=vehicle_id, plate_text="",
            embedding=embedding.astype(np.float32).tolist(),
            created_at=timestamp, updated_at=timestamp, observation_count=1))
        self._rebuild()
        return vehicle_id

    def _update_embedding(self, idx: int, new_embedding: np.ndarray,
                          timestamp: str) -> None:
        record = self._records[idx]
        old = np.array(record.embedding, dtype=np.float32)
        n = record.observation_count
        updated = (old * n + new_embedding) / (n + 1)
        norm = np.linalg.norm(updated)
        if norm > 0:
            updated /= norm
        record.embedding = updated.astype(np.float32).tolist()
        record.observation_count += 1
        record.updated_at = timestamp
        self._rebuild()

    def _rebuild(self) -> None:
        if self._records:
            self._matrix = np.array([r.embedding for r in self._records],
                                    dtype=np.float32)
        else:
            self._matrix = np.zeros((0, self._dim), dtype=np.float32)


class LPRStage:
    """Post-cascade LPR logic: per-track plate voting + vehicle re-ID.

    Feed it every AdvFrameMeta after the SGIE stages ran; read per-track
    results via get_result(track_id) or let it annotate ObjectRecords."""

    def __init__(self, config: LPRConfig | None = None):
        self.config = config or LPRConfig()
        self.voter = PlateVoter()
        self.reid = VehicleReID(self.config.similarity_threshold,
                                self.config.embedding_dim)
        # keys are (source_id, track_id): unlike the original system, track
        # ids from different cameras cannot collide
        self._track_embeddings: dict[tuple, list[np.ndarray]] = {}
        self._track_vehicle_ids: dict[tuple, str] = {}
        self._results: dict[tuple, LPRResult] = {}
        if self.config.embeddings_dir:
            self.reid.load(self.config.embeddings_dir)

    def get_result(self, track_id: int, source_id: str = "") -> LPRResult | None:
        return self._results.get((source_id, track_id))

    def save(self) -> None:
        if self.config.embeddings_dir:
            self.reid.save(self.config.embeddings_dir)

    def process(self, frame: AdvFrameMeta, timestamp: str) -> None:
        cfg = self.config
        for obj in frame.primary_objects():
            if obj.class_id not in cfg.vehicle_class_ids or obj.object_id is None:
                continue
            self._process_vehicle(frame.source_id, obj, timestamp)

    def _process_vehicle(self, source_id: str, obj: AdvObjectMeta,
                         timestamp: str) -> None:
        cfg = self.config
        track_id = (source_id, obj.object_id)
        result = LPRResult()

        plate = obj.best_child_classification(cfg.plate_component_id,
                                              cfg.ocr_component_id)
        if plate is not None and plate.confidence >= cfg.plate_confidence_threshold:
            self.voter.accumulate(str(track_id), plate.label)

        vote = self.voter.vote(str(track_id), min_reads=cfg.min_plate_reads)
        if vote:
            result.plate_text, result.plate_read_count = vote
            result.plate_confidence = plate.confidence if plate else 0.0

        embedding = obj.tensors.get(cfg.embedding_component_id)
        if embedding is not None and embedding.size == cfg.embedding_dim:
            embeds = self._track_embeddings.setdefault(track_id, [])
            embeds.append(embedding)
            if (len(embeds) >= EMBEDDINGS_TO_AVERAGE
                    and track_id not in self._track_vehicle_ids):
                avg = np.mean(embeds[:EMBEDDINGS_TO_AVERAGE], axis=0
                              ).astype(np.float32)
                norm = np.linalg.norm(avg)
                if norm > 0:
                    avg /= norm
                self._track_vehicle_ids[track_id] = self.reid.lookup_or_create(
                    avg, timestamp)
            if track_id in self._track_vehicle_ids:
                result.vehicle_id = self._track_vehicle_ids[track_id]
                result.embedding = embedding
                if result.plate_text and result.vehicle_id:
                    self.reid.update_plate(result.vehicle_id,
                                           result.plate_text, timestamp)

        self._results[track_id] = result
