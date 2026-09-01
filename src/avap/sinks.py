"""Output sinks and formatters.

Locations (parsed from output_location):
    /path/file.jsonl            local file
    s3://bucket/prefix/file     S3 (boto3)
    kafka://host:9092/topic     Kafka (kafka-python)
    kinesis://stream-name       Kinesis Data Streams (boto3); optional
    kinesis://region/stream     region override before the stream name
    sqs://https%3A//... or a plain https queue URL      SQS (boto3)

Formats: "json" (one object per frame-record), "csv" (one row per tracked
object), "parquet" (pyarrow, written on close/flush). An optional
output_format_template is a Python str.format template applied per record
(overrides the structured formats for line-oriented sinks), e.g.:
    "{source_id},{pts_us},{n_objects}"
Template fields: source_id, frame, pts_us, ts, n_objects, objects (list).
"""
from __future__ import annotations

import csv
import io
import json
import time
from typing import Any

FORMATS = ("json", "csv", "parquet")

CSV_FIELDS = ["source_id", "frame", "pts_us", "track_id", "label",
              "class_id", "conf", "x1", "y1", "x2", "y2"]


def frame_record(source_id: str, frame_idx: int, pts_us: int,
                 objects: list[dict[str, Any]]) -> dict[str, Any]:
    return {"source_id": source_id, "frame": frame_idx, "pts_us": pts_us,
            "ts": time.time(), "n_objects": len(objects), "objects": objects}


class Formatter:
    def __init__(self, fmt: str = "json", template: str | None = None):
        if fmt not in FORMATS:
            raise ValueError(f"output format must be one of {FORMATS}")
        self.fmt = fmt
        self.template = template

    def lines(self, rec: dict[str, Any]) -> list[str]:
        """Line-oriented rendering (local file, kafka, sqs)."""
        if self.template is not None:
            return [self.template.format(**rec)]
        if self.fmt == "json":
            return [json.dumps(rec, separators=(",", ":"))]
        if self.fmt == "csv":
            buf = io.StringIO()
            w = csv.writer(buf)
            for o in rec["objects"]:
                w.writerow([rec["source_id"], rec["frame"], rec["pts_us"],
                            o["track_id"], o["label"], o["class_id"],
                            o["conf"], *o["bbox"]])
            return buf.getvalue().splitlines()
        raise ValueError("parquet is only supported for file/S3 sinks")

    def rows(self, rec: dict[str, Any]) -> list[dict[str, Any]]:
        """Row-per-object rendering (parquet)."""
        return [{"source_id": rec["source_id"], "frame": rec["frame"],
                 "pts_us": rec["pts_us"], "track_id": o["track_id"],
                 "label": o["label"], "class_id": o["class_id"],
                 "conf": o["conf"], "x1": o["bbox"][0], "y1": o["bbox"][1],
                 "x2": o["bbox"][2], "y2": o["bbox"][3]}
                for o in rec["objects"]]


class Sink:
    def emit(self, rec: dict[str, Any]) -> None: ...
    def close(self) -> None: ...


class LocalFileSink(Sink):
    def __init__(self, path: str, formatter: Formatter):
        self.formatter = formatter
        self.path = path
        if formatter.fmt == "parquet" and formatter.template is None:
            self._rows: list[dict] = []
            self._fh = None
        else:
            self._rows = []
            self._fh = open(path, "w")
            if formatter.fmt == "csv" and formatter.template is None:
                self._fh.write(",".join(CSV_FIELDS) + "\n")

    def emit(self, rec):
        if self._fh is None:
            self._rows.extend(self.formatter.rows(rec))
        else:
            for line in self.formatter.lines(rec):
                self._fh.write(line + "\n")

    def close(self):
        if self._fh is not None:
            self._fh.close()
        else:
            import pyarrow as pa
            import pyarrow.parquet as pq
            pq.write_table(pa.Table.from_pylist(self._rows), self.path)


class S3Sink(Sink):
    """Buffers records and uploads one object on close (v1 semantics)."""

    def __init__(self, uri: str, formatter: Formatter):
        import boto3
        self._s3 = boto3.client("s3")
        rest = uri[len("s3://"):]
        self.bucket, _, self.key = rest.partition("/")
        if not self.bucket or not self.key:
            raise ValueError(f"bad S3 uri (want s3://bucket/key): {uri}")
        self.formatter = formatter
        self._lines: list[str] = []
        self._rows: list[dict] = []

    def emit(self, rec):
        if self.formatter.fmt == "parquet" and self.formatter.template is None:
            self._rows.extend(self.formatter.rows(rec))
        else:
            self._lines.extend(self.formatter.lines(rec))

    def close(self):
        if self._rows:
            import pyarrow as pa
            import pyarrow.parquet as pq
            buf = io.BytesIO()
            pq.write_table(pa.Table.from_pylist(self._rows), buf)
            body = buf.getvalue()
        else:
            body = ("\n".join(self._lines) + "\n").encode()
        self._s3.put_object(Bucket=self.bucket, Key=self.key, Body=body)


class KafkaSink(Sink):
    def __init__(self, uri: str, formatter: Formatter):
        from kafka import KafkaProducer  # pip install kafka-python
        rest = uri[len("kafka://"):]
        bootstrap, _, topic = rest.partition("/")
        if not bootstrap or not topic:
            raise ValueError(f"bad Kafka uri (want kafka://host:port/topic): {uri}")
        self.topic = topic
        self.formatter = formatter
        self._producer = KafkaProducer(bootstrap_servers=bootstrap)

    def emit(self, rec):
        for line in self.formatter.lines(rec):
            self._producer.send(self.topic, line.encode())

    def close(self):
        self._producer.flush()
        self._producer.close()


class KinesisSink(Sink):
    """kinesis://stream-name or kinesis://region/stream-name. Partition key
    is the record's source_id, preserving per-camera ordering per shard."""

    def __init__(self, uri: str, formatter: Formatter):
        import boto3
        rest = uri[len("kinesis://"):]
        region, _, stream = rest.partition("/")
        if not stream:  # no region segment: kinesis://stream-name
            region, stream = None, rest
        if not stream:
            raise ValueError(
                f"bad Kinesis uri (want kinesis://[region/]stream-name): {uri}")
        self.stream = stream
        self.formatter = formatter
        self._kinesis = boto3.client(
            "kinesis", **({"region_name": region} if region else {}))

    def emit(self, rec):
        key = str(rec.get("source_id", "avap"))
        for line in self.formatter.lines(rec):
            self._kinesis.put_record(StreamName=self.stream,
                                     Data=line.encode(), PartitionKey=key)

    def close(self):
        pass


class SQSSink(Sink):
    def __init__(self, uri: str, formatter: Formatter):
        import boto3
        self._sqs = boto3.client("sqs")
        self.queue_url = uri[len("sqs://"):] if uri.startswith("sqs://") else uri
        self.formatter = formatter

    def emit(self, rec):
        for line in self.formatter.lines(rec):
            self._sqs.send_message(QueueUrl=self.queue_url, MessageBody=line)

    def close(self):
        pass


def make_sink(output_location: str, output_format: str = "json",
              template: str | None = None) -> Sink:
    formatter = Formatter(output_format, template)
    if output_location.startswith("s3://"):
        return S3Sink(output_location, formatter)
    if output_location.startswith("kafka://"):
        return KafkaSink(output_location, formatter)
    if output_location.startswith("kinesis://"):
        return KinesisSink(output_location, formatter)
    is_sqs_https = (output_location.startswith("https://")
                    and output_location.split("/")[2].startswith("sqs."))
    if output_location.startswith("sqs://") or is_sqs_https:
        return SQSSink(output_location, formatter)
    return LocalFileSink(output_location, formatter)
