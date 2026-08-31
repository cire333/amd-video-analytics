# AMD Video Analytics Pipeline — Architecture

A DeepStream-equivalent multi-stream video analytics pipeline for AMD hardware.

Unlike NVIDIA's DeepStream — which is one SDK stitching decode → batch → infer →
track over a unified CUDA layer — the AMD equivalent integrates 3–4 separately
maintained stacks (VAAPI/Mesa for decode, ROCm/HIP for compute, GStreamer for
transport). The architecture below is designed around that reality, and around
two explicit goals that DeepStream handles poorly:

- **Heterogeneous streams** — varying fps, resolution, and dropped frames, with
  hot add/remove of sources, without pipeline restarts.
- **Hardware flexibility** — portability across AMD's structurally different GPU
  tiers rather than assuming one uniform "AMD GPU."

---

## 1. High-level data flow

```
[Stream Registry] <--- add / remove / reconnect (mutable, polled)
        |
        v
[Per-Stream Decoder Workers]   VAAPI decode on VCN, one worker per source
        |  dmabuf export + fence
        v
[GPU Bridge Layer]             dmabuf -> HIP import, detile, NV12->RGB, ROI crop
        |  HIP tensor (linear RGB, ROI-cropped)
        v
[Per-Stream Preprocessor]      crop-rect apply, color range/matrix fix, resize
        |
        v
[Dynamic Batcher]              timestamp-windowed, variable batch shape
        |
        v
[Model Execution Graph]        single model, or sequential/parallel composition
        |
        v
[Tracker Bank]                 per-source Kalman state, full-frame coordinates
        |
        v
[Output Sink]                  OSD / RTSP-out / message broker, per stream
```

Core structural choice: **everything above the decoder is keyed by `source_id`,
not by pipeline position.** Streams are independent state machines that happen to
share a batching stage. No stage assumes a fixed stream count or a fixed frame
shape — which is precisely what makes hot add/remove and mid-stream resolution
changes cheap instead of a rebuild.

---

## 2. Hardware abstraction — three tiers, not one

AMD's lineup is not one target with variations. The real splits:

- **CDNA (Instinct MI-series, datacenter)** vs **RDNA (Radeon, workstation/
  consumer)** — different ISAs. Some Instinct SKUs have *no video decode engine
  at all*, so "GPU" does not imply "can decode."
- **RDNA1 / RDNA2 / RDNA3 (GFX10 / GFX10.3 / GFX11)** — each changed the
  tiling/swizzle layout and compute-unit structure. Kernels tuned for one don't
  perform identically on another.
- **Ryzen AI NPU (edge/APU)** — a different runtime stack entirely (Vitis AI /
  ONNX-DirectML), not ROCm.

**What the driver abstracts for you:** the `amdgpu` kernel driver is unified
across CDNA/RDNA, and ROCm/HIP's compiler backend targets the right ISA per GFX
generation — so one HIP kernel runs across generations. That's real leverage.

**What lands on you regardless of driver:** (1) capability *presence* — the
driver reports whether a decode engine exists but won't invent one; (2)
*cross-subsystem bridging* — VAAPI<->HIP is outside the zone the driver unifies;
(3) *per-generation performance tuning* — correct code is generated for all
generations, but not equally fast code.

Design response — three layers, not one flag-driven abstraction:

```python
class DeviceCapabilities:
    has_decode_engine: bool     # VCN present? (False on some Instinct)
    gfx_generation: str         # "gfx10" / "gfx10.3" / "gfx11" / "gfx9"
    total_vram: int
    memory_bandwidth: float
    device_ordinal: int
    drm_render_node: str

def probe_devices() -> list[DeviceCapabilities]:
    # queried at startup — load-bearing, not cosmetic:
    # the pipeline literally cannot assume decode exists on every device
    ...
```

- **Capability-detection layer** — probes each device at startup (above).
- **Per-generation backend implementations** behind a common interface — separate
  detile logic and tuned launch configs per GFX gen, rather than one kernel with
  runtime branches.
- **Deployment profiles** — presets like *Instinct-only (no decode, frames fed
  from elsewhere)*, *single Radeon (decode + infer)*, *mixed fleet* — that map
  onto the capability probe and the device-ordinal checks in the bridge layer.

---

## 3. Stream Registry

Mutable and polled — not baked into a static GStreamer graph at construction.

```python
class StreamRegistry:
    def __init__(self):
        self.streams: dict[str, StreamHandle] = {}
        self.lock = threading.RLock()

    def add_stream(self, source_id, uri, device_ordinal=None):
        with self.lock:
            handle = StreamHandle(source_id, uri, device_ordinal)
            handle.start()                 # spins up its own decoder worker
            self.streams[source_id] = handle

    def remove_stream(self, source_id):
        with self.lock:
            self.streams.pop(source_id).stop()

    def snapshot(self):                    # cheap copy; batcher iterates lock-free
        with self.lock:
            return list(self.streams.values())
```

Reconnect logic (exponential backoff, distinguishing a source hiccup from a
source that's gone) lives inside `StreamHandle`; the registry never blocks on a
bad stream. Adding/removing a source is a registry update the batcher picks up
next cycle — no pipeline renegotiation.

---

## 4. Decoder Worker (per stream)

One worker per source; owns its VAAPI display and surface pool.

```python
class DecoderWorker:
    def __init__(self, source_id, uri, device_ordinal):
        self.source_id = source_id
        self.device_ordinal = device_ordinal        # pins to a render node
        self.va_display = vaapi_init(device_ordinal)
        self.frame_queue = RingBuffer(capacity=4)    # bounded -> backpressure

    def decode_loop(self):
        for packet in self.demux(uri):
            surface = self.va_display.decode(packet)
            vaSyncSurface(surface)                     # FENCE: decode complete
            crop_rect = surface.get_crop_rect()        # real dims, not padded
            dmabuf_fd, planes = vaExportSurfaceHandle(surface)  # per-plane off/pitch
            frame = RawFrame(
                source_id=self.source_id,
                pts=packet.pts,
                dmabuf_fd=dmabuf_fd,
                planes=planes,                # [(offset, pitch), ...] Y then UV
                crop_rect=crop_rect,
                color_range=surface.color_range,    # limited / full
                color_matrix=surface.color_matrix,  # BT.601 / BT.709
                tiling_mode=surface.tiling_mode,
                device_ordinal=self.device_ordinal,
            )
            if not self.frame_queue.try_push(frame):
                self.drop_and_close_fd(frame)          # explicit fd close on drop
```

---

## 5. GPU Bridge (dmabuf -> HIP) — the part DeepStream gives you free

VCN decode is exposed via VAAPI (Mesa/DRM stack); ROCm/HIP compute via KFD. A
VAAPI surface is **not** a HIP pointer. The bridge is DMA-BUF: VAAPI exports the
surface as an fd, HIP imports it as external memory.

```python
def import_frame_to_hip(frame: RawFrame, target_ordinal: int, roi_mask=None):
    if frame.device_ordinal != target_ordinal:
        raise DeviceMismatchError(frame.source_id)     # explicit, not silent wrong-bind

    hip_mem = hipImportExternalMemory(frame.dmabuf_fd, size=frame.buffer_size)

    if frame.tiling_mode != LINEAR:                     # detile if swizzled
        hip_mem = detile_kernel(hip_mem, frame.tiling_mode, frame.planes)

    y_plane, uv_plane = split_planes(hip_mem, frame.planes)   # honor per-plane offset/pitch

    # Fuse ROI crop INTO the NV12->RGB conversion: the full RGB frame is never
    # materialized, so full-res data never leaves the decode surface.
    rgb = nv12_to_rgb_kernel(
        y_plane, uv_plane,
        crop_rect=frame.crop_rect,
        color_range=frame.color_range,      # limited-range handling
        color_matrix=frame.color_matrix,    # BT.601/709 matrix
        roi=roi_mask,                        # see ROI section
    )

    close(frame.dmabuf_fd)                  # ownership transferred; release fd
    return rgb                              # linear HIP tensor
```

### Bridge edge cases (in rough order of "will bite you")

1. **Tiling / DRM format modifier** — VCN surfaces are usually tiled, not linear.
   Import-as-linear on a tiled buffer gives a garbled image, and the tiling
   scheme differs per GFX gen. *First move: force linear VAAPI export* (driver
   does the untangle via `addrlib`); only write a detile kernel if profiling
   shows that driver-side copy is a real bottleneck. Don't hand-derive the
   swizzle math — pull it from AMD's open-source `addrlib`.
2. **Stride/pitch padding** — row pitch is aligned (often 256B) and != width. Read
   the real per-plane pitch; assuming `width == stride` gives a diagonal-shear
   image people misdiagnose as a color bug.
3. **Multi-plane NV12 offsets** — Y then interleaved UV at half res; a wrong
   plane offset gives chroma noise while luma looks fine.
4. **Fence/sync races** — importing a dmabuf doesn't guarantee the decode write
   finished. Without `vaSyncSurface`/DRM sync, inference can race the decoder and
   read a half-written frame. Load-dependent — passes in dev, fails under multi-
   stream throughput.
5. **FD lifetime / pool reuse** — each export is a new fd; leak them and you hit
   the ulimit (looks like "crashes after ~10 min"). Holding a HIP import past the
   decoder reclaiming its pooled surface gives use-after-free/torn frames.
6. **Color range & matrix** — limited (16-235) vs full, BT.601 vs BT.709. Wrong
   matrix = washed-out/oversaturated, no crash. Municipal encoders tag (or
   mistag) this inconsistently — real risk for scraped public feeds.
7. **Odd resolutions** — chroma subsampling needs even dims; use the decoder's
   crop rect, don't assume decoded dims == source dims, or you get a green border
   on non-conforming streams.
8. **Multi-GPU render-node mismatch** — a dmabuf is tied to a specific DRM render
   node; importing into a HIP context on a different device fails or silently
   binds wrong. Explicit device-ordinal check (above).

---

## 6. Dynamic Batcher — the fix for DeepStream's rigidity

`nvstreammux` wants a single batch-level width/height/framerate; every source is
scaled/padded to a common frame, and a resolution/fps change or an added stream
means rebuilding the muxer config. That's batching-friendly, not variable-input-
friendly. This design inverts it.

```python
class DynamicBatcher:
    def __init__(self, window_ms=33):
        self.window_ms = window_ms

    def assemble_batch(self, registry: StreamRegistry):
        now = current_time_ms()
        items = []
        for stream in registry.snapshot():
            frame = stream.frame_queue.peek_latest_within(now - self.window_ms, now)
            if frame is None:
                continue                    # stalled source just sits out this cycle
            items.append(frame)             # per-item resolution may differ
        return items                        # variable-length, variable-shape batch
```

- **No stream blocks another** — a slow/stalled source contributes nothing that
  cycle instead of holding up the batch.
- **No canonical resolution** — shape normalization is per-item in preprocessing,
  not a pipeline-wide constraint.
- **Timestamp-driven** — every frame carries a real PTS; the batcher assembles
  "latest frame per stream within a window," which absorbs variable/dropped-frame
  sources (constant with municipal feeds).
- **Tradeoff (be explicit):** DeepStream's rigidity buys predictable throughput
  (fixed shapes -> fixed memory, no recompilation). This design trades some per-
  batch latency predictability for robustness against heterogeneous feeds — the
  right trade for scraped public feeds, not necessarily for a controlled camera
  farm. `window_ms` should be deployment-tunable.

---

## 7. Preprocessor (per item, before batch assembly)

```python
def preprocess(rgb_tensor, target_shape):
    # per-item resize/pad: batch-level shape is a property of the inference call,
    # not something sources are forced into upstream
    return resize_kernel(rgb_tensor, target_shape)
```

---

## 8. ROI-gated inference

Specify regions of interest so only in-region pixel data feeds inference.

**Crop vs mask — they buy different things:**
- **Crop** to the ROI bounding rect *reduces data in* — fewer pixels -> less memory
  bandwidth and inference compute, proportional to area dropped.
- **Mask** (zeroing outside the region) does *not* save compute — the model still
  processes a full-size tensor — it only stops out-of-region content distracting
  the detector.

Real intersection ROIs are **polygons** (lanes/approaches aren't axis-aligned),
so use both: crop to the polygon's bounding rect for the compute win, then apply
a polygon mask within that rect. Generate the mask **once** at ROI-config time,
not per frame.

**Placement:** fold the crop into the NV12->RGB kernel in the bridge (Section 5).
You already touch every pixel there, so cropping on convert means the full RGB
frame is never materialized — full-res data never leaves the decode surface.
Much cheaper than convert-then-crop.

**Define ROIs in normalized coordinates (0-1), not pixels.** If a source changes
resolution mid-stream, a pixel-defined ROI silently points at the wrong region; a
normalized ROI re-projects against whatever the current frame dimensions are —
directly consistent with the mid-stream-resolution-change robustness goal.

**Critical rule — ROI gates inference input, NOT tracking:**

```python
# WRONG: keying the tracker on (source_id, roi_id) partitions tracking per ROI.
# A vehicle turning from approach A to approach B looks like a car vanishing
# in A and a different car appearing in B — the turn is never connected.

# RIGHT: ROI decides what's fed to detection; detections are un-projected back
# to FULL-FRAME coordinates before the tracker sees them.
detections_local = detect(roi_cropped_tensor)
detections_full  = unproject_to_full_frame(detections_local, frame.roi_transform)
tracker_bank.update(frame.source_id, detections_full)   # tracking stays full-frame
```

**Accuracy caveat to test, not assume:** if ROIs are small and the detector was
trained on full-frame context, cropping shifts object scale (a 40px vehicle in a
full frame is a different input distribution when it fills a tight crop). Validate
detection quality on cropped input before committing; consider rescaling crops to
a consistent input size.

ROI cropping costs the batcher nothing new — different ROIs across streams are
just different per-item crop sizes, exactly the heterogeneity Section 6 already
absorbs.

---

## 9. Model Execution Graph — single, sequential, or parallel

One structure covers all composition patterns: a **DAG of model nodes**, edges
are on-GPU tensor handoffs. A single model is a one-node graph; a sequential
stack is a chain; parallel-then-merge is fan-in into a bridge node.

```python
class ModelNode:
    def __init__(self, node_id, model, input_nodes: list[str]):
        self.node_id = node_id
        self.model = model                 # HIP/ROCm-resident model
        self.input_nodes = input_nodes     # upstream node ids; [] = takes the frame

class ModelGraph:
    def add_node(self, node_id, model, input_nodes=None): ...
    def topological_order(self): ...       # exec order + which nodes may run concurrently
```

```python
# Sequential stack:
graph.add_node("detector",   detector_model,   input_nodes=[])
graph.add_node("classifier", classifier_model, input_nodes=["detector"])

# Parallel-then-bridge:
graph.add_node("detector_rgb",     rgb_model,     input_nodes=[])
graph.add_node("detector_thermal", thermal_model, input_nodes=[])
graph.add_node("fusion_bridge",    bridge_model,  input_nodes=["detector_rgb", "detector_thermal"])
```

**Executor** — the part that matters for AMD is keeping tensors on-device between
nodes and only forcing sync where branches converge:

```python
class GraphExecutor:
    def __init__(self, graph, device_ordinal):
        self.graph = graph
        self.order = graph.topological_order()
        # independent branches -> separate HIP streams (real on-device overlap);
        # a sequential chain shares one stream (no concurrency to gain, less sync)
        self.streams = self._allocate_streams(graph)

    def run(self, input_tensor):
        results = {}                         # node_id -> device tensor, never copied to host
        for node_id in self.order:
            node = self.graph.nodes[node_id]
            stream = self.streams[branch_of(node_id)]
            inputs = ([input_tensor] if not node.input_nodes
                      else [results[d] for d in node.input_nodes])
            if len(node.input_nodes) > 1:
                self._sync_branches(node.input_nodes, stream)   # fan-in join
            with hip_stream_context(stream):
                results[node_id] = node.model(*inputs)          # stays resident
        return results[self.order[-1]]

    def _sync_branches(self, dep_ids, target_stream):
        for d in dep_ids:                    # only block where branches converge
            hipStreamWaitEvent(target_stream, self.streams[branch_of(d)])
```

Constraints worth flagging: parallel branches keep both models' weights +
activations resident at once (gate via the capability layer on smaller cards);
HIP streams aren't free and hardware queues are finite (cap concurrent streams,
fall back to scheduled execution); fan-in sync is where race bugs hide (unit-test
with deliberately mismatched branch latencies); and dynamic-shape recompilation
cost applies at *every* node in a chain, not once.

---

## 10. Tracker Bank

```python
class TrackerBank:
    def __init__(self):
        self.trackers: dict[str, KalmanTracker] = {}   # keyed by source_id

    def update(self, source_id, detections_full_frame):
        if source_id not in self.trackers:
            self.trackers[source_id] = KalmanTracker()
        return self.trackers[source_id].update(detections_full_frame)
```

Per-source state means a stream dropping out for a few cycles doesn't corrupt
other streams' track history, and removing a stream is just deleting its entry.
Tracking operates in full-frame coordinates (see Section 8) so cross-ROI turns
stay connected. This layer is hardware-agnostic — your existing custom Kalman
tracker carries over unchanged.

---

## 11. Failure / backpressure summary

| Condition | Handling |
|---|---|
| Source disconnects | `DecoderWorker` backoff-retries; registry entry stays; batcher stops seeing its frames |
| Decode queue full | Drop oldest, explicitly close its dmabuf fd |
| Resolution/fps change mid-stream | No rebuild — next frame carries new crop_rect/pts; preprocessor resizes per-item; normalized ROIs re-project |
| New stream added | Registry spins up a `DecoderWorker`; batcher picks it up next cycle |
| Multi-GPU mismatch | Explicit `device_ordinal` check at import; fails loud |
| Device lacks decode engine | Capability layer flags it; deployment profile feeds frames from elsewhere |
| Tiled surface | Force linear export (v1) or per-GFX-gen detile kernel |

---

## 12. Open decisions to pin down before building

1. **Linear export vs. detile kernel** — start with forced linear VAAPI export;
   build the detile kernel only if the driver-side copy profiles as a bottleneck.
2. **Batch window size (`window_ms`)** — latency vs. throughput; tunable per
   deployment (camera farm vs. scraped feeds).
3. **Per-stream vs. shared HIP streams/contexts** — affects decode/inference
   overlap across sources; benchmark with 2-3 real streams.
4. **Static vs. runtime-reconfigurable model graph** — static (declared at
   startup) is far simpler; go reconfigurable only with a concrete need.
5. **Per-branch device placement** — pin independent branches to different
   physical GPUs (real parallelism, reintroduces cross-device ordinal matching)
   vs. one device (simpler, shares bandwidth).
6. **ROI crop rescaling** — whether to rescale crops to a consistent input size to
   protect detector accuracy on small regions.
