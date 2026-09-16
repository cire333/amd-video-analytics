#!/usr/bin/env python3
"""nvstreammux behavioral probe (runs inside the DeepStream 7.1 container).

Builds N sources -> nvstreammux -> fakesink, records:
  * every input buffer arriving at each mux sink pad (wall clock, pts, dims)
  * every output batch (wall clock, pts, duration, per-frame NvDsFrameMeta)
  * optionally dumps the first K output surfaces (RGBA) as .npy for
    scaling-geometry analysis
  * downstream custom events (pad-added, stream-eos, segment...) seen at the sink
Writes a JSONL trace.
"""
import argparse, json, os, sys, time, threading
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import numpy as np
import pyds

Gst.init(None)


def now_ns():
    return time.monotonic_ns()


class Probe:
    def __init__(self, args):
        self.args = args
        self.out = open(args.out, "w")
        self.n_batches = 0
        self.n_dumped = 0
        self.t0 = now_ns()
        self.loop = GLib.MainLoop()
        self.eos_seen = False
        self._lock = threading.Lock()

    def rec(self, kind, **kw):
        kw["kind"] = kind
        kw["t_ms"] = (now_ns() - self.t0) / 1e6
        line = json.dumps(kw) + "\n"
        with self._lock:
            self.out.write(line)

    # ---- input side ---------------------------------------------------
    def sink_probe(self, pad, info, idx):
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK
        caps = pad.get_current_caps()
        w = h = 0
        fmt = ""
        if caps:
            s = caps.get_structure(0)
            w, h = s.get_value("width"), s.get_value("height")
            fmt = s.get_value("format")
        self.rec("in", pad=idx, pts_ns=int(buf.pts) if buf.pts != Gst.CLOCK_TIME_NONE else -1,
                 dur_ns=int(buf.duration) if buf.duration != Gst.CLOCK_TIME_NONE else -1,
                 w=w, h=h, fmt=fmt)
        return Gst.PadProbeReturn.OK

    def sink_event_probe(self, pad, info, idx):
        ev = info.get_event()
        if ev is not None:
            self.rec("in_event", pad=idx, event=ev.type.get_name(ev.type) if hasattr(ev.type, "get_name") else str(ev.type))
        return Gst.PadProbeReturn.OK

    # ---- output side --------------------------------------------------
    def src_probe(self, pad, info):
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK
        caps = pad.get_current_caps()
        cw = ch = 0
        if caps:
            s = caps.get_structure(0)
            cw, ch = s.get_value("width"), s.get_value("height")
        bm = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        frames = []
        if bm is not None:
            l = bm.frame_meta_list
            while l is not None:
                fm = pyds.NvDsFrameMeta.cast(l.data)
                frames.append(dict(pad_index=fm.pad_index, source_id=fm.source_id,
                                   batch_id=fm.batch_id, frame_num=fm.frame_num,
                                   buf_pts=fm.buf_pts, ntp=fm.ntp_timestamp,
                                   src_w=fm.source_frame_width, src_h=fm.source_frame_height,
                                   nspf=fm.num_surfaces_per_frame))
                if self.args.dump and self.n_dumped < self.args.dump:
                    try:
                        arr = pyds.get_nvds_buf_surface(hash(buf), fm.batch_id)
                        a = np.array(arr, copy=True)
                        fn = os.path.join(self.args.dump_dir,
                                          f"b{self.n_batches:04d}_p{fm.pad_index}_f{fm.frame_num}.npy")
                        np.save(fn, a)
                        frames[-1]["dump"] = fn
                        frames[-1]["out_shape"] = list(a.shape)
                        self.n_dumped += 1
                    except Exception as e:  # noqa
                        frames[-1]["dump_err"] = str(e)
                l = l.next
        self.rec("batch", n=self.n_batches, pts_ns=int(buf.pts) if buf.pts != Gst.CLOCK_TIME_NONE else -1,
                 dur_ns=int(buf.duration) if buf.duration != Gst.CLOCK_TIME_NONE else -1,
                 caps_w=cw, caps_h=ch,
                 num_frames=bm.num_frames_in_batch if bm else -1,
                 max_frames=bm.max_frames_in_batch if bm else -1, frames=frames)
        self.n_batches += 1
        if self.args.max_batches and self.n_batches >= self.args.max_batches:
            GLib.idle_add(self.stop)
        return Gst.PadProbeReturn.OK

    def src_event_probe(self, pad, info):
        ev = info.get_event()
        if ev is None:
            return Gst.PadProbeReturn.OK
        name = Gst.EventType.get_name(ev.type) if isinstance(ev.type, Gst.EventType) else str(int(ev.type))
        extra = {}
        st = ev.get_structure()
        if st is not None:
            extra["struct"] = st.to_string()[:300]
        self.rec("out_event", event=name, type_int=int(ev.type), **extra)
        return Gst.PadProbeReturn.OK

    def stop(self):
        if self.loop.is_running():
            self.loop.quit()
        return False

    # ---- pipeline ------------------------------------------------------
    def build(self):
        a = self.args
        p = Gst.Pipeline.new("probe")
        mux = Gst.ElementFactory.make("nvstreammux", "mux")
        assert mux is not None
        mux.set_property("batch-size", a.batch_size)
        if a.timeout is not None:
            mux.set_property("batched-push-timeout", a.timeout)
        if a.mux == "old":
            mux.set_property("width", a.width)
            mux.set_property("height", a.height)
            mux.set_property("live-source", a.live)
            mux.set_property("enable-padding", a.padding)
            mux.set_property("nvbuf-memory-type", 3 if a.dump else 0)
            if a.interp is not None:
                mux.set_property("interpolation-method", a.interp)
            if a.sync_inputs:
                mux.set_property("sync-inputs", True)
            if a.max_latency:
                mux.set_property("max-latency", a.max_latency)
            if a.attach_sys_ts is not None:
                mux.set_property("attach-sys-ts", bool(a.attach_sys_ts))
        else:
            if a.config:
                mux.set_property("config-file-path", a.config)
            if a.sync_inputs:
                mux.set_property("sync-inputs", True)
            if a.max_latency:
                mux.set_property("max-latency", a.max_latency)
            if a.attach_sys_ts is not None:
                mux.set_property("attach-sys-ts", bool(a.attach_sys_ts))
        sink = Gst.ElementFactory.make("fakesink", "sink")
        sink.set_property("sync", a.sink_sync)
        sink.set_property("async", False)
        p.add(mux)
        p.add(sink)
        if a.post_rgba:
            # NV12 batch -> RGBA (1:1, no scaling) so the surface can be dumped
            pcv = Gst.ElementFactory.make("nvvideoconvert", "postconv")
            pcv.set_property("nvbuf-memory-type", 3)
            pcf = Gst.ElementFactory.make("capsfilter", "postcaps")
            pcf.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"))
            p.add(pcv); p.add(pcf)
            mux.link(pcv); pcv.link(pcf); pcf.link(sink)
            self.dump_pad = pcf.get_static_pad("src")
        else:
            mux.link(sink)
            self.dump_pad = mux.get_static_pad("src")

        for i, src in enumerate(a.source):
            self.add_source(p, mux, i, src)

        self.dump_pad.add_probe(Gst.PadProbeType.BUFFER, self.src_probe)
        mux.get_static_pad("src").add_probe(Gst.PadProbeType.EVENT_DOWNSTREAM, self.src_event_probe)
        self.pipeline = p
        bus = p.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_msg)

    def add_source(self, p, mux, i, spec):
        """spec: 'file:/path[:drop=N]' | 'live:W:H:FPS[:pattern]' | 'png:/path:FPS' | 'rtsp:url'"""
        parts = spec.split(":")
        kind = parts[0]
        bin_ = Gst.Bin.new(f"src{i}")
        if kind == "file":
            path = parts[1]
            drop = 0
            for extra in parts[2:]:
                if extra.startswith("drop="):
                    drop = int(extra[5:])
            fs = Gst.ElementFactory.make("filesrc"); fs.set_property("location", path)
            dm = Gst.ElementFactory.make("qtdemux")
            pa = Gst.ElementFactory.make("h264parse")
            dec = Gst.ElementFactory.make("nvv4l2decoder")
            if drop:
                dec.set_property("drop-frame-interval", drop)
            for e in (fs, dm, pa, dec):
                bin_.add(e)
            fs.link(dm)
            def on_demux_pad(d, pad, pa=pa):
                caps = pad.get_current_caps() or pad.query_caps(None)
                if not caps.get_structure(0).get_name().startswith("video/"):
                    return
                sp = pa.get_static_pad("sink")
                if sp.is_linked():
                    return
                pad.link(sp)
            dm.connect("pad-added", on_demux_pad)
            pa.link(dec)
            last = dec
            if self.args.rgba:
                cv = Gst.ElementFactory.make("nvvideoconvert")
                cv.set_property("nvbuf-memory-type", 3)
                cf = Gst.ElementFactory.make("capsfilter")
                cf.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"))
                bin_.add(cv); bin_.add(cf); dec.link(cv); cv.link(cf); last = cf
        elif kind in ("live", "test"):
            w, h, fps = int(parts[1]), int(parts[2]), parts[3]
            pattern = parts[4] if len(parts) > 4 else "smpte"
            ts = Gst.ElementFactory.make("videotestsrc")
            ts.set_property("is-live", kind == "live")
            ts.set_property("pattern", pattern)
            if self.args.num_buffers:
                ts.set_property("num-buffers", self.args.num_buffers)
            cf0 = Gst.ElementFactory.make("capsfilter")
            cf0.set_property("caps", Gst.Caps.from_string(
                f"video/x-raw,format=I420,width={w},height={h},framerate={fps}/1"))
            cv = Gst.ElementFactory.make("nvvideoconvert")
            cv.set_property("nvbuf-memory-type", 3)
            cf = Gst.ElementFactory.make("capsfilter")
            fmt = "RGBA" if self.args.rgba else "NV12"
            cf.set_property("caps", Gst.Caps.from_string(f"video/x-raw(memory:NVMM),format={fmt}"))
            for e in (ts, cf0, cv, cf):
                bin_.add(e)
            ts.link(cf0); cf0.link(cv); cv.link(cf)
            last = cf
        elif kind == "png":
            path, fps = parts[1], parts[2]
            fs = Gst.ElementFactory.make("filesrc"); fs.set_property("location", path)
            dec = Gst.ElementFactory.make("pngdec")
            fr = Gst.ElementFactory.make("imagefreeze")
            if self.args.num_buffers:
                fr.set_property("num-buffers", self.args.num_buffers)
            cf0 = Gst.ElementFactory.make("capsfilter")
            cf0.set_property("caps", Gst.Caps.from_string(f"video/x-raw,framerate={fps}/1"))
            vc = Gst.ElementFactory.make("videoconvert")
            cv = Gst.ElementFactory.make("nvvideoconvert")
            cv.set_property("nvbuf-memory-type", 3)
            cf = Gst.ElementFactory.make("capsfilter")
            fmt = "RGBA" if self.args.rgba else "NV12"
            cf.set_property("caps", Gst.Caps.from_string(f"video/x-raw(memory:NVMM),format={fmt}"))
            for e in (fs, dec, fr, cf0, vc, cv, cf):
                bin_.add(e)
            fs.link(dec); dec.link(fr); fr.link(cf0); cf0.link(vc); vc.link(cv); cv.link(cf)
            last = cf
        else:
            raise SystemExit(f"unknown source spec {spec}")
        gp = Gst.GhostPad.new("src", last.get_static_pad("src"))
        bin_.add_pad(gp)
        p.add(bin_)
        sinkpad = mux.request_pad_simple(f"sink_{i}")
        assert sinkpad is not None
        gp.link(sinkpad)
        sinkpad.add_probe(Gst.PadProbeType.BUFFER, self.sink_probe, i)
        sinkpad.add_probe(Gst.PadProbeType.EVENT_DOWNSTREAM, self.sink_event_probe, i)

    def on_msg(self, bus, msg):
        t = msg.type
        if t == Gst.MessageType.EOS:
            self.rec("bus", msg="EOS")
            self.eos_seen = True
            GLib.timeout_add(200, self.stop)
        elif t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            self.rec("bus", msg="ERROR", err=str(err), dbg=str(dbg)[:400])
            print("ERROR", err, dbg, file=sys.stderr)
            self.stop()
        elif t == Gst.MessageType.ELEMENT:
            st = msg.get_structure()
            if st is not None:
                self.rec("bus_element", struct=st.to_string()[:300])
        return True

    def run(self):
        self.build()
        self.pipeline.set_state(Gst.State.PLAYING)
        self.t0 = now_ns()
        self.rec("start", args=vars(self.args))
        if self.args.duration:
            GLib.timeout_add(int(self.args.duration * 1000), self.stop)
        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        self.rec("end", n_batches=self.n_batches, eos=self.eos_seen)
        self.pipeline.set_state(Gst.State.NULL)
        self.out.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mux", choices=["old", "new"], default="old")
    ap.add_argument("--source", action="append", default=[])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--live", type=int, default=0)
    ap.add_argument("--padding", type=int, default=0)
    ap.add_argument("--interp", type=int, default=None)
    ap.add_argument("--sync-inputs", type=int, default=0)
    ap.add_argument("--max-latency", type=int, default=0)
    ap.add_argument("--attach-sys-ts", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--rgba", action="store_true")
    ap.add_argument("--post-rgba", action="store_true")
    ap.add_argument("--dump", type=int, default=0)
    ap.add_argument("--dump-dir", default="/work/dump")
    ap.add_argument("--num-buffers", type=int, default=0)
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--sink-sync", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.dump:
        os.makedirs(args.dump_dir, exist_ok=True)
    Probe(args).run()


if __name__ == "__main__":
    main()
