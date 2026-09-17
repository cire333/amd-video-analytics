#!/usr/bin/env python3
"""Replay recorded nvstreammux input timelines through avap.streammux and
compare the batches it forms with what DeepStream formed.

For every `in` record (wall time, pad, pts) the frame is pushed at that
instant; the mux is polled whenever DS pushed a batch and at its own
deadlines. Compared: batch-size histogram, top compositions, frames per pad,
cadence, frame_num continuity.
"""
import argparse, collections, json, statistics as st, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from avap.streammux import MuxBatch, MuxConfig, SourceConfig, StreamMux  # noqa: E402

MS = 1_000_000


def load(fn):
    rows = []
    for l in open(fn, errors="ignore"):
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    return rows


def hist(batches):
    return dict(sorted(collections.Counter(len(b) for b in batches).items()))


def comps(batches, n=6):
    return collections.Counter(tuple(b) for b in batches).most_common(n)


def per_pad(batches):
    return dict(sorted(collections.Counter(p for b in batches for p in b).items()))


def cadence(ts):
    d = [b - a for a, b in zip(ts, ts[1:])]
    return (round(st.median(d), 1), round(st.mean(d), 1)) if d else (0, 0)


def config_from_args(args, n_pads):
    old = args["mux"] == "old"
    timeout = args.get("timeout")
    cfg = MuxConfig(batch_size=args["batch_size"],
                    batched_push_timeout_us=(timeout if timeout is not None else
                                             (33000 if old else 200000)),
                    live_source=bool(args.get("live", 0)) if old else False,
                    policy="legacy" if old else "new",
                    sync_inputs=bool(args.get("sync_inputs", 0)),
                    max_latency_ns=int(args.get("max_latency", 0)),
                    queue_depth=args_queue_depth(old, args))
    if not old:
        # DS 7.1 new mux: configure_module() resets min-fps to the 5 fps default
        # unless a config file is given (docs §3). Mirror that quirk.
        cfg.batched_push_timeout_us = 200000
        if args.get("config"):
            cfg.batched_push_timeout_us = 200000  # cfg files used min-fps 5 too
            if "repeats" in args["config"]:
                cfg.adaptive_batching = False
                cfg.max_same_source_frames = 4
                cfg.sources = {0: SourceConfig(max_frames_per_batch=4)}
            if "bs2" in args["config"]:
                cfg.adaptive_batching = False
    return cfg


def args_queue_depth(old, args):
    # Old mux live mode holds ~1 buffer per pad and blocks upstream; the probe's
    # videotestsrc then simply produced later frames -> emulate with depth 1
    # (drop-oldest) so the replay sees the same "latest frame per source".
    if old and args.get("live"):
        return 2
    return 4


def replay(fn, verbose=False):
    rows = load(fn)
    args = next(r["args"] for r in rows if r["kind"] == "start")
    ins = [r for r in rows if r["kind"] == "in"]
    ds_batches = [r for r in rows if r["kind"] == "batch"]
    pads = sorted({r["pad"] for r in ins})
    cfg = config_from_args(args, len(pads))
    now = [0]
    mux = StreamMux(cfg, clock=lambda: now[0], wall_clock=lambda: 0)
    mux.start()
    for p in pads:
        mux.add_pad(p)
    # merge timeline: inputs and DS batch instants (poll points)
    timeline = sorted([(r["t_ms"], 0, r) for r in ins] + [(r["t_ms"], 1, None) for r in ds_batches])
    my_batches, my_ts = [], []
    fn_by_pad = collections.defaultdict(list)

    def poll():
        for item in mux.drain():
            if isinstance(item, MuxBatch):
                my_batches.append([f.pad_index for f in item.frames])
                my_ts.append(now[0] / MS)
                for f in item.frames:
                    fn_by_pad[f.pad_index].append(f.frame_num)

    last_t = 0.0
    for t, kind, r in timeline:
        # honour the mux's own deadlines between events
        while True:
            dl = mux.next_deadline_ns()
            if dl is None or dl > int(t * MS):
                break
            now[0] = dl
            poll()
        now[0] = int(t * MS)
        if kind == 0:
            mux.push_frame(r["pad"], object(), r["pts_ns"], r["w"], r["h"])
        poll()
        last_t = t
    # EOS for file sources that ended in the trace
    for e in [r for r in rows if r["kind"] == "in_event" and r.get("event") == "eos"]:
        mux.push_eos(e["pad"])
    now[0] = int((last_t + 500) * MS)
    poll()

    ds = [[f["pad_index"] for f in b["frames"]] for b in ds_batches]
    ds_ts = [b["t_ms"] for b in ds_batches]
    print(f"\n##### {Path(fn).name}  mux={args['mux']} bs={args['batch_size']} timeout={args.get('timeout')} "
          f"live={args.get('live')} sync={args.get('sync_inputs')} pads={len(pads)}")
    print(f"  inputs/pad      : {dict(sorted(collections.Counter(r['pad'] for r in ins).items()))}")
    print(f"  batches         : DS {len(ds):5d}   avap {len(my_batches):5d}")
    print(f"  size histogram  : DS {hist(ds)}   avap {hist(my_batches)}")
    print(f"  frames per pad  : DS {per_pad(ds)}   avap {per_pad(my_batches)}")
    print(f"  cadence med/mean: DS {cadence(ds_ts)}   avap {cadence(my_ts)}")
    print(f"  compositions DS : {comps(ds)}")
    print(f"  compositions AV : {comps(my_batches)}")
    rep_ds = max((collections.Counter(b).most_common(1)[0][1] for b in ds if b), default=0)
    rep_av = max((collections.Counter(b).most_common(1)[0][1] for b in my_batches if b), default=0)
    print(f"  max same-source : DS {rep_ds}   avap {rep_av}")
    gaps = {p: sum(1 for a, b in zip(l, l[1:]) if b - a != 1) for p, l in fn_by_pad.items()}
    print(f"  avap frame_num gaps per pad: {gaps}")
    # exact match rate on composition multiset (order-insensitive) over aligned batches
    n = min(len(ds), len(my_batches))
    same = sum(1 for a, b in zip(ds[:n], my_batches[:n]) if sorted(a) == sorted(b))
    print(f"  aligned batches with identical source multiset: {same}/{n}")
    if verbose:
        for i in range(min(10, n)):
            print(f"    {i}: DS {ds[i]} @ {ds_ts[i]:.1f}   avap {my_batches[i]} @ {my_ts[i]:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    for fn in a.traces:
        replay(fn, a.v)
