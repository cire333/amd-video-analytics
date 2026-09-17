#!/usr/bin/env python3
"""Summarize a mux_probe JSONL trace: batch composition, timing, frame_num continuity."""
import json, sys, collections, statistics as st


def load(fn):
    rows, bad = [], 0
    with open(fn, errors="ignore") as f:
        for l in f:
            if not l.strip():
                continue
            try:
                rows.append(json.loads(l))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"[{fn}: {bad} corrupt lines skipped]")
    return rows


def summarize(fn, verbose=False):
    rows = load(fn)
    batches = [r for r in rows if r["kind"] == "batch"]
    ins = [r for r in rows if r["kind"] == "in"]
    evs = [r for r in rows if r["kind"] == "out_event"]
    args = next((r["args"] for r in rows if r["kind"] == "start"), {})
    print(f"\n##### {fn}")
    print(f"mux={args.get('mux')} bs={args.get('batch_size')} timeout={args.get('timeout')} live={args.get('live')} "
          f"sync_inputs={args.get('sync_inputs')} sources={len(args.get('source', []))} cfg={args.get('config')}")
    if not batches:
        print("NO BATCHES"); return
    n_in = collections.Counter(r["pad"] for r in ins)
    print(f"inputs per pad: {dict(sorted(n_in.items()))}   batches: {len(batches)}")
    sizes = collections.Counter(b["num_frames"] for b in batches)
    print(f"batch size histogram: {dict(sorted(sizes.items()))}   max_frames_in_batch={set(b['max_frames'] for b in batches)}")
    # composition patterns
    comp = collections.Counter(tuple(f["pad_index"] for f in b["frames"]) for b in batches)
    print("top compositions (pad order):", comp.most_common(8))
    # frames per pad emitted; repeats per batch
    out_per_pad = collections.Counter(f["pad_index"] for b in batches for f in b["frames"])
    print(f"output frames per pad: {dict(sorted(out_per_pad.items()))}  (dropped={ {p: n_in[p]-out_per_pad.get(p,0) for p in n_in} })")
    rep = max((collections.Counter(f["pad_index"] for f in b["frames"]).most_common(1)[0][1] for b in batches), default=0)
    print(f"max same-source frames in one batch: {rep}")
    # frame_num continuity per pad
    fn_by_pad = collections.defaultdict(list)
    for b in batches:
        for f in b["frames"]:
            fn_by_pad[f["pad_index"]].append(f["frame_num"])
    for p, lst in sorted(fn_by_pad.items()):
        gaps = [b - a for a, b in zip(lst, lst[1:])]
        print(f"  pad{p}: frame_num {lst[0]}..{lst[-1]} n={len(lst)} step_hist={dict(collections.Counter(gaps))}")
    # batch_id == position?
    bad = sum(1 for b in batches for i, f in enumerate(b["frames"]) if f["batch_id"] != i)
    print(f"batch_id != position count: {bad}")
    # output pts vs frame pts
    rel = []
    for b in batches:
        pts = [f["buf_pts"] for f in b["frames"]]
        if pts:
            rel.append((b["pts_ns"] - min(pts), b["pts_ns"] - max(pts), b["pts_ns"] - b["frames"][0]["buf_pts"]))
    if rel:
        print("out_pts - min(buf_pts) [first 6]:", [r[0] for r in rel[:6]], " out_pts-first_frame_pts uniq(first 6):", sorted(set(r[2] for r in rel))[:6])
    # inter-batch timing
    ts = [b["t_ms"] for b in batches]
    d = [b - a for a, b in zip(ts, ts[1:])]
    if d:
        print(f"inter-batch wall ms: mean={st.mean(d):.2f} median={st.median(d):.2f} min={min(d):.2f} max={max(d):.2f}")
    # source dims
    dims = collections.Counter((f["pad_index"], f["src_w"], f["src_h"]) for b in batches for f in b["frames"])
    print("source dims per pad:", sorted(dims))
    print("caps out:", set((b["caps_w"], b["caps_h"]) for b in batches))
    # ntp
    ntp = [f["ntp"] for b in batches for f in b["frames"]]
    print(f"ntp: first={ntp[0]} nonzero={sum(1 for n in ntp if n)}/{len(ntp)}")
    evn = collections.Counter(e["struct"].split(",")[0] if "struct" in e else e["event"] for e in evs)
    print("downstream events:", dict(evn))
    if verbose:
        for b in batches[:12]:
            print(f"  t={b['t_ms']:.1f} pts={b['pts_ns']/1e6:.1f}ms n={b['num_frames']} ",
                  [(f["pad_index"], f["frame_num"], round(f["buf_pts"]/1e6, 1)) for f in b["frames"]])
        # timeline of ins vs batches around the middle
    # last-batch behaviour near EOS
    tail = batches[-5:]
    print("last batches:", [[(f["pad_index"], f["frame_num"]) for f in b["frames"]] for b in tail])


if __name__ == "__main__":
    v = "-v" in sys.argv
    for fn in [a for a in sys.argv[1:] if not a.startswith("-")]:
        summarize(fn, v)
