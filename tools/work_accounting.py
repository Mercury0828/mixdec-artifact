#!/usr/bin/env python
"""One place that decides what a decoding policy COSTS. Round 11, corrected at gate 6.

🔴 It is NOT unbypassable. The exact guarantee is stated below and `self_test()` reports
the holes it cannot close. An earlier version of this line said "cannot be bypassed" and the
sixth gate showed four more ways through.

🔴 WHY THIS MODULE EXISTS. Four separate fatal findings in this project were the same error:

  1. the composed latency projection timed `WindowGraph.decode` and omitted the masking, boundary
     construction and parity a window must also perform;
  2. `parallel_runtime`'s batch throughput did the same on both sides, comparing a window KERNEL
     against a joint KERNEL and calling the ratio a decoder speedup;
  3. `enlarge_policy` called `joint.decode` for seam repair inside every policy and charged none of
     it, while its prose said the policy never needs a joint decoder;
  4. the "gap costs 3x" headline charged TWO constrained decodes per window when the ordinary decode
     already supplies the winning class, so only the opposite class needs one.

🔴🔴 AND THE FIRST VERSION OF THIS MODULE DID NOT FIX IT. The third gate found `CountingGraph` and
`check_complete` **were never used anywhere**, and that `check_complete` could not have detected an
omission anyway: it asked only whether a label appeared at least once, with no call counter. A
structural fix that is not wired in is not a fix.

SO: `instrument()` REPLACES the decode methods on the graph object **and on its matcher**, and
the independent counter lives on the MATCHER. Then a call that reaches around the graph --
`graph.m.decode(z)`, or `WindowGraph.decode(graph, z)` -- bumps the counter without charging the
ledger, and `audit()` fails. Both of those attacks are exercised by `self_test()` below.

🔴 **THE EXACT GUARANTEE, because a previous version of this docstring overstated it.**
Every decode issued *after* `instrument()` returns, through the graph or through its matcher, is
counted. A handle captured BEFORE instrumentation is not:

    raw = graph.m.decode        # captured first
    instrument(graph, ...)
    raw(z)                      # counted by nothing; audit(0) passes

The fifth gate demonstrated exactly that. There is no way to close it from inside Python, so the
mitigation is structural instead: use `build()`, which constructs and instruments in one call so no
window exists in which a handle can be taken. `audit()` is a check on the accounting, **not** a
proof that no decoder exists outside it.

UNITS. Decoded detector-layers per shot, summed over every decode. An implementation-independent
proxy, labelled as one everywhere it is used. It is **not** latency: this project measured the two
differing by more than 3x, and it is not a stand-in for the end-to-end benchmark.

Usage:  from work_accounting import WorkLedger, instrument, audit
"""


class WorkLedger:
    """Per-shot record of every decode performed, and the layers each covered."""

    def __init__(self):
        self.total_layers = 0.0
        self.by_label = {}
        self.n_decodes = 0

    def charge(self, label, layers):
        self.total_layers += float(layers)
        self.by_label[label] = self.by_label.get(label, 0.0) + float(layers)
        self.n_decodes += 1

    def reset(self):
        self.total_layers = 0.0
        self.by_label = {}
        self.n_decodes = 0


class _Counter:
    """Independent tally of real decoder calls, used to audit the ledger."""

    def __init__(self):
        self.calls = 0


RAW = _Counter()


def instrument(graph, ledger_box, label, span=None):
    """Replace `graph`'s decode methods in place so every call is charged. Returns `graph`.

    `ledger_box` is a one-element list holding the CURRENT ledger, so the same instrumented graph
    can be charged to a different ledger each shot without being re-wrapped.

    `span` is the layer count attributed to one call; it defaults to the graph's own span. A
    class-constrained decode is charged the same span, once per call -- so a complementary gap
    computed as two constrained decodes is charged twice, and one computed by reusing the ordinary
    solution for the winning class is charged once. That distinction is the point of finding 4.
    """
    if getattr(graph, "_instrumented", False):
        graph._wa_label = label
        graph._wa_span = span if span is not None else (graph.hi - graph.lo)
        graph._wa_box = ledger_box
        return graph

    graph._wa_label = label
    graph._wa_span = span if span is not None else (graph.hi - graph.lo)
    graph._wa_box = ledger_box
    raw_decode = graph.decode
    raw_class = getattr(graph, "decode_class", None)

    # 🔴 THE INDEPENDENT COUNTER LIVES ON THE MATCHER, NOT ON THE WRAPPER. The previous version
    # incremented `RAW.calls` inside the same wrapper that charged the ledger, so a call that
    # bypassed the wrapper -- `graph.m.decode(z)`, or a method handle taken before instrumentation,
    # or `WindowGraph.decode(graph, z)` -- incremented NEITHER and the audit passed on 0 == 0.
    # Gate-4 finding 1. Wrapping `m.decode` itself makes the counter genuinely independent: a raw
    # decode now bumps `RAW` without a charge, and `audit()` fails.
    if not getattr(graph.m, "_wa_counted", False):
        # 🔴 EVERY decoding entry point on the matcher, not just `decode`. The sixth gate
        # called `decode_batch`, `decode_to_edges_array`, `decode_to_matched_dets_array` and
        # `decode_to_matched_dets_dict` after instrumentation and every one passed `audit(0)`.
        for _nm in ("decode", "decode_batch", "decode_to_edges_array",
                    "decode_to_matched_dets_array", "decode_to_matched_dets_dict"):
            _raw = getattr(graph.m, _nm, None)
            if _raw is None:
                continue

            def _mk(f):
                def _w(*a, **k):
                    RAW.calls += 1
                    return f(*a, **k)
                return _w

            setattr(graph.m, _nm, _mk(_raw))
        graph.m._wa_counted = True

    def decode(*a, **k):
        graph._wa_box[0].charge(graph._wa_label, graph._wa_span)
        return raw_decode(*a, **k)

    graph.decode = decode

    # a weighted ordinary decode: same cost, but it also returns the solution's weight, which is
    # what lets the complementary gap be charged its MARGINAL cost instead of double.
    def decode_weighted(z):
        graph._wa_box[0].charge(graph._wa_label, graph._wa_span)
        return graph.m.decode(z, return_weight=True)

    graph.decode_weighted = decode_weighted

    if raw_class is not None:
        def decode_class(*a, **k):
            graph._wa_box[0].charge(graph._wa_label + ":class", graph._wa_span)
            return raw_class(*a, **k)
        graph.decode_class = decode_class

    graph._instrumented = True
    return graph


def build(cls, *args, ledger_box=None, label="graph", span=None, **kw):
    """Construct and instrument in one call, so no uninstrumented handle can be taken."""
    g = cls(*args, **kw)
    return instrument(g, ledger_box, label, span)


def self_test(cls, *args, **kw):
    """Run the bypasses the auditors named and confirm `audit()` catches them. Returns a report."""
    box = [WorkLedger()]
    g = build(cls, *args, ledger_box=box, label="selftest", **kw)
    import numpy as np
    d = np.zeros((g.hi - g.lo, g.n_anc), dtype=np.uint8)
    before = RAW.calls
    g.decode(d)
    ok_normal = (box[0].n_decodes == 1)
    g.m.decode(d.ravel())
    cls.decode(g, d)
    caught = False
    try:
        audit(box[0].n_decodes, "self test")
    except AssertionError:
        caught = True
    # the acknowledged hole: a handle captured before instrumentation
    box2 = [WorkLedger()]
    g2 = cls(*args, **kw)
    raw = g2.m.decode
    instrument(g2, box2, "selftest2")
    n0 = RAW.calls
    raw(d.ravel())
    hole = (RAW.calls == n0 and box2[0].n_decodes == 0)
    # every alternate matcher entry point, which gate 6 showed all escaped
    alts = {}
    for nm in ("decode_batch", "decode_to_edges_array", "decode_to_matched_dets_array",
               "decode_to_matched_dets_dict"):
        f = getattr(g.m, nm, None)
        if f is None:
            continue
        n0, c0 = RAW.calls, box[0].n_decodes
        try:
            f(d.ravel()[None, :] if "batch" in nm else d.ravel())
        except Exception:
            alts[nm] = "raised"
            continue
        alts[nm] = "counted" if RAW.calls > n0 else "ESCAPED"
        del c0
    return dict(normal_path_charged=bool(ok_normal),
                bypasses_caught=bool(caught),
                pre_instrumentation_handle_escapes=bool(hole),
                alternate_matcher_entry_points=alts,
                raw_calls_delta=RAW.calls - before)


def audit(total_charged_decodes, note=""):
    """Hard error if the ledgers charged fewer decodes than the decoders actually performed.

    This is the check the first version of this module lacked. `RAW.calls` is incremented inside the
    replacement itself, so the only way for the two to disagree is a decoder that was never
    instrumented -- which is exactly the defect being guarded against.
    """
    if total_charged_decodes != RAW.calls:
        raise AssertionError(
            f"work accounting mismatch{(' (' + note + ')') if note else ''}: ledgers charged "
            f"{total_charged_decodes} decodes, decoders performed {RAW.calls}. "
            f"{RAW.calls - total_charged_decodes} decode(s) were never instrumented.")
    return True
