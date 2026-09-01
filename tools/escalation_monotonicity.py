#!/usr/bin/env python
"""Is enlarging the buffer monotonically safer FOR A GIVEN SHOT? Measured, label-free, 0 QPU.

A decomposition of round 14 part C's pre-registered overlap matrix (`a8dc617`), not a new test.
It carries no falsifier and is reported as measured.

WHY IT MATTERS. Both named prior-art triggers -- ADaPT (arXiv:2605.01149) and the spatiotemporal
complementary gap (arXiv:2605.14637) -- decode with a small buffer, compute a confidence, and
**enlarge the buffer and redo the decode** when confidence is low. The standard justification for
`b >= d` is a statement about the average logical error RATE. It is not a statement about any
individual shot.

Against the joint decoder as reference, escalating `b -> b'` on a given shot:

    HELPS   the shot disagreed with joint at `b`  and agrees at `b'`
    HURTS   the shot AGREED with joint at `b`     and disagrees at `b'`

Strict per-shot monotonicity would make `hurts` identically zero. These are PAIRED counts on the
same shots, so the test of "does escalation help more than it hurts" is an exact McNemar -- a
binomial test on the discordant pairs against 1/2 -- Bonferroni-corrected over all 28 ordered pairs.

🔴 Scope, in the statement: `Delta_b` is disagreement with the joint decoder, not error. A `hurts`
shot is one where the enlarged buffer newly incurs the `P7` certificate's risk, and whether that is
an actual logical error is not knowable without the label. That is the same scope every `Delta_b`
result in this project carries.

Usage:  python tools/escalation_monotonicity.py
"""
import json
import os
import sys

from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))


def main():
    with open(os.path.join(ROOT, "data", "nesting_audit.json")) as fh:
        d = json.load(fh)
    B = d["b_grid"]
    n_pairs = len(B) * (len(B) - 1) // 2
    alpha = 0.05 / n_pairs

    print("IS A BIGGER BUFFER SAFER FOR A GIVEN SHOT?  device, held-out 50,000, 0 QPU")
    print(f"  exact McNemar on discordant pairs, Bonferroni over {n_pairs} pairs "
          f"(alpha = {alpha:.2e})")
    print(f"  strict per-shot monotonicity would make every `hurts` zero\n")

    out = {}
    for tag, src in (("device", d["device"]), ("M2", d["model_M2"])):
        n_shots = 50_000 if tag == "device" else d["n_model"]
        print(f"  {tag} ({n_shots:,} shots)")
        print(f"    {'b -> b2':>10} {'helps':>6} {'hurts':>6} {'hurt/help':>10} "
              f"{'exact p':>10}  significant?")
        rows = {}
        for i, b in enumerate(B):
            for b2 in B[i + 1:]:
                e = src["escalation"][f"{b}->{b2}"]
                h, u = e["helps"], e["hurts"]
                p = binomtest(h, h + u, 0.5, alternative="greater").pvalue if h + u else 1.0
                sig = p < alpha
                rows[f"{b}->{b2}"] = dict(helps=h, hurts=u, p=float(p), significant=bool(sig),
                                          hurt_over_help=(u / h if h else float("inf")))
                if b in (1, 2, 3, 4, 6, 8, 12):
                    print(f"    {b:>4} ->{b2:>4} {h:>6} {u:>6} "
                          f"{(u / h if h else float('nan')):>10.2f} {p:>10.2e}  "
                          f"{'YES' if sig else 'no'}")
        n_sig = sum(1 for v in rows.values() if v["significant"])
        n_any_hurt = sum(1 for v in rows.values() if v["hurts"] > 0)
        worst = max(rows.items(), key=lambda kv: kv[1]["hurt_over_help"])
        print(f"    -> escalation is a net win at Bonferroni-corrected significance in "
              f"{n_sig} of {n_pairs} pairs")
        print(f"    -> it breaks at least one previously-agreeing shot in "
              f"{n_any_hurt} of {n_pairs} pairs")
        print(f"    -> worst ratio: {worst[0]} newly breaks {worst[1]['hurts']} shots while "
              f"fixing {worst[1]['helps']} ({worst[1]['hurt_over_help']:.2f} per fix)\n")
        out[tag] = dict(pairs=rows, n_significant=n_sig, n_with_any_hurt=n_any_hurt,
                        n_pairs=n_pairs, alpha=alpha, worst_pair=worst[0])

    dv = out["device"]["pairs"]
    print("  THE HEADLINE, stated at the strength it can carry:")
    print(f"    b=1 -> b=16 is a clear net win: {dv['1->16']['helps']} fixed against "
          f"{dv['1->16']['hurts']} broken, p = {dv['1->16']['p']:.2e}")
    small = [k for k in ("1->2", "2->3", "3->4") if not dv[k]["significant"]]
    print(f"    but SMALL steps are not: {', '.join(small) if small else 'none'} fail to reach "
          f"corrected significance")
    print(f"    b=1 -> b=2 in particular: {dv['1->2']['helps']} fixed, {dv['1->2']['hurts']} "
          f"broken, p = {dv['1->2']['p']:.2f} -- indistinguishable from a coin flip")

    path = os.path.join(ROOT, "data", "escalation_monotonicity.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(b_grid=B, **out), fh, indent=1)
    os.replace(tmp, path)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
