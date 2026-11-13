#!/usr/bin/env python3
"""Local scorer for polymarket-nvidia-ceo-forecast-01.

There is no network and no official answer file, so this scorer validates the
forecast's shape exactly as the rules specify, and then estimates the pinball
loss against *realized* future moves taken from the input history itself: the
answer window of a practice case (at+1 .. at+240) is present as the start of the
next case's 72 hours of prices. That lets us score every practice case except the
last against actual prices, and separately sanity-check the per-market sigma
estimates against realized volatility.

Usage:
    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 scoring.py
"""
import glob
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "out")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def epoch_min(iso):
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() // 60)


def load_cases():
    cases = []
    for f in sorted(glob.glob(os.path.join(INPUT_DIR, "*.in"))):
        with open(f, "r", encoding="utf-8") as fh:
            cases.append(json.load(fh))
    return cases


def load_outputs():
    outs = {}
    for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.out"))):
        cid = os.path.basename(f)[:-4]
        with open(f, "r", encoding="utf-8") as fh:
            outs[cid] = json.load(fh)
    return outs


def validate(case, out):
    """Return a list of problems for one case's output (empty means valid)."""
    problems = []
    if "forecasts" not in out or list(out.keys()) != ["forecasts"]:
        problems.append("top-level key must be exactly ['forecasts']")
        return problems
    forecasts = out["forecasts"]
    ids = {m["id"] for m in case["markets"]}
    got = {}
    if not isinstance(forecasts, list) or len(forecasts) != len(ids):
        problems.append("forecasts length %s != %d markets" % (type(forecasts).__name__, len(ids)))
        return problems
    for f in forecasts:
        if set(f.keys()) != {"id", "q"}:
            problems.append("row keys != {id, q}: %s" % f.keys())
            continue
        if f["id"] in got:
            problems.append("duplicate id %s" % f["id"])
        got[f["id"]] = f
        if f["id"] not in ids:
            problems.append("unknown id %s" % f["id"])
    if got.keys() != ids:
        problems.append("missing ids: %s" % (ids - got.keys()))
    horizon = case["horizon_minutes"]
    for f in forecasts:
        q = f.get("q")
        if not isinstance(q, list) or len(q) != horizon:
            problems.append("%s q length %s != %d" % (f["id"], type(q).__name__, horizon))
            continue
        for h, row in enumerate(q):
            if not isinstance(row, list) or len(row) != 5:
                problems.append("%s row %d not 5 numbers" % (f["id"], h))
                continue
            prev = -1.0
            for v in row:
                if not isinstance(v, (int, float)) or not math.isfinite(v) or not (0.0 <= v <= 1.0):
                    problems.append("%s row %d has bad value %r" % (f["id"], h, v))
                if v < prev:
                    problems.append("%s row %d not non-decreasing" % (f["id"], h))
                prev = v
    return problems


def pinball(y, q, tau):
    return (tau - 1.0) * (y - q) if y < q else tau * (y - q)


def score_against_next(cases, outs):
    """Score practice cases 0..n-2 against the prices in the following case."""
    total = 0.0
    n_items = 0
    med_err = 0.0
    n_min = 0
    cov = 0
    cov_n = 0
    for i in range(len(cases) - 1):
        c = cases[i]
        cn = cases[i + 1]
        nxt = {m["id"]: m for m in cn["markets"]}
        out = outs.get(c["case"])
        if out is None:
            print("  missing output for %s" % c["case"])
            continue
        om = {f["id"]: f for f in out["forecasts"]}
        at_min = epoch_min(c["at"])
        horizon = c["horizon_minutes"]
        for m in c["markets"]:
            mid = m["id"]
            if mid not in nxt:
                continue
            s1 = epoch_min(nxt[mid]["start"])
            prices = nxt[mid]["prices"]
            rows = om[mid]["q"]
            for h in range(1, horizon + 1):
                idx = (at_min + h) - s1
                if not (0 <= idx < len(prices)):
                    continue
                y = prices[idx]
                row = rows[h - 1]
                for j, tau in enumerate(QUANTILES):
                    total += pinball(y, row[j], tau)
                    n_items += 1
                med_err += abs(y - row[2])
                n_min += 1
                if row[0] <= y <= row[4]:
                    cov += 1
                cov_n += 1
    return total / n_items * 100, med_err / n_min * 100, cov / cov_n


def sigma_check(cases):
    """Compare each case's realized 240-min-ahead move to the baseline's fixed sigma."""
    print("Sigma sanity check (realized |move over 240 min| vs. baseline 0.0025*sqrt(240)=%.4f):"
          % (0.0025 * math.sqrt(240)))
    for i in range(len(cases) - 1):
        c = cases[i]
        cn = cases[i + 1]
        nxt = {m["id"]: m for m in cn["markets"]}
        at_min = epoch_min(c["at"])
        moves = []
        for m in c["markets"]:
            mid = m["id"]
            if mid not in nxt:
                continue
            s1 = epoch_min(nxt[mid]["start"])
            idx = (at_min + 240) - s1
            if 0 <= idx < len(nxt[mid]["prices"]):
                moves.append(abs(nxt[mid]["prices"][idx] - m["prices"][-1]))
        if moves:
            print("  %s: median |240-min move| = %.5f, max = %.5f"
                  % (c["case"], statistics.median(moves), max(moves)))


def main():
    cases = load_cases()
    outs = load_outputs()
    print("Cases: %d, outputs: %d" % (len(cases), len(outs)))
    all_valid = True
    for c in cases:
        if c["case"] not in outs:
            print("  MISSING output for %s" % c["case"])
            all_valid = False
            continue
        problems = validate(c, outs[c["case"]])
        if problems:
            all_valid = False
            print("  %s INVALID: %s" % (c["case"], "; ".join(problems[:5])))
        else:
            print("  %s valid (%d markets)" % (c["case"], len(c["markets"])))
    print("All valid: %s" % all_valid)
    if len(cases) > 1:
        loss, med, cov = score_against_next(cases, outs)
        print("Estimated pinball loss (practice, vs realized next-case prices): %.6f" % loss)
        print("Median error: %.6f  90%% coverage: %.6f" % (med, cov))
    sigma_check(cases)


if __name__ == "__main__":
    sys.exit(main())
