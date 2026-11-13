#!/usr/bin/env python3
# polymarket-nvidia-ceo-forecast-01 entry (Python 3.13, standard library only).
#
# Improved persistence forecaster. For each market we forecast the Yes price over the
# 240 minutes after the checkpoint with five quantiles (0.05, 0.25, 0.5, 0.75, 0.95),
# shaped as a random walk around the checkpoint price:
#
#   q_tau(h) = clamp(center(h) + z_tau * sigma * sqrt(h) * tail_tau, 0, 1), rounded to 6 decimals
#
# with these differences from the house baseline (which uses a fixed sigma = 0.0025 and no drift):
#
#   * sigma is estimated per market from its own last ~4 hours of 1-minute price returns,
#     as a robust blend of the standard deviation and the median-absolute-deviation scale
#     (so one jump does not blow the estimate up), then shrunk toward a prior of 0.002 and
#     floored at 0.0005. Active markets get wider bands than the baseline, quiet ones narrower.
#   * a small mean reversion toward the market's 24-hour average (weight 0.05, ramped in over
#     the horizon) nudges the center of every quantile.
#   * the outer tails (q5 / q95) are widened by ~1.2x: pinball punishes a tail miss at the far
#     level far more than it rewards a tight band, and these prices move in steps with jumps.
#
# Deterministic: no randomness, no clock, no network. Every run reads the same case and writes
# the same bytes.

import json
import math
import os
import statistics

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# The contest's fixed quantile levels and the standard normal quantiles at them.
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
# How much each level's spread is scaled: widen the two outer tails, keep the core at 1.
TAIL_SCALE = [1.2, 1.0, 1.0, 1.0, 1.2]

# Sigma estimation knobs (fixed in advance, not tuned per case).
SIGMA_PRIOR = 0.002      # prior spread in price units per sqrt-minute (the baseline's value).
SIGMA_PRIOR_W = 0.25     # weight on the prior vs. the data estimate.
SIGMA_ALPHA = 0.3        # weight on the std vs. the robust (MAD) scale in the data estimate.
SIGMA_FLOOR = 0.0005     # never forecast a narrower spread than this.
RETURN_WINDOW = 240      # minutes of recent returns to estimate sigma from.

# Mean-reversion knobs.
MR_TARGET = 1440         # reversion target: the market's own mean over the last 24h (minutes).
MR_WEIGHT = 0.05         # how strongly the center is pulled toward the mean (full weight at h=240).


def price6(x):
    """Clamp x to [0, 1] and round to 6 decimals (never a negative zero)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return round(x * 1e6) / 1e6


def sigma_of(prices):
    """Per-market spread of one-minute moves, in price units per sqrt-minute.

    Uses the last few hours of simple (price) returns: a robust blend of their
    standard deviation and their median-absolute-deviation scale, shrunk toward a
    prior of 0.002, and floored. Simple returns (not log) keep low-price markets sane.
    """
    n = len(prices)
    if n < 2:
        return SIGMA_PRIOR
    win = prices[max(0, n - RETURN_WINDOW - 1):]
    diffs = [win[i] - win[i - 1] for i in range(1, len(win))]
    if not diffs:
        return SIGMA_PRIOR
    abs_diffs = sorted(abs(d) for d in diffs)
    mad = abs_diffs[len(abs_diffs) // 2]
    robust = 1.4826 * mad  # MAD -> std scale for a normal, but only used as a robust floor-ish term
    std = statistics.pstdev(diffs)
    est = SIGMA_ALPHA * std + (1.0 - SIGMA_ALPHA) * robust
    sigma = SIGMA_PRIOR_W * SIGMA_PRIOR + (1.0 - SIGMA_PRIOR_W) * est
    return max(SIGMA_FLOOR, sigma)


def center_target(prices):
    """The longer-run mean the forecast is pulled toward (the market's last ~24h average)."""
    win = prices[-MR_TARGET:] if len(prices) > MR_TARGET else prices
    return statistics.fmean(win)


def forecast_market(m, horizon):
    prices = m["prices"]
    if not prices:
        raise ValueError("market %s has no prices" % m["id"])
    x = prices[-1]
    sigma = sigma_of(prices)
    mu = center_target(prices)
    rows = []
    for h in range(1, horizon + 1):
        # Slight mean reversion: pull the center partway toward mu as the horizon grows.
        w = MR_WEIGHT * min(1.0, h / horizon)
        center = x + (mu - x) * w
        spread = sigma * math.sqrt(h)
        rows.append([price6(center + z * spread * ts) for z, ts in zip(Z, TAIL_SCALE)])
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Every *.in case, in name order, so two runs write the same bytes.
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fn in inputs:
        case_id = fn[:-3]  # strip ".in"
        with open(os.path.join(INPUT_DIR, fn), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        horizon = case["horizon_minutes"]
        forecasts = [
            {"id": m["id"], "q": forecast_market(m, horizon)} for m in case["markets"]
        ]
        payload = json.dumps({"forecasts": forecasts}, separators=(",", ":"))
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")


if __name__ == "__main__":
    main()
