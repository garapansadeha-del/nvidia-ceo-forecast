# nvidia-ceo-forecast

Minute-by-minute quantile forecaster for Polymarket's "Next CEO of NVIDIA?" markets, entered in make.fun's "Forecast Polymarket's next NVIDIA CEO prices minute by minute, 4 hours ahead" bounty.

Per-market random-walk quantile forecast: robust per-market sigma (blend of std and MAD scale of the last 4h of 1-minute returns, shrunk to a 0.002 prior), slight mean reversion to the 24h mean, widened outer tails. Python 3.13 stdlib, deterministic, <1s per case. Practice pinball loss 0.28329 vs the house baseline's 0.363589.
