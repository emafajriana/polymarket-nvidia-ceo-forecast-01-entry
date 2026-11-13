# polymarket-nvidia-ceo-forecast-01 — entry

Forecasts the Yes price of every trading market of Polymarket's "Next CEO of
NVIDIA?" event for the 240 minutes after a checkpoint: five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95) per market per minute, scored by mean pinball
loss against the recorded prices.

Approach: the case's markets share one pooled empirical distribution of
price changes — each change normalised by the level factor
sqrt(max(p(1-p), floor)) of its starting price — which is scaled back per
market and multiplied by that market's relative recent activity
(volmodel.py). quantshape.py shapes the rows: per-side deviation scales
for the four non-median quantiles, a mild median pull toward the 24h
mean, and a cross-market sum-reversion step — because the markets are
near-exhaustive, the sum of Yes prices decays from its current level
toward a long-run target, and the gap is distributed across markets in
proportion to their last prices (helpers in volmodel.py). main.py is the
I/O shell the grader runs. Python 3.13, standard library only,
deterministic, offline.
