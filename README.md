# polymarket-nvidia-ceo-forecast-01 — entry

Forecasts the Yes price of every trading market of Polymarket's "Next CEO of
NVIDIA?" event for the 240 minutes after a checkpoint: five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95) per market per minute, scored by mean pinball
loss against the recorded prices.

Approach: per market, the empirical distribution of historical price changes
at multiple horizons (volmodel.py) is blended with a robust normal random-walk
model, shaped (median shrink/reversion, mid/tail spread scaling) and kept
monotone and inside [0, 1] (quantshape.py). main.py is the I/O shell the
grader runs. Python 3.13, standard library only, deterministic, offline.
