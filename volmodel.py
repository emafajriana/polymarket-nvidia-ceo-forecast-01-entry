"""Empirical change distributions and robust volatility for one market.

This module looks at a single Polymarket-style price history -- a list of
minute-by-minute Yes prices in [0, 1], oldest first -- and summarises how
far the price tends to move over a range of horizons.  A downstream
forecaster uses those summaries to build quantile forecasts for the next
240 minutes, so everything here is descriptive: no modelling assumptions,
just quantiles of observed changes plus a couple of location/volatility
statistics.

Public interface
----------------
TAUS : tuple of float
    Quantile levels reported for every horizon.
ANCHORS : list of int
    Candidate horizon lengths, in minutes, in increasing order.
quantile_of_sorted(sorted_vals, tau)
    Type-7 quantile of an already sorted sequence.
market_stats(prices) -> dict with keys:

    'x'       Last (most recent) price, i.e. prices[-1].
    'anchors' Horizons actually computed.  Normally the sublist of
              ANCHORS whose change sample is large enough (see below);
              a single fallback horizon is used for very short series.
    'dq'      List parallel to 'anchors'.  Entry i holds the five
              empirical quantiles (at TAUS) of the h-minute price
              changes prices[t] - prices[t - h] for h = anchors[i],
              taken over every valid t in the series.
    'sig1'    Robust per-minute volatility: the median absolute
              deviation of the 1-minute changes, rescaled by 1.4826 so
              it is comparable to a standard deviation under normality.
              0.0 when fewer than two 1-minute changes exist.
    'mean24'  Mean of the last 1440 prices (24 h of minutes), or of
              the whole series when it is shorter.
    'mean6'   Mean of the last 360 prices (6 h of minutes), or of the
              whole series when it is shorter.

An anchor h only qualifies when it yields at least 20 change samples,
i.e. len(prices) - h >= 20; tiny samples give quantiles that are more
noise than signal.  If no anchor qualifies, a single fallback horizon
h = max(1, (len(prices) - 1) // 2) is computed from however many samples
exist.  A one-element series is degenerate: anchors == [1] and the
change quantiles are all zero.

The module is pure: no I/O, no randomness, no mutable global state.
"""

import math

TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)

ANCHORS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 240]

# Minimum number of h-minute changes before an anchor is trusted.
_MIN_SAMPLES = 20

# MAD -> sigma consistency factor for a normal distribution.
_MAD_SCALE = 1.4826


def quantile_of_sorted(sorted_vals, tau):
    """Return the type-7 quantile of an already sorted sequence.

    Type 7 (the R / NumPy default) places the quantile at the real-valued
    rank ``tau * (n - 1)`` and linearly interpolates between the two
    neighbouring order statistics when that rank falls between them.

    ``sorted_vals`` must be non-empty and sorted ascending; the caller
    owns sorting so a list can be reused for several taus.  A single
    element is its own quantile at every tau.
    """
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])

    # Real-valued position in [0, n - 1]; clamp guards against a tau
    # marginally outside [0, 1] from floating point callers.
    pos = tau * (n - 1)
    if pos <= 0.0:
        return float(sorted_vals[0])
    if pos >= n - 1:
        return float(sorted_vals[-1])

    lo = int(math.floor(pos))
    frac = pos - lo
    low = float(sorted_vals[lo])
    high = float(sorted_vals[lo + 1])
    return low + frac * (high - low)


def _median(sorted_vals):
    """Median of an already sorted, non-empty sequence."""
    return quantile_of_sorted(sorted_vals, 0.5)


def _mean_tail(prices, count):
    """Mean of the last ``count`` prices (fewer if the series is short)."""
    tail = prices[-count:] if len(prices) > count else prices
    return sum(tail) / len(tail)


def _change_quantiles(prices, h):
    """Quantiles at TAUS of the h-minute changes, sorted only once.

    Builds the full list of changes prices[t] - prices[t - h], sorts that
    one list, then reads all five quantiles off it.
    """
    changes = [prices[t] - prices[t - h] for t in range(h, len(prices))]
    changes.sort()
    return [quantile_of_sorted(changes, tau) for tau in TAUS]


def _robust_sig1(prices):
    """MAD-based volatility of the consecutive (1-minute) differences."""
    diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    if len(diffs) < 2:
        return 0.0

    ordered = sorted(diffs)
    centre = _median(ordered)

    # Absolute deviations from the median, themselves median-ed.  Reusing
    # the sorted-quantile helper keeps the median definition consistent
    # with everything else in the module.
    deviations = sorted(abs(d - centre) for d in diffs)
    return _MAD_SCALE * _median(deviations)


def market_stats(prices):
    """Compute the change-distribution summary for one price series.

    ``prices`` is a non-empty sequence of floats, oldest first.  See the
    module docstring for the meaning of each returned key.
    """
    n = len(prices)

    # Degenerate history: no movement can be observed, so report a flat
    # change distribution at the single fallback anchor.
    if n == 1:
        only = float(prices[0])
        return {
            'x': only,
            'anchors': [1],
            'dq': [[0.0, 0.0, 0.0, 0.0, 0.0]],
            'sig1': 0.0,
            'mean24': only,
            'mean6': only,
        }

    # Keep only anchors with enough change samples to be meaningful.
    anchors = [h for h in ANCHORS if n - h >= _MIN_SAMPLES]

    if not anchors:
        # Very short series: fall back to roughly half the available
        # span, which still leaves a handful of (overlapping) samples.
        anchors = [max(1, (n - 1) // 2)]

    dq = [_change_quantiles(prices, h) for h in anchors]

    return {
        'x': float(prices[-1]),
        'anchors': anchors,
        'dq': dq,
        'sig1': _robust_sig1(prices),
        'mean24': _mean_tail(prices, 1440),
        'mean6': _mean_tail(prices, 360),
    }


def level_factor(price, floor):
    """Scale of a price's typical move: sqrt(max(price*(1-price), floor)).

    A probability's moves are bounded by its distance to 0 and 1, so
    changes are compared across markets in units of this factor.
    """
    return math.sqrt(max(price * (1.0 - price), floor))


def pooled_change_quantiles(price_lists, window, floor):
    """Pooled, level-normalised change quantiles across several markets.

    For every anchor in ANCHORS, each series contributes its h-minute
    changes over its last ``window`` minutes, each change divided by the
    level factor of the price it started from.  The samples of all
    markets are combined and read at TAUS.  Returns (anchors, dq) in the
    same shape as market_stats: anchors is ANCHORS itself, and a dq row
    is all zeros when no market yields enough samples at that horizon.
    """
    dq = []
    for h in ANCHORS:
        samples = []
        for prices in price_lists:
            seg = prices[-window:] if len(prices) > window else prices
            if len(seg) - h > _MIN_SAMPLES:
                for t in range(h, len(seg)):
                    p0 = seg[t - h]
                    samples.append((seg[t] - p0) / level_factor(p0, floor))
        if samples:
            samples.sort()
            dq.append([quantile_of_sorted(samples, tau) for tau in TAUS])
        else:
            dq.append([0.0] * len(TAUS))
    return list(ANCHORS), dq


def activity(prices, window, floor):
    """Mean absolute level-normalised 1-minute change over the tail.

    Averages |prices[t] - prices[t-1]| / level_factor(prices[t-1], floor)
    over the last ``window`` prices (fewer when the series is short);
    0.0 when fewer than two prices exist.  One market's activity is
    comparable to another's because of the normalisation.
    """
    if len(prices) < 2:
        return 0.0
    tail = prices[-window:] if len(prices) > window else prices
    total = 0.0
    for i in range(1, len(tail)):
        total += abs(tail[i] - tail[i - 1]) / level_factor(tail[i - 1], floor)
    return total / (len(tail) - 1)
