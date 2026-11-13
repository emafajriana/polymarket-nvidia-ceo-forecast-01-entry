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
decayed_sum_target(current_sum, target_inf, tau, h)
    Target for the sum of all markets' Yes prices at horizon minute
    ``h``, assuming that sum decays exponentially from ``current_sum``
    toward ``target_inf`` with time constant ``tau`` (see the function
    docstring for the formula and edge cases).
share_weights(last_prices, floor)
    One weight per price in ``last_prices``, proportional to
    ``max(price, floor)`` and normalised to sum to 1.0 (see the
    function docstring for the empty and degenerate-total fallbacks).

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


def decayed_sum_target(current_sum, target_inf, tau, h):
    """Exponentially decayed target for the sum of Yes prices.

    Cross-market Yes prices need not sum to exactly 1 -- the sum drifts
    above or below it -- and a forecaster that expects that sum to
    revert uses this helper for the level it should have reached by
    horizon minute ``h``:

        target_inf + (current_sum - target_inf) * exp(-h / tau)

    ``current_sum`` is the sum now, ``target_inf`` is the long-run
    level the sum decays toward, ``tau`` is the decay time constant in
    minutes, and ``h`` is a non-negative horizon in minutes.  At
    ``h == 0`` the result is ``current_sum``; as ``h`` grows it
    approaches ``target_inf``, reaching about 63% of the way there at
    ``h == tau``.

    If ``tau`` is not positive, is NaN, or cannot be compared as a
    number at all, there is no meaningful decay rate, so the decay is
    treated as immediate and ``target_inf`` is returned.  (An infinite
    ``tau`` is usable: it means no decay, and the formula itself then
    returns ``current_sum``.)  A negative ``h`` is clamped to 0, i.e.
    treated as "no time has passed", rather than extrapolating the
    sum away from ``target_inf``.  The return value is always a float.
    The function is pure float math: no I/O and no state.
    """
    try:
        unusable = tau <= 0 or math.isnan(tau)
    except (TypeError, ValueError):
        unusable = True
    if unusable:
        return float(target_inf)
    horizon = h if h > 0 else 0.0
    return float(target_inf + (current_sum - target_inf)
                 * math.exp(-horizon / tau))


def share_weights(last_prices, floor):
    """Normalised weights proportional to ``max(price, floor)``.

    Given one last price per market, returns a list of weights, one
    per price and in the same order, proportional to
    ``max(price, floor)`` and normalised so they sum to 1.0.  A
    cross-market forecaster uses the weights to split a target sum
    (for example from decayed_sum_target) back across markets.

    ``floor`` is expected to be positive.  It keeps a market whose
    price is 0.0 -- or very small, or even negative -- in the split
    with a small positive share instead of dropping it to zero weight:
    every such price is treated as ``floor`` by the same ``max`` rule.

    An empty input returns an empty list.  If the total of the floored
    prices is not positive -- which can happen only when ``floor``
    itself is not positive -- no proportional split exists, so the
    fallback is uniform weights of ``1 / len(last_prices)`` each.
    A floored value that is still negative (possible only with a
    non-positive ``floor`` and a negative price) is clamped to 0.0,
    so no weight is ever negative.  The input may be any sequence;
    it is materialised once so its length is known exactly.  The
    function is pure: no I/O and no state.
    """
    prices = list(last_prices)
    if not prices:
        return []
    # max(price, floor) is the share rule; the outer max with 0.0 is a
    # no-op for a positive floor and only bites in the degenerate
    # non-positive-floor case, where it keeps weights non-negative.
    floored = [max(max(price, floor), 0.0) for price in prices]
    total = sum(floored)
    if total <= 0:
        return [1.0 / len(prices)] * len(prices)
    return [value / total for value in floored]
