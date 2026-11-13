"""Quantile forecast shaping for Polymarket-style Yes prices.

Approach: ``volmodel.market_stats`` gives, per market, the last
price ``x``, anchor horizons with empirical quantiles of the price
changes over each anchor, a robust 1-minute volatility ``sig1``,
and trailing means.  For each horizon minute ``h`` this module:

1. interpolates the empirical change quantiles linearly in
   ``sqrt(h)`` between bracketing anchors (nearest anchor outside
   the range) -- square-root time matches diffusion scaling;
2. builds Gaussian change quantiles from one effective sigma,
   blending the sigma implied by the largest anchor's 5%-95%
   spread with ``sig1``, clamped to ``[sig_floor, sig_cap]``;
3. averages empirical and Gaussian quantiles, shrinks the median,
   optionally pulls it toward the 24h mean, and caps its drift;
4. rescales the inner (25%/75%) and outer (5%/95%) deviations
   from the median separately, adds ``x``, then forces each row
   non-decreasing, clamps it to [0, 1] and rounds to 6 decimals.

``forecast_case`` does this for all markets in a case and, when
``sum_shrink > 0``, shifts every median at a minute by a common
amount toward restoring the sum of the current prices (useful
for near-exhaustive market sets), keeping each shifted median
inside its own row.

In the calibrated configuration ``forecast_case`` replaces each
market's own empirical quantiles with the case's pooled change
distribution: every market's recent changes, divided by the level
factor ``sqrt(max(p*(1-p), norm_floor))`` of the price they started
from, are pooled across markets (``volmodel.pooled_change_quantiles``)
and scaled back by each market's own level factor.  Each market's
spread is further multiplied by its relative activity -- how much it
has been moving lately versus the case average -- clipped and raised
to ``act_gamma``.  With ``w_emp`` at 1.0 the Gaussian component is
off and the pooled empirical distribution is the whole model.

Parameters (``DEFAULT_PARAMS``):
  w_emp -- weight on empirical quantiles (rest: Gaussian model).
  sig_blend -- weight on anchor-implied sigma (rest: ``sig1``).
  sig_floor / sig_cap -- bounds for the effective per-minute sigma.
  mid_scale -- multiplier on 25%/75% deviations from the median.
  tail_scale -- multiplier on 5%/95% deviations from the median.
  dn_tail_scale -- optional multiplier on the 5% (outer, below the
    median) deviation; None (default) falls back to ``tail_scale``.
  dn_mid_scale -- optional multiplier on the 25% (inner, below the
    median) deviation; None (default) falls back to ``mid_scale``.
  up_mid_scale -- optional multiplier on the 75% (inner, above the
    median) deviation; None (default) falls back to ``mid_scale``.
  up_tail_scale -- optional multiplier on the 95% (outer, above the
    median) deviation; None (default) falls back to ``tail_scale``.
  med_shrink -- multiplier on the blended median change.
  revert -- strength of median pull toward ``mean24`` (0 = off).
  revert_tau -- minutes; pull grows as ``1 - exp(-h/revert_tau)``.
  drift_cap -- absolute cap on the shaped median change.
  sum_shrink -- cross-market median-sum adjustment (0 = off).
  sum_target_inf -- long-run target for the sum of last prices
    across markets; when > 0 (and there are at least 2 markets),
    every horizon minute's rows are whole-row shifted so their
    medians sum to a target decaying from the current sum toward
    this value.  0.0 (default) = feature off.
  sum_tau -- minutes; decay timescale for that target
    (``volmodel.decayed_sum_target``).
  sum_weight_floor -- floor on each market's last price when
    computing the share weights that split the median-sum gap
    across markets (``volmodel.share_weights``).

Pure and deterministic: no randomness, no I/O, stdlib only.
"""

import bisect
import math

import volmodel

# Standard normal quantiles at the five levels, as in volmodel.TAUS.
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

DEFAULT_PARAMS = {
    # v3 calibration: a 10% Gaussian component whose sigma is the
    # market's own MAD-based sig1 alone (sig_blend 0.0) repairs the
    # pooled empirical shape at short horizons, where its inner
    # quantiles collapse onto the median; revert_tau 360 and a 0.8
    # deviation scale for sub-0.02-priced markets (pb_s1) are the
    # other round-3 changes.  Local scores vs v2 (0.258393 /
    # 0.340938): official 0.256001, extra 0.341345.
    "w_emp": 0.9, "sig_blend": 0.0, "sig_floor": 0.0004,
    "sig_cap": 0.02, "mid_scale": 0.62, "tail_scale": 0.62,
    "dn_tail_scale": 0.56, "dn_mid_scale": 1.15,
    "up_mid_scale": 0.78, "up_tail_scale": 0.47,
    "med_shrink": 1.0, "revert": 0.05, "revert_tau": 360.0,
    "drift_cap": 0.05, "sum_shrink": 0.0,
    # Sum-reversion (forecast_case): whole-row shifts so the median
    # sum decays from the current price sum toward sum_target_inf.
    "sum_target_inf": 0.98, "sum_tau": 240.0, "sum_weight_floor": 0.002,
    # Pooled cross-market empirical distribution (forecast_case):
    # changes of every market in the case, level-normalised and pooled,
    # replace each market's own change quantiles as the empirical part.
    "pool_window": 1080, "norm_floor": 0.004,
    # Per-market activity multiplier on the spread around the median:
    # g = clip(sqrt(activity / mean activity), act_clip_lo, act_clip_hi)
    #     ** act_gamma, activity measured over the last act_window prices.
    "act_window": 361, "act_gamma": 1.0,
    "act_clip_lo": 0.4, "act_clip_hi": 2.0,
    # Horizon-dependent deviation scale: every deviation from the
    # median is multiplied by clip((h / h_scale_ref) ** h_scale_pow,
    # h_scale_lo, h_scale_hi).  pow 0.0 (default) = exactly 1.0 = off.
    "h_scale_pow": 0.0, "h_scale_ref": 60.0,
    "h_scale_lo": 0.6, "h_scale_hi": 2.0,
    # Price-bucket deviation scale: multiplier chosen by the market's
    # last price x: x < pb_t1 -> pb_s1; pb_t1 <= x < pb_t2 -> pb_s2;
    # x >= pb_t2 -> pb_s3.  All scales 1.0 (default) = off.
    "pb_t1": 0.02, "pb_t2": 0.08,
    "pb_s1": 0.8, "pb_s2": 1.0, "pb_s3": 1.0,
}

_MED = 2  # median position in a five-quantile row


def _merged(params):
    out = dict(DEFAULT_PARAMS)
    if params:
        out.update(params)
    return out


def _z_for_taus(taus):
    known = dict(zip(volmodel.TAUS, Z))
    return [known.get(tau, Z[i] if i < len(Z) else 0.0)
            for i, tau in enumerate(taus)]


def _empirical_at(sqrt_h, sqrt_anchors, dq):
    """Empirical change quantiles at ``sqrt_h`` (nearest outside)."""
    if not dq:
        return [0.0] * 5
    if len(dq) == 1 or sqrt_h <= sqrt_anchors[0]:
        return list(dq[0])
    if sqrt_h >= sqrt_anchors[-1]:
        return list(dq[-1])
    j = bisect.bisect_right(sqrt_anchors, sqrt_h)  # bracket is j-1, j
    lo, hi = dq[j - 1], dq[j]
    span = sqrt_anchors[j] - sqrt_anchors[j - 1]
    frac = 0.0 if span <= 0 else (sqrt_h - sqrt_anchors[j - 1]) / span
    return [lo[k] + frac * (hi[k] - lo[k]) for k in range(len(lo))]


def _anchor_sigma(anchors, dq, sig1):
    """Per-sqrt-minute sigma implied by the largest anchor."""
    if not anchors or not dq or not anchors[-1] or anchors[-1] <= 0:
        return sig1
    last = dq[-1]
    if len(last) < 5 or last[4] - last[0] <= 0 or Z[4] - Z[0] <= 0:
        return sig1
    return (last[4] - last[0]) / (Z[4] - Z[0]) / math.sqrt(anchors[-1])


def _clean(value):
    rounded = round(value, 6)
    return 0.0 if rounded == 0 else rounded  # also normalises -0.0


def rows_for_market(prices, horizon=240,
                    taus=(0.05, 0.25, 0.5, 0.75, 0.95), params=None,
                    shared=None):
    """Return ``horizon`` quantile rows for one market's prices.

    ``shared`` optionally supplies the case-level pooled empirical
    distribution as (sqrt_anchors, dq, level_f, act_g): its change
    quantiles (in level-normalised units) replace the market's own as
    the empirical part, scaled back by ``level_f`` for this market's
    price level, and ``act_g`` additionally multiplies every deviation
    from the median (the market's relative activity).
    """
    p = _merged(params)
    taus = tuple(taus)
    n_tau = len(taus)
    if horizon <= 0:
        return []
    if not prices:
        return [[0.0] * n_tau for _ in range(horizon)]

    stats = volmodel.market_stats(prices)
    x = float(stats["x"])
    keep = min(len(stats.get("anchors") or []), len(stats.get("dq") or []))
    anchors = list(stats["anchors"][:keep])
    dq = [list(e) for e in stats["dq"][:keep]]
    sqrt_anchors = [math.sqrt(a) if a > 0 else 0.0 for a in anchors]
    sig1 = float(stats.get("sig1") or 0.0)
    mean24 = float(stats.get("mean24", x))

    # Effective sigma: blend anchor-implied with sig1, then clamp.
    blend = float(p["sig_blend"])
    sigma = blend * _anchor_sigma(anchors, dq, sig1) + (1 - blend) * sig1
    lo_sig, hi_sig = float(p["sig_floor"]), float(p["sig_cap"])
    if lo_sig > hi_sig:
        lo_sig, hi_sig = hi_sig, lo_sig
    sigma = max(lo_sig, min(hi_sig, sigma))

    z_vals = _z_for_taus(taus)
    w_emp, w_norm = float(p["w_emp"]), 1.0 - float(p["w_emp"])
    med_idx = min(range(n_tau), key=lambda i: abs(taus[i] - 0.5))
    revert, revert_tau = float(p["revert"]), float(p["revert_tau"])
    gap24, cap = mean24 - x, abs(float(p["drift_cap"]))

    # Price-bucket deviation multiplier for this market (1.0 = off).
    if x < float(p["pb_t1"]):
        pb_mult = float(p["pb_s1"])
    elif x < float(p["pb_t2"]):
        pb_mult = float(p["pb_s2"])
    else:
        pb_mult = float(p["pb_s3"])
    hs_pow = float(p["h_scale_pow"])
    hs_ref = float(p["h_scale_ref"]) or 60.0
    hs_lo, hs_hi = float(p["h_scale_lo"]), float(p["h_scale_hi"])

    rows = []
    for h in range(1, horizon + 1):
        sqrt_h = math.sqrt(h)
        if hs_pow != 0.0:
            hs_mult = min(hs_hi, max(hs_lo, (h / hs_ref) ** hs_pow))
        else:
            hs_mult = 1.0
        if shared is not None:
            s_sqrt_anchors, s_dq, s_f, act_mult = shared
            emp = [v * s_f for v in _empirical_at(sqrt_h, s_sqrt_anchors, s_dq)]
        else:
            act_mult = 1.0
            emp = _empirical_at(sqrt_h, sqrt_anchors, dq)
        if len(emp) < n_tau:  # pad/truncate to the requested width
            emp += [emp[-1] if emp else 0.0] * (n_tau - len(emp))
        emp = emp[:n_tau]
        changes = [w_emp * emp[k] + w_norm * z_vals[k] * sigma * sqrt_h
                   for k in range(n_tau)]

        # Median: shrink, optional reversion pull, then drift cap.
        pull = 1.0 - math.exp(-h / revert_tau) if revert_tau > 1e-12 else 1.0
        med = float(p["med_shrink"]) * changes[med_idx] + gap24 * revert * pull
        med = max(-cap, min(cap, med))
        changes[med_idx] = med

        # Rescale deviations from the median: near pair vs tails,
        # times the market's relative-activity multiplier.  A
        # per-side scale, when set (not None), overrides the shared
        # mid/tail scale for its side and distance from the median.
        for k in range(n_tau):
            if k != med_idx:
                inner = abs(k - med_idx) == 1
                scale = p["mid_scale"] if inner else p["tail_scale"]
                if k < med_idx:
                    side = p.get("dn_mid_scale") if inner \
                        else p.get("dn_tail_scale")
                else:
                    side = p.get("up_mid_scale") if inner \
                        else p.get("up_tail_scale")
                if side is not None:
                    scale = side
                changes[k] = med + float(scale) * act_mult * hs_mult \
                    * pb_mult * (changes[k] - med)

        row, running = [], None
        for k in range(n_tau):  # add x, monotonise, clamp, round
            val = x + changes[k]
            running = val if running is None else max(running, val)
            row.append(_clean(max(0.0, min(1.0, running))))
        rows.append(row)
    return rows


def forecast_case(case, params=None):
    """Map each market id in ``case`` to its forecast rows."""
    p = _merged(params)
    case = case or {}
    markets = list(case.get("markets") or [])
    horizon = int(case.get("horizon_minutes") or 240)
    taus = tuple(case.get("quantiles") or volmodel.TAUS)

    # Case-level pooled empirical distribution + activity multipliers.
    shared_by_id = {}
    floor = float(p.get("norm_floor") or 0.004)
    pool_window = int(p.get("pool_window") or 0)
    if pool_window > 0 and markets:
        anchors_p, dq_p = volmodel.pooled_change_quantiles(
            [list(m.get("prices") or []) for m in markets], pool_window, floor)
        sqrt_p = [math.sqrt(a) if a > 0 else 0.0 for a in anchors_p]
        act_win = int(p.get("act_window") or 361)
        acts = [volmodel.activity(list(m.get("prices") or []), act_win, floor)
                for m in markets]
        pool_act = sum(acts) / len(acts) if acts else 0.0
        gamma = float(p.get("act_gamma") or 0.0)
        clip_lo = float(p.get("act_clip_lo") or 0.4)
        clip_hi = float(p.get("act_clip_hi") or 2.0)
        for m, act in zip(markets, acts):
            prices_m = m.get("prices") or []
            x_m = float(prices_m[-1]) if prices_m else 0.0
            if pool_act > 0 and act > 0:
                g = min(clip_hi, max(clip_lo, math.sqrt(act / pool_act))) ** gamma
            elif pool_act > 0:
                g = clip_lo ** gamma
            else:
                g = 1.0
            shared_by_id[m["id"]] = (sqrt_p, dq_p,
                                     volmodel.level_factor(x_m, floor), g)

    result = {m["id"]: rows_for_market(m.get("prices") or [], horizon=horizon,
                                       taus=taus, params=p,
                                       shared=shared_by_id.get(m["id"]))
              for m in markets}

    shrink = float(p.get("sum_shrink") or 0.0)
    if shrink > 0 and markets and horizon > 0:
        ids = [m["id"] for m in markets]
        s0 = sum(float(m["prices"][-1]) if m.get("prices") else 0.0
                 for m in markets)
        med_idx = _MED if len(taus) > _MED else len(taus) // 2
        for h in range(horizon):
            med_sum = sum(result[i][h][med_idx] for i in ids)
            shift = (s0 - med_sum) * shrink / len(ids)
            if shift == 0.0:
                continue
            for i in ids:
                row = result[i][h]
                new_med = row[med_idx] + shift
                if med_idx > 0:  # stay inside this row's neighbours
                    new_med = max(new_med, row[med_idx - 1])
                if med_idx + 1 < len(row):
                    new_med = min(new_med, row[med_idx + 1])
                row[med_idx] = _clean(max(0.0, min(1.0, new_med)))

    # Sum-reversion: shift whole rows so the sum of medians at each
    # horizon minute decays from the current price sum toward
    # sum_target_inf, split across markets by last-price shares.
    sum_target_inf = float(p.get("sum_target_inf") or 0.0)
    if sum_target_inf > 0 and len(markets) >= 2 and horizon > 0:
        ids = [m["id"] for m in markets]
        last_prices = [float(m["prices"][-1]) if m.get("prices") else 0.0
                       for m in markets]
        s0 = sum(last_prices)
        sum_tau = float(p.get("sum_tau") or 240.0)
        weight_floor = float(p.get("sum_weight_floor") or 0.002)
        weights = volmodel.share_weights(last_prices, weight_floor)
        if weights:
            if 0.5 in taus:
                med_idx_sr = taus.index(0.5)
            else:
                med_idx_sr = min(range(len(taus)),
                                 key=lambda i: abs(taus[i] - 0.5))
            for idx in range(horizon):
                h = idx + 1  # rows are 0-indexed; minutes are 1-based
                target = volmodel.decayed_sum_target(
                    s0, sum_target_inf, sum_tau, h)
                present = []
                med_sum = 0.0
                for pos, i in enumerate(ids):
                    rows_i = result.get(i)
                    if not rows_i or idx >= len(rows_i):
                        continue
                    row = rows_i[idx]
                    if not row or med_idx_sr >= len(row):
                        continue
                    med_sum += row[med_idx_sr]
                    present.append(pos)
                if not present:
                    continue
                gap = target - med_sum
                if gap == 0.0:
                    continue
                for pos in present:
                    if pos >= len(weights):
                        continue
                    shift_i = gap * weights[pos]
                    row = result[ids[pos]][idx]
                    new_row, running = [], None
                    for val in row:  # shift, monotonise, clamp, round
                        val = val + shift_i
                        running = val if running is None \
                            else max(running, val)
                        new_row.append(_clean(max(0.0, min(1.0, running))))
                    result[ids[pos]][idx] = new_row
    return result
