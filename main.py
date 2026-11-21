"""GTA VI trophy rarity forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Reads one case (schema "makefun/gta6-trophies-case/v1") per *.in file in the
input directory and writes, for each case, five quantiles (5/25/50/75/95) of
every trophy's Exophase earn rate at the case's target, as
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.

Method (standard library only, deterministic, no network, no clock reads):
  1. Split each trophy's rate into its two moving parts: earners and the
     tracked-player count, rate = 100 * earners / players. Earners are
     what grows as players progress; the player count is what dilutes
     the rate (launch night's first refresh divided every rate by ~6
     while the denominator jumped 638 -> 3,868 players). Forecasting the
     two separately is the structural model; a single line on the rate
     conflates them.
  2. Work in log space: log earners and log players are each fitted with
     a recency-weighted least-squares line against
     x = ln(1 + hours since the game's release, 2026-11-19T00:00:00Z),
     over the case's recent reads. The player trend is one series shared
     by every trophy, so it is estimated once and precisely; each
     trophy's earner slope is shrunk toward its tier's mean earner slope
     (platinum / gold / silver / bronze), because a single trophy's short
     history is noisy.
  3. The median rate at the target is 100 * earners(target) /
     players(target), with each extrapolated move damped by
     kappa = 1 / (1 + horizon_days / 6), because growth decelerates as
     the reachable cohort is exhausted and the launch surge passes. The
     median move is capped at 4 logit units from the last read.
  4. The spread is honest: a random-walk term (0.5 logit per sqrt day),
     the slopes' own standard errors projected to the target, and a
     refresh-shock term for Exophase's discrete page refreshes, which
     dominates short horizons (the practice cases straddle one refresh)
     and fades over long ones.
  5. Degenerate histories (a single read, every read identical, rates
     missing, no player counts) fall back to a direct logit-rate line fit
     (v1), and a trophy with no rate anywhere falls back to a tier
     prior. A case with no usable history at all still yields a valid
     forecast, so a run never produces an invalid case.

Quantiles use the standard normal's ppf (statistics.NormalDist, exact) at
the case's own levels, so any level list works. Output is rounded to 4
decimals, half up, non-decreasing per row, and byte-identical across runs.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

_NORMAL_DIST = NormalDist()

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

RELEASE_EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Fallback medians (percent) per tier when a trophy has no rate anywhere.
TIER_PRIOR = {"platinum": 0.01, "gold": 0.05, "silver": 0.5, "bronze": 2.0}

# Random-walk spread of a rate's logit move, per square-root day.
RW_SIGMA = 0.5
# Damping scale (days) for the extrapolated trend.
DAMP_DAYS = 6.0
# Prior SD of a trophy's logit slope within its tier (for shrinkage).
SLOPE_PRIOR_SD = 0.5
# Refresh-shock spread: Exophase refreshes its page in discrete jumps.
SHOCK_SHORT = 0.85
SHOCK_FLOOR = 0.15
SHOCK_DECAY_DAYS = 1.5
# Spread floor / cap, in logit units.
SPREAD_MIN = 0.25
SPREAD_MAX = 3.0
# Largest logit move the median may make from the last read.
MAX_MEDIAN_MOVE = 4.0


def clamp_rate(pct):
    """Clamp a percent into the range the logit can take."""
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, as JavaScript Math.round does it."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def norm_ppf(p):
    """Standard normal quantile, via the standard library (exact to ~1e-15)."""
    if not 0.0 < p < 1.0:
        raise ValueError("norm_ppf needs 0 < p < 1")
    return _NORMAL_DIST.inv_cdf(p)


def parse_time(value):
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def hours_since_release(moment):
    return (moment - RELEASE_EPOCH).total_seconds() / 3600.0


def log_time(moment):
    """Log-time axis: ln(1 + hours since release), as the house curve uses."""
    return math.log1p(max(0.0, hours_since_release(moment)))


def wls_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, se_b, resid_var, n)."""
    n = len(xs)
    sw = sum(ws)
    if n == 0 or sw <= 0:
        return None
    xm = sum(w * x for w, x in zip(ws, xs)) / sw
    ym = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xm) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        # No spread in x: flat line at the weighted mean.
        resid = sum(w * (y - ym) ** 2 for w, y in zip(ws, ys)) / sw
        return (ym, 0.0, 0.0, resid, n)
    sxy = sum(w * (x - xm) * (y - ym) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = ym - b * xm
    if n > 2:
        resid_var = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
        resid_var *= n / (n - 2.0)
    else:
        resid_var = 0.0
    se_b = math.sqrt(resid_var / sxx) if resid_var > 0 else 0.0
    return (a, b, se_b, resid_var, n)


def trophy_series(case, index):
    """(times, logits) for one trophy, skipping reads that lack its rate."""
    times = []
    values = []
    for read in case.get("history", []):
        rates = read.get("rates") or []
        if index >= len(rates) or rates[index] is None:
            continue
        try:
            moment = parse_time(read["at"])
        except (KeyError, TypeError, ValueError):
            continue
        times.append(moment)
        values.append(logit(float(rates[index])))
    return times, values


def players_series(case):
    """(times, log players) for the case, skipping reads without a count."""
    times = []
    values = []
    for read in case.get("history", []):
        players = read.get("players_tracked")
        if not players or float(players) <= 0:
            continue
        try:
            moment = parse_time(read["at"])
        except (KeyError, TypeError, ValueError):
            continue
        times.append(moment)
        values.append(math.log(float(players)))
    return times, values


def earners_series(case, index):
    """(times, log earners) for one trophy: earners = rate * players / 100.

    Earners are what actually grows as players progress; the tracked-player
    count is what dilutes the rate. Splitting the two is the structural
    model: a trophy's rate moves with its earners against the player base.
    Reads that lack the trophy's rate, or a player count, are skipped.
    """
    times = []
    values = []
    for read in case.get("history", []):
        rates = read.get("rates") or []
        players = read.get("players_tracked")
        if index >= len(rates) or rates[index] is None:
            continue
        if not players or float(players) <= 0:
            continue
        try:
            moment = parse_time(read["at"])
        except (KeyError, TypeError, ValueError):
            continue
        earners = max(0.02, float(rates[index]) * float(players) / 100.0)
        times.append(moment)
        values.append(math.log(earners))
    return times, values


def forecast_trophy(trophy, times, values, x_target, horizon_days,
                    tier_slope, tier_slope_weight):
    """Forecast one trophy's five quantiles. Returns (median_logit, spread)."""
    if not times:
        prior = TIER_PRIOR.get(str(trophy.get("tier", "bronze")).lower(), 2.0)
        return logit(prior), 1.5
    last_logit = values[-1]
    last_time = times[-1]
    # Recent window: last 24h of reads, or all of them when fewer.
    window_start = last_time.timestamp() - 24.0 * 3600.0
    recent = [(t, v) for t, v in zip(times, values)
              if t.timestamp() >= window_start]
    if len(recent) < 3:
        recent = list(zip(times, values))
    xs = [log_time(t) for t, _ in recent]
    ys = [v for _, v in recent]
    # Recency weights: half-life 12h, floored so old reads still count.
    ws = [max(0.2, math.exp(-(last_time - t).total_seconds() / (12.0 * 3600.0)))
          for t, _ in recent]
    fit = wls_line(xs, ys, ws)
    if fit is None:
        return last_logit, 1.0
    _a, slope, se_b, _resid, _n = fit
    # Shrink the slope toward the tier mean, precision-weighted.
    se = max(se_b, 0.05)
    w_own = 1.0 / (se * se)
    w_tier = tier_slope_weight / (SLOPE_PRIOR_SD ** 2)
    if w_tier > 0:
        slope = (slope * w_own + tier_slope * w_tier) / (w_own + w_tier)
    # Extrapolate to the target, damped toward persistence.
    x_last = log_time(last_time)
    kappa = 1.0 / (1.0 + horizon_days / DAMP_DAYS)
    move = slope * (x_target - x_last) * kappa
    move = max(-MAX_MEDIAN_MOVE, min(MAX_MEDIAN_MOVE, move))
    median = last_logit + move
    # Spread: random walk + slope uncertainty projected to the target,
    # plus the Exophase refresh shock (large on short horizons).
    slope_term = (se * abs(x_target - x_last)) ** 2
    shock = SHOCK_SHORT * math.exp(-horizon_days / SHOCK_DECAY_DAYS) + SHOCK_FLOOR
    spread = math.sqrt(RW_SIGMA ** 2 * horizon_days + slope_term + shock ** 2)
    spread = max(SPREAD_MIN, min(SPREAD_MAX, spread))
    return median, spread


def tier_slopes(case, horizon_days):
    """Precision-weighted mean logit slope per tier, for shrinkage.

    Returns {tier: (mean_slope, weight)}; weight is 0 when a tier has no
    usable trophy, in which case callers fall back to the global mean.
    """
    per_tier = {}
    global_slopes = []
    for i, trophy in enumerate(case.get("trophies", [])):
        times, values = trophy_series(case, i)
        if len(times) < 3:
            continue
        xs = [log_time(t) for t in times]
        ys = values
        ws = [1.0] * len(xs)
        fit = wls_line(xs, ys, ws)
        if fit is None:
            continue
        _a, slope, se_b, _resid, _n = fit
        se = max(se_b, 0.05)
        tier = str(trophy.get("tier", "bronze")).lower()
        per_tier.setdefault(tier, []).append((slope, 1.0 / (se * se)))
        global_slopes.append((slope, 1.0 / (se * se)))
    out = {}
    for tier, items in per_tier.items():
        wsum = sum(w for _, w in items)
        mean = sum(s * w for s, w in items) / wsum if wsum > 0 else 0.0
        out[tier] = (mean, min(1.0, wsum / (wsum + 4.0)))
    if global_slopes:
        wsum = sum(w for _, w in global_slopes)
        gmean = sum(s * w for s, w in global_slopes) / wsum if wsum > 0 else 0.0
        out.setdefault("__global__", (gmean, 0.5))
    return out


def earner_slopes(case):
    """Precision-weighted mean log-earner slope per tier, for shrinkage."""
    per_tier = {}
    global_slopes = []
    for i, trophy in enumerate(case.get("trophies", [])):
        times, values = earners_series(case, i)
        if len(times) < 3:
            continue
        xs = [log_time(t) for t in times]
        fit = wls_line(xs, values, [1.0] * len(xs))
        if fit is None:
            continue
        _a, slope, se_b, _resid, _n = fit
        se = max(se_b, 0.05)
        tier = str(trophy.get("tier", "bronze")).lower()
        per_tier.setdefault(tier, []).append((slope, 1.0 / (se * se)))
        global_slopes.append((slope, 1.0 / (se * se)))
    out = {}
    for tier, items in per_tier.items():
        wsum = sum(w for _, w in items)
        mean = sum(s * w for s, w in items) / wsum if wsum > 0 else 0.0
        out[tier] = (mean, min(1.0, wsum / (wsum + 4.0)))
    if global_slopes:
        wsum = sum(w for _, w in global_slopes)
        gmean = sum(s * w for s, w in global_slopes) / wsum if wsum > 0 else 0.0
        out.setdefault("__global__", (gmean, 0.5))
    return out


def fit_recent(times, values, x_target, horizon_days):
    """WLS log-time line over the recent window; damped extrapolation.

    Returns (median_at_target, slope_se_projected, last_value, x_last).
    The recent window is the last 24h of reads (or all of them when fewer);
    recency weights have a 12h half-life. The extrapolated move from the
    last value is damped by kappa = 1 / (1 + horizon_days / 6).
    """
    last_time = times[-1]
    last_value = values[-1]
    window_start = last_time.timestamp() - 24.0 * 3600.0
    recent = [(t, v) for t, v in zip(times, values)
              if t.timestamp() >= window_start]
    if len(recent) < 3:
        recent = list(zip(times, values))
    xs = [log_time(t) for t, _ in recent]
    ys = [v for _, v in recent]
    ws = [max(0.2, math.exp(-(last_time - t).total_seconds() / (12.0 * 3600.0)))
          for t, _ in recent]
    fit = wls_line(xs, ys, ws)
    if fit is None:
        return last_value, 0.0, last_value, log_time(last_time)
    _a, slope, se_b, _resid, _n = fit
    x_last = log_time(last_time)
    kappa = 1.0 / (1.0 + horizon_days / DAMP_DAYS)
    move = slope * (x_target - x_last) * kappa
    move = max(-MAX_MEDIAN_MOVE, min(MAX_MEDIAN_MOVE, move))
    se_proj = max(se_b, 0.0) * abs(x_target - x_last)
    return last_value + move, se_proj, last_value, x_last


def forecast_case_decomp(case):
    """Forecast via the earners/players decomposition (v2).

    rate(target) = 100 * earners(target) / players(target), with log
    earners and log players each extrapolated on the log-time axis. The
    player count is one series shared by every trophy, so its trend is
    estimated once and precisely; each trophy's earners carry its own
    trend, shrunk toward its tier's mean earner slope. When a trophy has
    no usable earner history, or the case has no player counts, that
    trophy falls back to the direct logit-rate fit.
    """
    levels = list(case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95])
    zscores = [norm_ppf(level) for level in levels]
    horizon_days = float(case.get("horizon_minutes", 0)) / 1440.0
    try:
        target = parse_time(case["target"])
    except (KeyError, TypeError, ValueError):
        target = parse_time(case["at"])
        horizon_days = float(case.get("horizon_minutes", 0)) / 1440.0
    x_target = log_time(target)
    kappa = 1.0 / (1.0 + horizon_days / DAMP_DAYS)
    shock = SHOCK_SHORT * math.exp(-horizon_days / SHOCK_DECAY_DAYS) + SHOCK_FLOOR
    # Shared player trend.
    p_times, p_values = players_series(case)
    if len(p_times) >= 2:
        logp_star, se_p, logp_last, _x = fit_recent(
            p_times, p_values, x_target, horizon_days)
    else:
        logp_star, se_p, logp_last = None, 0.0, None
    e_tiers = earner_slopes(case)
    global_mean, global_weight = e_tiers.get("__global__", (0.0, 0.0))
    tiers = tier_slopes(case, horizon_days)
    d_global_mean, d_global_weight = tiers.get("__global__", (0.0, 0.0))
    rows = []
    for i, trophy in enumerate(case.get("trophies", [])):
        times, values = earners_series(case, i)
        tier = str(trophy.get("tier", "bronze")).lower()
        median = None
        spread = None
        if len(times) >= 2 and logp_star is not None:
            tier_mean, tier_weight = e_tiers.get(tier, (global_mean, global_weight))
            # Shrink this trophy's earner slope toward its tier mean.
            xs = [log_time(t) for t in times]
            ys = values
            last_time = times[-1]
            window_start = last_time.timestamp() - 24.0 * 3600.0
            recent = [(t, v) for t, v in zip(times, values)
                      if t.timestamp() >= window_start]
            if len(recent) < 3:
                recent = list(zip(times, values))
            rxs = [log_time(t) for t, _ in recent]
            rys = [v for _, v in recent]
            rws = [max(0.2, math.exp(-(last_time - t).total_seconds() / (12.0 * 3600.0)))
                   for t, _ in recent]
            fit = wls_line(rxs, rys, rws)
            if fit is not None:
                _a, slope, se_b, _resid, _n = fit
                se = max(se_b, 0.05)
                w_own = 1.0 / (se * se)
                w_tier = tier_weight / (SLOPE_PRIOR_SD ** 2)
                if w_tier > 0:
                    slope = (slope * w_own + tier_mean * w_tier) / (w_own + w_tier)
                x_last = log_time(last_time)
                move = slope * (x_target - x_last) * kappa
                move = max(-MAX_MEDIAN_MOVE, min(MAX_MEDIAN_MOVE, move))
                loge_star = values[-1] + move
                se_e = max(se_b, 0.0) * abs(x_target - x_last)
                rate_star = 100.0 * math.exp(loge_star - logp_star)
                median = logit(rate_star)
                slope_term = se_e ** 2 + se_p ** 2
                spread = math.sqrt(RW_SIGMA ** 2 * horizon_days
                                   + slope_term + shock ** 2)
                spread = max(SPREAD_MIN, min(SPREAD_MAX, spread))
        if median is None:
            # No usable decomposition for this trophy: direct logit fit.
            t_times, t_values = trophy_series(case, i)
            d_tier_mean, d_tier_weight = tiers.get(
                tier, (d_global_mean, d_global_weight))
            median, spread = forecast_trophy(
                trophy, t_times, t_values, x_target, horizon_days,
                d_tier_mean, d_tier_weight)
        quantiles = []
        previous = None
        for z in zscores:
            value = round4(sigmoid(median + z * spread))
            value = min(99.99, max(0.01, value))
            if previous is not None and value < previous:
                value = previous
            quantiles.append(value)
            previous = value
        rows.append({"id": trophy["id"], "q": quantiles})
    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def forecast_case_direct(case):
    """Forecast via the direct logit-rate line fit (v1)."""
    levels = list(case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95])
    zscores = [norm_ppf(level) for level in levels]
    horizon_days = float(case.get("horizon_minutes", 0)) / 1440.0
    try:
        target = parse_time(case["target"])
    except (KeyError, TypeError, ValueError):
        target = parse_time(case["at"])
        horizon_days = float(case.get("horizon_minutes", 0)) / 1440.0
    x_target = log_time(target)
    tiers = tier_slopes(case, horizon_days)
    global_mean, global_weight = tiers.get("__global__", (0.0, 0.0))
    rows = []
    for i, trophy in enumerate(case.get("trophies", [])):
        times, values = trophy_series(case, i)
        tier = str(trophy.get("tier", "bronze")).lower()
        tier_mean, tier_weight = tiers.get(tier, (global_mean, global_weight))
        median, spread = forecast_trophy(
            trophy, times, values, x_target, horizon_days,
            tier_mean, tier_weight)
        quantiles = []
        previous = None
        for z in zscores:
            value = round4(sigmoid(median + z * spread))
            value = min(99.99, max(0.01, value))
            if previous is not None and value < previous:
                value = previous
            quantiles.append(value)
            previous = value
        rows.append({"id": trophy["id"], "q": quantiles})
    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def forecast_case(case):
    """Forecast every trophy of one case. Returns the output object.

    Primary model is the earners/players decomposition; the direct
    logit-rate fit is the fallback for any case the decomposition cannot
    handle, so a run never produces an invalid case.
    """
    try:
        return forecast_case_decomp(case)
    except Exception:
        return forecast_case_direct(case)


def fallback_case(case):
    """A valid forecast from tier priors only; used if a case is malformed."""
    levels = list(case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95])
    zscores = [norm_ppf(level) for level in levels]
    rows = []
    for trophy in case.get("trophies", []):
        prior = TIER_PRIOR.get(str(trophy.get("tier", "bronze")).lower(), 2.0)
        median = logit(prior)
        quantiles = []
        previous = None
        for z in zscores:
            value = round4(sigmoid(median + z * 1.5))
            value = min(99.99, max(0.01, value))
            if previous is not None and value < previous:
                value = previous
            quantiles.append(value)
            previous = value
        rows.append({"id": trophy["id"], "q": quantiles})
    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(INPUT_DIR.glob("*.in"))
    if not inputs:
        print("no case files in %s" % INPUT_DIR, file=sys.stderr)
    for path in inputs:
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception as exc:  # keep every case valid, never crash a run
            print("case %s failed (%s); using tier-prior fallback"
                  % (path.name, exc), file=sys.stderr)
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                forecast = fallback_case(case)
            except Exception:
                print("case %s unreadable; skipping" % path.name, file=sys.stderr)
                continue
        out_path = OUTPUT_DIR / ("%s.out" % path.stem)
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
