"""GTA VI trophy rarity forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Reads one case (schema "makefun/gta6-trophies-case/v1") per *.in file in the
input directory and writes, for each case, five quantiles (5/25/50/75/95) of
every trophy's Exophase earn rate at the case's target, as
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.

Method (standard library only, deterministic, no network, no clock reads):
  1. Work in logit space, logit(p) = ln(p / (100 - p)), p in percent.
  2. Per trophy, fit a weighted least-squares line of logit(rate) against
     x = ln(1 + hours since the game's release, 2026-11-19T00:00:00Z),
     over the case's recent reads, with recency weights. This is the same
     family as the house curve baseline, but with three corrections:
       a. the slope is shrunk toward its tier's mean slope (precision
          weighted), because a single trophy's short history is noisy;
       b. the extrapolated move is damped, kappa = 1 / (1 + horizon_days/6),
          because trophy growth decelerates as the reachable cohort is
          exhausted and the tracked-player base matures; the house curve
          extrapolates its line undamped, which overshoots when growth
          saturates;
       c. the spread is honest: a random-walk term (0.5 logit per sqrt day),
          the slope's own standard error projected to the target, and a
          refresh-shock term for Exophase's discrete page refreshes, which
          dominates short horizons (the practice cases straddle one refresh)
          and fades over long ones.
  3. Degenerate histories (a single read, every read identical, rates
     missing) fall back to a flat forecast at the last rate with the same
     spread machinery; a trophy with no rate in any read falls back to a
     tier prior. A case with no usable history at all still yields a valid
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


def forecast_case(case):
    """Forecast every trophy of one case. Returns the output object."""
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
