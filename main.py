#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates (Exophase, PS5) at a future target time.

make.fun competition gta6-trophy-rarity-forecast-01.

Reads every *.in case file in $MAKEFUN_INPUT_DIR (else /input) and writes one
<id>.out per case to $MAKEFUN_OUTPUT_DIR (else /output). Each output row holds
five quantiles of a trophy's earn rate (percent) at the case's target time.
Scoring is mean pinball loss on the logit scale, so all modelling happens on
the logit scale and the written quantiles are the predictive quantiles there.

Model summary
-------------
The page a case's history reads is Exophase's, and Exophase refreshes its
counts on its own schedule: between refreshes the page is frozen. So the
history is first collapsed to DISTINCT STATES (reads where players_tracked or
any rate actually changed). Fitting the raw reads would weight one frozen
page by its read count.

On the distinct states, per trophy:
  * players_tracked is forecast to the target with a recency-weighted
    log-linear growth fit whose growth rate is damped (launch growth decays)
    and blended with the undamped projection;
  * the trophy's logit rate (its penetration: earners / players, whose trend
    already nets dilution against progression) is anchored at a smoothed
    recent level and extrapolated by a per-trophy trend on ln(1 + hours
    since release), fit on the distinct states with recency weights. Slopes
    are pooled per tier and globally, and each trophy's slope shrinks toward
    its tier's with weight n_eff / (n_eff + k);
  * an earners floor keeps the median from implying fewer earners than the
    last state already shows (earners are cumulative);
  * ultra-rare trophies whose printed rate has rounded to ~0 are forecast in
    earners space instead (rate = forecast earners / forecast players);
  * a trophy sitting at 100% with earners == players throughout is treated
    as automatic: median ~100, downside-only risk.

Spread per trophy combines the shrunk fit residual, an extrapolation term in
ln-time distance, a count-noise term 1/sqrt(earners), and a horizon floor.
With a single frozen state (the practice shape) the forecast shades medians
down for the documented launch-refresh dilution and uses a wide spread.

Everything is deterministic pure math: no randomness, no wall clock.
Standard library only.
"""

import glob
import json
import math
import os
import sys
from datetime import datetime, timezone
from statistics import NormalDist

_NORM = NormalDist()

# ---------------------------------------------------------------------------
# Tunable constants (chosen by scenario backtests against the house baselines)
# ---------------------------------------------------------------------------
FIT_RECENCY_HL = 14.0        # hours; recency half-life for fit weights
PLAYER_RECENCY_HL = 48.0     # hours; recency half-life for the players fit
GROWTH_DAMP_HL = 30.0        # hours; half-life of the player growth rate
SHRINK_K = 3.0               # pseudo-count pulling trophy slopes to tier mean
TIER_SHRINK_K = 4.0          # pseudo-count pulling tier slopes to global mean
SIGMA_EXT_COEF = 0.45        # spread growth per unit of ln-time extrapolation
SIGMA_FLOOR_COEF = 0.22      # spread floor: coef * sqrt(horizon in days)
FLOOR_W = 0.7                # weight of the earners floor when it binds
LEVEL_TAIL = 2               # observations in the smoothed recent level
BASELINE_WIDEN = 1.15        # fallback spread vs the house baseline's
TIER_PRIOR_RATE = {"platinum": 0.5, "gold": 0.7, "silver": 0.7, "bronze": 0.9}
FROZEN_TIER_EARNERS_B = {"platinum": 0.5, "gold": 0.9, "silver": 1.2, "bronze": 1.6}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def parse_ts(text):
    """Parse an ISO-8601 timestamp to epoch seconds; None when unparsable."""
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def logit_pct(x):
    """logit of a percent, with the scoring clamp to [0.01, 99.99]."""
    p = clamp(float(x), 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def logit_raw(p):
    """logit of a percent without the scoring clamp (internal forecasts)."""
    p = clamp(float(p), 1e-6, 99.999999) / 100.0
    return math.log(p / (1.0 - p))


def logistic_pct(l):
    if l >= 0.0:
        return 100.0 / (1.0 + math.exp(-l))
    z = math.exp(l)
    return 100.0 * z / (1.0 + z)


def release_epoch(case):
    """Epoch seconds of the game's release midnight UTC (the ln-time origin)."""
    game = case.get("game") or {}
    rel = game.get("release_date")
    if rel:
        ts = parse_ts(str(rel) + "T00:00:00Z")
        if ts is not None:
            return ts
    return datetime(2026, 11, 19, tzinfo=timezone.utc).timestamp()


def hours_between(epoch, ts):
    return (ts - epoch) / 3600.0


def quantiles_from(median_logit, sigma, levels):
    qs = []
    for tau in levels:
        z = _NORM.inv_cdf(clamp(float(tau), 1e-6, 1.0 - 1e-6))
        q = logistic_pct(median_logit + z * sigma)
        qs.append(round(clamp(q, 0.0, 100.0), 4))
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def baseline_row(rate, horizon_days, levels, widen=BASELINE_WIDEN):
    """House-baseline-shaped fallback: last rate, random-walk spread."""
    sigma = 0.5 * math.sqrt(max(horizon_days, 0.0)) * widen
    sigma = max(sigma, 0.05)
    return quantiles_from(logit_pct(rate), sigma, levels)


# ---------------------------------------------------------------------------
# History -> distinct states
# ---------------------------------------------------------------------------
def build_states(case, epoch):
    """Collapse reads to states where (players, rates) actually change."""
    states = []
    for read in case.get("history") or []:
        try:
            players = read.get("players_tracked")
            rates = read.get("rates")
            ts = parse_ts(read.get("at") or read.get("slot"))
            if players is None or rates is None or ts is None:
                continue
            players = float(players)
            key = (players, tuple(rates))
            t_h = hours_between(epoch, ts)
            if states and states[-1]["key"] == key:
                states[-1]["t_read"] = t_h
                continue
            states.append(
                {"t": t_h, "t_read": t_h, "players": players,
                 "rates": list(rates), "key": key}
            )
        except Exception:
            continue
    return states


def trophy_obs(states, idx):
    """(t_hours, players, rate) rows for one trophy over the states."""
    obs = []
    for st in states:
        rates = st["rates"]
        if idx >= len(rates):
            continue
        r = rates[idx]
        if r is None:
            continue
        try:
            obs.append((st["t"], st["players"], float(r)))
        except (TypeError, ValueError):
            continue
    return obs


# ---------------------------------------------------------------------------
# Weighted least squares (intercept + one slope)
# ---------------------------------------------------------------------------
def wls1(rows, weights):
    """rows: (x, y). Returns (a, b, resid_sd, n_eff) or None."""
    sw = sum(weights)
    if sw <= 0:
        return None
    mx = sum(w * x for (x, _), w in zip(rows, weights)) / sw
    my = sum(w * y for (_, y), w in zip(rows, weights)) / sw
    den = sum(w * (x - mx) ** 2 for (x, _), w in zip(rows, weights))
    if den < 1e-9:
        return None
    num = sum(w * (x - mx) * (y - my) for (x, y), w in zip(rows, weights))
    b = num / den
    a = my - b * mx
    rnum = sum(w * (y - (a + b * x)) ** 2 for (x, y), w in zip(rows, weights))
    n_eff = _n_eff(weights)
    if sw > 0:
        resid = math.sqrt((rnum / sw) * (n_eff / max(n_eff - 2.0, 1.0)))
    else:
        resid = 0.0
    return a, b, resid, n_eff


def _n_eff(weights):
    sw = sum(weights)
    sw2 = sum(w * w for w in weights)
    if sw2 <= 0:
        return 0.0
    return min(sw * sw / sw2, float(len(weights)))


def recency_weights(times, t_ref, half_life):
    return [0.5 ** (max(t_ref - t, 0.0) / half_life) for t in times]


# ---------------------------------------------------------------------------
# Players forecast
# ---------------------------------------------------------------------------
def forecast_players(states, t_star):
    last = states[-1]
    p_last = max(last["players"], 1.0)
    dt = t_star - last["t"]
    if dt <= 0 or len(states) < 2:
        return p_last
    pts = [(st["t"], math.log(max(st["players"], 1.0))) for st in states]
    if len({t for t, _ in pts}) < 2:
        return p_last

    def slope(rows, weights):
        sw = sum(weights)
        mx = sum(w * x for (x, _), w in zip(rows, weights)) / sw
        my = sum(w * y for (_, y), w in zip(rows, weights)) / sw
        num = sum(w * (x - mx) * (y - my) for (x, y), w in zip(rows, weights))
        den = sum(w * (x - mx) ** 2 for (x, _), w in zip(rows, weights))
        return num / den if den > 1e-9 else 0.0

    w_all = recency_weights([t for t, _ in pts], pts[-1][0], PLAYER_RECENCY_HL)
    g1 = slope(pts, w_all)
    tail = pts[-min(5, len(pts)):]
    g2 = slope(tail, [1.0] * len(tail)) if len({t for t, _ in tail}) >= 2 else g1
    g = clamp(0.5 * g1 + 0.5 * g2, 0.0, 0.05)
    lam = math.log(2.0) / GROWTH_DAMP_HL
    tau_eff = (1.0 - math.exp(-lam * dt)) / lam
    ln_damped = math.log(p_last) + g * tau_eff
    ln_linear = math.log(p_last) + g * dt
    ln_star = 0.65 * ln_damped + 0.35 * ln_linear
    ln_star = min(ln_star, math.log(p_last) + math.log(60.0))
    return max(math.exp(ln_star), p_last)


# ---------------------------------------------------------------------------
# Frozen-page forecast (a single distinct state: the practice shape)
# ---------------------------------------------------------------------------
def frozen_forecast(case, trophies, states, t_at, t_star, levels):
    horizon_h = max(t_star - t_at, 0.0)
    horizon_d = horizon_h / 24.0
    st = states[-1] if states else None
    p_last = st["players"] if st else 0.0
    # A refresh backlog jump is a launch-window phenomenon.
    amp = 5.5 if t_at <= 36.0 else 1.5
    m_players = 1.0 + amp * (1.0 - math.exp(-horizon_h / 3.0)) if horizon_h > 0 else 1.0
    sigma = 0.85 + 0.25 * math.sqrt(horizon_d)
    rows = []
    for idx, tr in enumerate(trophies):
        tier = tr.get("tier") or "bronze"
        r = None
        if st is not None and idx < len(st["rates"]):
            r = st["rates"][idx]
        if r is None:
            r = TIER_PRIOR_RATE.get(tier, 0.9)
            rows.append({"id": tr.get("id"),
                         "q": quantiles_from(logit_pct(r), 1.2, levels)})
            continue
        r = float(r)
        if r >= 99.99:
            rows.append({"id": tr.get("id"),
                         "q": quantiles_from(logit_pct(99.9), 0.5, levels)})
            continue
        if r <= 0.0:
            med = max(0.01, 30.0 / max(p_last * m_players, 1.0))
            rows.append({"id": tr.get("id"),
                         "q": quantiles_from(logit_raw(med), 1.2, levels)})
            continue
        b = FROZEN_TIER_EARNERS_B.get(tier, 1.6)
        m_earners = 1.0 + b * (1.0 - math.exp(-horizon_h / 6.0)) if horizon_h > 0 else 1.0
        if m_players > 2.0:
            m_earners = min(m_earners, 0.9 * m_players)
        med_rate = clamp(r * m_earners / m_players, 0.01, 99.99)
        rows.append({"id": tr.get("id"),
                     "q": quantiles_from(logit_pct(med_rate), sigma, levels)})
    return rows


# ---------------------------------------------------------------------------
# Main model (two or more distinct states)
# ---------------------------------------------------------------------------
def _shrink(value, n_eff, pooled, k=SHRINK_K):
    if value is None:
        return pooled
    return (n_eff * value + k * pooled) / (n_eff + k)


def model_forecast(case, trophies, states, t_star, levels):
    t_last = states[-1]["t"]
    horizon_d = max((t_star - t_last) / 24.0, 0.0)
    u_star = math.log1p(max(t_star, 0.0))
    p_star = forecast_players(states, t_star)

    all_obs = [trophy_obs(states, i) for i in range(len(trophies))]

    # ---- pass 1: per-trophy fits of logit rate on ln(1 + hours since release)
    fits = []
    for obs in all_obs:
        fit = None
        if len(obs) >= 4:
            times = [o[0] for o in obs]
            us = [math.log1p(max(t, 0.0)) for t in times]
            if max(us) - min(us) > 0.05:
                rows = []
                weights = []
                tw = recency_weights(times, times[-1], FIT_RECENCY_HL)
                for k, ((t, p, r), w0) in enumerate(zip(obs, tw)):
                    r_eff = r if r > 0 else max(0.01, 20.0 / max(p, 1.0))
                    earners = r * p / 100.0
                    w = w0 * min(1.0, (earners + 2.0) / 30.0)
                    rows.append((us[k], logit_pct(r_eff)))
                    weights.append(w)
                res = wls1(rows, weights)
                if res is not None:
                    _, b, resid, n_eff = res
                    fit = {"b": clamp(b, -2.5, 2.5),
                           "resid": resid, "n_eff": n_eff}
        fits.append(fit)

    # ---- pooled slopes / residuals (tier shrunk toward global)
    def pooled(key, subset):
        num = 0.0
        den = 0.0
        for i in subset:
            f = fits[i]
            if f is not None:
                num += f["n_eff"] * f[key]
                den += f["n_eff"]
        return (num / den) if den > 0 else None

    tiers = [tr.get("tier") or "bronze" for tr in trophies]
    all_idx = list(range(len(trophies)))
    glob = {k: pooled(k, all_idx) for k in ("b", "resid")}
    if glob["b"] is None:
        glob["b"] = 0.35
    if glob["resid"] is None:
        glob["resid"] = 0.6
    tier_pool = {}
    for tier in set(tiers):
        sub = [i for i in all_idx if tiers[i] == tier]
        entry = {}
        for k in ("b", "resid"):
            raw = pooled(k, sub)
            n_tot = sum(fits[i]["n_eff"] for i in sub if fits[i] is not None)
            if raw is None:
                entry[k] = glob[k]
            else:
                entry[k] = (n_tot * raw + TIER_SHRINK_K * glob[k]) / (n_tot + TIER_SHRINK_K)
        tier_pool[tier] = entry

    # ---- pooled earners growth (for the rounded-to-zero path)
    earn_slopes = []
    for i, obs in enumerate(all_obs):
        if len(obs) >= 4:
            pts = [(math.log1p(max(t, 0.0)), math.log(r * p / 100.0 + 0.5))
                   for (t, p, r) in obs if r > 0]
            if len(pts) >= 4 and max(x for x, _ in pts) - min(x for x, _ in pts) > 0.05:
                w = recency_weights([o[0] for o in obs if o[2] > 0],
                                    obs[-1][0], FIT_RECENCY_HL)
                sw = sum(w)
                mx = sum(wi * x for (x, _), wi in zip(pts, w)) / sw
                my = sum(wi * y for (_, y), wi in zip(pts, w)) / sw
                den = sum(wi * (x - mx) ** 2 for (x, _), wi in zip(pts, w))
                if den > 1e-9:
                    num = sum(wi * (x - mx) * (y - my) for (x, y), wi in zip(pts, w))
                    earn_slopes.append((tiers[i], num / den))
    g_earn_glob = (sum(s for _, s in earn_slopes) / len(earn_slopes)) if earn_slopes else 0.35
    g_earn_tier = {}
    for tier in set(tiers):
        ss = [s for tname, s in earn_slopes if tname == tier]
        g_earn_tier[tier] = (sum(ss) / len(ss)) if ss else g_earn_glob

    # ---- pass 2: per-trophy forecast
    rows = []
    for idx, tr in enumerate(trophies):
        tid = tr.get("id")
        tier = tiers[idx]
        obs = all_obs[idx]
        pool = tier_pool[tier]
        try:
            rows.append({"id": tid, "q": _forecast_one(
                obs, fits[idx], pool, tier, g_earn_tier[tier],
                t_star, u_star, p_star, horizon_d, levels)})
        except Exception:
            r_last = obs[-1][2] if obs else TIER_PRIOR_RATE.get(tier, 0.9)
            rows.append({"id": tid, "q": baseline_row(r_last, horizon_d, levels)})
    return rows


def _forecast_one(obs, fit, pool, tier, g_earn, t_star, u_star, p_star,
                  horizon_d, levels):
    if not obs:
        med = logit_pct(TIER_PRIOR_RATE.get(tier, 0.9)) + pool["b"] * 0.5
        return quantiles_from(med, 1.3, levels)

    t_last, p_last, r_last = obs[-1]
    u_last = math.log1p(max(t_last, 0.0))
    earners_last = r_last * p_last / 100.0

    # Automatic trophy: at 100% with earners == players throughout.
    if r_last >= 99.99 and all(r >= 99.99 for _, _, r in obs):
        return quantiles_from(logit_pct(99.9), 0.5, levels)

    # Rounded-to-zero path: forecast earners, divide by forecast players.
    if r_last <= 0.011 and len(obs) >= 2:
        ref = None
        for (t, p, r) in obs:
            if r >= 0.05:
                ref = (t, r * p / 100.0)
        if ref is not None:
            t_ref, e_ref = ref
            e_star = e_ref * math.exp(g_earn * (u_star - math.log1p(max(t_ref, 0.0))))
            med_rate = clamp(e_star / max(p_star, 1.0) * 100.0, 0.002, 99.99)
            return quantiles_from(logit_raw(med_rate), 1.15, levels)
        if r_last <= 0.0:
            med_rate = max(0.005, 40.0 / max(p_star, 1.0))
            return quantiles_from(logit_raw(med_rate), 1.25, levels)

    # Smoothed recent level (last few observations, recency weighted).
    tail = obs[-LEVEL_TAIL:]
    tw = recency_weights([o[0] for o in tail], t_last, FIT_RECENCY_HL)
    level_num = 0.0
    level_den = 0.0
    for (t, p, r), w in zip(tail, tw):
        r_eff = r if r > 0 else max(0.01, 20.0 / max(p, 1.0))
        level_num += w * logit_pct(r_eff)
        level_den += w
    level = level_num / level_den if level_den > 0 else logit_pct(r_last)

    if fit is not None:
        b = _shrink(fit["b"], fit["n_eff"], pool["b"])
        resid = fit["resid"]
        n_eff = fit["n_eff"]
    else:
        b, resid, n_eff = pool["b"], pool["resid"], 0.0
    med = level + b * (u_star - u_last)

    # Earners floor: the median must not imply earners collapse.
    implied = logistic_pct(med) / 100.0 * p_star
    if earners_last > 0 and implied < earners_last:
        floor_med = logit_raw(clamp(earners_last / max(p_star, 1.0) * 100.0, 0.001, 99.99))
        med = FLOOR_W * floor_med + (1.0 - FLOOR_W) * med

    # Spread.
    sig_fit = math.sqrt((n_eff * resid ** 2 + SHRINK_K * pool["resid"] ** 2)
                        / (n_eff + SHRINK_K))
    sig_fit = clamp(sig_fit, 0.22, 1.6)
    sig_ext = SIGMA_EXT_COEF * max(u_star - u_last, 0.0)
    sig_cnt = min(0.8, 1.0 / math.sqrt(max(earners_last, 0.5)))
    sigma = math.sqrt(sig_fit ** 2 + sig_ext ** 2 + sig_cnt ** 2)
    sigma = max(sigma, SIGMA_FLOOR_COEF * math.sqrt(horizon_d))
    sigma = min(sigma, 2.4)
    return quantiles_from(med, sigma, levels)


# ---------------------------------------------------------------------------
# Case driver
# ---------------------------------------------------------------------------
def forecast_case(case):
    trophies = case.get("trophies") or []
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    epoch = release_epoch(case)
    ts_at = parse_ts(case.get("at"))
    ts_target = parse_ts(case.get("target"))
    if ts_at is None:
        ts_at = epoch
    if ts_target is None:
        ts_target = ts_at + 60.0 * float(case.get("horizon_minutes") or 0)
    t_at = hours_between(epoch, ts_at)
    t_star = hours_between(epoch, ts_target)
    states = build_states(case, epoch)

    if len(states) <= 1:
        rows = frozen_forecast(case, trophies, states, t_at, t_star, levels)
    else:
        rows = model_forecast(case, trophies, states, t_star, levels)
    return {"trophies": rows}


def fallback_case(case):
    """Baseline-shaped output used if anything unexpected goes wrong."""
    trophies = case.get("trophies") or []
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    epoch = release_epoch(case)
    states = build_states(case, epoch)
    ts_at = parse_ts(case.get("at")) or epoch
    ts_target = parse_ts(case.get("target"))
    if ts_target is None:
        ts_target = ts_at + 60.0 * float(case.get("horizon_minutes") or 0)
    horizon_d = max((ts_target - ts_at) / 86400.0, 0.0)
    rows = []
    for idx, tr in enumerate(trophies):
        r = None
        if states and idx < len(states[-1]["rates"]):
            r = states[-1]["rates"][idx]
        if r is None:
            r = TIER_PRIOR_RATE.get(tr.get("tier") or "bronze", 0.9)
        rows.append({"id": tr.get("id"),
                     "q": baseline_row(float(r), horizon_d, levels)})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(in_dir, "*.in")))
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        case_id = os.path.splitext(os.path.basename(path))[0]
        try:
            out = forecast_case(case)
        except Exception:
            out = fallback_case(case)
        out_path = os.path.join(out_dir, str(case_id) + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
