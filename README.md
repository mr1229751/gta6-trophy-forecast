# GTA VI Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mr1229751/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mr1229751/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for the make.fun bounty **gta6-trophy-rarity-forecast-01** — "Forecast how
many GTA VI players earn each trophy in week one".

`main.py` (Python 3, standard library only, no network) reads every `*.in`
case file in `$MAKEFUN_INPUT_DIR` and writes one `<id>.out` per case to
`$MAKEFUN_OUTPUT_DIR`: for each of the case's 55 Grand Theft Auto VI (PS5)
trophies, five quantiles (5/25/50/75/95%) of its Exophase earn rate at the
case's target time. Scoring is mean pinball loss on the logit scale, so all
modelling is done on the logit scale.

## Approach

Exophase refreshes its counts on its own schedule, so a case's history —
reads every 30 minutes — is mostly one frozen page repeated. The forecaster
therefore works on **distinct states**: reads where `players_tracked` or any
rate actually changed.

- **Penetration trend.** Per trophy, the logit of its earn rate (earners /
  players — a trend that already nets dilution against progression) is
  anchored at a smoothed recent level and extrapolated along
  ln(1 + hours since release), fit on the distinct states with recency
  weights. Per-trophy slopes are pooled per trophy tier and globally, and
  each trophy's slope shrinks toward its tier's with weight n/(n+k), so
  erratic or barely-observed trophies borrow the pooled trend.
- **Players forecast.** `players_tracked` is projected to the target with a
  recency-weighted log-linear fit whose growth rate is exponentially damped
  (launch growth decays), blended with the undamped projection and capped.
- **Earners floor.** Earners are cumulative: if a median would imply fewer
  earners at the target than the last state already shows, it is pulled up
  toward the flat-earners line.
- **Rounded-to-zero trophies.** At large player counts a rare trophy's rate
  prints as 0.00%. Those are forecast in earners space (pooled earners
  growth) and divided by the forecast player count.
- **The 100% trophy.** A trophy at 100% with earners equal to players in
  every state is treated as automatic: median ~100, downside-only spread.
- **Frozen page (one distinct state).** The known next event is an Exophase
  refresh with a player backlog jump, so medians shade down for dilution by
  tier and the spread is wide — the regime the practice cases grade.
- **Spread.** Per trophy: shrunk fit residual + extrapolation distance in
  ln-time + count noise 1/sqrt(earners), with a horizon floor. Deterministic
  throughout: no randomness, no wall clock, byte-identical repeat runs.

The design was validated by scenario backtests: synthetic refresh dynamics
anchored on the documented launch facts (frozen page of 638 players; first
refresh ×6.06 players with every non-100% rate lower; saturating player
growth; cohort-convolution earners) were used to cut final-like cases
(checkpoint +59.5 h, target +168 h). Across 60 runs in 10 scenario families
the forecaster's mean pinball logit loss was **0.230**, against **0.490** for
the house baseline and **0.297** for the house curve baseline — ahead of both
in every run — and **0.359 vs 0.853** on practice-like cuts.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<dir-with-.in-files> MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Each run is far inside the sandbox limits (a 120-read case forecasts in
well under a second; the limit is 8 s per case).
