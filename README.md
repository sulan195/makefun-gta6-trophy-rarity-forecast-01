# makefun-gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sulan195/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sulan195/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for Grand Theft Auto VI's PlayStation 5 trophy earn rates on Exophase: given the house's recording of the trophy page up to a case's checkpoint, it forecasts every trophy's earn rate a week after launch (2026-11-26 00:00 UTC) as five quantiles (5%, 25%, 50%, 75%, 95%), scored by pinball loss on the logit scale. It beats the house baselines by splitting each trophy's rate into its two moving parts — earners and the tracked-player count — and forecasting the two separately: earners grow as players progress, the player count dilutes the rate (launch night's first refresh divided every rate by ~6 while the denominator jumped 638 → 3,868 players), and a single line on the rate conflates them.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The program reads every `<id>.in` case in the input directory and writes one `<id>.out` forecast per case to the output directory. It uses only the Python 3.13 standard library, needs no network, is deterministic (two runs write byte-identical output), and runs in well under the 8-second limit.

## How it works

- **Input.** One case file per run: the game's 55 trophies and every valid Exophase read before the checkpoint (`history`, oldest first, with `players_tracked` and per-trophy earn rates).
- **Model.** Per trophy, log earners and log players are each fitted with a recency-weighted least-squares line against `x = ln(1 + hours since release)`, over the last 24 hours of reads (12-hour half-life weights). The player trend is one series shared by every trophy, so it is estimated once and precisely; each trophy's earner slope shrinks toward its tier mean (platinum / gold / silver / bronze), precision-weighted, so a noisy single trophy cannot run away on its own. The median rate is `100 * earners(target) / players(target)`, with each extrapolated move damped by `kappa = 1 / (1 + horizon_days / 6)`: a 90-minute practice horizon is barely damped, a 4.5-day final horizon keeps about 40% of the line's move. The median move is capped at 4 logit units from the last read. Degenerate histories fall back to a direct logit-rate line fit.
- **Spread.** `s = sqrt(0.5^2 * horizon_days + (slope_se * dx)^2 + shock^2)`, where the slope term combines the earner and player fits' uncertainties, and the shock term `0.85 * exp(-horizon_days / 1.5) + 0.15` covers Exophase's discrete page refreshes (launch night's first refresh divided every rate by ~6 while the denominator jumped 638 → 3,868 players). Quantiles are `sigmoid(median + z_tau * s)` at the case's own levels, rounded to 4 decimals, half up, non-decreasing.
- **Edge cases.** Flat or single-read histories forecast the last rate; trophies with no rate anywhere fall back to tier priors (platinum 0.01%, gold 0.05%, silver 0.5%, bronze 2%); rates are clamped to 0.01–99.99% before the logit; a malformed case falls back to tier priors instead of crashing the run, so every case stays valid.

## Validation

- `tests/test_forecast.py` — 20 tests (stdlib `unittest` only): output shape and the 65,536-byte cap, byte-identical determinism across runs, the RULES.txt worked pinball example (loss 0.403404), flat-history medians, refresh-shock spread, the earners/players decomposition (dilution captured; fallback to the direct fit when no player counts exist), and 10+ edge cases (empty history, single read, null rates, 100% and 0% rates, zero horizon, custom levels, unsorted trophy ids, malformed input). Run with `python3 -m unittest discover -s tests -v`.
- A synthetic backtest (kept outside the repository) scores the model against the house baseline and the undamped log-time curve on four known-answer regimes — a refresh shock, saturating growth, continuing log-time-linear growth, and a rich history with discrete Exophase refreshes: the model wins or ties the direct-fit v1 everywhere and beats the house baseline in all four (refresh shock 1.08 vs 1.69; saturating 0.22 vs 0.23; linear 0.25 vs 0.70; refresh-history 0.23 vs 0.25).

## Files

- `main.py` — the entrypoint (Python 3.13, standard library only).
- `tests/test_forecast.py` — the test suite.
- `LICENSE` — MIT.

---

<sub>Entered on [make.fun](https://make.fun/?ref=mmtmfkx5&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
