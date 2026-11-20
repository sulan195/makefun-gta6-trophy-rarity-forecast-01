"""Tests for the GTA VI trophy rarity forecaster (stdlib unittest only).

Run: python3 -m unittest discover -s tests -v   (from the repository root)
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main as forecast


def make_case(trophies, history, horizon_minutes=90,
              target="2026-11-19T04:30:00Z", at="2026-11-19T03:00:00Z",
              levels=None):
    return {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": "t00",
        "at": at,
        "horizon_minutes": horizon_minutes,
        "target": target,
        "levels": levels or [0.05, 0.25, 0.5, 0.75, 0.95],
        "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                 "release_date": "2026-11-19", "source": "exophase"},
        "trophies": trophies,
        "history": history,
        "notes": "test case",
    }


def trophy(tid, tier="bronze", name="T"):
    return {"id": tid, "name": name, "description": "d", "tier": tier,
            "points": 15, "secret": False}


def read(at, players, rates):
    return {"slot": at, "at": at, "players_tracked": players,
            "trophies_earned": 0, "rates": rates}


def pinball_loss(quantiles, rate, levels=(0.05, 0.25, 0.5, 0.75, 0.95)):
    """Mean pinball loss on the logit scale, per RULES.txt's formula."""
    def logit(pct):
        p = min(99.99, max(0.01, pct))
        return math.log(p / (100.0 - p))
    total = 0.0
    for tau, q in zip(levels, quantiles):
        d = logit(rate) - logit(q)
        total += 2.0 * max(tau * d, (tau - 1.0) * d)
    return total / len(levels)


class TestScoringFormula(unittest.TestCase):
    def test_worked_example_from_rules(self):
        # RULES.txt: rows [10,15,20,25,35], rates 22% and 8% -> loss 0.403404.
        rows = [[10, 15, 20, 25, 35], [10, 15, 20, 25, 35]]
        rates = [22.0, 8.0]
        losses = [pinball_loss(q, r) for q, r in zip(rows, rates)]
        mean_loss = sum(losses) / len(losses)
        self.assertAlmostEqual(mean_loss, 0.403404, places=5)

    def test_norm_ppf(self):
        self.assertAlmostEqual(forecast.norm_ppf(0.5), 0.0, places=9)
        self.assertAlmostEqual(forecast.norm_ppf(0.95), 1.6448536269514722, places=6)
        self.assertAlmostEqual(forecast.norm_ppf(0.05), -1.6448536269514722, places=6)
        self.assertAlmostEqual(forecast.norm_ppf(0.25), -0.6744897501960817, places=6)

    def test_round4_half_up(self):
        self.assertEqual(forecast.round4(1.23455), 1.2346)
        self.assertEqual(forecast.round4(2.0), 2)
        self.assertEqual(forecast.round4(0.12345), 0.1235)


class TestOutputShape(unittest.TestCase):
    def setUp(self):
        kit = Path("/home/hatch/workspace/makefun/kit/gta6-trophy-rarity-forecast-01"
                   "/gta6-trophy-rarity-forecast-01-kit/input")
        self.p00 = json.loads((kit / "p00.in").read_text())
        self.p01 = json.loads((kit / "p01.in").read_text())

    def check_shape(self, case, out):
        trophies = out["trophies"]
        self.assertEqual(set(out.keys()), {"trophies"})
        self.assertEqual(len(trophies), len(case["trophies"]))
        self.assertEqual(sorted(r["id"] for r in trophies),
                         sorted(t["id"] for t in case["trophies"]))
        for row in trophies:
            self.assertEqual(set(row.keys()), {"id", "q"})
            self.assertEqual(len(row["q"]), 5)
            for value in row["q"]:
                self.assertIsInstance(value, (int, float))
                self.assertTrue(math.isfinite(value))
                self.assertGreaterEqual(value, 0.01)
                self.assertLessEqual(value, 99.99)
            self.assertEqual(row["q"], sorted(row["q"]))
        blob = json.dumps(out, separators=(",", ":")) + "\n"
        self.assertLessEqual(len(blob.encode()), 65536)

    def test_practice_cases_shape(self):
        for case in (self.p00, self.p01):
            self.check_shape(case, forecast.forecast_case(case))

    def test_practice_flat_history_median_is_last_rate(self):
        # Every practice read is identical, so the slope is 0 and the
        # median must equal the last read's rate (clamped to 99.99).
        out = forecast.forecast_case(self.p00)
        last = self.p00["history"][-1]["rates"]
        for row, rate in zip(out["trophies"], last):
            self.assertAlmostEqual(row["q"][2], min(rate, 99.99), places=3)

    def test_practice_spread_wider_than_house_baseline(self):
        # The house baseline's 90-minute spread is ~0.125 logit; ours must
        # be wide enough to cover an Exophase refresh shock.
        out = forecast.forecast_case(self.p00)
        row = out["trophies"][3]  # a trophy last at 1.10%
        self.assertLess(row["q"][0], 0.6)
        self.assertGreater(row["q"][4], 2.0)


class TestEdgeCases(unittest.TestCase):
    def test_empty_history_uses_tier_prior(self):
        case = make_case([trophy(1, "platinum"), trophy(2, "bronze")], [])
        out = forecast.forecast_case(case)
        self.assertEqual(len(out["trophies"]), 2)
        self.assertAlmostEqual(out["trophies"][0]["q"][2], 0.01, places=2)
        self.assertAlmostEqual(out["trophies"][1]["q"][2], 2.0, places=2)

    def test_single_read_is_flat(self):
        case = make_case(
            [trophy(1)], [read("2026-11-19T02:30:00Z", 638, [3.5])])
        out = forecast.forecast_case(case)
        self.assertAlmostEqual(out["trophies"][0]["q"][2], 3.5, places=3)

    def test_null_rates_skipped(self):
        history = [
            read("2026-11-19T00:00:00Z", 100, [None]),
            read("2026-11-19T01:00:00Z", 100, [2.0]),
            read("2026-11-19T02:00:00Z", 100, [2.5]),
        ]
        case = make_case([trophy(1)], history)
        out = forecast.forecast_case(case)
        self.assertGreaterEqual(out["trophies"][0]["q"][2], 2.0)

    def test_rate_100_stays_near_100(self):
        history = [read("2026-11-19T02:00:00Z", 638, [100.0]),
                   read("2026-11-19T02:30:00Z", 638, [100.0])]
        case = make_case([trophy(1)], history)
        out = forecast.forecast_case(case)
        self.assertGreater(out["trophies"][0]["q"][2], 99.0)
        self.assertLessEqual(out["trophies"][0]["q"][4], 99.99)

    def test_rate_zero_clamped(self):
        history = [read("2026-11-19T02:00:00Z", 638, [0.0]),
                   read("2026-11-19T02:30:00Z", 638, [0.0])]
        case = make_case([trophy(1)], history)
        out = forecast.forecast_case(case)
        self.assertEqual(out["trophies"][0]["q"][0], 0.01)

    def test_zero_horizon(self):
        # At horizon 0 the median is the last rate; the spread stays wide
        # on purpose, because the answering read comes up to an hour after
        # the target and an Exophase refresh can still land inside it.
        history = [read("2026-11-19T02:00:00Z", 638, [5.0]),
                   read("2026-11-19T02:30:00Z", 638, [5.0])]
        case = make_case([trophy(1)], history, horizon_minutes=0,
                         target="2026-11-19T03:00:00Z")
        out = forecast.forecast_case(case)
        row = out["trophies"][0]["q"]
        self.assertAlmostEqual(row[2], 5.0, places=2)
        self.assertEqual(row, sorted(row))

    def test_custom_levels(self):
        history = [read("2026-11-19T02:00:00Z", 638, [5.0]),
                   read("2026-11-19T02:30:00Z", 638, [5.0])]
        case = make_case([trophy(1)], history, levels=[0.1, 0.5, 0.9])
        out = forecast.forecast_case(case)
        self.assertEqual(len(out["trophies"][0]["q"]), 3)

    def test_unsorted_and_non55_trophies(self):
        trophies = [trophy(9), trophy(3), trophy(7)]
        history = [read("2026-11-19T02:00:00Z", 638, [1.0, 2.0, 3.0]),
                   read("2026-11-19T02:30:00Z", 638, [1.0, 2.0, 3.0])]
        case = make_case(trophies, history)
        out = forecast.forecast_case(case)
        self.assertEqual([r["id"] for r in out["trophies"]], [3, 7, 9])

    def test_rising_trend_moves_median_up_but_damped(self):
        # Rates rising steadily in log-time before the checkpoint: the
        # median must sit above the last rate (damped trend), not at it.
        history = []
        for half_hour in range(0, 6):
            hour = half_hour * 0.5
            at = "2026-11-19T%02d:%02d:00Z" % (int(hour), int((hour % 1) * 60))
            rate = 1.0 * math.exp(0.35 * math.log1p(hour))
            history.append(read(at, 1000, [round(rate, 2)]))
        case = make_case([trophy(1)], history, horizon_minutes=90)
        out = forecast.forecast_case(case)
        last = history[-1]["rates"][0]
        self.assertGreater(out["trophies"][0]["q"][2], last)

    def test_fallback_case_is_valid(self):
        case = make_case([trophy(1, "gold"), trophy(2, "silver")], [])
        out = forecast.fallback_case(case)
        self.assertEqual(len(out["trophies"]), 2)
        for row in out["trophies"]:
            self.assertEqual(row["q"], sorted(row["q"]))


class TestEndToEnd(unittest.TestCase):
    def test_cli_writes_outputs_and_is_deterministic(self):
        kit = Path("/home/hatch/workspace/makefun/kit/gta6-trophy-rarity-forecast-01"
                   "/gta6-trophy-rarity-forecast-01-kit/input")
        repo = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            out1 = Path(tmp) / "out1"
            out2 = Path(tmp) / "out2"
            in_dir.mkdir()
            for name in ("p00.in", "p01.in"):
                (in_dir / name).write_text((kit / name).read_text())
            env = dict(os.environ)
            for out_dir in (out1, out2):
                env["MAKEFUN_INPUT_DIR"] = str(in_dir)
                env["MAKEFUN_OUTPUT_DIR"] = str(out_dir)
                proc = subprocess.run(
                    [sys.executable, str(repo / "main.py")],
                    env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(proc.returncode, 0, proc.stderr)
            for name in ("p00.out", "p01.out"):
                first = (out1 / name).read_bytes()
                second = (out2 / name).read_bytes()
                self.assertEqual(first, second, "%s not deterministic" % name)
                self.assertLessEqual(len(first), 65536)

    def test_malformed_case_does_not_crash_run(self):
        repo = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            out_dir = Path(tmp) / "out"
            in_dir.mkdir()
            (in_dir / "bad.in").write_text("{not json")
            kit = Path("/home/hatch/workspace/makefun/kit/gta6-trophy-rarity-forecast-01"
                       "/gta6-trophy-rarity-forecast-01-kit/input")
            (in_dir / "p00.in").write_text((kit / "p00.in").read_text())
            env = dict(os.environ)
            env["MAKEFUN_INPUT_DIR"] = str(in_dir)
            env["MAKEFUN_OUTPUT_DIR"] = str(out_dir)
            proc = subprocess.run(
                [sys.executable, str(repo / "main.py")],
                env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue((out_dir / "p00.out").exists())


if __name__ == "__main__":
    unittest.main()
