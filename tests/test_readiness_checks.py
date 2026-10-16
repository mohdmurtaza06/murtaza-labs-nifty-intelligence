"""Regression tests for fail-closed prediction safety checks."""
import unittest
from datetime import date

import pandas as pd

from src.readiness_checks import (
    target_from_next_returns,
    validate_market_data_freshness,
    validate_probability_pair,
)


class TargetLabelTests(unittest.TestCase):
    def test_missing_next_return_is_not_labeled_down(self):
        returns = pd.Series([0.01, -0.02, float("nan")])
        labels = target_from_next_returns(returns)
        self.assertEqual(labels.iloc[0], 1.0)
        self.assertEqual(labels.iloc[1], 0.0)
        self.assertTrue(pd.isna(labels.iloc[2]))

    def test_infinite_return_is_not_labeled(self):
        labels = target_from_next_returns(pd.Series([float("inf"), 0.0]))
        self.assertTrue(pd.isna(labels.iloc[0]))
        self.assertEqual(labels.iloc[1], 0.0)


class FreshnessTests(unittest.TestCase):
    def test_current_and_normal_weekend_gap_pass(self):
        self.assertEqual(
            validate_market_data_freshness(
                "2026-10-16", now=date(2026, 10, 16), max_age_days=4
            ),
            0,
        )
        self.assertEqual(
            validate_market_data_freshness(
                "2026-10-16", now=date(2026, 10, 19), max_age_days=4
            ),
            3,
        )

    def test_stale_and_future_observations_fail(self):
        with self.assertRaisesRegex(ValueError, "stale"):
            validate_market_data_freshness(
                "2026-10-01", now=date(2026, 10, 16), max_age_days=4
            )
        with self.assertRaisesRegex(ValueError, "future"):
            validate_market_data_freshness(
                "2026-10-17", now=date(2026, 10, 16), max_age_days=4
            )


class ProbabilityTests(unittest.TestCase):
    def test_valid_pair_passes(self):
        self.assertEqual(validate_probability_pair(0.7, 0.3), (0.7, 0.3))

    def test_invalid_pairs_fail(self):
        for up, down in ((1.2, -0.2), (float("nan"), 1.0), (0.7, 0.2)):
            with self.subTest(up=up, down=down):
                with self.assertRaises(ValueError):
                    validate_probability_pair(up, down)


if __name__ == "__main__":
    unittest.main()
