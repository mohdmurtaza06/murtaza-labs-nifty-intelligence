"""Tests for disk-backed readiness evidence and fail-closed defaults."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.release_readiness import assess_readiness


class ReleaseReadinessTests(unittest.TestCase):
    def test_empty_project_is_not_ready_and_orders_remain_disabled(self):
        with tempfile.TemporaryDirectory() as temp:
            report = assess_readiness(temp)

        self.assertEqual(report["readiness_status"], "NOT_READY")
        self.assertTrue(report["release_blocked"])
        self.assertEqual(report["trading_mode"], "RESEARCH_ONLY")
        self.assertEqual(report["trade_status"], "NO_TRADE")
        self.assertFalse(report["production_eligible"])
        self.assertFalse(report["live_orders_enabled"])
        self.assertEqual(
            report["gates"]["fresh_market_data"]["status"], "NOT_VERIFIED"
        )
        self.assertEqual(
            report["gates"]["canonical_172_feature_model"]["status"], "NOT_VERIFIED"
        )
        self.assertEqual(
            report["gates"]["live_orders"]["status"], "DISABLED"
        )
        self.assertEqual(
            report["gates"]["prospective_paper_signals"]["verified_count"], 0
        )

    def _write_paper_ledger(self, base: Path, rows: int = 200, duplicate_ids=False):
        data = base / "data"
        data.mkdir(parents=True, exist_ok=True)
        items = []
        for i in range(rows):
            signal_id = f"signal-{i}"
            if duplicate_ids and i == rows - 1:
                signal_id = f"signal-0"
            items.append({
                "signal_id": signal_id,
                "signal_time": "2026-10-01T16:00:00+05:30",
                "data_through_date": "2026-10-01",
                "model_hash": "model-sha256",
                "feature_hash": "feature-sha256",
                "target_horizon": "next_session_open_to_close_direction",
                "is_prospective": True,
                "outcome_verified_at": "2026-10-02T17:00:00+05:30",
                "actual_open": 25000.0,
                "actual_close": 25050.0,
                "actual_return": 0.002,
                "simulated_costs": 0.002,
            })
        pd.DataFrame(items).to_csv(data / "paper_trading_ledger.csv", index=False)

    def test_exactly_200_auditable_prospective_closed_signals_pass_count_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            self._write_paper_ledger(Path(temp), rows=200)
            gate = assess_readiness(temp)["gates"]["prospective_paper_signals"]

        self.assertEqual(gate["status"], "PASS")
        self.assertEqual(gate["verified_count"], 200)

    def test_duplicate_signal_ids_do_not_count_toward_200(self):
        with tempfile.TemporaryDirectory() as temp:
            self._write_paper_ledger(Path(temp), rows=200, duplicate_ids=True)
            gate = assess_readiness(temp)["gates"]["prospective_paper_signals"]

        self.assertEqual(gate["status"], "FAIL")
        self.assertEqual(gate["verified_count"], 198)
        self.assertFalse(gate["unique_signal_ids"])

    def test_malformed_schema_fails_closed_without_crashing(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "data"
            data.mkdir()
            (data / "production_model_schema.json").write_text(
                '{"status":"CANONICAL","feature_count":"not-a-number","features":[]}',
                encoding="utf-8",
            )
            gate = assess_readiness(temp)["gates"]["canonical_172_feature_model"]

        self.assertEqual(gate["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
