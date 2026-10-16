"""API safety regression tests.

These tests exercise guard functions directly; none starts the scheduler,
background pipeline, broker, or order execution.
"""
import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from starlette.requests import Request

from backend.main import health, production_prediction, readiness, run_pipeline


def make_request(headers=None):
    headers = headers or {}
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/run-pipeline",
            "raw_path": b"/api/run-pipeline",
            "query_string": b"",
            "headers": [
                (str(key).lower().encode("latin-1"), str(value).encode("latin-1"))
                for key, value in headers.items()
            ],
            "client": ("testclient", 123),
            "server": ("testserver", 80),
        }
    )


class ReadinessEndpointTests(unittest.TestCase):
    def test_health_is_not_production_approval(self):
        value = health()
        self.assertEqual(value["trading_mode"], "RESEARCH_ONLY")
        self.assertFalse(value["production_eligible"])
        self.assertFalse(value["live_orders_enabled"])

    def test_readiness_fails_closed_when_release_evidence_is_unverified(self):
        value = readiness()
        self.assertEqual(value["readiness_status"], "NOT_READY")
        self.assertTrue(value["release_blocked"])
        self.assertEqual(value["trade_status"], "NO_TRADE")
        self.assertFalse(value["production_eligible"])
        self.assertFalse(value["live_orders_enabled"])
        self.assertEqual(
            value["gates"]["prospective_paper_signals"]["required_count"], 200
        )
        self.assertEqual(
            value["gates"]["prospective_paper_signals"]["verified_count"], 0
        )
        self.assertEqual(value["gates"]["real_money_approval"]["status"], "NOT_APPROVED")

    def test_production_prediction_is_unavailable_without_approval(self):
        with self.assertRaises(HTTPException) as raised:
            production_prediction()
        self.assertEqual(raised.exception.status_code, 503)

    def test_manual_pipeline_is_disabled_without_server_token(self):
        with patch.dict(os.environ, {"PIPELINE_TRIGGER_TOKEN": ""}):
            with self.assertRaises(HTTPException) as raised:
                run_pipeline(make_request())
        self.assertEqual(raised.exception.status_code, 503)

    def test_manual_pipeline_rejects_wrong_token(self):
        with patch.dict(os.environ, {"PIPELINE_TRIGGER_TOKEN": "server-only-secret"}):
            with self.assertRaises(HTTPException) as raised:
                run_pipeline(make_request({"X-Pipeline-Token": "wrong"}))
        self.assertEqual(raised.exception.status_code, 403)


if __name__ == "__main__":
    unittest.main()
