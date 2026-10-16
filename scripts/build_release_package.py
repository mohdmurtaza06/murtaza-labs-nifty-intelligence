#!/usr/bin/env python3
"""Build one reproducible, source-only research release ZIP from tracked files.

Runtime market data and root model binaries are intentionally not fabricated or
replaced with legacy artifacts. The older bundled app is preserved in a clearly
labelled legacy directory inside the ZIP.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
PACKAGE_PREFIX = "Murtaza_Labs_NIFTY_Intelligence"
REQUIRED_PATHS = {
    "README.md",
    "RELEASE_READINESS.md",
    "requirements.txt",
    ".github/workflows/ci.yml",
    "backend/main.py",
    "notebooks/15_live_prediction_production.py",
    "notebooks/27_build_production_model.py",
    "notebooks/28_time_split_validation.py",
    "src/readiness_checks.py",
    "tests/test_api_safety.py",
    "web/package.json",
    "web/package-lock.json",
    "web/src/App.jsx",
    "vercel.json",
    "web/vercel.json",
}

EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "dist",
    "build",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tmp", ".swp", ".bak"}
SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}
SECRET_NAME_PARTS = {"credentials", "secrets", "id_rsa", "id_ed25519"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tracked_paths() -> list[str]:
    raw = subprocess.check_output(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
    )
    return sorted(item.decode("utf-8") for item in raw.split(b"\0") if item)


def included_path(path: str) -> str | None:
    pure = PurePosixPath(path)
    lower_name = pure.name.lower()

    if path == "StockMarket_UI_Updated.zip":
        return None  # The uncompressed legacy snapshot is preserved separately.
    if any(part in EXCLUDED_PARTS for part in pure.parts):
        return None
    if pure.suffix.lower() in EXCLUDED_SUFFIXES:
        return None
    if lower_name == ".env" or (lower_name.startswith(".env.") and lower_name != ".env.example"):
        return None
    if pure.suffix.lower() in SECRET_SUFFIXES:
        return None
    if any(term in lower_name for term in SECRET_NAME_PARTS):
        return None

    if path.startswith("StockMarket_UI_Updated/"):
        return "legacy/" + path
    return path


def main() -> int:
    missing = sorted(path for path in REQUIRED_PATHS if not (ROOT / path).is_file())
    if missing:
        print("Cannot package release; required source files are missing:", file=sys.stderr)
        print("\n".join(f"  - {path}" for path in missing), file=sys.stderr)
        return 2

    tracked = tracked_paths()
    selected: list[tuple[str, Path, bytes]] = []
    seen_archive_paths: set[str] = set()

    for repository_path in tracked:
        archive_path = included_path(repository_path)
        if archive_path is None:
            continue
        source = ROOT / repository_path
        if not source.is_file():
            continue
        payload = source.read_bytes()
        if archive_path in seen_archive_paths:
            raise RuntimeError(f"Duplicate archive path: {archive_path}")
        seen_archive_paths.add(archive_path)
        selected.append((archive_path, source, payload))

    legacy_note_path = "legacy/README.md"
    legacy_note = (
        "# Preserved legacy snapshot\n\n"
        "This directory contains the historical `StockMarket_UI_Updated` "
        "snapshot preserved for reference. It is not the active application "
        "and its bundled model/market data must not be treated as current or "
        "compatible with the root research pipeline. The active source is at "
        "the package root.\n"
    ).encode("utf-8")
    if legacy_note_path not in seen_archive_paths:
        selected.append((legacy_note_path, ROOT / "RELEASE_READINESS.md", legacy_note))
        seen_archive_paths.add(legacy_note_path)

    paths = {archive_path for archive_path, _, _ in selected}
    active_data_present = "data/live_market_data.csv" in paths
    canonical_schema_present = "data/production_model_schema.json" in paths
    active_research_model_present = "models/nifty50_research_candidate.pkl" in paths
    validation_report_present = "reports/validation/nifty50_time_split_validation.json" in paths
    paper_ledger_present = "data/prediction_history.csv" in paths

    generated_at = datetime.now(timezone.utc).isoformat()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError:
        commit = "unknown"

    manifest = {
        "schema_version": 1,
        "package_type": "SOURCE_RELEASE_RESEARCH_ONLY",
        "generated_at_utc": generated_at,
        "git_commit": commit,
        "ci_status": os.environ.get("RELEASE_CI_STATUS", "not_recorded"),
        "active_application_roots": ["backend/", "notebooks/", "src/", "tests/", "web/"],
        "legacy_snapshot": "legacy/StockMarket_UI_Updated/",
        "file_count_excluding_manifest": len(selected),
        "files_sha256": {
            f"{PACKAGE_PREFIX}/{archive_path}": sha256(payload)
            for archive_path, _, payload in sorted(selected, key=lambda item: item[0])
        },
        "evidence": {
            "active_root_market_data_included": active_data_present,
            "canonical_172_feature_manifest_included": canonical_schema_present,
            "active_research_model_binary_included": active_research_model_present,
            "chronological_validation_report_included": validation_report_present,
            "active_prospective_paper_ledger_included": paper_ledger_present,
        },
        "readiness": {
            "trading_mode": "RESEARCH_ONLY",
            "trade_status": "NO_TRADE",
            "production_eligible": False,
            "live_orders_enabled": False,
            "real_money_approval": "NO",
            "automatic_vercel_deployments_enabled": False,
        },
        "limitations": [
            "Source and tests are packaged; active-root runtime data/model artifacts are not available in the repository snapshot.",
            "The legacy snapshot is preserved for reference only and is not a substitute for the active model.",
            "A passing CI build is not proof of model profitability or investment readiness.",
            "Fresh NSE-session data, canonical feature parity, untouched holdout, calibration, 200 prospective closed paper signals, independent NIFTYBEES reconciliation, and operational/compliance approval remain evidence gates.",
        ],
        "manifest_note": "File hashes cover packaged source/data files other than this manifest itself and the ZIP checksum sidecar.",
    }

    DIST.mkdir(parents=True, exist_ok=True)
    date_part = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    zip_path = DIST / f"MURTAZA_LABS_NIFTY_INTELLIGENCE_SOURCE_RELEASE_{date_part}.zip"
    manifest_path = DIST / "RELEASE_MANIFEST.json"
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    members = selected + [("RELEASE_MANIFEST.json", manifest_path, manifest_bytes)]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for archive_path, _source, payload in sorted(members, key=lambda item: item[0]):
            info = zipfile.ZipInfo(
                f"{PACKAGE_PREFIX}/{archive_path}",
                date_time=datetime.now().timetuple()[:6],
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)

    digest = sha256(zip_path.read_bytes())
    checksum_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")

    # Verify the ZIP central directory, CRCs, required release markers and manifest.
    with zipfile.ZipFile(zip_path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP integrity check failed at {bad_member}")
        names = set(archive.namelist())
        expected = {
            f"{PACKAGE_PREFIX}/README.md",
            f"{PACKAGE_PREFIX}/RELEASE_READINESS.md",
            f"{PACKAGE_PREFIX}/RELEASE_MANIFEST.json",
            f"{PACKAGE_PREFIX}/backend/main.py",
            f"{PACKAGE_PREFIX}/web/src/App.jsx",
            f"{PACKAGE_PREFIX}/legacy/README.md",
        }
        if not expected.issubset(names):
            raise RuntimeError(f"ZIP missing release members: {sorted(expected - names)}")
        stored_manifest = json.loads(
            archive.read(f"{PACKAGE_PREFIX}/RELEASE_MANIFEST.json").decode("utf-8")
        )
        if stored_manifest.get("git_commit") != commit:
            raise RuntimeError("Release manifest commit mismatch.")

    print(f"Release package : {zip_path}")
    print(f"SHA-256         : {digest}")
    print(f"Tracked source  : {len(selected)} files")
    print(f"Git commit      : {commit}")
    print(f"CI status       : {manifest['ci_status']}")
    print("Trading state   : RESEARCH_ONLY / NO_TRADE")
    print("Live orders     : DISABLED")
    print("ZIP integrity   : PASS")
    print(f"Checksum sidecar: {checksum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
