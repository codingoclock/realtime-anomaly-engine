"""Integration test: run consumer with small count and confirm persistence."""
from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path
from time import sleep

from storage import database


def test_consumer_writes_db(tmp_path: Path):
    dbfile = str(tmp_path / "out_consumer.db")

    cmd = [
        sys.executable,
        "consumer/consumer.py",
        "--interval",
        "0.01",
        "--count",
        "3",
        "--seed",
        "42",
        "--db-path",
        dbfile,
        "--fill-missing-with-zero",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    proc = subprocess.run(cmd, env=env, cwd=".", capture_output=True, text=True, check=True)

    # Give a moment for file flush
    sleep(0.1)

    results = database.fetch_recent_anomalies(db_path=dbfile)
    assert len(results) == 3
