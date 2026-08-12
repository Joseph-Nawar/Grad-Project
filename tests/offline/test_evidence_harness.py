from __future__ import annotations

from scripts.run_stage4_evidence import run_campaigns


def test_stage4_campaign_harness_reports_zero_loss_and_duplicate_effects(tmp_path) -> None:
    report = run_campaigns(tmp_path, cycles=2, replays=3)

    assert report["disconnect_reconnect"]["lost_queued_submissions"] == 0
    assert report["duplicate_replay"]["duplicate_central_effects"] == 0
    assert report["disconnect_reconnect"]["cycles"] == 2
    assert report["duplicate_replay"]["replays"] == 3
