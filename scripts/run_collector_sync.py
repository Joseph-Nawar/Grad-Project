"""Run one synchronization worker for one named collector store."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import time

from rural_stroke_assist.client import ApiClient
from rural_stroke_assist.offline.http import ApiClientTransport
from rural_stroke_assist.offline.store import SQLiteOfflineStore
from rural_stroke_assist.offline.token import FileTokenProvider, StaticTokenProvider
from rural_stroke_assist.offline.worker import SyncWorker


def main() -> None:
    database = Path(os.getenv("RURALSTROKE_OFFLINE_DB", "/var/lib/ruralstroke/collector.sqlite3"))
    media = Path(os.getenv("RURALSTROKE_OFFLINE_MEDIA", "/var/lib/ruralstroke/media"))
    store = SQLiteOfflineStore(database, media)
    store.initialize()
    client = ApiClient()
    token_file = os.getenv("RURALSTROKE_API_TOKEN_FILE")
    token_provider = (
        FileTokenProvider(Path(token_file)) if token_file else StaticTokenProvider(client.token)
    )
    worker = SyncWorker(
        store=store,
        transport=ApiClientTransport(client),
        token_provider=token_provider,
        worker_id=f"collector-sync-{os.getpid()}",
    )
    while True:
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        worker.run_once(now=now)
        store.cleanup_synchronized_media(
            grace_seconds=float(os.getenv("RURALSTROKE_MEDIA_GRACE_SECONDS", "86400")),
            now=time.time(),
        )
        time.sleep(float(os.getenv("RURALSTROKE_SYNC_INTERVAL_SECONDS", "2")))


if __name__ == "__main__":
    main()
