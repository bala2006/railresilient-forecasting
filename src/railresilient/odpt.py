"""Token-gated ODPT access checks and snapshot collection."""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

TOKYO_METRO_STATIC = "https://api.odpt.org/api/v4/files/TokyoMetro/data/TokyoMetro-Train-GTFS.zip"
TOKYO_METRO_ALERT = "https://api.odpt.org/api/v4/gtfs/realtime/tokyometro_odpt_train_alert"


class ODPTAccessError(RuntimeError):
    """Raised when ODPT access is attempted without an approved token."""


def access_token() -> str:
    token = os.environ.get("ODPT_ACCESS_TOKEN", "").strip()
    if not token:
        raise ODPTAccessError(
            "ODPT_ACCESS_TOKEN is not set. Register with ODPT and review provider terms before collection."
        )
    return token


def check_access(timeout: int = 30) -> dict[str, Any]:
    token = access_token()
    results: dict[str, Any] = {}
    for name, url in {"tokyo_metro_static": TOKYO_METRO_STATIC, "tokyo_metro_alert": TOKYO_METRO_ALERT}.items():
        response = requests.get(url, params={"acl:consumerKey": token}, timeout=timeout)
        results[name] = {
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type"),
            "content_length": len(response.content),
        }
    return results


def collect_alert_snapshot(output_root: str | Path = "data/private/odpt", timeout: int = 30) -> Path:
    token = access_token()
    response = requests.get(
        TOKYO_METRO_ALERT, params={"acl:consumerKey": token}, timeout=timeout
    )
    response.raise_for_status()
    now = datetime.now(UTC)
    digest = hashlib.sha256(response.content).hexdigest()
    output = Path(output_root) / now.strftime("%Y/%m/%d") / f"{now:%H%M%S}_{digest[:12]}.pb"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(response.content)
    return output
