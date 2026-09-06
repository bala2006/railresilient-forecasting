"""Local HTTP service for the RailResilient interactive simulation demo.

The service is intentionally dependency-light. It always supports a deterministic
simulation fallback and only loads R3S-MoE when an explicit compatible checkpoint
and normalization file are provided.
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
QUALITY_FIELDS = (
    "missing_fraction",
    "stale_fraction",
    "observation_age_saturating",
    "declared_delay_saturating",
    "inconsistent_flag",
    "duplicate_flag",
    "no_fresh_observation",
)
QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def number(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def build_quality(scenario: dict[str, Any], current_delay: float) -> list[float]:
    loss = clamp(number(scenario.get("packetLoss"), 0.0) / 100.0, 0.0, 1.0)
    stale = clamp(number(scenario.get("staleMinutes"), 0.0) / 10.0, 0.0, 1.0)
    outage = clamp(number(scenario.get("outageMinutes"), 0.0) / 15.0, 0.0, 1.0)
    inconsistent = clamp(number(scenario.get("inconsistentRate"), 0.0) / 100.0, 0.0, 1.0)
    duplicate = clamp(number(scenario.get("duplicateRate"), 0.0) / 100.0, 0.0, 1.0)
    declared = clamp(abs(current_delay) / 900.0, 0.0, 1.0)
    no_fresh = 1.0 if outage > 0.55 or stale >= 1.0 else 0.0
    return [loss, stale, max(stale, outage), declared, inconsistent, duplicate, no_fresh]


def simulated_router(current_delay: float, trend: float, quality: list[float]) -> dict[str, Any]:
    """Provide honest, labeled routing visuals when no learned checkpoint is available."""
    scores = [
        0.35 + max(0.0, 1.0 - quality[0] - quality[1]),
        0.25 + min(1.0, abs(current_delay) / 300.0) + max(0.0, trend / 90.0),
        0.20 + quality[0] + quality[1] + quality[2],
        0.20 + max(0.0, -trend / 60.0),
    ]
    total = sum(scores)
    probabilities = [score / total for score in scores]
    selected = sorted(range(4), key=lambda index: probabilities[index], reverse=True)[:2]
    selected_total = sum(probabilities[index] for index in selected)
    experts = [
        {"id": index + 1, "weight": round(probabilities[index] / selected_total, 3)}
        for index in selected
    ]
    return {
        "expert": experts[0]["id"],
        "experts": experts,
        "topK": 2,
        "probabilities": [round(value, 3) for value in probabilities],
    }


class ModelAdapter:
    """Optional CPU checkpoint adapter with a deterministic fallback."""

    def __init__(
        self,
        config_path: Path | None,
        checkpoint_path: Path | None,
        normalization_path: Path | None,
    ) -> None:
        self.model: Any = None
        self.normalization: Any = None
        self.torch_batch: Any = None
        self.mode = "simulation fallback"
        self.model_label = "Persistence + uncertainty simulator"
        self.message = "No checkpoint supplied; using a transparent persistence-based simulator."
        if checkpoint_path is not None:
            self._try_load(config_path, checkpoint_path, normalization_path)

    def _try_load(
        self,
        config_path: Path | None,
        checkpoint_path: Path,
        normalization_path: Path | None,
    ) -> None:
        if not checkpoint_path.exists():
            self.message = f"Checkpoint not found: {checkpoint_path}; using simulation fallback."
            return
        if config_path is None or not config_path.exists():
            self.message = "A model checkpoint needs its matching JSON config; using simulation fallback."
            return
        if normalization_path is None or not normalization_path.exists():
            self.message = "A model checkpoint needs matching normalization metadata; using simulation fallback."
            return
        try:
            sys.path.insert(0, str(ROOT / "src"))
            import torch
            from railresilient.models import Normalization, R3SMoE, torch_batch

            config = json.loads(config_path.read_text(encoding="utf-8"))
            normalization = Normalization(**json.loads(normalization_path.read_text(encoding="utf-8")))
            model = R3SMoE(config)
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            self.model = model
            self.normalization = normalization
            self.torch_batch = torch_batch
            self.mode = "R4S-MoE checkpoint" if int(config["model"].get("routing_top_k", 1)) == 2 and int(config["model"].get("num_experts", 0)) == 4 else "R3S-MoE checkpoint"
            self.model_label = self.mode.removesuffix(" checkpoint")
            self.message = f"Loaded the supplied CPU {self.model_label} checkpoint and matching normalization metadata."
        except Exception as error:  # A demo should remain usable if local artifacts are incompatible.
            self.model = None
            self.normalization = None
            self.torch_batch = None
            self.mode = "simulation fallback"
            self.model_label = "Persistence + uncertainty simulator"
            self.message = f"Checkpoint could not be loaded ({type(error).__name__}); using simulation fallback."

    def predict(
        self,
        current_delay: float,
        trend: float,
        mask: list[float],
        quality: list[float],
    ) -> tuple[list[list[float]], dict[str, Any], float] | None:
        if self.model is None or self.normalization is None or self.torch_batch is None:
            return None
        try:
            import numpy as np
            import torch
            from railresilient.data import ArrayDataset

            context = 8
            horizons = 4
            past = np.asarray(
                [current_delay - trend * (context - 1 - index) for index in range(context)],
                dtype=np.float32,
            )[None, :]
            data = ArrayDataset(
                past_delay=past,
                past_planned_delta=np.zeros((1, context), dtype=np.float32),
                past_observed_delta=np.zeros((1, context), dtype=np.float32),
                past_event_type=np.tile(np.asarray([0, 1, 2, 1, 0, 1, 2, 1], dtype=np.int64), (1, 1)),
                past_station=np.tile(np.asarray([12, 24, 36, 48, 60, 72, 84, 96], dtype=np.int64), (1, 1)),
                observation_mask=np.asarray([mask], dtype=np.float32),
                future_planned_delta=np.asarray([[300.0, 600.0, 900.0, 1200.0]], dtype=np.float32),
                future_observed_ns=np.zeros((1, horizons), dtype=np.int64),
                target_delay=np.zeros((1, horizons), dtype=np.float32),
                forecast_origin_ns=np.zeros(1, dtype=np.int64),
                day_ordinal=np.zeros(1, dtype=np.int64),
                regime=np.zeros(1, dtype=np.int64),
                current_delay=np.asarray([current_delay], dtype=np.float32),
            )
            quality_array = np.asarray([quality], dtype=np.float32)
            batch = self.torch_batch(
                data, quality_array, np.asarray([0]), self.normalization, torch.device("cpu")
            )
            started = time.perf_counter()
            with torch.no_grad():
                output = self.model(batch)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            raw = output["quantiles"].cpu().numpy()[0]
            quantiles = (raw * self.normalization.delay_std + self.normalization.delay_mean).tolist()
            assignments = output["router_assignments"].cpu().numpy().tolist()
            probabilities = output["router_probabilities"].cpu().numpy()[0].tolist()
            topk_indices = output.get("router_topk_assignments")
            topk_weights = output.get("router_topk_weights")
            experts = []
            if topk_indices is not None and topk_weights is not None:
                indices = topk_indices.cpu().numpy()[0].tolist()
                weights = topk_weights.cpu().numpy()[0].tolist()
                experts = [{"id": int(index) + 1, "weight": round(float(weight), 3)} for index, weight in zip(indices, weights)]
            return quantiles, {
                "expert": int(assignments[0]) + 1,
                "experts": experts,
                "topK": len(experts),
                "probabilities": [round(float(item), 3) for item in probabilities],
            }, elapsed_ms
        except Exception:
            return None


class DemoService:
    def __init__(
        self,
        config_path: Path | None = None,
        checkpoint_path: Path | None = None,
        normalization_path: Path | None = None,
    ) -> None:
        self.adapter = ModelAdapter(config_path, checkpoint_path, normalization_path)

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        scenario = payload.get("scenario") or {}
        current_delay = clamp(number(scenario.get("currentDelay"), 74.0), -120.0, 900.0)
        trend = clamp(number(scenario.get("trendPerEvent"), 12.0), -90.0, 90.0)
        quality = build_quality(scenario, current_delay)
        loss = quality[0]
        stale = quality[1]
        outage = quality[2]
        mask = [1.0] * 8
        missing_count = min(7, round(loss * 8))
        for index in range(missing_count):
            mask[index] = 0.0
        outage_count = min(7, round(outage * 8))
        for index in range(8 - outage_count, 8):
            mask[index] = 0.0
        quality_score = clamp(
            100.0
            * (1.0 - (0.28 * loss + 0.24 * stale + 0.22 * outage + 0.14 * quality[4] + 0.12 * quality[5])),
            0.0,
            100.0,
        )

        model_result = self.adapter.predict(current_delay, trend, mask, quality)
        if model_result is not None:
            quantiles, router, latency_ms = model_result
            mode = self.adapter.mode
            model_label = self.adapter.model_label
        else:
            uncertainty = 18.0 + abs(trend) * 0.22 + loss * 92.0 + stale * 32.0 + outage * 58.0
            uncertainty += quality[4] * 45.0 + quality[5] * 18.0 + quality[6] * 32.0
            quantiles = []
            for horizon in range(1, 5):
                median = current_delay + trend * horizon
                width = uncertainty * (1.0 + 0.08 * (horizon - 1))
                quantiles.append(
                    [
                        median - width * 1.95,
                        median - width * 1.55,
                        median - width * 0.82,
                        median,
                        median + width * 0.82,
                        median + width * 1.55,
                        median + width * 1.95,
                    ]
                )
            router = simulated_router(current_delay, trend, quality)
            latency_ms = 0.3
            mode = "simulation fallback"
            model_label = "Persistence + uncertainty simulator"

        medians = [round(float(row[3]), 1) for row in quantiles]
        spreads = [round(float(row[6] - row[0]), 1) for row in quantiles]
        alert = quality_score < 65.0 or max(spreads) > 500.0
        if quality_score < 65.0:
            alert_text = "Feed quality is degraded; treat the forecast as advisory."
        elif max(spreads) > 500.0:
            alert_text = "Forecast interval is wide; use the output for triage only."
        else:
            alert_text = "No immediate reliability alert."
        return {
            "mode": mode,
            "modelLabel": model_label,
            "modelMessage": self.adapter.message,
            "quality": {
                name: round(float(value), 3) for name, value in zip(QUALITY_FIELDS, quality)
            },
            "qualityScore": round(quality_score, 1),
            "mask": mask,
            "quantileLevels": QUANTILES,
            "quantiles": [[round(float(value), 1) for value in row] for row in quantiles],
            "medians": medians,
            "spreads": spreads,
            "router": router,
            "latencyMs": round(float(latency_ms), 2),
            "alert": alert,
            "alertText": alert_text,
        }


class RequestHandler(BaseHTTPRequestHandler):
    service: DemoService

    def _headers(self, content_type: str) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")

    def _json(self, status: HTTPStatus, value: dict[str, Any]) -> None:
        body = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self._headers("application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/health":
            self._json(HTTPStatus.OK, {"ok": True, "model": self.service.adapter.mode})
            return
        if path in {"/", "/index.html"}:
            target = STATIC_ROOT / "index.html"
        elif path.startswith("/static/"):
            target = STATIC_ROOT / path.removeprefix("/static/")
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        try:
            resolved = target.resolve()
            resolved.relative_to(STATIC_ROOT.resolve())
            body = resolved.read_bytes()
        except (OSError, ValueError):
            self._json(HTTPStatus.NOT_FOUND, {"error": "Asset not found"})
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self._headers(content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/predict":
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 64_000:
                raise ValueError("request is too large")
            payload = json.loads(self.rfile.read(length) or b"{}")
            result = self.service.predict(payload)
            self._json(HTTPStatus.OK, result)
        except (ValueError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def log_message(self, format: str, *args: Any) -> None:
        # Keep the demo terminal readable while retaining useful request diagnostics.
        print(f"[demo] {self.address_string()} - {format % args}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the RailResilient simulation demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "pilot_v3_seed20260908.json")
    parser.add_argument("--checkpoint", type=Path, help="Optional trained r3s_moe.pt checkpoint")
    parser.add_argument("--normalization", type=Path, help="Optional matching normalization.json")
    args = parser.parse_args()
    service = DemoService(args.config, args.checkpoint, args.normalization)
    RequestHandler.service = service
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f"RailResilient demo: http://{args.host}:{args.port}")
    print(f"Forecast mode: {service.adapter.mode} — {service.adapter.message}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping demo.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
