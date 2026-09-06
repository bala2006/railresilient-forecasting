const QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95];
const QUALITY_FIELDS = [
  "missing_fraction",
  "stale_fraction",
  "observation_age_saturating",
  "declared_delay_saturating",
  "inconsistent_flag",
  "duplicate_flag",
  "no_fresh_observation",
];

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, Number.isFinite(value) ? value : low));
}

function valueOf(input, fallback) {
  const value = Number(input);
  return Number.isFinite(value) ? value : fallback;
}

function buildQuality(scenario, currentDelay) {
  const loss = clamp(valueOf(scenario.packetLoss, 0) / 100, 0, 1);
  const stale = clamp(valueOf(scenario.staleMinutes, 0) / 10, 0, 1);
  const outage = clamp(valueOf(scenario.outageMinutes, 0) / 15, 0, 1);
  const inconsistent = clamp(valueOf(scenario.inconsistentRate, 0) / 100, 0, 1);
  const duplicate = clamp(valueOf(scenario.duplicateRate, 0) / 100, 0, 1);
  const declared = clamp(Math.abs(currentDelay) / 900, 0, 1);
  const noFresh = outage > 0.55 || stale >= 1 ? 1 : 0;
  return [loss, stale, Math.max(stale, outage), declared, inconsistent, duplicate, noFresh];
}

function forecast(payload) {
  const scenario = payload?.scenario || {};
  const currentDelay = clamp(valueOf(scenario.currentDelay, 74), -120, 900);
  const trend = clamp(valueOf(scenario.trendPerEvent, 12), -90, 90);
  const quality = buildQuality(scenario, currentDelay);
  const loss = quality[0];
  const stale = quality[1];
  const outage = quality[2];
  const mask = Array(8).fill(1);
  const missingCount = Math.min(7, Math.round(loss * 8));
  const outageCount = Math.min(7, Math.round(outage * 8));
  for (let index = 0; index < missingCount; index += 1) mask[index] = 0;
  for (let index = 8 - outageCount; index < 8; index += 1) mask[index] = 0;

  const qualityScore = clamp(
    100 * (1 - (0.28 * loss + 0.24 * stale + 0.22 * outage + 0.14 * quality[4] + 0.12 * quality[5])),
    0,
    100,
  );
  const uncertainty = 18 + Math.abs(trend) * 0.22 + loss * 92 + stale * 32 + outage * 58 + quality[4] * 45 + quality[5] * 18 + quality[6] * 32;
  const quantiles = Array.from({ length: 4 }, (_, index) => {
    const median = currentDelay + trend * (index + 1);
    const width = uncertainty * (1 + 0.08 * index);
    return [
      median - width * 1.95,
      median - width * 1.55,
      median - width * 0.82,
      median,
      median + width * 0.82,
      median + width * 1.55,
      median + width * 1.95,
    ].map((value) => Math.round(value * 10) / 10);
  });
  const medians = quantiles.map((row) => row[3]);
  const spreads = quantiles.map((row) => Math.round((row[6] - row[0]) * 10) / 10);
  const alert = qualityScore < 65 || Math.max(...spreads) > 500;
  const alertText = qualityScore < 65
    ? "Feed quality is degraded; treat the forecast as advisory."
    : Math.max(...spreads) > 500
      ? "Forecast interval is wide; use the output for triage only."
      : "No immediate reliability alert.";

  return {
    mode: "Cloudflare demo fallback",
    modelLabel: "Persistence + uncertainty simulator",
    modelMessage: "This public preview runs the deterministic fallback at the Cloudflare edge; no private checkpoint is exposed.",
    quality: Object.fromEntries(QUALITY_FIELDS.map((name, index) => [name, Math.round(quality[index] * 1000) / 1000])),
    qualityScore: Math.round(qualityScore * 10) / 10,
    mask,
    quantileLevels: QUANTILES,
    quantiles,
    medians,
    spreads,
    router: { expert: null, probabilities: [] },
    latencyMs: 0.2,
    alert,
    alertText,
  };
}

function jsonResponse(value, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
      "access-control-allow-origin": "*",
    },
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/api/health") {
      return jsonResponse({ ok: true, model: "Cloudflare demo fallback" });
    }
    if (url.pathname === "/api/predict") {
      if (request.method !== "POST") return jsonResponse({ error: "POST required" }, 405);
      try {
        return jsonResponse(forecast(await request.json()));
      } catch (error) {
        return jsonResponse({ error: String(error) }, 400);
      }
    }
    if (url.pathname === "/" || url.pathname === "/index.html") {
      return env.ASSETS.fetch(new Request(new URL("/index.html", request.url), request));
    }
    if (url.pathname.startsWith("/static/")) {
      const assetUrl = new URL(request.url);
      assetUrl.pathname = url.pathname.slice("/static".length) || "/index.html";
      return env.ASSETS.fetch(new Request(assetUrl, request));
    }
    return env.ASSETS.fetch(request);
  },
};
