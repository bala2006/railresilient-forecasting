# RailResilient interactive simulation

This is a local, browser-based demonstration of the RailResilient idea. It renders a small railway network, moves simulated services through it, injects unreliable-feed scenarios, and visualizes uncertainty-aware delay forecasts.

## Run locally

From the repository root:

```bash
uv run python demo/server.py
```

Then open <http://127.0.0.1:8765>.

The clean checkout uses a deterministic **simulation fallback** so the demo does not require private data or a checkpoint. To use a compatible trained checkpoint, supply the checkpoint and matching normalization metadata:

```bash
uv run python demo/server.py \
  --config configs/pilot_v4_r4s_top2_seed20260908.json \
  --checkpoint artifacts/<run>/checkpoints/r3s_moe.pt \
  --normalization artifacts/<run>/normalization.json
```

The v4 checkpoint must be trained from `configs/pilot_v4_r4s_top2_seed20260908.json`; older three-expert/top-1 checkpoints are intentionally not treated as compatible. The UI identifies whether it is using R4S-MoE or the fallback simulator.

## Temporary public preview

With [Cloudflare `cloudflared`](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) installed, run:

```bash
bash demo/public-tunnel.sh
```

The command starts the local server and prints a random `trycloudflare.com` URL. Keep that terminal running while sharing the preview; Quick Tunnels are temporary and have no uptime guarantee.

## What to try

1. Click **Run simulation** to animate services across the network.
2. Select **Packet loss**, **Stale feed**, **Outage**, or **Combined** in the editor.
3. Move the sliders to create a custom disruption.
4. Click a train on the map to inspect its service.
5. Watch the feed-quality vector, confidence, uncertainty band, and event stream update.

This is an interactive research demonstration, not a live railway connection. It must not be used for train control, signaling, dispatch automation, or safety-critical decisions.
