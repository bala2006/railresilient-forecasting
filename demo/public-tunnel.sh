#!/usr/bin/env bash
set -euo pipefail

# Run this from the repository root. cloudflared prints a random public URL.
if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared is required: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/" >&2
  exit 1
fi

uv run python demo/server.py >/tmp/railresilient-demo.log 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT

sleep 1
exec cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8765
