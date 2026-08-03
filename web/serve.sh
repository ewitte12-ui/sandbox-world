#!/usr/bin/env bash
# Serves the built web/dist/ over HTTP.
#
# Required because browsers refuse to load ES modules and WebAssembly from
# file:// — opening web/dist/index.html by double-clicking will not work.
#
# Usage:  web/serve.sh [port]
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${1:-8080}"
DIST="web/dist"

if [[ ! -f "$DIST/metalworld.js" || ! -f "$DIST/metalworld_bg.wasm" ]]; then
  echo "error: $DIST is missing the built engine. Run web/build.sh first." >&2
  exit 1
fi

echo "Serving $DIST at http://localhost:${PORT}/"
echo "(Ctrl-C to stop)"
exec python3 -m http.server -d "$DIST" "$PORT"
