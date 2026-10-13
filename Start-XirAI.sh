#!/usr/bin/env sh
set -u
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_DIR"
if ! command -v node >/dev/null 2>&1; then
  printf '%s\n' 'Startup failed: Node.js is missing.' 'Please run Setup-XirAI first.'
  exit 1
fi
exec node scripts/start.mjs
