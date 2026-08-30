#!/usr/bin/env sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_DIR"
if ! command -v node >/dev/null 2>&1; then
  printf '%s\n' 'Node.js 22.21 or newer is required.'
  exit 1
fi
printf '%s\n' 'XiriaCanvas AI Environment Setup'
printf '%s\n' 'A browser window will open. Choose Auto or Manual setup, then click Start.'
printf '%s\n' 'The wizard queries the PyTorch catalog first — nothing is installed yet.'
printf '%s\n' 'Downloads use a direct connection by default; use --use-proxy only when needed.'
exec node scripts/setup-gui.mjs "$@"
