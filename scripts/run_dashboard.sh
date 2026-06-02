#!/usr/bin/env bash
# Run with: ./scripts/run_dashboard.sh
# Do NOT: source scripts/run_dashboard.sh  (exec would replace your shell)
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  echo "Do not source this script — it will replace your terminal." >&2
  echo "Run: ./scripts/run_dashboard.sh" >&2
  return 1 2>/dev/null || exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck source=/dev/null
source .venv/bin/activate

python -m pip install -q -e ".[dashboard]" 2>/dev/null || python -m pip install -q -e .
export PYTHONPATH=src

exec streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
