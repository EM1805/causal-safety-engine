#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 demo.py

echo ""
echo "Done. Outputs are in the ./out folder."
