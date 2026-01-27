#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 demo.py
python3 ../../tools/authority_build.py --out out
python3 ../../tools/do_readiness.py --out out
python3 ../../tools/authority_propagate_do_check.py --out out

echo ""
echo "Done. Outputs are in the ./out folder."
