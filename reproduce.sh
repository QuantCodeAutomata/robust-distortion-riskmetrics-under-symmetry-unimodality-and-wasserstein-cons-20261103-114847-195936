#!/usr/bin/env bash
# Run experiments from the repository root.

set -u

python -m pytest -m "not market_data" -v
python -m exp.exp1
python -m exp.exp2
