#!/usr/bin/env bash
set -euo pipefail

python -m pytest -q \
  tests/test_rain_lab_launcher.py \
  tests/test_rain_lab_runtime.py \
  tests/test_rain_health_check.py \
  tests/test_rain_lab_backup.py \
  tests/test_rain_first_run.py
