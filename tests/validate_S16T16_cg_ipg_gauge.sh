#!/usr/bin/env bash

conda activate pygpt
python scripts/validate_cg_ipg.py \
  --cg-dir ensemble/S16T16_cg/gauge \
  --ipg-dir ensemble/S16T16_cg_ipg/gauge \
  --glob 'wilson_b6.cg.1e-08.*' \
  --projection-method cabibbo-marinari \
  --spread-tol 1e-10 \
  --boundary-tol 1e-10
