#!/usr/bin/env bash
set -e
# Usage: ./run_predict_inverse.sh <run_id> <input_npz> <out_npz>
python -m nfs_ml predict --run "$1" --input "$2" --out "$3"
