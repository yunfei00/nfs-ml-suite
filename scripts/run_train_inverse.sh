#!/usr/bin/env bash
set -e
python -m nfs_ml train --task inverse --config configs/model_inverse.yaml
