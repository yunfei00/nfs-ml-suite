@echo off
setlocal
python -m nfs_ml train --task inverse --config configs\model_inverse.yaml
endlocal
