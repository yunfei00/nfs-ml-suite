# Install & Environment

## Python
- Python 3.10+ recommended
- GPU optional (CUDA-enabled PyTorch if needed)

## Install
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -U pip
pip install -e .
```

## Workdir (must be outside repo)
Set `NFS_WORKDIR` to a separate disk path and run:
```bash
python -m nfs_ml init-workdir
```
