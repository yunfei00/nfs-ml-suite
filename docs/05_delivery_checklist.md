# Delivery Checklist (Commercial)

## Repo must contain only:
- src/ (code)
- configs/ (yaml)
- docs/ (md)
- scripts/ (run helpers)
- README, pyproject, requirements

## Must NOT contain:
- raw/ processed/ runs/ exports/ cache/
- any .npz/.pt/.csv/.png large artifacts

## Workdir packaging for a customer (optional)
If customer needs example data/runs, deliver as a separate archive:
- `workdir_sample.zip` containing:
  - processed/small_sample.npz
  - runs/demo_run_id/ (optional)
  - exports/ (optional)
