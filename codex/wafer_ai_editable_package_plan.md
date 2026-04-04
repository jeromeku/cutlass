## Goal

Make `wafer-ai/wafer-ai` editable-installable by adding a `pyproject.toml` that reconstructs package metadata from the extracted wheel.

## Findings

- The extracted package lives at `wafer-ai/wafer-ai/wafer`.
- Wheel metadata exists under `wafer-ai/wafer-ai/wafer_ai-0.0.87.dist-info/`.
- Console scripts are `wafer` and `wafer-ai`, both targeting `wafer.cli.app:main`.
- Non-Python package data that must remain available:
  - `wafer/cli/GUIDE.md`
  - `wafer/cli/skills/**/*.md`
  - `wafer/core/config/templates/*.toml`
- The closest packaging reference is `wafer-ai/wafer-ext/extension/bundled/wafer-ai/pyproject.toml`, but the extracted wheel has newer version metadata (`0.0.87`) and a slightly different dependency layout.

## Chunked Plan

### Chunk 1

Create `wafer-ai/wafer-ai/pyproject.toml` with:
- `setuptools.build_meta`
- `project.name = "wafer-ai"`
- wheel-matching version, Python requirement, dependencies, optional `dev` dependencies
- console scripts for `wafer` and `wafer-ai`
- setuptools package discovery for `wafer*`
- package-data rules for markdown guides/skills and TOML templates

### Chunk 2

Validate locally with the repo virtualenv:
- inspect `pip` or `python -m pip` editable install metadata generation
- confirm entry points resolve
- note any dependency resolver issues separately from packaging correctness

## Open Decision

Default choice is to package version `0.0.87` to match the extracted wheel metadata, even though `wafer/__init__.py` currently reports `0.3.0`.
