# AI Agent Guide

This is the operational entry point for AI agents working in this repository.
Read the root `AGENTS.md` first: it is normative and takes precedence over this
guide. Use `docs/experiments.md` for experiment-specific scientific context.

## Repository purpose

The project evaluates motion dissimilarity with shape subspaces. Inputs are C3D
motion-capture data and 3D poses estimated from video. Each executable
experiment has an explicit `src/ex*.py` entry point and a matching YAML file in
`configs/`.

The primary design goal is reproducibility. Preserve numerical operations,
random-number consumption, frame selection, array orientation, filenames, and
existing outputs unless the user explicitly requests a change.

## First actions in every task

1. Run `git status --short`. The worktree may contain user changes and
   untracked experiment files. Never discard or overwrite them.
2. Read the relevant `src/ex*.py`, matching `configs/ex*.yaml`, tests, and the
   corresponding section in `docs/experiments.md`.
3. Use `rg` to find all callers, configuration keys, and output paths before
   editing shared code.
4. Prefer the smallest change that leaves the experiment order visible in the
   experiment's `main()`.
5. Do not delete, move, or bulk-regenerate existing results without explicit
   authorization.

`data/` and `out/` are ignored by Git. Their absence from `git status` does not
mean they are disposable. Treat both as user data.

## Environment and standard commands

The project requires Python 3.12 or newer and uses `uv`.

```powershell
uv sync
$env:PYTHONPATH='src'
uv run python src/ex3.py configs/ex3.yaml
```

For the repository's existing virtual environment, validation is:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\flake8.exe --jobs 1 src tests
.\.venv\Scripts\python.exe -m compileall -q src tests
git diff --check
```

Run GIF/PNG generation only when output behavior is part of the change. Pose
inference and GIF rendering can take much longer than unit tests.

## Architecture and dependency direction

```text
src/ex*.py  ->  src/experiments  ->  src/subspace
     |                                  ^
     +----------->  src/utils           |
```

- `src/ex*.py`: experiment-specific orchestration. Keep data loading,
  preprocessing, subspace construction, analysis, and output visibly ordered in
  `main()`.
- `src/experiments`: reusable experiment-domain operations, including YAML
  loading, common-subspace construction, dissimilarity, noise, pose estimation,
  coordinate projection, and joint correspondence.
- `src/subspace`: mathematical primitives. It must not depend on `experiments`
  or `utils`; consumers should import its public API from `subspace`.
- `src/utils`: domain-independent I/O and visualization. Japanese-capable
  Matplotlib font selection is centralized in `utils/plot_style.py`.

Do not add a generic experiment registry or runner that hides an experiment's
flow. Batch execution belongs in `scripts/run_all.ps1`, which calls each
`src/ex*.py` directly.

## Configuration contract

- A runnable script and YAML file use the same stem, for example
  `src/ex11.py` and `configs/ex11.yaml`.
- Every experiment defines `DEFAULT_CONFIG`; invocation without YAML remains
  supported.
- YAML-relative paths resolve from the YAML file's directory.
- Unknown keys raise an error.
- When adding a setting, update `DEFAULT_CONFIG`, YAML, documentation, and tests
  in the same change.

Standard invocation:

```powershell
uv run python src/ex11.py configs/ex11.yaml
```

## Experiment map

| Entry point | Main responsibility |
| --- | --- |
| `ex1.py` | C3D common-subspace eigenvalue and projection-error analysis |
| `ex2_*` | First/second differences and geodestic decomposition |
| `ex3*.py` | Noisy C3D projection experiments |
| `ex4.py` | MPI-INF-3DHP video pose estimation |
| `ex5.py` | GaHu S001 video pose estimation |
| `ex6.py` | Self-video pose estimation followed by ex2-style analysis |
| `ex6_analysis.py` | Numerical ex6 analysis from saved NPZ, without inference |
| `ex6_pose.py` | Pose comparison GIF redraw from saved ex6 NPZ, without inference |
| `ex7.py` | Eigenvalue analysis of ex6 pose estimates |
| `ex8.py` | 3D visualization of noisy C3D projection |
| `ex9.py` | 3D visualization of video-pose projection |
| `ex10.py` | Video joints projected into a noisy-C3D subspace |
| `ex11.py` | Video difference analysis in a clean-C3D subspace |
| `ex12.py` | Static overlay comparison of matching ex6/ex11 scores |

## ex6, ex11, and ex12 data flow

```text
self videos
   -> ex6 inference -> out/ex6/<video-name>/<video-name>.npz
                         |          |
                         |          +-> ex6_analysis -> pose-derived scores/GIFs
                         |
clean data/07_01.c3d ----+-> ex11 -> clean-C3D-derived scores/GIFs
                         |
                         +-> ex12 -> overlaid ex6/ex11 score PNGs
```

Use numerical-only ex6 analysis when NPZ files already exist:

```powershell
uv run python src/ex6_analysis.py configs/ex6_analysis.yaml
```

Current condition-oriented output layout:

```text
out/ex6/
  <video-name>/
    <video-name>.npz
    <video-name>_pose.gif
    fiN/origin/*.gif
    fiN/dN/*.gif

out/ex11/
  <video-name>/
    fiN/origin/*.gif
    fiN/dN/*.gif

out/ex12/
  fiN/dN/*.png
```

`N` in `dN` is the common-subspace dimension; `N` in `fiN` is the frame
interval. Those conditions are encoded only by directories, not repeated in
filenames. The video title is always the first directory level for ex6 and
ex11 condition outputs.

ex12 does not read graphs produced by ex6/ex11. It deliberately reuses their
calculation functions for matching dimension/frame-interval conditions and
draws only overlaid score graphs. It does not run pose inference.

## Numerical and output safety checklist

Before completing a change, confirm:

- array shapes and coordinate orientation are unchanged unless requested;
- failed video detections remain aligned as NaN frames in saved NPZ files;
- randomness distribution, seed handling, call count, and call order remain
  stable;
- public/output spelling such as `geodestic` remains compatible;
- existing result paths are not deleted or silently repurposed;
- new result conditions coexist through dimension and frame-interval tags;
- relevant tests cover orchestration and output paths;
- `docs/experiments.md` reflects experiment-level behavior.

## Handoff expectations

Report what changed, exact commands executed, validation results, and generated
output counts/locations. Mention warnings that affect interpretation, even when
execution succeeds. Do not claim generated media was visually verified unless
it was actually inspected.
