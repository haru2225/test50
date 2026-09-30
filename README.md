# test50 — test38's model, unchanged, on an SiO2 Si-only coarse-graining

test38's sigma-conditioned NequIP (`NequIP_TimeEmbed`) + reverse variance-exploding SDE sampler,
applied to a new system. **The model, training objective and sampler code are copied unchanged
from test38.py** — see `test50.py`'s own module docstring for exactly which functions are
byte-for-byte identical. What's new is the data: an SiO2 (beta-cristobalite) coarse-graining that
keeps only the Si of every SiO4 tetrahedron.

## Coarse-graining

One CG particle per original Si atom (index-preserving subset selection) — every O atom is
dropped outright, not averaged or merged into anything. This is simpler than test37.py's general
weighted CG `prepare` (there's nothing to average: each retained site already *is* a real atom).
The consequence: the model never sees O, so the SiO4 tetrahedral bonding geometry is only implicit
in the resulting Si-Si network (corner-sharing tetrahedra set the Si-Si next-nearest-neighbor
distance, not a real Si-Si bond).

`test50.py prepare` builds the CG dataset from the beta-cristobalite NVT reference frames already
validated for test47/48/49 (`simu_data/reference_frames.npz`, bundled here — 184 frames, 192 atoms
= 64 Si + 128 O, cubic cell 13.573 Å, 300 K NVT) by selecting the 64 Si indices per frame and
discarding the 128 O positions; the cell is carried through unchanged (only the atom set shrinks),
unless `--replicate` tiles it larger (below).

## `prepare --replicate N`: tiling the box for a bigger `--cutoff`

The bundled box is only 13.573 Å across (half-box 6.7865 Å), which puts a **hard ceiling** on
`--cutoff`/`--large-cutoff` regardless of tuning: anything ≥ half the box connects the same atom
pair through 2+ periodic images at once (the test33/48 duplicate-periodic-image bug). `--cutoff 8`
is unsafe on this box no matter what.

`--replicate N` tiles the Si positions `N x N x N` (exact periodic copies) and scales the cell by
`N`, so the box (and its safe cutoff ceiling) grows by the same factor — `--replicate 2` turns
13.573 Å into 27.146 Å (half-box 13.573 Å), comfortably covering `--cutoff 8`. **This is not a
substitute for a real larger-box MD run**: the `N^3` copies per frame are perfectly correlated
exact duplicates of the same configuration, not independent thermal samples — it only exists to
let a bigger cutoff see further without new simulation data. `train` refuses (see the guard below)
to run with a `--cutoff`/`--large-cutoff` that isn't strictly safe for whatever dataset was
actually built, replicated or not, so this can't be silently misused.

```bash
python test50.py prepare --replicate 2 --output sio2-si-only/dataset-2x2x2   # 64 -> 512 Si sites
```

## Numeric differences from test38.py's own argparse defaults

Every default not listed here is untouched from test38.py.

- **`--cutoff`/`--large-cutoff`: 8.0 / 8.2** (was 5.0/5.0, then 6.5/6.7). test38's own default is
  10.0/10.0, tuned for the clay system's larger box. These new defaults assume a `--replicate 2`
  dataset (half-box 13.573 Å, so 8.0/8.2 has 5+ Å of margin to spare) — **do not use the default
  `train` invocation against a `--replicate 1` (untiled) dataset**; `train` now checks the actual
  box size from the dataset's own `cells.npy` and raises a clear error instead of silently
  corrupting the graph if `--large-cutoff` isn't strictly less than half of it, so this mismatch is
  caught immediately rather than producing a silently-wrong checkpoint.
- **`--sigma-max`: 1.5** (was 0.75, test38/DM2's shared default). Si-Si nearest-neighbor distance
  is ~2.97 Å, so this already reaches "the atom is effectively scrambled" territory — training now
  covers a genuinely wider corruption range. Match this with `generate --start-sigma 1.5` (and
  `--init crystal-noised`, see below) to actually exercise the new range at generation time —
  `generate`'s own `--start-sigma` default (0.75) is untouched, so it must be passed explicitly.
- **`--updates`: 50000** (was 6000, then 20000) — a longer training budget.

A checkpoint trained with these values cannot `--resume` into a directory whose checkpoint used
different ones (`train`'s resume path checks the full settings dict and refuses on any mismatch,
`cutoff`/`sigma_max`/`updates` included) — retraining with the new defaults means a **fresh**
`--output` directory, not resuming an old one.

## `--irreps-hidden` / `--irreps-edge`: raising the spherical-harmonics l_max

test38's `NequIP_TimeEmbed` hardcodes its irreps (l≤1 hidden, l≤2 edge spherical harmonics) inside
`architecture()`; test50 exposes them as `train` flags instead (default unchanged: `"64x0e + 32x1e"`
/ `"4x0e + 4x1e + 2x2e"`, identical to test38 unless overridden). `generate` needs no equivalent
flag — it always rebuilds the model from whatever irreps the checkpoint itself was trained with.

To raise l_max to 4 (more angular/orientational detail per edge and node — one candidate fix for
the "can't tell which lattice site" problem, since Si-Si-Si angles alone may be too coarse a
signal at l≤1):

```bash
python test50.py train --dataset ... --output ... \
    --irreps-hidden "64x0e + 32x1e + 16x2e + 8x3e + 4x4e" \
    --irreps-edge   "4x0e + 4x1e + 2x2e + 2x3e + 1x4e"
```

(test47/48's own l≤5 stack truncated by one order.) **This is the same lever that caused test49's
CUDA out-of-memory issue** — e3nn's tensor-product path count and intermediate-tensor size grow
steeply with l_max (measured there: l≤5 vs l≤1 was a ~17x larger per-edge intermediate tensor), so
raising this may need a smaller `--batch-size` or more GPU memory. Untested at l≤4 specifically;
only smoke-tested here for a few CPU updates to confirm it trains and checkpoints/reloads
correctly.

## Files

- `test50.py` — prepare/train/generate, single self-contained script. Unlike test45-49, it never
  imports `graphite` (the model code is vendored the same way test38.py vendors it from DM2), so
  **no DM2 checkout or DM2_ROOT bind-mount is needed at run time**.
- `simu_data/reference_frames.npz` + `..._metadata.json` — the same reference MD frames test47/48/
  49 train against (bundled here so `prepare` has a default to point at).
- `run_test50.pbs` — one PBS script, three stages via `STAGE=prepare|train|generate` (qsub `-v`
  variables are translated to `test50.py`'s CLI flags; anything not set falls back to `test50.py`'s
  own argparse defaults).
- `Singularity.def` — PyTorch + e3nn 0.4.4 + torch_geometric + ase runtime. No DM2/graphite
  install step (nothing to bind-mount either).

## Usage

```bash
git clone git@github.com:haru2225/test50.git
cd test50

module load singularity
singularity build test50.sif Singularity.def
# Apptainer: apptainer build test50.sif Singularity.def

qsub -P PROJECT_ID -v STAGE=prepare,OUTPUT=sio2-si-only/dataset-2x2x2,REPLICATE=2 run_test50.pbs

qsub -P PROJECT_ID -v STAGE=train,DATASET=sio2-si-only/dataset-2x2x2,OUTPUT=sio2-si-only/checkpoint3 \
    run_test50.pbs
# cutoff=8.0/8.2, sigma-max=1.5, updates=50000 are now the defaults (assume the replicated
# dataset above) -- override any test50.py train flag via qsub -v, e.g.:
# qsub -P PROJECT_ID -v STAGE=train,DATASET=...,OUTPUT=...,UPDATES=20000,BATCH_SIZE=16 run_test50.pbs

qsub -P PROJECT_ID -v STAGE=generate,CHECKPOINT=sio2-si-only/checkpoint1/checkpoint.pt,OUTPUT=sio2-si-only/checkpoint1/generated,REVERSE_STEPS=3000,DETERMINISTIC_STEPS=300 \
    run_test50.pbs
```

Locally (no PBS/Singularity), the same three commands directly:

```bash
python test50.py prepare --replicate 2 --output sio2-si-only/dataset-2x2x2   # uses bundled simu_data/
python test50.py train --dataset sio2-si-only/dataset-2x2x2 --output sio2-si-only/checkpoint3 --device cuda
python test50.py generate --checkpoint sio2-si-only/checkpoint3/checkpoint.pt \
    --output sio2-si-only/checkpoint3/generated --reverse-steps 3000 --deterministic-steps 300 --device cuda
```

## Starting `generate` from an actually-noised structure: `--init crystal-noised`

test38's original `generate()` always starts from the clean training frame
(`ck["start_positions_angstrom"]`) and just runs the annealed-Langevin schedule against it — the
schedule's own noise injection at the early (high-sigma) steps is the *only* corruption that ever
happens, and it's easy to misread a run as "generation from noise" when `deterministic_steps` is
set too high and that injection barely happens at all (see the test50-v2 case below).

`--init crystal-noised` fixes this by explicitly forward-diffusing the clean frame *before* the
loop starts: `pos += start_sigma * randn_like(pos)` — the exact same noise model `RattleParticles`
uses during training. This is a real "start from a genuinely sigma=start_sigma-noised structure and
watch the reverse SDE denoise it back into a crystal" test, not an implicit one. Use it together
with a real stochastic annealing budget (`--deterministic-steps` well below `--reverse-steps`) to
actually see the Si sublattice condense in `trajectory.extxyz`:

```bash
python test50.py generate --checkpoint sio2-si-only/checkpoint1/checkpoint.pt \
    --output sio2-si-only/checkpoint1/generated_noised --init crystal-noised \
    --reverse-steps 3000 --deterministic-steps 300 --trajectory-stride 10 --device cuda
```

`--init crystal` (unchanged default) reproduces the original test38 behavior exactly, for
backward compatibility / A-B comparison.

## Watching the structure form: `trajectory.extxyz`

`generate` always writes `final.extxyz` (last frame only) and, unless `--trajectory-stride 0` is
passed, also `output/trajectory.extxyz` — every `--trajectory-stride`-th reverse-diffusion step
(default: every step) as one multi-frame extended-XYZ trajectory, built by re-reading the
`positions.npy` array `generate` already saves. Open it in OVITO/VMD/ASE's own GUI to watch the Si
sublattice condense from the injected noise back toward a crystal-like structure step by step; each
frame's `ase.Atoms.info["step"]` records which reverse-diffusion step it came from. For a long run
(e.g. `--reverse-steps 3000`), raise `--trajectory-stride` (e.g. to 10) to keep the file smaller.

## Status

A real GPU training + `generate --reverse-steps 300 --deterministic-steps 30` (test38's own
defaults) run has completed end-to-end. Analysis against the training reference (Si-Si nearest-
neighbor distance and Si-Si-Si angle) shows the sampler moves in the right direction — starting
from an over-noised configuration (step 30: NN distance collapsed to a 1.4 Å minimum) and
recovering toward the reference's ~2.97 Å spacing and ~108° tetrahedral-like angle by the final
step — but has **not fully converged** at only 300 steps: the final structure's bond-length spread
(σ≈0.11 Å) and angle spread (σ≈29°) are still ~2-3x broader than the real MD reference (σ≈0.04 Å /
13°). test47-49's own SiO2 generation uses 2900+100 steps for the same box; **increasing
`--reverse-steps` well above test38's clay-tuned default of 300 is the most likely lever to close
this gap** (untested at higher step counts so far).

**Pitfall found the hard way**: a follow-up run set `--deterministic-steps` equal to
`--reverse-steps` (both 300), which makes `stochastic_steps = reverse_steps - deterministic_steps
= 0` — every step runs the deterministic DDIM branch, no annealed noise is ever injected. Starting
from `--init crystal` (the default, a clean frame), that run barely moved (total displacement 0.14
Å) and scored *better* than real MD on every metric — not because generation succeeded, but because
it was never actually asked to recover from noise. Keep `--deterministic-steps` a small tail
fraction of `--reverse-steps` (e.g. 10%), and use `--init crystal-noised` (above) if you want an
unambiguous "did it actually reconstruct a crystal from noise" test.
