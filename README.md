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
discarding the 128 O positions; the cell is carried through unchanged (only the atom set shrinks).

## The one numeric change from test38.py

test38's own `--cutoff`/`--large-cutoff` default to 10.0 (tuned for the clay system's larger box).
This SiO2 box is only 13.573 Å across (half-box ~6.79 Å) — 10 Å would hit the same
duplicate-periodic-image bug test33/48 document (a cutoff bigger than half the box connects the
same atom pair through 2+ periodic images at once). **test50's `train` subcommand changes only
this one default, to 5.0/5.0** (test47/48/49's own fix for the same box). Every other argparse
default is untouched from test38.py. `generate` has no cutoff flag of its own — it reads whatever
cutoff the checkpoint was trained with.

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

qsub -P PROJECT_ID -v STAGE=prepare,OUTPUT=sio2-si-only/dataset-pilot run_test50.pbs

qsub -P PROJECT_ID -v STAGE=train,DATASET=sio2-si-only/dataset-pilot,OUTPUT=sio2-si-only/checkpoint1 \
    run_test50.pbs
# override any test50.py train flag via qsub -v, e.g.:
# qsub -P PROJECT_ID -v STAGE=train,DATASET=...,OUTPUT=...,UPDATES=6000,BATCH_SIZE=16 run_test50.pbs

qsub -P PROJECT_ID -v STAGE=generate,CHECKPOINT=sio2-si-only/checkpoint1/checkpoint.pt,OUTPUT=sio2-si-only/checkpoint1/generated \
    run_test50.pbs
```

Locally (no PBS/Singularity), the same three commands directly:

```bash
python test50.py prepare --output sio2-si-only/dataset-pilot   # uses bundled simu_data/
python test50.py train --dataset sio2-si-only/dataset-pilot --output sio2-si-only/checkpoint1 --device cuda
python test50.py generate --checkpoint sio2-si-only/checkpoint1/checkpoint.pt \
    --output sio2-si-only/checkpoint1/generated --device cuda
```

## Status

`prepare` → `train` (a handful of updates) → `generate` (a handful of steps) has been run
end-to-end on CPU as a smoke test (no crash, correct shapes: 64 Si sites, 184 frames, single "Si"
species) with the new 5.0/5.0 cutoff defaults. **Not yet trained for real** (6000+ updates) or run
on GPU/HPC; generation quality from a real checkpoint is unverified.
