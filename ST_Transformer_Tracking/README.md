# ST Transformer Tracking

This directory contains the DeepMTT / ST-Transformer target-tracking workspace.
It is organized so code, scripts, data, checkpoints, logs, and figures are
separated while old entry-point filenames remain as compatibility wrappers.

## Runtime

Use the `deepmtt` conda environment:

```bash
conda activate deepmtt
cd /home/wanqiang/wanqiang_space/deepmtt/ST_Transformer_Tracking
```

Current environment notes:

- Python: 3.9.23
- PyTorch: 1.10.0
- NumPy: 2.0.2, which emits a PyTorch compatibility warning in this env
- TensorFlow: not installed, so the legacy TensorFlow simulation is not runnable

## Layout

```text
ST_Transformer_Tracking/
  src/deepmtt/              Core project code
  src/filterpy/             Vendored FilterPy dependency
  scripts/                  Preferred command-line entry points
  scripts/legacy/           Original TensorFlow simulation entry point
  data/                     Local training data, ignored by Git
  checkpoints/              Local model weights, ignored by Git
  logs/                     Local training logs, ignored by Git
  outputs/figures/          Generated figures
  docs/PROJECT_MAP.md       Detailed file roles and relationships
```

## Main Workflows

Generate training samples and train the turn-rate model:

```bash
python scripts/train_turnrate_model.py
```

Validate adaptive UKF tracking with a trained checkpoint:

```bash
python scripts/validate_adaptive_ukf.py \
  --num-traj 4 \
  --data-len 1000 \
  --turn-rates "0,3,-5,2" \
  --out outputs/figures/adaptive_ukf_segment4_customrates.png
```

Plot training curves:

```bash
python scripts/plot_train_log.py
```

Plot samples from the generated training data:

```bash
python scripts/plot_real_trajectories.py
```

## Compatibility Wrappers

These old filenames still exist at the project root and forward to the new
locations:

- `Trajectory_data_generator2.py`
- `batchdata_derive3.py`
- `LMTT_deta_bidr_0818.py`
- `adaptive_ukf_turnratev1.py`
- `plot_train_log.py`
- `plot_real_trajectories.py`
- `maxout.py`
- `Simulations_of_DeepMTT.py`

