# Project Map

## File Roles

### Root Files

- `README.md`: current project guide and recommended commands.
- `README`: original upstream DeepMTT notes. It describes the old TensorFlow
  workflow and is kept for provenance.
- `.gitignore`: local ignore rules for caches, generated data, logs, and
  checkpoints.
- `Trajectory_data_generator2.py`: compatibility wrapper for
  `src/deepmtt/trajectory_data_generator.py`.
- `batchdata_derive3.py`: compatibility wrapper for
  `src/deepmtt/training_data.py`.
- `LMTT_deta_bidr_0818.py`: compatibility wrapper for
  `src/deepmtt/st_transformer_model.py`.
- `adaptive_ukf_turnratev1.py`: compatibility wrapper for
  `src/deepmtt/adaptive_ukf_turnrate.py`.
- `plot_train_log.py`: compatibility wrapper for `scripts/plot_train_log.py`.
- `plot_real_trajectories.py`: compatibility wrapper for
  `scripts/plot_real_trajectories.py`.
- `maxout.py`: compatibility wrapper for `src/deepmtt/tf_maxout.py`.
- `Simulations_of_DeepMTT.py`: compatibility wrapper for the legacy TensorFlow
  simulation in `scripts/legacy/`.

### Core Code

- `src/deepmtt/trajectory_data_generator.py`
  - Generates single-turn synthetic maneuvering target trajectories.
  - Produces true state sequences, polar observations, transition matrices, and
    turn-rate labels.
  - Main output function: `trajectory_batch_generator(batch_size, data_len)`.

- `src/deepmtt/training_data.py`
  - Calls `trajectory_batch_generator` to create ground-truth trajectories.
  - Runs a baseline UKF over generated observations to create tracked
    trajectories.
  - Returns `(true_traj, observations, ukf_traj, true_minus_ukf, turn_rate)`.
  - Main output function: `creat_batch3(...)`.

- `src/deepmtt/st_transformer_model.py`
  - Defines the PyTorch ST-Transformer turn-rate predictor.
  - Contains model blocks: attention, spatial/temporal transformer, entangle
    module, prediction MLP.
  - Defines training loop, training-data saving, and checkpoint writing.
  - Reads samples from `training_data.py`.
  - Writes generated `.mat` data to `data/MTT_TrainingData.mat`.
  - Writes checkpoints to `checkpoints/pytorch/Save_Model/`.

- `src/deepmtt/adaptive_ukf_turnrate.py`
  - Generates multi-segment validation trajectories.
  - Loads an ST-Transformer checkpoint and predicts turn rate.
  - Compares a constant-velocity UKF against adaptive turn-rate UKF.
  - Writes validation figures to `outputs/figures/`.
  - Note: `run_adaptive_ukf_aligned` currently uses true trajectory history as
    model input, so it is useful for controlled analysis but optimistic for a
    strict online-tracking benchmark.

- `src/deepmtt/tf_maxout.py`
  - Original TensorFlow maxout helper used by the legacy simulation.
  - Requires TensorFlow 1.x style APIs.

- `src/filterpy/`
  - Vendored FilterPy source used by both training-data generation and UKF
    validation.
  - Main imported APIs are `filterpy.kalman.UnscentedKalmanFilter` and
    `filterpy.kalman.JulierSigmaPoints`.

### Scripts

- `scripts/train_turnrate_model.py`
  - Preferred training entry point.
  - Calls `deepmtt.st_transformer_model.main()`.

- `scripts/validate_adaptive_ukf.py`
  - Preferred adaptive UKF validation entry point.
  - Calls `deepmtt.adaptive_ukf_turnrate.main()`.

- `scripts/plot_train_log.py`
  - Parses `logs/train.log`.
  - Plots loss and turn-rate RMSE to `outputs/figures/train_curves.png`.

- `scripts/plot_real_trajectories.py`
  - Loads `data/MTT_TrainingData.mat`.
  - Plots sampled true trajectories to
    `outputs/figures/real_trajectories.png`.

- `scripts/legacy/Simulations_of_DeepMTT.py`
  - Original TensorFlow DeepMTT simulation.
  - Uses the legacy checkpoint under `checkpoints/tensorflow/`.
  - Not runnable in the current `deepmtt` env because TensorFlow is absent.

### Local Artifacts

- `data/MTT_TrainingData.mat`
  - Generated training data. Ignored by Git.

- `checkpoints/pytorch/Save_Model/*.pth`
  - PyTorch ST-Transformer checkpoints. Ignored by Git.

- `checkpoints/tensorflow/`
  - Original TensorFlow checkpoint files. Ignored by Git.

- `logs/train.log`
  - Training log used by `scripts/plot_train_log.py`. Ignored by Git.

- `outputs/figures/*.png`
  - Generated visualizations. Some figures are tracked as project examples.

## Workflow Relationships

### Training Data Generation

```text
trajectory_data_generator.py
  -> training_data.py
      -> creat_batch3(...)
          -> true trajectory set
          -> noisy polar observations
          -> baseline UKF trajectory set
          -> turn-rate labels
```

The generated dataset is saved by the training script as:

```text
data/MTT_TrainingData.mat
```

### Model Training

```text
training_data.py
  -> st_transformer_model.py
      -> Network
      -> MSELoss(predicted_turn_rate, true_turn_rate)
      -> checkpoints/pytorch/Save_Model/ST_Transformer_0418_iter_*.pth
```

Recommended command:

```bash
python scripts/train_turnrate_model.py
```

### Adaptive UKF Validation

```text
adaptive_ukf_turnrate.py
  -> load Network checkpoint
  -> generate multi-segment validation trajectory
  -> run Standard UKF
  -> run Adaptive UKF
  -> outputs/figures/adaptive_ukf_validation.png
```

Recommended command:

```bash
python scripts/validate_adaptive_ukf.py \
  --num-traj 4 \
  --data-len 1000 \
  --turn-rates "0,3,-5,2"
```

### Visualization

```text
logs/train.log
  -> scripts/plot_train_log.py
  -> outputs/figures/train_curves.png

data/MTT_TrainingData.mat
  -> scripts/plot_real_trajectories.py
  -> outputs/figures/real_trajectories.png
```

