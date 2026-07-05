# Project Map

## File Roles

### Root Files

- `README.md`: current project guide and recommended commands.
- `README`: short pointer to the current PyTorch-only workflow.
- `.gitignore`: local ignore rules for caches, generated data, logs, and
  checkpoints.

The old root-level Python entry points were removed after the project was
organized. Use `scripts/` for executable workflows and `src/deepmtt/` for
implementation modules.

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

### Local Artifacts

- `data/MTT_TrainingData.mat`
  - Generated training data. Ignored by Git.

- `checkpoints/pytorch/Save_Model/*.pth`
  - PyTorch ST-Transformer checkpoints. Ignored by Git.

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
