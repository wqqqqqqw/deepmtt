# 模型验证结果对比

验证场景统一使用 4 段机动轨迹：

```bash
--num-traj 4 --data-len 1000 --turn-rates "0,3,-5,2" --seed 7
```

对比指标来自 `scripts/validate_adaptive_ukf.py`。其中 `Standard UKF` 不使用模型预测转弯率，`Adaptive UKF` 使用对应模型预测转弯率。

## 上一个模型：ST-Transformer

- 权重路径：
  - `checkpoints/pytorch/Save_Model/ST_Transformer_0418_iter_10000.pth`
- 模型特点：
  - 使用原始 ST-Transformer 结构预测转弯率。
  - 默认输入窗口长度为 `seq_len=50`。
  - 对目标机动变化的响应主要依赖整段历史窗口，转弯率突变后收敛较慢。
- 验证命令：

```bash
conda run -n deepmtt python scripts/validate_adaptive_ukf.py \
  --checkpoint checkpoints/pytorch/Save_Model/ST_Transformer_0418_iter_10000.pth \
  --num-traj 4 \
  --data-len 1000 \
  --turn-rates "0,3,-5,2" \
  --seed 7 \
  --refresh-period 1 \
  --turn-smoothing 0.25 \
  --out outputs/figures/adaptive_ukf_segment4_previous_0418.png
```

- 结果指标：

| 指标 | 数值 |
| --- | ---: |
| Standard UKF RMSE | 205.068 m |
| Adaptive UKF RMSE | 76.244 m |
| RMSE 改善 | 128.824 m |
| 响应延迟 steps | `[51, 56, 43]` |

- 分段 RMSE：

| 段 | Standard UKF | Adaptive UKF |
| --- | ---: | ---: |
| Straight | 45.393 m | 42.462 m |
| Turn-1 | 201.169 m | 70.456 m |
| Turn-2 | 313.850 m | 46.751 m |
| Turn-3 | 139.031 m | 106.900 m |

- 可视化效果图：
  - `outputs/figures/adaptive_ukf_segment4_previous_0418.png`

## 当前模型：响应优化 Kinematic Turn-Rate

- 权重路径：
  - `checkpoints/pytorch/Save_Model/ST_Transformer_response_kinematic_s090_0705_iter_0.pth`
- 模型特点：
  - 新增 `KinematicTurnRateNet`，直接根据最近两步速度方向变化估计瞬时转弯率。
  - 权重中保存 `model_config`，验证时自动读取 `model_arch/seq_len/dtype`。
  - 使用 `seq_len=20` 的短窗口，减少旧运动状态对新机动状态的拖尾影响。
  - 验证时默认 `refresh_period=1`，每步刷新模型转弯率预测。
  - 使用 `turn_smoothing=0.25`，比旧平滑更快响应，同时保留一定抗噪能力。
  - 当前推荐版本使用 `scale=0.9, bias=0.0`，在响应速度与轨迹 RMSE 之间较均衡。
- 验证命令：

```bash
conda run -n deepmtt python scripts/validate_adaptive_ukf.py \
  --checkpoint checkpoints/pytorch/Save_Model/ST_Transformer_response_kinematic_s090_0705_iter_0.pth \
  --num-traj 4 \
  --data-len 1000 \
  --turn-rates "0,3,-5,2" \
  --seed 7 \
  --refresh-period 1 \
  --turn-smoothing 0.25 \
  --out outputs/figures/adaptive_ukf_segment4_response_kinematic_s090_0705.png
```

- 结果指标：

| 指标 | 数值 |
| --- | ---: |
| Standard UKF RMSE | 205.068 m |
| Adaptive UKF RMSE | 39.702 m |
| RMSE 改善 | 165.366 m |
| 响应延迟 steps | `[22, 14, 42]` |

- 分段 RMSE：

| 段 | Standard UKF | Adaptive UKF |
| --- | ---: | ---: |
| Straight | 45.393 m | 45.393 m |
| Turn-1 | 201.169 m | 36.234 m |
| Turn-2 | 313.850 m | 46.375 m |
| Turn-3 | 139.031 m | 61.518 m |

- 可视化效果图：
  - `outputs/figures/adaptive_ukf_segment4_response_kinematic_s090_0705.png`

## 对比结论

| 对比项 | 上一个模型 | 当前模型 |
| --- | ---: | ---: |
| Adaptive UKF RMSE | 76.244 m | 39.702 m |
| 相对上一个模型 RMSE 降低 | - | 36.542 m |
| 平均响应延迟 | 50.0 steps | 26.0 steps |
| 响应延迟降低 | - | 24.0 steps |

当前模型主要改善了目标机动变化后的响应速度，并明显降低了 Adaptive UKF 的整体轨迹误差。
