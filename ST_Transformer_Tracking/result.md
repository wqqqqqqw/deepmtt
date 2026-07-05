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

## 当前模型：响应优化 ST-Transformer

- 权重路径：
  - `checkpoints/pytorch/Save_Model/ST_Transformer_response_feature_transformer_0705_iter_60.pth`
- 模型特点：
  - 保留 ST-Transformer 作为转弯率学习主模型。
  - 新增 `FeatureTransformerNet`，在原始轨迹状态 `[x, y, vx, vy]` 基础上拼接瞬时航向变化特征，再交给 Transformer 学习转弯率。
  - 权重中保存 `model_config`，验证时自动读取 `model_arch/seq_len/dtype`。
  - 使用 `seq_len=20` 的短窗口，减少旧运动状态对新机动状态的拖尾影响。
  - 验证时默认 `refresh_period=1`，每步刷新模型转弯率预测。
  - 使用 `turn_smoothing=0.25`，比旧平滑更快响应，同时保留一定抗噪能力。
  - 与直接运动学估计不同，当前模型仍由 Transformer 从轨迹上下文中学习转弯率，运动特征只作为输入增强。
- 验证命令：

```bash
conda run -n deepmtt python scripts/validate_adaptive_ukf.py \
  --checkpoint checkpoints/pytorch/Save_Model/ST_Transformer_response_feature_transformer_0705_iter_60.pth \
  --num-traj 4 \
  --data-len 1000 \
  --turn-rates "0,3,-5,2" \
  --seed 7 \
  --refresh-period 1 \
  --turn-smoothing 0.25 \
  --out outputs/figures/adaptive_ukf_segment4_response_feature_transformer_0705.png
```

- 结果指标：

| 指标 | 数值 |
| --- | ---: |
| Standard UKF RMSE | 205.068 m |
| Adaptive UKF RMSE | 42.993 m |
| RMSE 改善 | 162.075 m |
| 响应延迟 steps | `[19, 11, 14]` |

- 分段 RMSE：

| 段 | Standard UKF | Adaptive UKF |
| --- | ---: | ---: |
| Straight | 45.393 m | 37.241 m |
| Turn-1 | 201.169 m | 69.154 m |
| Turn-2 | 313.850 m | 41.913 m |
| Turn-3 | 139.031 m | 44.293 m |

- 可视化效果图：
  - `outputs/figures/adaptive_ukf_segment4_response_feature_transformer_0705.png`

## 对照模型：Kinematic Turn-Rate Baseline

- 权重路径：
  - `checkpoints/pytorch/Save_Model/ST_Transformer_response_kinematic_s090_0705_iter_0.pth`
- 说明：
  - 该模型直接根据最近两步速度方向变化估计瞬时转弯率。
  - 它用于说明运动学先验能提供很强的响应能力，但不作为本项目的主要创新模型。
  - 项目主线仍是使用 Transformer 学习轨迹转弯率。
- 对照结果：
  - Adaptive UKF RMSE: 39.702 m
  - 响应延迟 steps: `[22, 14, 42]`
  - 可视化效果图：`outputs/figures/adaptive_ukf_segment4_response_kinematic_s090_0705.png`

## 对比结论

| 对比项 | 上一个模型 | 当前模型 |
| --- | ---: | ---: |
| Adaptive UKF RMSE | 76.244 m | 42.993 m |
| 相对上一个模型 RMSE 降低 | - | 33.251 m |
| 平均响应延迟 | 50.0 steps | 14.7 steps |
| 响应延迟降低 | - | 35.3 steps |

当前模型在保持 Transformer 学习转弯率这一主线的基础上，通过响应型多段机动训练、短窗口输入、运动特征增强和低滞后更新策略，显著提升了目标机动变化后的响应速度，并降低了 Adaptive UKF 的整体轨迹误差。
