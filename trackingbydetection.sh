#!/usr/bin/env bash
set -e

source /home/wanqiang/wanqiang_space/software/anaconda3/etc/profile.d/conda.sh
conda activate deepmtt

python -u ST_Transformer_Tracking/scripts/run_detection_sequence_ukf.py \
  --result-root detection/result-0705 \
  --sequence WestAfrica-9_49 \
  --output-root detection/result-0705/detection_txt_multi \
  --checkpoint ST_Transformer_Tracking/checkpoints/pytorch/Save_Model/ST_Transformer_response_feature_transformer_0705_iter_60.pth \
  --min-area 1 \
  --max-link-distance 35 \
  --velocity-weight 0.5 \
  --max-gap 12 \
  --gap-distance-growth 8 \
  --velocity-gate-weight 0.8 \
  --merge-gap 30 \
  --merge-distance 45 \
  --min-track-length 20 \
  --show \
  --video-out ST_Transformer_Tracking/outputs/figures/detection_tracking/WestAfrica-9_49_tracking.mp4 \
  --show-frame-dir detection/result-0705/detection_txt_multi/WestAfrica-9_49/visualization_frames \
  --out ST_Transformer_Tracking/outputs/figures/detection_tracking/detection_ukf_WestAfrica-9_49_all_tracks.png
