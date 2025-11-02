#!/bin/bash
#python chess_pipeline.py \
#  --video ./data/aca02073-output.mp4 \
#  --json  ./data/project-1-at-2025-10-26-16-17-9bfd7434.json \
#  --out   runs/exp1 \
#  --classes white_pawn white_rook white_knight white_bishop white_queen white_king \
#           black_pawn black_rook black_knight black_bishop black_queen black_king \
#  --lazy_extract \
#  --export_yolo --export_dir yolo_export \
#  --train_yolo --yolo_model yolov8n.pt --epochs 50 --imgsz 1280 --batch 8
  
# R-CNN with YOLO formatting  
#python pytorch_pipeline.py \
#  --format yolo \
#  --data_yaml chess_pieces.yolov8/data.yaml \
#  --out runs/chess_exp \
#  --epochs 20 \
#  --batch_size 8 \
#  --freeze_backbone

# Output detections to an image
#python test_chess_detector.py     --model runs/detect/train/weights/#best.pt     --source data/test_frames/test_video_frame1335.jpg     --#detector yolo     --save

python test_chess_detector.py \
	--model runs/detect/train/weights/best.pt \
	--source data/preliminary_videos/orig_20251027_212308.mp4 \
	--detector yolo \
	--save_video
