#!/bin/bash
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

# Detector
# python test_chess_detector.py \
# 	--model runs/detect/train/weights/best.pt \
# 	--source data/preliminary_videos/orig_20251027_212308.mp4 \
# 	--detector yolo \
# 	--save_video

# Homography
#python3 chess_homography.py --source data/preliminary_videos/test.mp4 --model runs/detect/train/weights/best.pt

python3 chess_homography.py --source data/preliminary_videos/phone_video-6.mp4 --model runs/detect/train/weights/best.pt --use-bottom --nms 0.3 --conf 0.5 --save-video