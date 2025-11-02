# 2025-10-05
## 1149
Problems:
- Though it has only been tested with an unstable platform (i.e. I'm holding the camera with my hands and it has not yet been mounted), `detect_board.py` cannot yet consistently keep track of the board. Suspect this is due to similarity of color between the board and table.
- The video feed crashes randomly, sometimes right at the start of the feed. This in spite of the fact that it does output video. Code produces an error like this:
```
QObject::moveToThread: Current thread (0x456eee0) is not the object's thread (0x4ad68d0).
Cannot move to target thread (0x456eee0)

Camera read failed.
[ WARN:0@161.098] global cap_v4l.cpp:803 requestBuffers VIDEOIO(V4L2:/dev/video4): failed VIDIOC_REQBUFS: errno=19 (No such device)
```

## 1248
Problems:
- `detect_board.py` consistently chooses a region of the video feed that is too small, merely a portion of the board.
- Once that region is locked in, it seems that no updates are possible.

## 1352
Added the ability to manually recalibrate for the board. Something about the auto detector is too spotty; must remember to investigate later.

## 1508
Added video capture. The code is set up to capture both the original and the rectified output. One might be better for identifying final board configurations than the other, but we'll see.

Currently wondering if the code for detection of the board should just be removed, since manual calibration has been needed for every test so far. Next steps:
- Attempt auto-calibration on other surfaces and lighting conditions.
- Try different tape or bright corner markers on the board.
- Adjust thresholds in the code itself.

# 2025-10-26
## 2133
Added the pipeline for actually training. We have some preliminary data based on a 52-second video of just setting up the board. The next step will be collecting data in earnest.

# 2025-10-27
## 2044
Working out a recording pipeline. Worth noting that no matter what order we select the coordinates for the board, it places them... I don't want to say it's always upside-down, but it *is* something that we'll need to pay attention to.

The recordings need to be moved, renamed, etc. Right now, we are taking examples from Bobby Fischer Plays Chess. The following is a list of the exercies:
- 13 (noticed board was upside down)
- 14 (x2, becasue we had bowls of pieces behind and don't want the model to get confused.)
- 15
- 16
- 17
- 17, but the pieces were already set and we just rotated the board a bit.
- 18 (x3, because we are unsure how to handle when we place a piece a little on a line, and had a cup of coffee in the background)
- 19 (first time moving pieces on the board, recorded several of with different board rotations)
- 22 (x2, from a different angle)
- 23 (x2 normal, one simple rotation)
- 24

## 2158
TODO: organize video files output from `record_with_homography.py`, JSON files output from LabelStudio, etc.
TODO: Rewrite where code outputs any files
TODO: Reconsider the need for rectified versions of videos.

Recorded 26 short baseline videos, some of which include board rotation and piece movement.

# 2025-11-01
## 1159
Downloaded a decently sized chess dataset from Roboflow and trained a model on it. Will validate tomorrow

# 2025-11-02
## 1459
Trained YOLOv8 to this standard:

50 epochs completed in 2.081 hours.
Optimizer stripped from runs/detect/train/weights/last.pt, 6.2MB
Optimizer stripped from runs/detect/train/weights/best.pt, 6.2MB

Validating runs/detect/train/weights/best.pt...
Ultralytics YOLOv8.0.145 🚀 Python-3.7.7 torch-1.13.0+cu117 CUDA:0 (NVIDIA GeForce RTX 2070 SUPER, 7982MiB)
Model summary (fused): 168 layers, 3007988 parameters, 0 gradients
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   3%|▎         | 1/29 [00:00<00:08,  3.42it/s]WARNING ⚠️ NMS time limit 2.100s exceeded
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):   7%|▋         | 2/29 [00:04<01:05,  2.44s/it]WARNING ⚠️ NMS time limit 2.100s exceeded
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  10%|█         | 3/29 [00:07<01:13,  2.82s/it]WARNING ⚠️ NMS time limit 2.100s exceeded
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 29/29 [00:16<00:00,  1.74it/s]
                   all        910      16125      0.954       0.82      0.854      0.607
          black_bishop        910       1292      0.963      0.798      0.849      0.567
            black_king        910        790      0.947      0.851      0.906      0.693
          black_knight        910        987      0.966      0.861      0.891      0.631
            black_pawn        910       3831      0.973      0.893      0.917      0.627
           black_queen        910        786      0.913      0.764      0.806      0.623
            black_rook        910       1171      0.947      0.833      0.853      0.599
          white_bishop        910       1204       0.96      0.814      0.848      0.563
            white_king        910        887       0.94      0.763      0.794      0.607
          white_knight        910       1125      0.968      0.908      0.928      0.651
            white_pawn        910       2363      0.962      0.833       0.86       0.54
           white_queen        910        931       0.96      0.704      0.755      0.585
            white_rook        910        758      0.946      0.823      0.846      0.598
Speed: 0.2ms preprocess, 2.3ms inference, 0.0ms loss, 8.8ms postprocess per image
Results saved to runs/detect/train
wandb: Waiting for W&B process to finish... (success).
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:                  lr/pg0 ▃▆████▇▇▇▇▇▆▆▆▆▆▅▅▅▅▅▅▄▄▄▄▄▃▃▃▃▃▂▂▂▂▂▁▁▁
wandb:                  lr/pg1 ▃▆████▇▇▇▇▇▆▆▆▆▆▅▅▅▅▅▅▄▄▄▄▄▃▃▃▃▃▂▂▂▂▂▁▁▁
wandb:                  lr/pg2 ▃▆████▇▇▇▇▇▆▆▆▆▆▅▅▅▅▅▅▄▄▄▄▄▃▃▃▃▃▂▂▂▂▂▁▁▁
wandb:        metrics/mAP50(B) ▁▃▅▆▇▇▇████████████████████████████████▆
wandb:     metrics/mAP50-95(B) ▁▃▄▅▆▆▇▇▇▇▇▇▇██████████████████████████▇
wandb:    metrics/precision(B) ▁▃▅▆▇▇▇▇▇▇▇▇████████████████████████████
wandb:       metrics/recall(B) ▁▂▃▄▅▆▆▆▇▇▇▇▇▇▇▇█▇█████████████████████▅
wandb:            model/GFLOPs ▁
wandb:        model/parameters ▁
wandb: model/speed_PyTorch(ms) ▁
wandb:          train/box_loss █▆▆▆▅▅▅▄▄▄▄▄▄▄▄▃▃▃▃▃▃▃▃▃▃▃▃▂▂▂▂▂▂▂▂▁▁▁▁▁
wandb:          train/cls_loss █▅▄▄▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁
wandb:          train/dfl_loss █▆▆▆▅▄▄▄▄▃▃▃▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▂▂▂▂▁▁▁▁
wandb:            val/box_loss ▇▇█▆▅▅▄▄▄▄▃▃▃▂▂▂▂▂▁▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:            val/cls_loss █▇▆▄▃▃▃▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:            val/dfl_loss ▇▆█▆▅▅▅▄▄▄▃▃▃▃▂▃▃▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb: 
wandb: Run summary:
wandb:                  lr/pg0 0.0005
wandb:                  lr/pg1 0.0005
wandb:                  lr/pg2 0.0005
wandb:        metrics/mAP50(B) 0.85446
wandb:     metrics/mAP50-95(B) 0.60687
wandb:    metrics/precision(B) 0.95371
wandb:       metrics/recall(B) 0.82038
wandb:            model/GFLOPs 0.0
wandb:        model/parameters 3013188
wandb: model/speed_PyTorch(ms) 1.706
wandb:          train/box_loss 0.95785
wandb:          train/cls_loss 0.51165
wandb:          train/dfl_loss 0.97841
wandb:            val/box_loss 1.03857
wandb:            val/cls_loss 0.43343
wandb:            val/dfl_loss 1.01223

Two notes:
1. The author of the dataset, like me, accidentally confused the king and queen pieces. TODO: just reverse the labels.
2. We're early in testing, but it looks like these models in general will sometimes have difficulty distinguishing black pawns and black bishops.