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