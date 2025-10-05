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