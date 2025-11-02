# image_inference_test.py
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ckpt_path = "./runs/chess_exp/best.pt"
ckpt = torch.load(ckpt_path, map_location=device)

# The checkpoint is a dict, not a bare state_dict
state_dict = ckpt["model"] if "model" in ckpt else ckpt

# --- infer number of classes from checkpoint ---
# this key name matches torchvision's Faster R-CNN heads
num_classes = state_dict["roi_heads.box_predictor.cls_score.weight"].shape[0]

# Recreate the model with the SAME number of classes
model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# --- run a test image ---
img = Image.open("./data/test_frames/orig_20251027_204250_frame0225.jpg").convert("RGB")
img_tensor = to_tensor(img).unsqueeze(0).to(device)

with torch.no_grad():
    pred = model(img_tensor)[0]

# We don't actually know your real label map, so let's not guess.
# We'll print IDs, and you can map them to names later.
score_thresh = 0.5
print(f"Found {len(pred['boxes'])} detections (before threshold)")
for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
    if score >= score_thresh:
        print(f"  class_id={label.item()}  score={score:.3f}  box={box.tolist()}")