import os
import yaml
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# ----------------------
# config-ish stuff
# ----------------------
CKPT_PATH = "./runs/chess_exp/best.pt"
IMG_PATH = "./data/test_frames/chess_test.jpg"
YAML_PATH = "./chess_pieces.yolov8/data.yaml"
OUT_DIR = "./data/test_frame_detections"
OUT_PATH = os.path.join(OUT_DIR, "test_frame_vis.jpg")
SCORE_THRESH = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# 1) load checkpoint
# ----------------------
ckpt = torch.load(CKPT_PATH, map_location=device)
state_dict = ckpt["model"] if "model" in ckpt else ckpt

# infer num_classes from the head
num_classes = state_dict["roi_heads.box_predictor.cls_score.weight"].shape[0]

# ----------------------
# 2) rebuild model
# ----------------------
model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ----------------------
# 3) load label names from YAML
# ----------------------
def load_names(yaml_path, num_classes_guess=None):
    """
    Supports:
      names: ['pawn', 'rook', ...]
    or:
      names: {0: pawn, 1: rook, ...}
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # YOLO-style is usually under "names"
    names = data.get("names", None)
    if names is None:
        # fallback: maybe "classes"
        names = data.get("classes", None)

    if names is None:
        # last resort: make dummies
        if num_classes_guess is None:
            return None
        return {i: f"class_{i}" for i in range(num_classes_guess)}

    # dict or list?
    if isinstance(names, dict):
        return names  # already index -> name
    elif isinstance(names, list):
        # turn into dict
        return {i: name for i, name in enumerate(names)}
    else:
        # weird format
        if num_classes_guess is None:
            return None
        return {i: f"class_{i}" for i in range(num_classes_guess)}

names_map = load_names(YAML_PATH, num_classes_guess=num_classes)

# NOTE on indexing:
# - torchvision detection models usually output labels starting at 1 (0 is background)
# - YOLO YAMLs usually define names starting at 0
# so we'll try name = names_map[label-1] if label-1 in map
def label_to_name(label_id: int) -> str:
    if names_map is None:
        return f"class_{label_id}"
    # try YOLO-style (0-based) first
    if (label_id - 1) in names_map:
        return str(names_map[label_id - 1])
    # otherwise try 1-based
    if label_id in names_map:
        return str(names_map[label_id])
    # fallback
    return f"class_{label_id}"

# ----------------------
# 4) run inference
# ----------------------
img = Image.open(IMG_PATH).convert("RGB")
img_tensor = to_tensor(img).unsqueeze(0).to(device)

with torch.no_grad():
    pred = model(img_tensor)[0]

# ----------------------
# 5) draw boxes
# ----------------------
# make sure output dir exists
os.makedirs(OUT_DIR, exist_ok=True)

draw_img = img.copy()
draw = ImageDraw.Draw(draw_img)

# try to get a font; if fails, PIL default
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 18)
except:
    font = ImageFont.load_default()

for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
    score = float(score)
    if score < SCORE_THRESH:
        continue

    box = [float(b) for b in box]
    x1, y1, x2, y2 = box

    class_name = label_to_name(int(label))
    text = f"{class_name} {score:.2f}"

    # box
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

    # text background
    text_w, text_h = draw.textsize(text, font=font)
    text_bg = [x1, y1 - text_h - 4, x1 + text_w + 4, y1]
    draw.rectangle(text_bg, fill=(255, 0, 0))
    # text
    draw.text((x1 + 2, y1 - text_h - 2), text, fill=(255, 255, 255), font=font)

# ----------------------
# 6) save
# ----------------------
draw_img.save(OUT_PATH)
print(f"Saved detections to: {OUT_PATH}")