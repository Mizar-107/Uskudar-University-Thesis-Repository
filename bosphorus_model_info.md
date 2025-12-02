# Bosphorus Ship Detection Model

## Model Details
- **Architecture:** YOLO11s (small variant, ~9M parameters)
- **Task:** Object detection (ships/vessels)
- **Classes:** 1 class - `gemiler` (ships)
- **Input Size:** 1088x1088 (optimized for 1920x1080 source images)

## Training Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Optimizer | AdamW |
| Learning Rate | 0.001 → 0.01 (cosine) |
| Warmup | 3 epochs |
| Batch Size | Auto |
| Rectangular Training | Yes |
| AMP | Enabled |

## File Locations
```
# Trained model weights
/content/drive/MyDrive/Uskudar Uni Master/YOLO Model klasor/runs/bosphorus_yolo11s_10884/weights/best.pt

# Dataset
/content/drive/MyDrive/Uskudar Uni Master/YOLO Model klasor/Custom Dataset/bogaz_v_1.v3i.yolov12/

# data.yaml location
/content/drive/MyDrive/Uskudar Uni Master/YOLO Model klasor/Custom Dataset/bogaz_v_1.v3i.yolov12/data.yaml
```

## Quick Usage
```python
from ultralytics import YOLO

# Load model
model = YOLO("/content/drive/MyDrive/Uskudar Uni Master/YOLO Model klasor/runs/bosphorus_yolo11s_10884/weights/best.pt")

# Inference
results = model.predict(source="image.jpg", conf=0.25, imgsz=1088)

# Validation
metrics = model.val(data="/content/drive/MyDrive/Uskudar Uni Master/YOLO Model klasor/Custom Dataset/bogaz_v_1.v3i.yolov12/data.yaml")
```

## Notes
- Model trained on Google Colab with T4 GPU
- Dataset is Roboflow export (YOLOv12 format, compatible with YOLO11)
- Use `rect=True` for inference on non-square images
- Recommended confidence threshold: 0.25
