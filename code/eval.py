from ultralytics import YOLO

model = YOLO('best.pt')

# Runs validation over everything listed in all_eval_images.txt
metrics = model.val(
    data='data.yaml',
    imgsz=1024,
    batch=8,
    device='cpu'
)