import os
from ultralytics import YOLO

# Note that I am considering only my own annotations set
# Parameter are made according to using google colab
def train_yolo_model():
  base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  model_path = os.path.join(base_dir, 'yolo11s-seg.pt')
  data_yaml = os.path.join(base_dir, 'data', 'data.yaml')

  # Fallback to current working directory if not found relative to script
  if not os.path.exists(model_path):
    model_path = 'yolo11s-seg.pt'
  if not os.path.exists(data_yaml):
    data_yaml = 'data.yaml'

  # 1. Load a pre-trained YOLO11 segmentation model (e.g., small version 's')
  model = YOLO(model_path)

  results = model.train(
      data=data_yaml, 
      epochs=50,
      patience=20,
      imgsz=1024,
      batch=8,
      device=0,
      workers=2,
      augment=True,
      cache=True
  )
  print(results)
  print('Training completed! Best weights saved in runs/segment/train/weights/best.pt')


if __name__ == '__main__':
  train_yolo_model()

