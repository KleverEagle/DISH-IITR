from ultralytics import YOLO

# Note that I am considering only my own annotations set
#Parameter are made accordingt t using google colab
def train_yolo_model():
  # 1. Load a pre-trained YOLO11 segmentation model (e.g., small version 's')
  model = YOLO('yolo11s-seg.pt')

  results = model.train(
      data='data.yaml', 
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

