#Takes and image, and draws the boxes as the output of the YOLO model
# YOLO model outputs labels -> which drawn on the image

import cv2
import numpy as np
from ultralytics import YOLO


def draw_and_save_detections(
    model_path, image_path, output_path='annotated_output.png', conf=0.25
):
  # 1. Load trained model
  model = YOLO(model_path)

  # 2. Run inference
  results = model(image_path, conf=conf)
  result = results[0]

  # Load base image (BGR format)
  img = cv2.imread(image_path)
  if img is None:
    raise FileNotFoundError(f'Could not load image at {image_path}')

  # Optional: To draw on a BLANK canvas instead of overlaying on the original image,
  # uncomment the next line:
  # img = np.zeros_like(img)

  # Create a transparent overlay for filled polygons
  overlay = img.copy()

  # Define color scheme (BGR format)
  COLORS = {
      'Nucleus': (255, 100, 0),  # Blueish-Cyan outline/fill
      'black(Her2)': (0, 0, 0),  # Black
      'pink(Chr17)': (203, 192, 255),  # Light Pink
      'fusion(Her2+Chr17)': (0, 255, 255),  # Yellow
      'HER2_cluster3': (50, 50, 50),
      'HER2_cluster6': (30, 30, 30),
      'HER2_cluster12': (10, 10, 10),
  }

  if result.boxes is not None:
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    # --- Step 1: Draw Nuclei Segmentation Masks ---
    if result.masks is not None:
      masks = result.masks.xy
      for i, cls_id in enumerate(classes):
        class_name = model.names[int(cls_id)]
        if class_name == 'Nucleus' and i < len(masks):
          pts = masks[i].astype(np.int32)
          if len(pts) >= 3:
            # Fill polygon interior on overlay
            cv2.fillPoly(overlay, [pts], COLORS['Nucleus'])
            # Draw sharp polygon border on main image
            cv2.polylines(
                img, [pts], isClosed=True, color=(255, 50, 0), thickness=2
            )

    # Blend filled overlay with original image (40% opacity)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # --- Step 2: Draw Signal Detections ---
    for box, cls_id in zip(boxes, classes):
      class_name = model.names[int(cls_id)]

      if class_name != 'Nucleus' and class_name != 'exclude':
        x1, y1, x2, y2 = map(int, box)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        color = COLORS.get(class_name, (0, 0, 255))

        # Option A: Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Option B: Draw center dot for small signals (easier to see)
        cv2.circle(img, (center_x, center_y), radius=3, color=color, thickness=-1)

  # 3. Save the new image
  cv2.imwrite(output_path, img)
  print(f'Annotated image saved successfully to: {output_path}')


if __name__ == '__main__':
  draw_and_save_detections(
      model_path='',
      image_path='',
      output_path='mapped_nuclei_signals.png',
  )
