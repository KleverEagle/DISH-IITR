#Takes a image and outputs the corresponding nucleus wise signal count (incl cords)

from ultralytics import YOLO

def detect_all_objects_globally(model_path, image_path):
  # 1. Load your trained model weights
  model = YOLO(model_path)

  # 2. Run inference with a low confidence threshold to catch weak signals
  results = model(image_path, conf=0.05)
  result = results[0]

  # 3. Initialize overall counters
  total_nuclei = 0
  signal_counts = {'black': 0, 'pink': 0, 'fusion': 0}
  detected_signals = []

  # 4. Parse all detections in the image
  if result.boxes is not None:
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    for box, cls_id, conf in zip(boxes, classes, confs):
      class_name = model.names[int(cls_id)]

      if class_name == 'Nucleus':
        total_nuclei += 1
      elif class_name in ['black(Her2)', 'pink(Chr17)', 'fusion(Her2+Chr17)']:
        if 'black' in class_name:
          sig_type = 'black'
        elif 'pink' in class_name:
          sig_type = 'pink'
        else:
          sig_type = 'fusion'

        signal_counts[sig_type] += 1
        detected_signals.append({
            'type': sig_type,
            'box': box,
            'conf': conf
        })

  # 5. Print the global image results
  print('\n--- Global Image Detection Results ---')
  print(f'Total Nuclei Detected: {total_nuclei}')
  print(f'Total Black (HER2) Signals: {signal_counts["black"]}')
  print(f'Total Pink (Chr17) Signals: {signal_counts["pink"]}')
  print(f'Total Fusion Signals: {signal_counts["fusion"]}')

  # Optional: Print coordinates and confidence of the found signals
  if detected_signals:
    print('\n--- Signal Details ---')
    for i, sig in enumerate(detected_signals):
      print(
          f'Signal #{i+1}: {sig["type"].upper()} | '
          f'Conf: {sig["conf"]:.2f} | Box: {sig["box"]}'
      )
  else:
    print('\nNo signals were detected anywhere in the image.')

  return total_nuclei, signal_counts, detected_signals


if __name__ == '__main__':
  trained_weights = 'best.pt'
  test_image = 'C:/Dev/DISH/data/images/test/503763SL4_y1200_x1024.png'

  detect_all_objects_globally(trained_weights, test_image)
