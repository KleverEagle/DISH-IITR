#Takes in 2 label files as from CVAT (one seg and one bounding box) and converts them into the seg
# way (,ie. converting the detection box format to the polygon format)

import glob
import os

# Paths to your exported CVAT label folders
SEG_LABELS_DIR = ''
DET_LABELS_DIR = ''
OUTPUT_DIR = ''

os.makedirs(OUTPUT_DIR, exist_ok=True)


def convert_box_to_polygon(cls_id, x_c, y_c, w, h):
  # Converts 5-value bbox into an 8-value polygon rectangle
  x1, y1 = max(0.0, x_c - w / 2), max(0.0, y_c - h / 2)
  x2, y2 = min(1.0, x_c + w / 2), max(0.0, y_c - h / 2)
  x3, y3 = min(1.0, x_c + w / 2), min(1.0, y_c + h / 2)
  x4, y4 = max(0.0, x_c - w / 2), min(1.0, y_c + h / 2)
  return f'{cls_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}\n'


# Process each text file
for seg_file in glob.glob(os.path.join(SEG_LABELS_DIR, '*.txt')):
  filename = os.path.basename(seg_file)
  det_file = os.path.join(DET_LABELS_DIR, filename)

  merged_lines = []

  # 1. Keep Nuclei polygons from Segmentation export (Class 6)
  if os.path.exists(seg_file):
    with open(seg_file, 'r') as f:
      for line in f:
        parts = line.strip().split()
        if parts and int(parts[0]) == 6:  # Nucleus
          merged_lines.append(line)

  # 2. Add Signal boxes from Detection export (Classes 0 to 5) converted to polygons
  if os.path.exists(det_file):
    with open(det_file, 'r') as f:
      for line in f:
        parts = line.strip().split()
        if parts and int(parts[0]) in [0, 1, 2, 3, 4, 5]:  # Signals
          cls_id, x_c, y_c, w, h = map(float, parts)
          poly_line = convert_box_to_polygon(int(cls_id), x_c, y_c, w, h)
          merged_lines.append(poly_line)

  # 3. Save merged label file
  with open(os.path.join(OUTPUT_DIR, filename), 'w') as f:
    f.writelines(merged_lines)

print('Successfully merged Detection and Segmentation labels!')
