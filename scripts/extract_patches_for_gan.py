import os
import cv2
import glob

# Rutas
IMAGES_DIR = "/home/javier/Javier_Alvarado_Proyecto/dataset/images/train"
LABELS_DIR = "/home/javier/Javier_Alvarado_Proyecto/dataset/labels/train"

OUT_CUBETA = "/home/javier/Javier_Alvarado_Proyecto/gan_patches/cubeta"
OUT_MACETA = "/home/javier/Javier_Alvarado_Proyecto/gan_patches/maceta"

os.makedirs(OUT_CUBETA, exist_ok=True)
os.makedirs(OUT_MACETA, exist_ok=True)

# IDs de clases
ID_CUBETA = 1
ID_MACETA = 4

def yolo_to_xyxy(w, h, cx, cy, bw, bh):
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    return x1, y1, x2, y2

image_paths = glob.glob(os.path.join(IMAGES_DIR, "*.jpg")) + \
              glob.glob(os.path.join(IMAGES_DIR, "*.png"))

patch_count = {ID_CUBETA: 0, ID_MACETA: 0}

for img_path in image_paths:
    name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(LABELS_DIR, name + ".txt")
    if not os.path.exists(label_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])

        if cls_id not in (ID_CUBETA, ID_MACETA):
            continue

        x1, y1, x2, y2 = yolo_to_xyxy(w, h, cx, cy, bw, bh)

        # margen para no cortar mal
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        patch = img[y1:y2, x1:x2]
        if patch.size == 0:
            continue

        # redimensionar a tamaño fijo para el GAN
        patch = cv2.resize(patch, (128, 128))

        if cls_id == ID_CUBETA:
            out_dir = OUT_CUBETA
        else:
            out_dir = OUT_MACETA

        idx = patch_count[cls_id]
        out_name = f"{name}_cls{cls_id}_{idx:04d}.png"
        cv2.imwrite(os.path.join(out_dir, out_name), patch)
        patch_count[cls_id] += 1

print("Parches extraídos:")
print("Cubeta:", patch_count[ID_CUBETA])
print("Maceta:", patch_count[ID_MACETA])
