import os
import cv2
import glob

# Rutas
IMAGES = "/home/javier/Javier_Alvarado_Proyecto/dataset/images/train/"
LABELS = "/home/javier/Javier_Alvarado_Proyecto/dataset/labels/train/"

OUTPUT = "/home/javier/augmentation/copies/"
os.makedirs(OUTPUT, exist_ok=True)

TARGET_CLASSES = [1, 4]  # cubeta, maceta

count = 0

for label_path in glob.glob(LABELS + "*.txt"):
    image_path = IMAGES + os.path.basename(label_path).replace(".txt", ".png")

    if not os.path.exists(image_path):
        image_path = IMAGES + os.path.basename(label_path).replace(".txt", ".jpg")

    img = cv2.imread(image_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        for line in f.readlines():
            cls, x, y, bw, bh = map(float, line.split())

            cls = int(cls)
            if cls not in TARGET_CLASSES:
                continue

            # Convertir a píxeles
            x1 = int((x - bw/2) * w)
            y1 = int((y - bh/2) * h)
            x2 = int((x + bw/2) * w)
            y2 = int((y + bh/2) * h)

            # Expandir bbox ×2 para contexto
            cx, cy = (x1+x2)//2, (y1+y2)//2
            bw2, bh2 = int((x2-x1)*2), int((y2-y1)*2)

            x1e = max(cx - bw2//2, 0)
            y1e = max(cy - bh2//2, 0)
            x2e = min(cx + bw2//2, w)
            y2e = min(cy + bh2//2, h)

            crop = img[y1e:y2e, x1e:x2e]

            save_path = f"{OUTPUT}/{os.path.basename(image_path)}_cls{cls}_{count}.png"
            cv2.imwrite(save_path, crop)
            count += 1

print(f"Total recortes generados: {count}")
