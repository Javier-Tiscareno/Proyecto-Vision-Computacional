import os
import cv2
import glob
import random

COPIES = "/home/javier/augmentation/copies/"
BACKGROUNDS = "/home/javier/augmentation/backgrounds/"

OUT_IMG = "/home/javier/augmentation/output/images/"
OUT_LAB = "/home/javier/augmentation/output/labels/"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LAB, exist_ok=True)

copy_paths = glob.glob(COPIES + "*.png")
bg_paths = glob.glob(BACKGROUNDS + "*.png") + glob.glob(BACKGROUNDS + "*.jpg")

new_count = 0

for i in range(300):  # generar 300 nuevas imágenes sintéticas
    bg_path = random.choice(bg_paths)
    bg = cv2.imread(bg_path)
    h, w = bg.shape[:2]

    # Insertar entre 1 y 3 objetos
    num_objects = random.randint(1, 3)
    yolo_lines = []

    for _ in range(num_objects):
        cp_path = random.choice(copy_paths)      # <-- CORRECCIÓN
        cp = cv2.imread(cp_path)
        if cp is None:
            continue

        ch, cw = cp.shape[:2]

        # Posición aleatoria
        x = random.randint(0, w - cw)
        y = random.randint(0, h - ch)

        # Mezcla con el fondo (más realista)
        roi = bg[y:y+ch, x:x+cw]
        blended = cv2.addWeighted(roi, 0.7, cp, 0.3, 0)
        bg[y:y+ch, x:x+cw] = blended

        # Coordenadas YOLO
        xc = (x + cw/2) / w
        yc = (y + ch/2) / h
        bw = cw / w
        bh = ch / h

        # Obtener clase del nombre del archivo
        cls = 1 if "_cls1_" in cp_path else 4   # <-- CORRECCIÓN

        yolo_lines.append(f"{cls} {xc} {yc} {bw} {bh}\n")

    # Guardar imagen sintética y label
    out_img = f"{OUT_IMG}/aug_{new_count}.jpg"
    out_lab = f"{OUT_LAB}/aug_{new_count}.txt"

    cv2.imwrite(out_img, bg)
    with open(out_lab, "w") as f:
        f.writelines(yolo_lines)

    new_count += 1

print(f"Generadas {new_count} imágenes sintéticas.")
