"""
extract_objects.py
---------------------------------------------------------
Autor: Javier Alvarado
Descripción:
    Este script recorta objetos específicos (clases objetivo)
    a partir de imágenes anotadas en formato YOLO. 
    
    El objetivo principal es generar recortes individuales
    de objetos (cubetas, macetas, etc.) para futuros procesos
    de augmentación, análisis o entrenamiento.

Flujo del script:
    1. Leer imágenes y anotaciones .txt del dataset.
    2. Convertir coordenadas YOLO (normalizadas) a píxeles.
    3. Extraer cada objeto perteneciente a las clases objetivo.
    4. Expandir el bounding box original para capturar contexto.
    5. Guardar cada recorte como archivo PNG en OUTPUT.

Dependencias:
    - OpenCV
    - glob
    - os
---------------------------------------------------------
"""

import os
import cv2
import glob

# ============================================================
# Configuración de rutas
# ============================================================

# Directorio donde se encuentran las imágenes originales
IMAGES = "/home/javier/Javier_Alvarado_Proyecto/dataset/images/train/"

# Directorio donde se encuentran las etiquetas YOLO
LABELS = "/home/javier/Javier_Alvarado_Proyecto/dataset/labels/train/"

# Directorio donde se guardarán los recortes generados
OUTPUT = "/home/javier/augmentation/copies/"
os.makedirs(OUTPUT, exist_ok=True)

# Clases objetivo según índice YOLO
# 1 = cubeta, 4 = maceta
TARGET_CLASSES = [1, 4]

count = 0  # Contador de recortes generados

# ============================================================
# Procesamiento de cada archivo de etiqueta (.txt)
# ============================================================

for label_path in glob.glob(LABELS + "*.txt"):

    # Obtener la imagen correspondiente al archivo .txt
    image_path = IMAGES + os.path.basename(label_path).replace(".txt", ".png")

    # Si no existe .png, intentar con .jpg
    if not os.path.exists(image_path):
        image_path = IMAGES + os.path.basename(label_path).replace(".txt", ".jpg")

    img = cv2.imread(image_path)

    # Saltar si la imagen no pudo cargarse
    if img is None:
        continue

    h, w = img.shape[:2]  # Tamaño de la imagen

    # ========================================================
    # Procesar cada línea del archivo de etiqueta
    # ========================================================
    with open(label_path, "r") as f:
        for line in f.readlines():

            # Formato YOLO: cls x_center y_center bw bh (normalizados)
            cls, x, y, bw, bh = map(float, line.split())
            cls = int(cls)

            # Saltar objetos que no sean de las clases objetivo
            if cls not in TARGET_CLASSES:
                continue

            # -----------------------------------------------
            # Convertir coordenadas normalizadas a píxeles
            # -----------------------------------------------
            x1 = int((x - bw/2) * w)
            y1 = int((y - bh/2) * h)
            x2 = int((x + bw/2) * w)
            y2 = int((y + bh/2) * h)

            # =====================================================
            # Expandir bounding box ×2 para incluir más contexto
            # =====================================================
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            bw2, bh2 = int((x2 - x1) * 2), int((y2 - y1) * 2)

            x1e = max(cx - bw2 // 2, 0)
            y1e = max(cy - bh2 // 2, 0)
            x2e = min(cx + bw2 // 2, w)
            y2e = min(cy + bh2 // 2, h)

            crop = img[y1e:y2e, x1e:x2e]  # Recorte final

            # =====================================================
            # Guardar el recorte con nombre único
            # =====================================================
            save_path = f"{OUTPUT}/{os.path.basename(image_path)}_cls{cls}_{count}.png"
            cv2.imwrite(save_path, crop)
            count += 1

print(f"Total recortes generados: {count}")

