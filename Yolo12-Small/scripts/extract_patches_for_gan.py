"""
extract_patches_for_gan.py
---------------------------------------------------------
Autor: Javier Alvarado
Descripción:
    Este script extrae parches (crops) de objetos específicos
    del dataset, con el propósito de alimentar una red GAN
    que generaría imágenes sintéticas de clases minoritarias.

    Aunque finalmente la GAN no se utilizó en el proyecto,
    este script documenta el proceso original de preparación
    del dataset para GAN y permite reproducirlo si se desea.

Flujo del script:
    1. Recorrer imágenes del dataset de entrenamiento.
    2. Leer anotaciones YOLO para ubicar objetos específicos.
    3. Convertir coordenadas normalizadas a píxeles absolutas.
    4. Extraer el parche correspondiente al bounding box.
    5. Redimensionarlo a 128x128 (tamaño típico para GAN).
    6. Guardarlo en carpetas separadas por clase.

Clases trabajadas:
    - ID 1: cubeta
    - ID 4: maceta

Dependencias:
    - OpenCV (cv2)
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

IMAGES_DIR = "/home/javier/Javier_Alvarado_Proyecto/dataset/images/train"
LABELS_DIR = "/home/javier/Javier_Alvarado_Proyecto/dataset/labels/train"

# Directorios donde se guardarán los parches por clase
OUT_CUBETA = "/home/javier/Javier_Alvarado_Proyecto/gan_patches/cubeta"
OUT_MACETA = "/home/javier/Javier_Alvarado_Proyecto/gan_patches/maceta"

os.makedirs(OUT_CUBETA, exist_ok=True)
os.makedirs(OUT_MACETA, exist_ok=True)

# Índices de clases según el dataset YOLO
ID_CUBETA = 1
ID_MACETA = 4


# ============================================================
# Función: Conversión de formato YOLO → coordenadas XYXY
# ============================================================

def yolo_to_xyxy(w, h, cx, cy, bw, bh):
    """
    Convierte coordenadas YOLO normalizadas a formato absoluto (x1, y1, x2, y2).

    Parámetros:
        w, h : ancho y alto de la imagen
        cx, cy : centro del bounding box (normalizado)
        bw, bh : ancho y alto del bounding box (normalizado)

    Retorna:
        x1, y1, x2, y2 : bounding box en coordenadas de imagen
    """
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    return x1, y1, x2, y2


# ============================================================
# Obtener lista de imágenes JPG y PNG
# ============================================================

image_paths = glob.glob(os.path.join(IMAGES_DIR, "*.jpg")) + \
              glob.glob(os.path.join(IMAGES_DIR, "*.png"))

# Contadores por clase
patch_count = {ID_CUBETA: 0, ID_MACETA: 0}

# ============================================================
# Procesamiento imagen por imagen
# ============================================================

for img_path in image_paths:

    name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(LABELS_DIR, name + ".txt")

    # Si no hay anotación correspondiente, saltar
    if not os.path.exists(label_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    # Leer anotaciones del archivo YOLO
    with open(label_path, "r") as f:
        lines = f.readlines()

    # ========================================================
    # Procesar cada línea de anotación
    # ========================================================
    for line in lines:
        parts = line.strip().split()

        # Validar formato correcto
        if len(parts) != 5:
            continue

        cls_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])

        # Tomar solo cubeta y maceta
        if cls_id not in (ID_CUBETA, ID_MACETA):
            continue

        # Convertir a píxeles
        x1, y1, x2, y2 = yolo_to_xyxy(w, h, cx, cy, bw, bh)

        # Limitar los bordes a la imagen
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        patch = img[y1:y2, x1:x2]

        # Evitar errores si el recorte está vacío
        if patch.size == 0:
            continue

        # Redimensionar a 128x128 para la GAN
        patch = cv2.resize(patch, (128, 128))

        # Seleccionar carpeta de salida
        if cls_id == ID_CUBETA:
            out_dir = OUT_CUBETA
        else:
            out_dir = OUT_MACETA

        # Guardar parche con índice incremental
        idx = patch_count[cls_id]
        out_name = f"{name}_cls{cls_id}_{idx:04d}.png"

        cv2.imwrite(os.path.join(out_dir, out_name), patch)
        patch_count[cls_id] += 1

# ============================================================
# Estadísticas finales
# ============================================================

print("Parches extraídos:")
print("Cubeta:", patch_count[ID_CUBETA])
print("Maceta:", patch_count[ID_MACETA])
