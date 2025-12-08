"""
split_yolo_dataset.py
---------------------------------------------------------
Autor: Javier Alvarado
Descripción:
    Divide un conjunto de datos YOLO en dos particiones:
    entrenamiento (80%) y validación (20%). Copia tanto las
    imágenes como sus archivos de etiqueta correspondientes.

    Este script asume que en la carpeta origen los nombres de
    las imágenes y las etiquetas corresponden entre sí, es decir:
        imagen.jpg  → etiqueta.txt

Flujo del script:
    1. Obtener todas las imágenes (.jpg/.png) del origen.
    2. Mezclarlas aleatoriamente.
    3. Separarlas en train/val según proporción elegida.
    4. Copiar imágenes y etiquetas al nuevo destino.

Dependencias:
    - shutil
    - os
    - random
---------------------------------------------------------
"""

import os
import random
import shutil

# ============================================================
# Configuración de rutas
# ============================================================

# Carpeta donde están actualmente las imágenes y etiquetas
source_path = "/home/usercimatmty/Javier_Alvarado_Proyecto/mbg-v2_sub"

# Carpeta destino donde se formará el dataset YOLO
dest_path = "/home/usercimatmty/Javier_Alvarado_Proyecto/dataset"

# Subdirectorios esperados por YOLO
images_train = os.path.join(dest_path, "images/train")
images_val   = os.path.join(dest_path, "images/val")
labels_train = os.path.join(dest_path, "labels/train")
labels_val   = os.path.join(dest_path, "labels/val")

# Crear carpetas destino si no existen
os.makedirs(images_train, exist_ok=True)
os.makedirs(images_val, exist_ok=True)
os.makedirs(labels_train, exist_ok=True)
os.makedirs(labels_val, exist_ok=True)

# ============================================================
# Obtener lista de imágenes disponibles
# ============================================================

all_images = [
    f for f in os.listdir(source_path)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Total de imágenes encontradas: {len(all_images)}")

# Mezclar aleatoriamente para evitar sesgos
random.shuffle(all_images)

# División 80/20
train_size = int(len(all_images) * 0.8)
train_files = all_images[:train_size]
val_files   = all_images[train_size:]


# ============================================================
# Función para copiar imágenes y etiquetas correspondientes
# ============================================================

def copy_files(file_list, img_dest, label_dest):
    """
    Copia imágenes y sus etiquetas asociadas a los directorios destino.

    Parámetros:
        file_list : lista de nombres de archivos de imagen
        img_dest  : ruta destino para las imágenes
        label_dest: ruta destino para las etiquetas
    """
    for img_file in file_list:

        img_src = os.path.join(source_path, img_file)
        img_dst = os.path.join(img_dest, img_file)

        # Archivo de etiqueta asociado
        label_name = img_file.rsplit(".", 1)[0] + ".txt"
        label_src = os.path.join(source_path, label_name)
        label_dst = os.path.join(label_dest, label_name)

        # Copiar solo si existe la etiqueta
        if os.path.exists(label_src):
            shutil.copy(img_src, img_dst)
            shutil.copy(label_src, label_dst)
        else:
            print(f"ADVERTENCIA: No se encontró etiqueta para {img_file}")


# ============================================================
# Copiar archivos a las carpetas de train y val
# ============================================================

print("Copiando archivos de entrenamiento (80%)...")
copy_files(train_files, images_train, labels_train)

print("Copiando archivos de validación (20%)...")
copy_files(val_files, images_val, labels_val)

print("\nDivisión completada exitosamente.")
print(f"Imágenes train: {len(train_files)}")
print(f"Imágenes val:   {len(val_files)}")
