import os
import random
import shutil

# Rutas que tú usas
source_path = "/home/usercimatmty/Javier_Alvarado_Proyecto/mbg-v2_sub"
dest_path   = "/home/usercimatmty/Javier_Alvarado_Proyecto/dataset"

# Carpetas destino ya creadas
images_train = os.path.join(dest_path, "images/train")
images_val   = os.path.join(dest_path, "images/val")
labels_train = os.path.join(dest_path, "labels/train")
labels_val   = os.path.join(dest_path, "labels/val")

# Crear por si acaso
os.makedirs(images_train, exist_ok=True)
os.makedirs(images_val, exist_ok=True)
os.makedirs(labels_train, exist_ok=True)
os.makedirs(labels_val, exist_ok=True)

# Obtener todos los archivos .jpg/.png de la carpeta origen
all_images = [
    f for f in os.listdir(source_path)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Total de imágenes encontradas: {len(all_images)}")

# Mezclar aleatoriamente
random.shuffle(all_images)

# División 80/20
train_size = int(len(all_images) * 0.8)
train_files = all_images[:train_size]
val_files   = all_images[train_size:]

def copy_files(file_list, img_dest, label_dest):
    for img_file in file_list:
        img_src = os.path.join(source_path, img_file)
        img_dst = os.path.join(img_dest, img_file)

        # Label correspondiente
        label_name = img_file.rsplit(".", 1)[0] + ".txt"
        label_src = os.path.join(source_path, label_name)
        label_dst = os.path.join(label_dest, label_name)

        if os.path.exists(label_src):
            shutil.copy(img_src, img_dst)
            shutil.copy(label_src, label_dst)
        else:
            print(f"⚠️ WARNING: No se encontró etiqueta para {img_file}")

# Copiar train y val
print("Copiando archivos de entrenamiento (80%)...")
copy_files(train_files, images_train, labels_train)

print("Copiando archivos de validación (20%)...")
copy_files(val_files, images_val, labels_val)

print("\n✅ División completada exitosamente")
print(f"Imágenes train: {len(train_files)}")
print(f"Imágenes val:   {len(val_files)}")
