import os
import random
import shutil
from collections import defaultdict

# --- RUTAS A TU DATASET ---
ROOT = "/home/francisco/Javier_Tiscareno_Proyecto/dataset"
IM_TRAIN = os.path.join(ROOT, "images", "train")
LB_TRAIN = os.path.join(ROOT, "labels", "train")

# Número de clases actuales (SIN charco)
NUM_CLASSES = 4  # 0:cubeta, 1:llanta, 2:tinaco, 3:maceta

# Factor respecto a la clase mayoritaria
# 1.0 = igualar totalmente; 0.6 = llegar al 60% de la mayoritaria
TARGET_FACTOR = 0.6

IMG_EXTS = [".jpg", ".jpeg", ".png"]


def find_image_file(stem):
    """Devuelve la ruta de imagen para un 'stem' (nombre sin extensión)."""
    for ext in IMG_EXTS:
        path = os.path.join(IM_TRAIN, stem + ext)
        if os.path.exists(path):
            return path
    return None


def main():
    # -------------------------------------------------
    # 1) Contar instancias por clase y mapear imagen -> clases
    # -------------------------------------------------
    counts = [0] * NUM_CLASSES
    images_by_class = defaultdict(set)  # class_id -> set(stems)

    label_files = [f for f in os.listdir(LB_TRAIN) if f.endswith(".txt")]
    print(f"Label files encontrados: {len(label_files)}")

    for lf in label_files:
        stem = os.path.splitext(lf)[0]
        with open(os.path.join(LB_TRAIN, lf), "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                if 0 <= cid < NUM_CLASSES:
                    counts[cid] += 1
                    images_by_class[cid].add(stem)

    print("\n### Conteo de instancias por clase (antes del balanceo) ###")
    for cid, c in enumerate(counts):
        print(f"Clase {cid}: {c} instancias")

    max_count = max(counts)
    target = int(max_count * TARGET_FACTOR)
    print(f"\nClase mayoritaria = {counts.index(max_count)} "
          f"con {max_count} instancias")
    print(f"Objetivo para minoritarias (TARGET_FACTOR={TARGET_FACTOR}): {target}")

    # -------------------------------------------------
    # 2) Oversampling: duplicar imágenes de clases minoritarias
    # -------------------------------------------------
    created = 0

    for cid in range(NUM_CLASSES):
        current = counts[cid]
        if current >= target:
            print(f"\nClase {cid} ya tiene {current} instancias (>= {target}), "
                  "no se sobremuestrea.")
            continue

        stems = list(images_by_class[cid])
        if not stems:
            print(f"\nClase {cid} no tiene imágenes, se omite.")
            continue

        print(f"\nSobremuestreando clase {cid}: {current} -> {target}")
        # Necesitamos crear (target - current) instancias adicionales.
        # Cada copia mantiene las mismas etiquetas (Ultralytics después hace augmentación).
        while counts[cid] < target:
            stem = random.choice(stems)

            src_img = find_image_file(stem)
            src_lbl = os.path.join(LB_TRAIN, stem + ".txt")

            if src_img is None or not os.path.exists(src_lbl):
                print(f"Saltando {stem}, no se encontró imagen o label.")
                continue

            # Nuevo nombre
            new_stem = f"{stem}_aug{counts[cid]}"
            img_ext = os.path.splitext(src_img)[1]
            dst_img = os.path.join(IM_TRAIN, new_stem + img_ext)
            dst_lbl = os.path.join(LB_TRAIN, new_stem + ".txt")

            # Copiar archivos
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)

            # Actualizar contadores
            with open(dst_lbl, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    c2 = int(parts[0])
                    if 0 <= c2 < NUM_CLASSES:
                        counts[c2] += 1
                        images_by_class[c2].add(new_stem)

            created += 1

    print(f"\nSe crearon {created} nuevas imágenes (copias).")

    print("\n### Conteo de instancias por clase (después del balanceo) ###")
    for cid, c in enumerate(counts):
        print(f"Clase {cid}: {c} instancias")


if __name__ == "__main__":
    main()
