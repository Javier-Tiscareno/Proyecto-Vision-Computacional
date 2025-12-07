import glob
import os

base_dir = "dataset/labels"  # relativo a /home/francisco/Javier_Tiscareno_Proyecto
splits = ["train", "val"]

for split in splits:
    pattern = os.path.join(base_dir, split, "*.txt")
    files = glob.glob(pattern)
    print(f"Procesando {len(files)} archivos en {split}...")

    for path in files:
        new_lines = []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])

                # Aquí asumimos que NO hay clase 0 (charco)
                if cls == 0:
                    # Si pasa esto, algo raro hay en tus datos
                    print(f"[ADVERTENCIA] Encontré clase 0 en {path}: {line.strip()}")
                    continue

                # Restar 1: 1->0, 2->1, 3->2, 4->3
                cls = cls - 1
                parts[0] = str(cls)
                new_lines.append(" ".join(parts))

        with open(path, "w") as f:
            if new_lines:
                f.write("\n".join(new_lines) + "\n")
            else:
                # Si el archivo queda vacío, lo dejamos vacío
                f.write("")
print("✅ Remapeo de clases terminado.")
