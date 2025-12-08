import os
from pathlib import Path
from PIL import Image

# === CONFIG ===
ROOT = Path("/home/francisco/Javier_Tiscareno_Proyecto/dataset")
IMG_DIR = ROOT / "images" / "train"
LBL_DIR = ROOT / "labels" / "train"

OUT_ROOT = Path("/home/francisco/Javier_Tiscareno_Proyecto/gan_patches")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Clases que queremos para GAN (ids YOLO actuales)
CLASS_NAMES = {
    0: "cubeta",
    1: "llanta",
    2: "tinaco",
    3: "maceta",
}

TARGET_SIZE = 64  # tamaño de patch cuadrado para DCGAN


def find_image(stem: str):
    """Busca imagen por stem sin extensión."""
    exts = [".png", ".jpg", ".jpeg", ".JPG", ".JPEG"]
    for ext in exts:
        p = IMG_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    label_files = sorted(LBL_DIR.glob("*.txt"))
    print(f"Encontré {len(label_files)} archivos de etiquetas.")

    counts = {cid: 0 for cid in CLASS_NAMES.keys()}

    for lbl_path in label_files:
        stem = lbl_path.stem

        # Ignorar datos augmentados
        if "_aug" in stem:
            continue

        img_path = find_image(stem)
        if img_path is None:
            print(f"[WARN] No encontré imagen para {lbl_path.name}, salto.")
            continue

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        with open(lbl_path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id not in CLASS_NAMES:
                continue

            # bbox YOLO: cx, cy, w, h (normalizados)
            cx, cy, bw, bh = map(float, parts[1:5])
            x_c = cx * W
            y_c = cy * H
            box_w = bw * W
            box_h = bh * H

            x1 = int(x_c - box_w / 2)
            y1 = int(y_c - box_h / 2)
            x2 = int(x_c + box_w / 2)
            y2 = int(y_c + box_h / 2)

            # recortar dentro de la imagen
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W - 1, x2)
            y2 = min(H - 1, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img.crop((x1, y1, x2, y2))

            # descartar parches demasiado pequeños
            if crop.size[0] < 16 or crop.size[1] < 16:
                continue

            # redimensionar a TARGET_SIZE x TARGET_SIZE
            crop = crop.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)

            out_dir = OUT_ROOT / CLASS_NAMES[cls_id]
            out_dir.mkdir(parents=True, exist_ok=True)

            out_name = f"{stem}_obj{i}.png"
            out_path = out_dir / out_name
            crop.save(out_path)

            counts[cls_id] += 1

    print("\nParches por clase:")
    for cid, name in CLASS_NAMES.items():
        print(f"Clase {cid} ({name}): {counts[cid]} parches")


if __name__ == "__main__":
    main()
