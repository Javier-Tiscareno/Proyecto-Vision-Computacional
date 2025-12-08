#!/usr/bin/env python3
import subprocess
from pathlib import Path

# Ruta base del proyecto
BASE = Path("/home/francisco/Javier_Tiscareno_Proyecto")

PYTHON = "python3"
SCRIPT = BASE / "train_dcgan.py"

def run_dcgan(data_subdir: str, out_subdir: str,
              epochs: int = 50, batch_size: int = 64, image_size: int = 64):
    data_dir = BASE / "gan_patches" / data_subdir
    out_dir = BASE / "gan_models" / out_subdir

    cmd = [
        PYTHON,
        str(SCRIPT),
        "--data_dir", str(data_dir),
        "--out_dir", str(out_dir),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--image_size", str(image_size),
    ]

    print("\nEjecutando:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("✅ Terminado:", data_subdir)


def main():
    # DCGAN para CUBETA
    run_dcgan(data_subdir="cubeta", out_subdir="cubeta",
              epochs=50, batch_size=64, image_size=64)

    # DCGAN para MACETA
    run_dcgan(data_subdir="maceta", out_subdir="maceta",
              epochs=50, batch_size=64, image_size=64)


if __name__ == "__main__":
    main()
