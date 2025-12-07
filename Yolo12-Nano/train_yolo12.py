
from ultralytics import YOLO
import torch


def main():
    # === RUTAS PRINCIPALES (AJUSTA AQUÍ SI CAMBIAS ALGO) ===
    dir_weights = "yolo12m.pt"  # pesos preentrenados (o ruta completa si lo tienes en otra carpeta)
    dir_yaml = "/home/francisco/Javier_Tiscareno_Proyecto/data.yaml"

    # === DETECCIÓN DEL DISPOSITIVO ===
    if torch.cuda.is_available():
        device = 0          # GPU 0
        batch_size = 4
        amp_flag = True
    else:
        device = "cpu"
        batch_size = 4      # bajar batch en CPU
        amp_flag = False

    print(f"Usando dispositivo: {device}")

    # === CARGAR MODELO ===
    model = YOLO(dir_weights)

    # === ENTRENAMIENTO ===
    model.train(
        data=dir_yaml,
        epochs=200,
        imgsz=1024,
        batch=batch_size,
        device=device,
        amp=amp_flag,
        workers=4,
        patience=100,                 # puedes subirlo si quieres early stopping más largo
        project="runs_mosquitos",
        name="train_yolov12m",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
