from ultralytics import YOLO
import torch


def main():
    # Ruta a los pesos entrenados (ajusta si el nombre cambia)
    dir_weights = "/home/francisco/Javier_Tiscareno_Proyecto/runs_mosquitos/train_yolov12m/weights/best.pt"
    dir_yaml = "/home/francisco/Javier_Tiscareno_Proyecto/data.yaml"

    # Dispositivo
    device = 0 if torch.cuda.is_available() else "cpu"

    # Cargar modelo
    model = YOLO(dir_weights)

    # Validación
    results = model.val(
        data=dir_yaml,
        imgsz=1024,
        batch=4,
        conf=0.5,
        iou=0.7,
        device=device,
        split="val",
        project="runs_mosquitos",
        name="val_yolov12m",
        exist_ok=True,
    )

    print(results)


if __name__ == "__main__":
    main()
