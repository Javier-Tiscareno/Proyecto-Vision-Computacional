from ultralytics import YOLO
import torch


def main():
    # Usa tus pesos entrenados
    dir_weights = "/home/francisco/Javier_Tiscareno_Proyecto/runs_mosquitos/train_yolov12m/weights/best.pt"

    # Carpeta o lista de imágenes a predecir
    source = "/home/francisco/Javier_Tiscareno_Proyecto/dataset/images/val"

    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(dir_weights)

    model.predict(
        source=source,
        imgsz=1024,
        conf=0.5,
        device=device,
        save=True,                    # guarda imágenes con cajas
        project="runs_mosquitos",
        name="pred_yolov12m",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
