from ultralytics import YOLO

# -------------------------------------------------------
# 1. Cargar modelo YOLOv12s
# -------------------------------------------------------

# Opción A: Cargar arquitectura desde YAML (opcional)
# model = YOLO("yolo12s.yaml")

# Opción B: Cargar pesos preentrenados (recomendado)
model = YOLO("/home/usercimatmty/Javier_Alvarado_Proyecto/yolo12s.pt")


# Si quieres: arquitectura YAML + pesos (transfer learning)
# model = YOLO("yolo12s.yaml").load("yolo12s.pt")


# -------------------------------------------------------
# 2. Entrenamiento del modelo
# -------------------------------------------------------

model.train(
    data="/home/usercimatmty/Javier_Alvarado_Proyecto/data.yaml",
    epochs=200,
    imgsz=1024,
    batch=8,
    device=0,          # GPU 0 (o cambia a 1 si tu servidor usa otra)
    patience=100,      # early stopping
    workers=8,         # threads para cargar datos (ajusta si te da error)
    optimizer="auto"   # deja que YOLO elija el mejor optimizador
)

# -------------------------------------------------------
# 3. (Opcional) Guardar resultados adicionales
# -------------------------------------------------------
print("\n🚀 Entrenamiento finalizado. Modelo entrenado disponible en:")
print("   runs/detect/train/weights/best.pt\n")
