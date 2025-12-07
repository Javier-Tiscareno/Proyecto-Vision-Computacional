from ultralytics import YOLO

# 🚀 Cargar modelo base YOLOv12s
model = YOLO("/home/javier/Javier_Alvarado_Proyecto/yolo12s.pt")

# 🚀 Entrenamiento
model.train(
    data="/home/javier/Javier_Alvarado_Proyecto/data.yaml",
    epochs=200,
    batch=8,
    imgsz=1024,
    device=0,
    name="train_optimized",

    # --------------------------------------------------------
    # 🔥 OPTIMIZACIÓN PARA CLASES MINORITARIAS
    # --------------------------------------------------------
    cls=0.6,          # aumenta penalización a la clasificación incorrecta
    kobj=1.4,         # refuerza objetos pequeños/minoritarios

    # --------------------------------------------------------
    # 🔥 WARMUP CUSTOMIZADO
    # --------------------------------------------------------
    warmup_epochs=5,          # antes era 3 → más estabilidad
    warmup_momentum=0.8,      # establece momentum inicial
    warmup_bias_lr=0.1,       # evita explosiones de gradiente al inicio

    # --------------------------------------------------------
    # 🔥 SCHEDULER CUSTOMIZADO
    # --------------------------------------------------------
    lr0=0.01,      # learning rate inicial
    lrf=0.001,     # learning rate final más bajo → refinamiento fino
    cos_lr=True,   # activa cosine decay

    # --------------------------------------------------------
    # 🔥 AUGMENTACIONES CONTROLADAS
    # --------------------------------------------------------
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    flipud=0.05,
    degrees=10,
    translate=0.1,
    scale=0.5,
    copy_paste=0.2,       # moderado (tu dataset ya tiene GAN y copy-paste)
    auto_augment="randaugment",
    close_mosaic=10,

    # --------------------------------------------------------
    # 🔥 EARLY STOPPING
    # --------------------------------------------------------
    patience=30,          # detiene si no mejora en 30 épocas
    deterministic=True,   # reproducibilidad total

    # --------------------------------------------------------
    # 🔥 OTROS
    # --------------------------------------------------------
    workers=8,
    save=True,
    val=True
)
