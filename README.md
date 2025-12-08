# Detección de Criaderos de Mosquitos con YOLO y Data Augmentation

Este proyecto busca detectar potenciales criaderos de mosquitos en imágenes aéreas capturadas con drones, utilizando modelos de detección de objetos basados en **YOLO**.  
Inicialmente se propuso entrenar una **red GAN** para generar imágenes sintéticas de la clase minoritaria; sin embargo, debido a problemas de convergencia, falta de datos y baja calidad en las imágenes generadas, el enfoque fue reemplazado por un esquema robusto de **data augmentation tradicional**, logrando mejorar el balance entre clases y evaluar su impacto en el rendimiento del modelo.

Este repositorio contiene el pipeline completo para procesar los datos, aplicar aumentación, entrenar los modelos YOLO y comparar los resultados.

# Objetivo del Proyecto

Evaluar el impacto de la **aumentación de datos** en el rendimiento de modelos YOLO para la detección de potenciales criaderos de mosquitos en imágenes aéreas.

Para ello, se entrenaron dos modelos:

1. **Modelo base (sin augmentación):**  
   Entrenado únicamente con las imágenes originales.

2. **Modelo con augmentación:**  
   Entrenado con un dataset aumentado mediante transformaciones tradicionales
   (rotaciones, flips, escalado, jittering, etc.) que incrementan la variabilidad
   visual y ayudan a balancear clases minoritarias.

El objetivo principal es **comparar el desempeño entre ambos modelos**, evaluando mejoras en:

- Precisión (Precision)  
- Recall  
- mAP50  
- mAP50–95  
- Desempeño en clases minoritarias  

y demostrar cómo un esquema de augmentación bien diseñado puede mejorar la capacidad del modelo para detectar objetos relevantes en escenarios reales.

---

#  Estructura del Repositorio
```bash
Proyecto-Vision-Computacional/
│
├── Yolo12-Small/ # Pipeline principal del modelo YOLOv12-Small
│ ├── data.yaml # Configuración del dataset YOLO
│ ├── yolo12s.pt # Pesos finales del modelo entrenado
│ ├── extract_objects.py # Extracción de objetos recortados
│ ├── extract_patches_for_gan.py # Preparación de parches para GAN
│ ├── split_yolo_dataset.py # División de imágenes en train/val
│ ├── train_dcgan_cubeta.py # GAN experimental (descartada)
│ ├── train_dcgan_maceta.py # GAN experimental (descartada)
│ └── (agregar scripts finales de entrenamiento YOLO si aplica)
│
└── README.md
```
