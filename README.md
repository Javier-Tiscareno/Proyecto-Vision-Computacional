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
│ ├── train_yolo12s.py # Modelo base de YOLO
│ ├── copy_paste.py # Copy-Paste Augmentación
| └── train_yolo12_optimized.py # Modeo YOLO con augmentacion de clases minoritarias
└── README.md
```
#  Pipeline Completo del Proyecto

El flujo de trabajo del proyecto se diseñó para comparar dos enfoques de entrenamiento:

1. **Modelo base YOLOv12-Small** (sin augmentación)  
2. **Modelo YOLOv12-Small optimizado** (con augmentación orientada a clases minoritarias)

A continuación se describe el pipeline completo, organizado según los scripts dentro del repositorio.

---

## 1. Organización y División del Dataset  
**Script:** `split_yolo_dataset.py`

- Toma las imágenes y etiquetas originales.  
- Genera las carpetas requeridas por YOLO:  
  - `images/train`, `images/val`  
  - `labels/train`, `labels/val`  
- Realiza una división **80% entrenamiento / 20% validación**.  
- Verifica que cada imagen tenga su archivo `.txt` correspondiente.

Este paso prepara el dataset base para ambos modelos (con y sin augmentación).

---

## 2. Extracción de Objetos para Análisis y Pruebas  
**Scripts:**  
- `extract_objects.py`  
- `extract_patches_for_gan.py`

Propósitos:

- Extraer recortes de objetos, especialmente **clases minoritarias** (cubeta, maceta).  
- Generar parches para analizar la distribución visual de las clases.  
- Crear datasets auxiliares para experimentos con GAN (posteriormente descartados).

Este paso **no afecta directamente al entrenamiento**, pero permitió estudiar el desbalance de clases.

---

## 3. Intento de Generación Sintética con GAN (Descartado)  
**Scripts:**  
- `train_dcgan_cubeta.py`  
- `train_dcgan_maceta.py`

Se entrenaron dos DCGAN independientes para generar imágenes sintéticas de:

- **cubeta**  
- **maceta**

Motivación: aumentar clases minoritarias.

Problemas encontrados:

- Convergencia inestable  
- Artefactos y falta de realismo  
- Dataset pequeño → GAN poco robusto  

Por ello, este enfoque fue **descartado**, manteniendo los scripts como evidencia experimental.

---

## 4. Entrenamiento del Modelo Base YOLOv12-Small  
**Script:** `train_yolo12s.py`

Este modelo se entrena **únicamente con el dataset original**, sin aumentación adicional.  
Sirve como punto de comparación para medir la mejora del modelo optimizado.

Configuración:

- Arquitectura: YOLOv12-Small  
- Dataset definido en `data.yaml`  
- Hiperparámetros estándar  
- Entrenamiento desde cero o desde pesos base

Salida principal:

- `yolo12s.pt` → pesos del modelo base

---

## 5. Estrategias de Aumentación para Corregir el Desbalance
Se incluye una estrategia de augmentación específica para incrementar la representatividad de:

- **cubeta**  
- **maceta**

### 5.1 Copy-Paste Augmentation  
**Script:** `copy_paste.py`

- Toma recortes de objetos minoritarios.  
- Los inserta estratégicamente en nuevas imágenes.  
- Genera nuevos ejemplos realistas sin afectar la distribución de fondo.

### 5.2 Aumentación Combinada y Entrenamiento Optimizado  
**Script:** `train_yolo12_optimized.py`

Este es el **pipeline final de entrenamiento**, que incorpora:

- Copy-Paste  
- Transformaciones geométricas  
- Jitter de color  
- Variaciones aleatorias en escala, posición y orientación  
- Mezcla con imágenes originales

Este modelo es evaluado contra el modelo base para medir la mejora en:

- precisión  
- recall  
- mAP50  
- mAP50–95  
- detección de clases minoritarias

---

## 6. Evaluación Final de Modelos  
Ambos modelos (base y optimizado) se evalúan con el mismo conjunto de validación.

Métricas principales:

- Precision  
- Recall  
- mAP@50  
- mAP@50–95  
- Análisis específico por clase  

Este paso permite cuantificar el impacto de la aumentación en clases minoritarias.

---

# 🧠 Resumen del Pipeline

1. **División del dataset** → `split_yolo_dataset.py`  
2. **Análisis de clases minoritarias** → `extract_objects.py`, `extract_patches_for_gan.py`  
3. **Intento de GAN (documentado, pero descartado)** → `train_dcgan_*`  
4. **Entrenamiento del modelo base** → `train_yolo12s.py`  
5. **Aumentación de datos** → `copy_paste.py`  
6. **Modelo YOLO optimizado con augmentación** → `train_yolo12_optimized.py`  
7. **Evaluación comparativa** → métricas de validación YOLO  

---

Este pipeline refleja fielmente la estructura y metodología del proyecto, destacando la comparación entre modelos con y sin aumentación.

# Requerimientos e Instalación

Este proyecto utiliza Python y librerías especializadas en visión computacional y deep learning.
Para instalar las dependencias principales, utilice el archivo requirements.txt:

pip install -r requirements.txt


Nota: El archivo requirements.txt no incluye PyTorch, ya que la versión adecuada depende de si se utilizará GPU y de la versión de CUDA instalada.
Esto es una práctica estándar en proyectos profesionales.

## Instalación de PyTorch con GPU (CUDA)

Para entrenar YOLO con aceleración por GPU, instale la versión adecuada de PyTorch según su sistema.

## Instalación recomendada (CUDA 12.1)

Si su GPU NVIDIA soporta CUDA moderno:

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

## Alternativa (CUDA 11.8)

Si su entorno utiliza drivers más antiguos:

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

✔ Verificar que PyTorch detecta la GPU
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


Si ambos comandos devuelven información válida, su entorno está correctamente configurado para entrenamiento acelerado.

