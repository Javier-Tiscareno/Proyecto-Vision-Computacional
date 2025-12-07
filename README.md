# Detección de Criaderos de Mosquitos con YOLO y Data Augmentation

Este proyecto busca detectar potenciales criaderos de mosquitos en imágenes aéreas capturadas con drones, utilizando modelos de detección de objetos basados en **YOLO**.  
Inicialmente se propuso entrenar una **red GAN** para generar imágenes sintéticas de la clase minoritaria; sin embargo, debido a problemas de convergencia, falta de datos y baja calidad en las imágenes generadas, el enfoque fue reemplazado por un esquema robusto de **data augmentation tradicional**, logrando mejorar el balance entre clases y evaluar su impacto en el rendimiento del modelo.

Este repositorio contiene el pipeline completo para procesar los datos, aplicar aumentación, entrenar los modelos YOLO y comparar los resultados.
