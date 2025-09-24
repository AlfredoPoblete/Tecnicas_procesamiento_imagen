24 de Septiembre del 2025

* Alumno: Alfredo Poblete
* Profesor: Barreto Matias
* Ifts 24

---

# Procesamiento de Imágenes – Trabajo Integrador

## 📌 Introducción

Este proyecto integra los conceptos aprendidos sobre **procesamiento digital de imágenes** mediante el uso de **Python**, **OpenCV** y **Matplotlib**. La actividad consiste en trabajar con una imagen real tomada en un acuario de San Juan (un pez de la especie *Pseudocrenilabrus* macho) para aplicar diferentes técnicas de mejora, transformación y segmentación. El entorno de desarrollo utilizado fue **Google Colab**.

---

## 🎯 Objetivos

* Leer y visualizar una imagen propia.
* Mejorar el brillo y el contraste mediante **ecualización de histograma**.
* Realizar transformaciones geométricas como la **rotación**.
* Segmentar **regiones de interés (ROI)** del pez.
* Aplicar máscaras para resaltar y mejorar áreas específicas.
* Presentar de forma clara los resultados obtenidos.

---

## 🛠️ Pasos realizados

1. **Importación de librerías**
   Se cargaron las herramientas necesarias (`cv2`, `numpy`, `matplotlib`, etc.).

2. **Definición de funciones auxiliares**
   Se implementaron funciones para mostrar imágenes, crear máscaras de color y segmentar por HSV/K-means.

3. **Lectura y exploración de la imagen**

   * Carga de la imagen original del pez.
   * Conversión de BGR a RGB para visualización correcta.

4. **Mejora de brillo y contraste**

   * Uso de ecualización de histograma para resaltar detalles.
   * Ajuste del contraste para obtener una imagen más definida.

5. **Transformaciones geométricas**

   * Rotación de la imagen para alinear mejor al pez en el lienzo.

6. **Segmentación y máscaras**

   * Segmentación por rango de color en espacio HSV para detectar tonos específicos.
   * Segmentación automática con K-means para agrupar regiones cromáticas.
   * Creación de máscaras en blanco y negro para aislar el pez.

7. **Detección de contornos y anotaciones**

   * Uso de `cv2.findContours` para ubicar los objetos segmentados.
   * Dibujar rectángulos y agregar etiquetas a los objetos principales.

8. **Visualización de resultados**

   * Organización de los diferentes pasos en una grilla de subplots (`plt.subplots`).
   * Inclusión de títulos, espaciado y presentación clara para comparar resultados.

---

## 💡 Reflexión final

Este trabajo integrador permitió consolidar conocimientos de **procesamiento digital de imágenes** al aplicar de manera práctica técnicas fundamentales como ecualización de histograma, segmentación por color y clustering, rotaciones y manejo de máscaras.
El proceso destacó la importancia de **ajustar parámetros según la iluminación y el color del objeto**, así como la utilidad de **visualizaciones interactivas** para validar resultados. Además, demuestra cómo herramientas abiertas como OpenCV y Matplotlib son poderosas para resolver problemas de visión por computadora de manera accesible y reproducible.

---
