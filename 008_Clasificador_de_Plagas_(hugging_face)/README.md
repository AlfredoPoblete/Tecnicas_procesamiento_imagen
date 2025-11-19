
# Proyecto desarrollado para la materia Procesamiento Digital de Imágenes y Visión por Computadora.

**Autor**: Alfredo Poblete  
**Año**: 2025  
**Institución**: IFTS24

# Clasificador De Plagas En Cultivos Agricolas

## Descripción

Este proyecto aborda un problema real y de gran impacto en la agricultura: la detección temprana y clasificación de plagas o enfermedades en las hojas de las plantas, lo que permite tomar medidas a tiempo para salvar la cosecha.

## Modelo utilizado

- **Modelo preentrenado**: CLIP (Contrastive Language-Image Pre-training)
- **Tarea**: Clasificador Específico de Enfermedades/Plagas
- **Framework**: Transformers (Hugging Face)

## Manifestaciones Visuales de las Categorías de Hojas
### 1. Hoja sana 🍃
Apariencia: Debe ser el estándar de comparación.

**Características:**

* Color Uniforme: Verde intenso y brillante (el tono exacto dependerá de la especie de planta, pero debe ser el color esperado).

* Textura Lisa: Sin manchas, lesiones, agujeros o decoloraciones visibles.

* Forma y Tamaño: Completa, sin deformaciones ni rizado anormal, con bordes definidos.

* Importante: La imagen debe mostrar una hoja en pleno vigor y desarrollo.

### 2. Hoja dañada por insectos 🐞
Apariencia: El daño es físico y generalmente localizado.

**Características:**

* Agujeros: Presencia de perforaciones irregulares o grandes mordidas en el borde de la hoja.

* Marcas de Raspado: Cicatrices superficiales, a menudo en forma de líneas o parches plateados/blanquecinos (causados por trips o ácaros).

* Túneles: Si el daño es por larvas minadoras, se verán líneas o túneles claros dentro de la hoja.

* Puntos Negros/Marrón: Pequeñas manchas que pueden ser excrementos de insectos.

* A veces: El insecto (pulgón, oruga) puede estar presente en la imagen.

### 3. Hoja con presencia de hongos 🍄
Apariencia: El daño suele manifestarse como manchas o crecimientos superficiales.

**Características:**

* Manchas Circulares: Lesiones de forma definida, a menudo con un centro claro y un borde oscuro (o viceversa).

* Polvo/Vello: Presencia de una capa superficial que parece polvo blanco o gris (mildiu polvoroso) o vellosidad debajo de la hoja (mildiu velloso).

* Cambio de Color: Amarillamiento o pardeamiento que se extiende progresivamente desde las manchas.

* Textura: La hoja puede verse húmeda o, por el contrario, seca y quebradiza en las zonas afectadas (como el óxido de las royas).

### 4. Hoja con deficiencia de nutrientes 🟡
Apariencia: El daño es sistémico, afectando grandes áreas o la hoja entera de manera difusa.

**Características:**

* Clorosis (Amarillamiento): Es el signo más común. Puede ser:

* Generalizada: Toda la hoja amarillea (a menudo por falta de Nitrógeno).

* Intervenal: Solo el tejido entre las venas se pone amarillo, dejando las venas verdes (clásico de deficiencia de Magnesio o Hierro).

* Necrosis (Muerte del Tejido): Las puntas o los bordes de la hoja se vuelven marrones y secos (común en deficiencia de Potasio).

* Crecimiento Anormal: Hojas pequeñas o rizado generalizado, sin las marcas específicas de insectos u hongos.

## Cómo usar

1. Subí una imagen o usá tu cámara
2. Presioná "Clasificar"
3. Observá los resultados

## Instalación local

```bash
# Clonar repositorio
git clone [url-del-espacio]

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: .\venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
python app.py
```

## Comparación de Modelos

### Modelo Preentrenado (CLIP)

**Ventajas**:
  * No requiere entrenamiento
  * Funciona con cualquier categoría en lenguaje natural
  * Generaliza bien a diferentes contextos

**Desventajas**:
  * Menor precisión en tareas específicas
  * No se adapta al dominio particular

**Resultados en mi dataset**:
  * Precisión aproximada: 50% de los casos
  * Casos donde funciona bien: En categorias de deficiencia de Nutrientes
  * Casos donde falla: En hojas "afectadas por insectos" las detecta como "con presencia de hongos" esto posiblemente es devido a las similitudes que tienen los hongos con por ejemplo, los pulgones que tambien son de color blanco


### Modelo Personalizado (Teachable Machine)

**Ventajas**:
  * Alta precisión en la tarea específica
  * Adaptado al dominio de interés
  * Más rápido en inferencia

**Desventajas**:
  * Requiere recolectar y etiquetar datos
  * Solo funciona para las clases entrenadas
  * Puede sufrir overfitting con pocos datos

**Resultados en mi dataset**:
  * Precisión aproximada: 90% de los casos
  * Tamaño del dataset de entrenamiento: 70 imagenes por cada clase, en total 280 imagenes
  * Mejora respecto a CLIP: Obtenemos una mejora aproximadamente del 40%


# Análisis y Conclusiones del Producto Mínimo Viable (MVP)
Este MVP de clasificación de enfermedades de hojas, desarrollado con Gradio y modelos de Hugging Face/Keras, nos permite establecer una base de trabajo y validar diferentes estrategias de Visión por Computadora.

## 1. ¿Cuándo usar modelos preentrenados vs. personalizados?
* Preentrenados (CLIP): Para una prueba inicial rápida (Zero-Shot), cuando no se dispone de datos de entrenamiento o las categorías son amplias y conceptuales. Es ideal para validar rápidamente la viabilidad.
* Personalizados (Keras): Cuando se necesita la máxima precisión en un conjunto de clases fijo y específico, o cuando las características visuales son sutiles (ej. deficiencias nutricionales). Requiere datos etiquetados y tiempo de entrenamiento.

## 2. Aprendizaje clave del proceso de desarrollo
El proceso nos permite aprender sobre el ciclo de vida completo de un proyecto de Machine Learning para imágenes: desde la preparación del modelo (adaptación de pre-procesamiento, como la normalización [-1, 1] para Teachable Machine), la carga de activos (modelos y etiquetas), hasta la implementación en una interfaz web funcional (Gradio) y la comparación de rendimiento entre diferentes arquitecturas (modelo generalista vs. modelo especializado).

## 3. Posibles mejoras futuras
La mejora más inmediata para aumentar la utilidad agronómica del proyecto es incrementar la granularidad de las clases. Por ejemplo:
* Subclasificar las Deficiencias Nutricionales: En lugar de una sola clase genérica, expandir a deficiencia de Hierro, Nitrógeno, Potasio, etc., para ofrecer recomendaciones de tratamiento más precisas.
* Mejorar la robustez del modelo personalizado con un conjunto de datos más amplio y diverso.

## 4. Aplicaciones reales del proyecto
Este proyecto tiene aplicaciones directas en la agricultura de precisión :
* Diagnóstico Temprano para Agricultores: Permite identificar problemas (plagas o enfermedades) de manera inmediata en el campo con una simple foto, reduciendo la necesidad de análisis de laboratorio.
* Monitoreo Automatizado de Cultivos: Integración en sistemas de drones o cámaras estáticas para escanear grandes áreas y alertar sobre brotes de enfermedades.
* Herramienta Educativa: Puede servir como plataforma interactiva para estudiantes o nuevos técnicos agrícolas en el reconocimiento de patologías de plantas.
