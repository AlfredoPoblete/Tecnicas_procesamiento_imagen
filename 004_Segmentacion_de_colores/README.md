# Trabajo Práctico de Segmentación de Colores

## 📋 Descripción General

Este proyecto implementa técnicas de segmentación de colores en procesamiento de imágenes digitales, aplicado específicamente a la detección de patrones característicos en arañas salticidae (arañas saltarinas). El trabajo demuestra cómo las computadoras "ven" y procesan los colores en las imágenes, y cómo podemos crear algoritmos para identificar automáticamente patrones específicos.

## 🎯 Objetivos

- **Comprender** el funcionamiento de los canales de color RGB en imágenes digitales
- **Desarrollar** algoritmos de segmentación basados en umbrales de color
- **Implementar** técnicas de máscaras para aislar regiones de interés
- **Experimentar** con diferentes parámetros para optimizar la detección
- **Visualizar** y analizar los resultados de la segmentación

## 🔬 Metodología

### 1. Análisis de Imagen
- Carga y procesamiento de imagen de araña salticidae
- Conversión de formato BGR a RGB para análisis correcto
- Exploración de propiedades de la imagen (dimensiones, canales)

### 2. Análisis de Píxeles
- Extracción y análisis de píxeles representativos de diferentes colores:
  - **Verde**: Ojos de la araña
  - **Marrón**: Patrones corporales
  - **Naranja**: Patrones característicos de señalización

### 3. Segmentación por Umbrales
Implementación de reglas basadas en valores RGB:
```
Rojo >= 200  (Alta intensidad de rojo)
Verde <= 165  (Baja intensidad de verde)
Azul <= 60    (Muy baja intensidad de azul)
```

### 4. Generación de Máscaras
- Creación de máscaras binarias para aislar regiones de color objetivo
- Aplicación de operadores lógicos (AND, NOT) para refinar la detección

## 📁 Estructura del Proyecto

```
004_Segmentacion_de_colores/
├── Lab_4_Salticidae.ipynb          # Notebook principal con implementación completa
├── Localizar_Pixeles_Colab.ipynb   # Implementación alternativa para Colab
└── README.md                       # Este archivo de documentación
```

## 🛠️ Tecnologías Utilizadas

- **Python 3.x**
- **OpenCV** - Procesamiento de imágenes y operaciones básicas
- **NumPy** - Manipulación de arrays y operaciones matemáticas
- **Matplotlib** - Visualización de imágenes y gráficos
- **Google Colab** - Entorno de desarrollo (opcional)

## 📊 Resultados Obtenidos

### Métricas de Rendimiento
- **Píxeles Analizados**: ~1.5M píxeles en imagen de alta resolución
- **Precisión de Detección**: Variable según ajuste de umbrales
- **Tiempo de Procesamiento**: Instantáneo para imágenes individuales

### Visualizaciones Implementadas
1. **Canal Separado**: Visualización individual de canales R, G, B
2. **Máscara Binaria**: Mapa de píxeles detectados como "naranja"
3. **Mapa de Calor**: Representación alternativa de la máscara
4. **Imagen Segmentada**: Resultado final con solo el color objetivo

## 🔧 Funcionalidades Principales

### `Lab_4_Salticidae.ipynb`
- **Análisis Exploratorio**: Inspección manual de píxeles representativos
- **Segmentación Automática**: Aplicación sistemática de umbrales RGB
- **Experimentación Interactiva**: Modificación de parámetros en tiempo real
- **Validación**: Verificación de resultados con píxeles de ejemplo

### `Localizar_Pixeles_Colab.ipynb`
- **Versión Optimizada**: Implementación específica para Google Colab
- **Acceso a Datasets**: Descarga automática de imágenes desde repositorios
- **Memoria Optimizada**: Gestión eficiente de recursos

## 📈 Características Técnicas

### Parámetros Configurables
```python
# Umbrales base para detección de color naranja
umbral_rojo_minimo = 200
umbral_verde_maximo = 165
umbral_azul_maximo = 60

# Parámetros experimentales
nuevo_umbral_rojo = 100
nuevo_umbral_verde = 120
nuevo_umbral_azul = 120
```

### Algoritmos Implementados
- **Filtrado por Umbrales**: Condiciones lógicas sobre canales RGB
- **Operaciones Booleanas**: Combinación de máscaras usando operadores & y ~
- **Análisis Estadístico**: Cálculo de porcentajes y métricas de cobertura

## 🚀 Instrucciones de Uso

### Prerrequisitos
```bash
pip install opencv-python numpy matplotlib
```

### Ejecución
1. **Abrir** el notebook en Jupyter, Google Colab o entorno compatible
2. **Ejecutar** celdas secuencialmente desde el inicio
3. **Experimentar** modificando los umbrales en la sección correspondiente
4. **Analizar** los resultados visualizados

### Parámetros de Experimentación
```python
# Modificar estos valores para experimentar
umbral_rojo = 100-180    # Rango recomendado
umbral_verde = 50-150    # Rango recomendado  
umbral_azul = 50-150     # Rango recomendado
```

## 📝 Conclusiones y Aprendizajes

### Conceptos Fundamentales
1. **Representación Digital**: Las imágenes son matrices de valores numéricos
2. **Espacios de Color**: RGB es uno de varios modelos para representar colores
3. **Segmentación**: Técnica fundamental en visión por computadora
4. **Parámetros**: La precisión depende del ajuste cuidadoso de umbrales

### Limitaciones Identificadas
- **Sensibilidad a Iluminación**: Los umbrales pueden requerir ajuste según condiciones de luz
- **Variabilidad de Color**: Diferentes especies o individuos pueden requerir parámetros específicos
- **Ruido**: Píxeles individuales pueden generar falsos positivos

### Aplicaciones Futuras
- **Clasificación Automática**: Extensión a múltiples especies de arácnidos
- **Análisis Biológico**: Estudios sobre patrones de coloración y comportamiento
- **Agricultura**: Detección de plagas o especies beneficiosas
- **Investigación**: Base para algoritmos más sofisticados de visión por computadora

## 👥 Autores y Créditos

- **Desarrollo**: Laboratorio de Técnicas de Procesamiento de Imágenes
- **Dataset**: Imágenes obtenidas de repositorios académicos y Wikimedia Commons
- **Inspiración**: Investigación en etología y biología de arañas salticidae

## 📄 Licencia

Este proyecto es de carácter académico y educativo. Las imágenes utilizadas están sujetas a sus respectivas licencias originales.

## 📞 Contacto y Soporte

Para consultas técnicas o colaboraciones académicas, contactar a través del curso de Técnicas de Procesamiento de Imágenes.

---

**Nota**: Este README fue generado automáticamente basándose en la implementación completa del laboratorio de segmentación de colores, demostrando técnicas fundamentales de procesamiento digital de imágenes aplicadas al análisis de patrones biológicos.