# Laboratorio Práctico Integrador - Cámara Oscura y Figuras Geométricas

## 📋 Descripción

Este proyecto integra conceptos teóricos y prácticos de procesamiento de imágenes digitales, modelos de color y óptica fundamental. Se exploran los modelos de color RGB y HSV, los fundamentos ópticos de la cámara oscura y la manipulación de imágenes mediante la librería **py5**.

El laboratorio conecta la generación de imágenes digitales desde cero con la captura de imágenes del mundo real a través de un dispositivo óptico construido manualmente.

---

## 🎯 Objetivos

Al finalizar este laboratorio, se logra:

- ✅ Implementar código en py5 para generar y guardar imágenes simples
- ✅ Aplicar los modelos de color RGB y HSV en un entorno de programación
- ✅ Construir un dispositivo óptico funcional (cámara oscura)
- ✅ Capturar y digitalizar una imagen proyectada para su posterior procesamiento
- ✅ Analizar cómo las condiciones de iluminación afectan la captura de una imagen

---

## 🛠️ Tecnologías y Herramientas

- **Python 3.x**
- **py5** - Librería para procesamiento creativo y generación de gráficos
- **py5_tools** - Herramientas complementarias para py5
- **Java 17** - Requerido por py5
- **Jupyter Notebook** - Entorno de desarrollo interactivo

---

## 📦 Instalación

### Requisitos Previos

```bash
# Instalar Java Development Kit
pip install install-jdk

# Configurar Java 17
python -c "import jdk; print('Java installed to', jdk.install('17'))"
```

### Instalación de Dependencias

```bash
# Instalar py5
pip install py5

# En sistemas Linux, instalar dependencias adicionales
apt-get install ca-certificates-java libxcursor1 libxrandr2 libxrender1 libxtst6 libxi6 xvfb
```

---

## 🚀 Estructura del Proyecto

```
002_Camara_Oscura-Figuras_geometricas/
│
├── Trabajo_de_Laboratorio_1.ipynb    # Notebook principal con todos los ejercicios
├── figura_rgb.jpg                     # Figura generada en espacio RGB
├── figura_hsv.jpg                     # Figura generada en espacio HSV
├── 001.jpg                            # Imagen original de referencia
├── 002.jpeg                           # Imagen capturada con cámara oscura
├── Observaciones.txt                  # Notas y observaciones del experimento
└── README.md                          # Este archivo
```

---

## 📚 Contenido del Laboratorio

### Ejercicio 1: Figura en Espacio de Color RGB

Generación de una figura geométrica simple (triángulo) utilizando el modelo de color RGB:

- **Canvas**: 400x400 píxeles
- **Modo de color**: RGB (255, 255, 255)
- **Figura**: Triángulo con color magenta (255, 0, 255)
- **Salida**: [`figura_rgb.jpg`](figura_rgb.jpg)

```python
py5.color_mode(py5.RGB, 255)
py5.fill(255, 0, 255)
py5.triangle(200, 100, 100, 300, 300, 300)
```

### Ejercicio 2: Figura en Espacio de Color HSV

Generación de una figura geométrica (elipse) utilizando el modelo de color HSV:

- **Canvas**: 400x400 píxeles
- **Modo de color**: HSV (360°, 100%, 100%)
- **Figura**: Elipse con valores HSV (215, 30, 70)
- **Salida**: [`figura_hsv.jpg`](figura_hsv.jpg)

```python
py5.color_mode(py5.HSB, 360, 100, 100)
py5.fill(215, 30, 70)
py5.ellipse(200, 200, 300, 300)
```

### Ejercicio 3: Construcción y Uso de la Cámara Oscura

#### Materiales Necesarios

- Caja de zapatos con tapa
- Cinta adhesiva opaca (cinta aisladora negra)
- Papel manteca o de calcar
- Alfiler o aguja
- Tijera o cúter
- Teléfono celular con cámara

#### Proceso de Construcción

1. **Estenopo**: Realizar un orificio pequeño con alfiler en un extremo de la caja
2. **Pantalla de proyección**: Recortar ventana en el extremo opuesto y pegar papel manteca
3. **Sellado**: Cubrir con cinta todas las entradas de luz
4. **Captura**: Dirigir el estenopo hacia una fuente de luz intensa y fotografiar la imagen proyectada

#### Imagen Capturada

La imagen fue capturada desde una pantalla de computadora con intensidad de luz al máximo para obtener una mejor visualización. La imagen original proviene de [Unsplash](https://unsplash.com/es/fotos/un-arbol-rosado-en-un-campo-cubierto-de-hierba-junto-a-un-cuerpo-de-agua-NCe2hR_2pps).

### Ejercicio 4: Procesamiento de Imágenes

Se aplicaron diversos filtros y transformaciones a la imagen capturada:

#### 4.1 Diferentes Tintes (HSV)
Aplicación de tintes en diferentes tonalidades utilizando el espacio de color HSV.

#### 4.2 Alto Contraste
Aplicación de filtro de umbral (threshold) con valor de 0.6 para mejorar la separación entre luces y sombras.

```python
py5.apply_filter(py5.THRESHOLD, 0.6)
```

#### 4.3 Blanco y Negro
Conversión a escala de grises para análisis de luminosidad.

```python
py5.apply_filter(py5.GRAY)
```

#### 4.4 Colores Invertidos
Inversión de colores para análisis de contraste.

```python
py5.apply_filter(py5.INVERT)
```

---

## 🔍 Observaciones y Resultados

### Análisis de Procesamiento

- **Diferentes tonos**: La imagen es visible pese a no estar tan nítida
- **Más contraste**: Mejora la separación entre luces y sombras; se percibe más la forma del árbol
- **Blanco y Negro**: Todo parece más suave, con menos definición y profundidad

### Conclusiones

El algoritmo detecta perfectamente los colores de la imagen capturada. Es interesante notar que el aumento de nitidez logrado por contraste no requiere mejorar la resolución, sino realzar contornos. Es fundamental equilibrar estos ajustes para mantener la fidelidad y autenticidad visual de la imagen.

---

## 💻 Uso

### Ejecutar el Notebook

```bash
# Abrir Jupyter Notebook
jupyter notebook Trabajo_de_Laboratorio_1.ipynb
```

### Ejecutar Celdas Individuales

El notebook está organizado en secciones. Ejecutar las celdas secuencialmente para:

1. Instalar dependencias
2. Generar figuras RGB y HSV
3. Cargar y procesar imágenes capturadas
4. Aplicar filtros y transformaciones

---

## 📸 Galería de Resultados

| Ejercicio | Descripción | Archivo |
|-----------|-------------|---------|
| RGB | Triángulo magenta | `figura_rgb.jpg` |
| HSV | Elipse azul-gris | `figura_hsv.jpg` |
| Original | Imagen de referencia | `001.jpg` |
| Cámara Oscura | Imagen capturada | `002.jpeg` |

---

## 🤝 Contribuciones

Este es un proyecto académico de laboratorio. Las observaciones y mejoras son bienvenidas.

---

## 📄 Licencia

Este proyecto es de uso académico y educativo.

---

## 👥 Autor

Desarrollado como parte del curso de **Técnicas de Procesamiento de Imagen**.

---

## 📞 Contacto

Para consultas sobre este laboratorio, referirse a las observaciones documentadas en [`Observaciones.txt`](Observaciones.txt).

---

**Fecha de realización**: 2025  
**Versión**: 1.0
