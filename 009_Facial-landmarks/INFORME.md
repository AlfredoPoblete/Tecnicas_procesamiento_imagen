# Informe del Proyecto: Detector de Landmarks Faciales

## Introducción: ¿Qué son los Landmarks y su Importancia?

Los **landmarks faciales** son puntos de referencia específicos que se identifican en rostros humanos para mapear características anatómicas clave. En este proyecto, utilizamos **MediaPipe Face Mesh**, una solución desarrollada por Google que detecta **478 puntos específicos** en rostros humanos.

### Importancia de los Landmarks:

1. **Análisis de Expresiones**: Permiten detectar y cuantificar emociones mediante la medición de distancias entre puntos faciales (apertura de boca, ojos, inclinación de cabeza).

2. **Aplicaciones de Realidad Aumentada**: Son fundamentales para filtros faciales, maquillaje virtual y máscaras AR en plataformas como Instagram y Snapchat.

3. **Animación Facial**: Proporcionan la base para la creación de avatares animados y sistemas de captura de movimiento facial.

4. **Autenticación Biométrica**: Los patrones únicos de landmarks faciales pueden utilizarse para identificación personal.

5. **Análisis Médico**: En investigación de parálisis facial, cirugías reconstructivas y estudios ergonómicos.

### Puntos Clave Detectados:

- **Ojos**: Iris, párpados superiores e inferiores (aproximadamente 70 puntos)
- **Nariz**: Puente nasal y fosas nasales (aproximadamente 40 puntos)
- **Boca**: Labios superior e inferior, comisuras (aproximadamente 80 puntos)
- **Contorno Facial**: Perfil completo del rostro (aproximadamente 180 puntos)
- **Región de Mejillas**: Puntos de soporte facial (aproximadamente 108 puntos)

## Arquitectura: Diagrama de la Estructura del Proyecto

```
proyecto-landmarks/
├── app.py                    # Interfaz de usuario Streamlit
├── requirements.txt          # Dependencias del proyecto
├── readme.md                # Documentación del proyecto
├── .gitignore               # Archivos ignorados por Git
└── src/                     # Código fuente modular
    ├── __init__.py          # Inicializador del paquete
    ├── config.py            # Configuración centralizada
    ├── detector.py          # Lógica de detección con MediaPipe
    └── utils.py             # Funciones auxiliares y métricas
```

### Componentes Principales:

1. **app.py**: Aplicación principal de Streamlit
   - Interfaz de usuario intuitiva
   - Selector de estilos de visualización
   - Upload de imágenes y videos
   - Métricas en tiempo real
   - Visualización de resultados

2. **src/detector.py**: Núcleo de detección
   - Clase `FaceLandmarkDetector`
   - Integración con MediaPipe Face Mesh
   - 4 estilos de visualización diferentes
   - Gestión de recursos y memoria

3. **src/config.py**: Configuración centralizada
   - Parámetros de MediaPipe
   - Configuración de colores y estilos
   - Constantes del proyecto

4. **src/utils.py**: Utilidades y métricas
   - Conversión entre formatos PIL ↔ OpenCV
   - Redimensionado inteligente
   - Cálculo de métricas de expresión:
     - Apertura de boca
     - Apertura de ojos (izquierdo/derecho)
     - Inclinación de cabeza

## Decisiones de Diseño: Por qué elegí esta Estructura

### 1. **Arquitectura Modular**

**Decisión**: Separar el código en módulos especializados (`detector`, `utils`, `config`)

**Justificación**:
- **Mantenibilidad**: Cada módulo tiene una responsabilidad específica
- **Reutilización**: Las funciones utilitarias pueden usarse en otros proyectos
- **Testabilidad**: Fácil de probar componentes individualmente
- **Escalabilidad**: Agregar nuevas funcionalidades sin modificar código existente

### 2. **Patrón MVC Aproximado**

**Decisión**: Separar presentación (app.py), lógica de negocio (detector.py) y datos (config.py)

**Justificación**:
- **Claridad**: Código fácil de entender y navegar
- **Modificación**: Cambios en UI no afectan la lógica de detección
- **Colaboración**: Diferentes desarrolladores pueden trabajar en módulos distintos

### 3. **Configuración Centralizada**

**Decisión**: Todas las constantes en `config.py`

**Justificación**:
- **Consistencia**: Evita valores hardcodeados dispersos
- **Mantenimiento**: Cambios de configuración en un solo lugar
- **Flexibilidad**: Fácil ajuste de parámetros sin tocar código

### 4. **Múltiples Estilos de Visualización**

**Decisión**: Ofrecer 4 estilos diferentes de mostrar landmarks

**Justificación**:
- **Versatilidad**: Diferentes casos de uso requieren diferentes visualizaciones
- **Educativo**: Los usuarios pueden elegir el nivel de detalle
- **Performance**: Opciones optimizadas para diferentes necesidades

### 5. **Compatibilidad de Formatos**

**Decisión**: Conversión automática entre PIL y OpenCV

**Justificación**:
- **Usabilidad**: El usuario puede subir cualquier formato soportado
- **Flexibilidad**: Backend robusto que maneja diferentes fuentes de imagen
- **Robustez**: Manejo automático de conversiones internas

### 6. **Métricas Cuantificables**

**Decisión**: Calcular valores numéricos específicos para expresiones

**Justificación**:
- **Objetividad**: Datos cuantitativos en lugar de observaciones subjetivas
- **Análisis**: Permitir análisis estadístico de expresiones
- **Aplicaciones**: Útil para investigación y desarrollo de aplicaciones

## Desafíos: Problemas Encontrados y Soluciones

### 1. **Problema de Compatibilidad con Python 3.13**

**Descripción**: Inicialmente desarrollé el proyecto en Python 3.13, pero MediaPipe no era compatible con esta versión más reciente.

**Errores encontrados**:
```
ERROR: Could not find a version that satisfies the requirement mediapipe
ERROR: Package 'mediapipe' requires a different Python: 3.13.0 not in '>=3.7, <3.11'
```

**Solución implementada**:
- **Migración a Python 3.10**: Versión estable y completamente compatible
- **Actualización de requirements.txt**: Especificación clara de dependencias compatibles
- **Documentación**: Registro de la versión de Python requerida

**Lecciones aprendidas**:
- Verificar compatibilidad de librerías antes de elegir versión de Python
- Mantener documentación actualizada sobre versiones soportadas
- Usar entornos virtuales para control de dependencias

### 2. **Gestión de Formatos de Imagen**

**Descripción**: MediaPipe trabaja con RGB, OpenCV con BGR, y Streamlit con PIL, requiriendo múltiples conversiones.

**Problemas específicos**:
- Colores incorrectos por conversión BGR↔RGB
- Pérdida de calidad en conversiones múltiples
- Errores de tipo de datos en arrays numpy

**Solución implementada**:
```python
def pil_to_cv2(pil_image):
    # PIL → RGB → BGR (OpenCV)
    rgb_array = np.array(pil_image.convert('RGB'))
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_array

def cv2_to_pil(cv2_image):
    # BGR → RGB → PIL
    rgb_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_array)
    return pil_image
```

### 3. **Optimización de Rendimiento**

**Descripción**: Procesamiento de video en tiempo real causaba lentitud y uso excesivo de memoria.

**Problemas**:
- Frames muy grandes ralentizando el procesamiento
- Acumulación de memoria sin liberación
- UI bloqueada durante procesamiento

**Soluciones implementadas**:
- **Redimensionado inteligente**: `resize_image()` con límite de ancho máximo
- **Gestión de recursos**: Método `close()` para liberar memoria de MediaPipe
- **Procesamiento asíncrono**: Spinners de Streamlit para feedback al usuario

### 4. **Precisión de Landmarks**

**Descripción**: Algunos landmarks eran detectados de manera inconsistente dependiendo de la iluminación y ángulo facial.

**Problemas**:
- Landmarks saltando entre frames en videos
- Detección fallida en rostros parcialmente ocultos
- Métricas inexactas por landmarks mal posicionados

**Soluciones**:
- **Configuración optimizada**: `min_detection_confidence: 0.5` balancea precisión y sensibilidad
- **Modo estático**: `static_image_mode: True` para mayor precisión en imágenes individuales
- **Validación de resultados**: Verificación de detección exitosa antes de calcular métricas

### 5. **Experiencia de Usuario**

**Descripción**: Interfaz inicial confusa para usuarios sin conocimiento técnico.

**Problemas**:
- Upload de archivos grandes sin feedback
- Falta de explicación sobre qué son los landmarks
- Sin ejemplos visuales

**Soluciones implementadas**:
- **Sidebar informativo**: Explicaciones contextuales sobre landmarks y métricas
- **Ejemplo visual**: Imagen de referencia de MediaPipe
- **Mensajes de ayuda**: Tips para mejorar la detección
- **Progreso visual**: Spinners y métricas en tiempo real

## Conclusiones: Aprendizajes Principales

### 1. **Importancia de la Compatibilidad de Dependencias**

**Aprendizaje**: La selección de versiones de Python y librerías debe basarse en compatibilidad verificada, no en la última versión disponible.

**Impacto**: La migración de Python 3.13 a 3.10 resolvió todos los problemas de instalación y ejecución, demostrando que la estabilidad supera a la novedad.

### 2. **Valor de la Arquitectura Modular**

**Aprendizaje**: Separar responsabilidades en módulos especializados mejora significativamente la mantenibilidad y escalabilidad del proyecto.

**Impacto**: La estructura modular permitió identificar y resolver problemas específicos sin afectar otros componentes, acelerando el desarrollo y debugging.

### 3. **Gestión Cuidadosa de Formatos de Datos**

**Aprendizaje**: El procesamiento de imágenes requiere especial atención a conversiones de formato y tipos de datos.

**Impacto**: Implementar funciones robustas de conversión evitó errores sutiles de colores y tipos que podrían haber pasado desapercibidos.

### 4. **Balance entre Funcionalidad y Rendimiento**

**Aprendizaje**: Ofrecer múltiples funcionalidades (4 estilos de visualización, análisis de video) requiere optimizaciones cuidadosas para mantener fluidez.

**Impacto**: Las optimizaciones de redimensionado y gestión de memoria permitieron una experiencia fluida incluso en hardware modesto.

### 5. **Documentación y Experiencia de Usuario**

**Aprendizaje**: Proyectos técnicos requieren explicaciones claras para usuarios no especializados.

**Impacto**: El sidebar informativo y ejemplos visuales hicieron la aplicación accesible para usuarios con diferentes niveles de conocimiento técnico.

### 6. **Validación y Testing de Casos Límite**

**Aprendizaje**: Los sistemas de visión por computadora requieren validación exhaustiva de diferentes condiciones (iluminación, ángulos, oclusiones).

**Impacto**: Las configuraciones de MediaPipe y validaciones implementadas aseguran robustez en condiciones variadas.

### 7. **Aplicabilidad Real del Proyecto**

**Aprendizaje**: Los landmarks faciales tienen aplicaciones concretas en AR, análisis de emociones y biometría.

**Impacto**: El proyecto no solo es un ejercicio académico, sino una base sólida para aplicaciones comerciales de visión por computadora.

### Reflexión Final

Este proyecto demostró la importancia de:

- **Investigación previa**: Verificar compatibilidad antes de implementar
- **Diseño thinking**: Considerar experiencia del usuario desde el inicio
- **Testing exhaustivo**: Validar en múltiples condiciones y casos de uso
- **Documentación clara**: Facilitar uso y mantenimiento futuro
- **Código limpio**: Arquitectura que facilita extensiones y modificaciones

La combinación de MediaPipe con Streamlit resultó en una herramienta poderosa y accesible para análisis de expresiones faciales, estableciendo una base sólida para futuras aplicaciones de visión por computadora.

---

**Fecha**: 29 de octubre de 2025  
**Proyecto**: Laboratorio 2 - IFTS24  
**Tecnología**: MediaPipe + Streamlit + Python 3.10