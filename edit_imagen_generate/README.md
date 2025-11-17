**Proyecto**: MVP Edición Generativa de Imágenes - Versión Optimizada  
**Curso**: Procesamiento Digital de Imágenes - IFTS24  
**Alumno**: Alfredo Poblete  
**Profesor**: Matias Barreto  

**Año**: 2025  

---

# 🎨 MVP: Edición Generativa de Imágenes con Análisis Inteligente - Versión Optimizada

**Procesamiento Digital de Imágenes - IFTS24**

Una aplicación avanzada que integra modelos de difusión de última generación para manipulación de imágenes y análisis inteligente con Gemini 2.0. **Versión optimizada sin función Inpainting**.

## 🚀 Características Principales

### 🖼️ Edición Generativa de Imágenes
- **Outpainting**: Extender imágenes más allá de sus bordes con calidad optimizada
- **Style Transfer**: Transferir estilos artísticos manteniendo contenido con mejores parámetros
- **Object Removal**: Eliminación precisa de objetos específicos con detección inteligente mejorada
- **Background Replacement**: Cambiar fondos preservando sujetos principales con prompts optimizados
- **Composición Inteligente**: Combinar elementos de múltiples imágenes con algoritmos mejorados

### 🧠 Análisis Visual Avanzado
- **Comprensión Espacial 2D**: Detección y localización de objetos
- **Análisis Comparativo**: Evaluación antes/después del procesamiento
- **Métricas de Calidad**: PSNR, SSIM, similitud visual
- **Recomendaciones Inteligentes**: Sugerencias automáticas de mejora

### 🎮 Público Objetivo
**Desarrolladores de videojuegos y Concept Artists**
- Generación rápida de escenarios y texturas
- Outpainting para mundos expandidos
- Style transfer para consistencia gráfica
- Object Removal para limpieza de assets

### ✨ Nuevas Optimizaciones
- **Galería de Ejemplos Mejorada**: Miniaturas optimizadas con mejor UX
- **Parámetros Optimizados**: Configuraciones mejoradas para mejor calidad
- **Interfaz Refinada**: Mejor organización y experiencia de usuario
- **Rendimiento Mejorado**: Algoritmos optimizados para velocidad

## 🏗️ Arquitectura del Proyecto

```
010_tp_final_integrado/
├── app.py                    # Aplicación Streamlit principal (optimizada)
├── requirements.txt          # Dependencias
├── README.md                 # Esta documentación actualizada
├── models/
│   ├── __init__.py
│   ├── diffusion.py         # Procesamiento sin Inpainting (optimizado)
│   └── analysis.py          # Análisis visual con Gemini 2.0
├── utils/
│   ├── __init__.py
│   ├── image_utils.py       # Utilidades de procesamiento
│   └── ui_utils.py          # Componentes de interfaz (sin Inpainting)
├── config/
│   ├── __init__.py
│   └── config.py            # Configuración del proyecto
└── assets/
    └── ejemplos/            # Imágenes de ejemplo
```

## 🛠️ Instalación y Configuración

### Prerrequisitos
- Python 3.8+
- GPU recomendada (NVIDIA CUDA compatible)
- 8GB+ RAM
- 10GB+ espacio libre en disco

### Instalación

1. **Clonar el repositorio**
```bash
git clone <repository-url>
cd 010_tp_final_integrado
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno (opcional)**
```bash
# Crear archivo .env
GOOGLE_API_KEY=tu_api_key_aqui
```

### Configuración de GPU (Recomendado)

**Para NVIDIA GPUs:**
```bash
# Instalar CUDA Toolkit 11.8+ y PyTorch compatible
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Para verificar GPU:**
```python
import torch
print("CUDA disponible:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## 🚀 Uso de la Aplicación

### Ejecutar la aplicación
```bash
streamlit run app.py
```

### Interfaz Principal

La aplicación sigue el diseño especificado con mejoras optimizadas:

1. **Panel Izquierdo**: Carga y configuración
   - Galería de ejemplos mejorada con miniaturas
   - Selector de método de procesamiento (sin Inpainting)
   - Configuración de parámetros optimizados

2. **Panel Derecho**: Resultados y análisis
   - Imagen procesada
   - Análisis automático con Gemini 2.0
   - Métricas de calidad

3. **Sección Inferior**: Comparación
   - Vista lado a lado antes/después
   - Descripción de cambios detectados
   - Recomendaciones de mejora

### Métodos de Procesamiento (SIN INPAINTING)

#### 🔄 Outpainting Optimizado
```python
# Extender la imagen con parámetros mejorados
params = {
    'prompt': 'extended natural landscape seamlessly',
    'num_inference_steps': 50,  # Aumentado para mejor calidad
    'guidance_scale': 8.5,      # Optimizado
    'extension_factor': 1.5     # Factor de extensión
}
```

#### 🎭 Style Transfer Optimizado
```python
# Aplicar estilo artístico con configuraciones mejoradas
params = {
    'style_prompt': 'artistic painting style',
    'strength': 0.6,            # Optimizado
    'num_inference_steps': 35,  # Aumentado para mejor calidad
    'guidance_scale': 7.5       # Balanceado
}
```

#### 🗑️ Object Removal Inteligente
```python
# Eliminación de objetos con detección mejorada
params = {
    'object_description': 'unwanted object',
    'context_prompt': 'natural seamless background',
    'num_inference_steps': 50,  # Aumentado para mejor integración
    'guidance_scale': 9.0       # Mayor adherencia al contexto
}
```

#### 🖼️ Background Replacement Optimizado
```python
# Cambio de fondo con prompts mejorados
params = {
    'background_prompt': 'beautiful sunset landscape',
    'num_inference_steps': 45,  # Optimizado
    'guidance_scale': 8.5       # Balanceado
}
```

#### 🧩 Composición Inteligente Mejorada
```python
# Composición de elementos con parámetros optimizados
params = {
    'elements_prompt': 'harmonious artistic composition',
    'strength': 0.5,            # Control de intensidad
    'num_inference_steps': 40,  # Aumentado para mejor calidad
    'guidance_scale': 8.0       # Optimizado
}
```

## 🔧 APIs y Modelos Utilizados

### Modelos de Difusión
- **Stable Diffusion Inpainting**: `runwayml/stable-diffusion-inpainting` (usado para Outpainting)
- **Stable Diffusion v1.5**: `runwayml/stable-diffusion-v1-5` 
- **Stable Diffusion Upscaler**: `stabilityai/stable-diffusion-x4-upscaler`

### Análisis Visual
- **Gemini 2.0 Flash**: Análisis multimodal avanzado
- **Comprensión espacial**: Detección de objetos y bounding boxes
- **Razonamiento visual**: Descripción y comparación de imágenes

### Métricas de Calidad
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **Similitud**: Coeficiente de correlación visual

## 🎯 Casos de Uso Específicos

### Para Concept Artists
```markdown
1. **Expansión de Conceptos**: Usar outpainting para crear worlds más grandes
2. **Unificación de Estilo**: Style transfer para portfolios consistentes  
3. **Composición Creativa**: Intelligent composition para escenas complejas
```

### Para Desarrolladores de Juegos
```markdown
1. **Backgrounds Dinámicos**: Background replacement para diferentes niveles
2. **Asset Cleanup**: Object removal para limpiar recursos
3. **Environment Design**: Composición para crear escenas complejas
4. **Style Consistency**: Style transfer para mantener coherencia visual
```

## 📊 Parámetros de Rendimiento

### Configuraciones Recomendadas (OPTIMIZADAS)

**Para Desarrollo Rápido:**
- Steps: 25-35 (aumentado para mejor calidad)
- Guidance Scale: 6.5-8.0
- Resolución: 512x512
- Tiempo: 15-25 segundos (GPU)

**Para Calidad Máxima:**
- Steps: 45-60 (optimizado para resultados superiores)
- Guidance Scale: 8.0-9.5
- Resolución: 512x512+
- Tiempo: 30-45 segundos (GPU)

**Para CPU:**
- Steps: 20-30 (balanceado por velocidad y calidad)
- Tiempo: 2-4 minutos por imagen

## 🔍 Análisis Visual

### Capacidades de Gemini 2.0
- **Detección de Objetos**: Bounding boxes y clasificación
- **Análisis de Calidad**: Evaluación de mejoras
- **Comprensión Contextual**: Descripción detallada
- **Comparación**: Análisis antes/después automático

### Ejemplo de Análisis
```python
analysis = analyzer.analyze(processed_image, "comparison_analysis")

# Resultado típico:
{
    "changes_description": "Mejoras en claridad y definición...",
    "quality_metrics": {
        "mejora_nitidez": "Alta",
        "preservación_detalles": "Buena",
        "integración_natural": "Excelente"
    },
    "recommendations": [
        "Resultado exitoso del procesamiento",
        "Cambios coherentes con imagen original"
    ]
}
```

## 🛠️ Solución de Problemas

### Problemas Comunes

**Error: "CUDA out of memory"**
```python
# Reducir resolución de imagen
image = image.resize((512, 512))

# Reducir batch size
torch.cuda.empty_cache()
```

**Procesamiento muy lento**
```python
# Verificar GPU disponible
import torch
print(torch.cuda.is_available())

# Reducir parámetros
params['num_inference_steps'] = 20
```

**Error de API Gemini**
```bash
# Configurar API key
export GOOGLE_API_KEY=tu_api_key

# O crear archivo .env
echo "GOOGLE_API_KEY=tu_api_key" > .env
```

## ☁️ Despliegue en Streamlit Cloud (modo recomendado para modelos grandes)

Si la app se cae en Streamlit Cloud al intentar descargar pesos grandes, evita cargar modelos localmente y usa la Inference API de Hugging Face.

- Paso 1: Activar modo remoto con variable de entorno `USE_HF_API=true`.
- Paso 2: Configurar el token de Hugging Face en los secretos de Streamlit: `HUGGINGFACE_API_TOKEN`.

En Streamlit Cloud: Settings → Secrets, añadir:

```
USE_HF_API=true
HUGGINGFACE_API_TOKEN=hf_...TuTokenAquí...
```

La app detecta `USE_HF_API` y enviará las imágenes a la Inference API de Hugging Face en lugar de descargar pesos locales, evitando problemas de RAM y disco en la instancia.

Notas:
- Esta opción usa la API remota y generará tráfico hacia tu cuenta de HF (puede incurrir en uso de cuota/pagos según tu cuenta).
- Asegúrate de tener el token con permisos para usar los modelos públicos o privados que necesites.

## 🚀 Próximas Mejoras

- [ ] **Segment Anything Model (SAM)** para máscaras automáticas
- [ ] **ControlNet** para mayor control espacial
- [ ] **IP-Adapter** para control de estilo mejorado
- [ ] **Batch Processing** para múltiples imágenes
- [ ] **Export de máscaras** y metadatos
- [ ] **API REST** para integración externa
- [ ] **Galería Dinámica** con previsualización en tiempo real
- [ ] **Templates de Procesamiento** para casos de uso específicos

## 📚 Referencias

### Papers Importantes
- "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022)
- "Segment Anything" (Kirillov et al., 2023)

### Documentación
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/)
- [Gemini API Documentation](https://ai.google.dev/)
- [Stable Diffusion Models](https://stability.ai/)

## 👥 Contribución

Este proyecto es parte del curso **Procesamiento Digital de Imágenes - IFTS24**.

Para contribuir:
1. Fork el proyecto
2. Crear una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abrir un Pull Request

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

### 🎮 ¡Listo para crear contenido increíble!

Esta aplicación optimizada integra las tecnologías más avanzadas de IA para ofrecer capacidades profesionales de edición de imágenes, diseñada específicamente para desarrolladores de videojuegos y concept artists que necesitan generar contenido de alta calidad de manera eficiente.

**Versión optimizada sin Inpainting - Mejor rendimiento y UX mejorada**