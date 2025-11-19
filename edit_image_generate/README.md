**Proyecto**: MVP Edición Generativa de Imágenes  
**Curso**: Procesamiento Digital de Imágenes - IFTS24  
**Alumno**: Alfredo Poblete
**Profesor**: Matias Barreto

**Año**: 2025  

---

# 🎨 MVP: Edición Generativa de Imágenes con Análisis Inteligente

**Procesamiento Digital de Imágenes - IFTS24**

Una aplicación avanzada que integra modelos de difusión de última generación para manipulación de imágenes y análisis inteligente con Gemini 2.0.

## 🚀 Características Principales

### 🖼️ Edición Generativa de Imágenes
- **Outpainting**: Extender imágenes más allá de sus bordes
- **Style Transfer**: Transferir estilos artísticos manteniendo contenido
- **Object Removal**: Eliminación precisa de objetos específicos (reemplazo inteligente)
- **Background Replacement**: Cambiar fondos preservando sujetos principales
- **Composición Inteligente**: Combinar elementos de múltiples imágenes

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

## 🏗️ Arquitectura del Proyecto

```
010_tp_final_integrado/
├── app.py                    # Aplicación Streamlit principal
├── requirements.txt          # Dependencias
├── README.md                 # Esta documentación
├── .gitignore               # Ignorar archivos sensibles
├── models/
│   ├── __init__.py
│   ├── diffusion.py         # Procesamiento con modelos de difusión
│   └── analysis.py          # Análisis visual con Gemini 2.0
├── utils/
│   ├── __init__.py
│   ├── image_utils.py       # Utilidades de procesamiento
│   └── ui_utils.py          # Componentes de interfaz
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

La aplicación sigue el diseño especificado:

1. **Panel Izquierdo**: Carga y configuración
   - Botón para subir imagen
   - Selector de método de procesamiento
   - Configuración de parámetros

2. **Panel Derecho**: Resultados y análisis
   - Imagen procesada
   - Análisis automático con Gemini 2.0
   - Métricas de calidad

3. **Sección Inferior**: Comparación
   - Vista lado a lado antes/después
   - Descripción de cambios detectados
   - Recomendaciones de mejora

### Métodos de Procesamiento

#### 🗑️ Object Removal (Inpainting deshabilitado)
La funcionalidad de inpainting ha sido deshabilitada en esta versión para evitar la descarga de pesos y la ejecución de modelos locales en entornos con recursos limitados (por ejemplo, Streamlit Cloud). En su lugar, se recomienda usar **Object Removal** con máscara o la detección inteligente integrada.

Ejemplo de uso — `object_removal` (prompt de contexto):
```python
# Eliminar objeto específico y rellenar con fondo natural
params = {
    'context_prompt': 'natural background texture, photorealistic',
    'num_inference_steps': 45,
    'guidance_scale': 9.0
}
```

#### 🔄 Outpainting
```python
# Extender la imagen
params = {
    'prompt': 'extended natural landscape',
    'num_inference_steps': 50,
    'guidance_scale': 8.0
}
```

#### 🎭 Style Transfer
```python
# Aplicar estilo artístico
params = {
    'style_prompt': 'artistic painting style',
    'strength': 0.6,
    'num_inference_steps': 35
}
```

## 🔧 APIs y Modelos Utilizados

### Modelos de Difusión
- **Stable Diffusion v1.5**: `runwayml/stable-diffusion-v1-5` 
- **Stable Diffusion Upscaler**: `stabilityai/stable-diffusion-x4-upscaler`

Nota: El modelo `runwayml/stable-diffusion-inpainting` aparece en la documentación histórica del proyecto, pero la funcionalidad de inpainting fue deshabilitada para evitar descargas locales en entornos con recursos limitados. Las operaciones de eliminación de objetos y reemplazo de fondo se realizan mediante la Inference API de Hugging Face usando flujos `img2img` seguros.

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
3. **Limpieza Rápida**: Object Removal para refinar bocetos y assets
```

### Para Desarrolladores de Juegos
```markdown
1. **Backgrounds Dinámicos**: Background replacement para diferentes niveles
2. **Asset Cleanup**: Object removal para limpiar recursos
3. **Environment Design**: Composición para crear escenas complejas
```

## 📊 Parámetros de Rendimiento

### Configuraciones Recomendadas

**Para Desarrollo Rápido:**
- Steps: 20-25
- Guidance Scale: 6-7
- Resolución: 512x512
- Tiempo: 10-15 segundos (GPU)

**Para Calidad Máxima:**
- Steps: 50-80
- Guidance Scale: 8-10
- Resolución: 512x512+
- Tiempo: 30-60 segundos (GPU)

**Para CPU:**
- Steps: 15-20 (reducir por velocidad)
- Tiempo: 2-5 minutos por imagen

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
- La primera opción usa la API remota y generará tráfico hacia tu cuenta de HF (puede incurrir en uso de cuota/pagos según tu cuenta).
- Asegúrate de tener el token con permisos para usar los modelos públicos o privados que necesites.


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

## 🚀 Próximas Mejoras

- [ ] **Segment Anything Model (SAM)** para máscaras automáticas
- [ ] **ControlNet** para mayor control espacial
- [ ] **IP-Adapter** para control de estilo mejorado
- [ ] **Batch Processing** para múltiples imágenes
- [ ] **Export de máscaras** y metadatos
- [ ] **API REST** para integración externa

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

Esta aplicación integra las tecnologías más avanzadas de IA para ofrecer capacidades profesionales de edición de imágenes, diseñada específicamente para desarrolladores de videojuegos y concept artists que necesitan generar contenido de alta calidad de manera eficiente.
