**Proyecto**: MVP Edición Generativa de Imágenes  
**Curso**: Procesamiento Digital de Imágenes - IFTS24  
**Alumno**: Alfredo Poblete
**Profesor**: Matias Barreto

**Año**: 2025  

---


> **🌟 VERSIÓN CLOUD-OPTIMIZADA**  
> Esta aplicación ha sido migrada para funcionar perfectamente en **Streamlit Cloud** usando **HuggingFace Inference API**.  
> ✅ Sin modelos pesados | ✅ Sin GPU requerida | ✅ Deploy instantáneo  
> 📖 Ver [MIGRACION_CLOUD.md](MIGRACION_CLOUD.md) para detalles técnicos completos.

---


# 🎨 MVP: Edición Generativa de Imágenes con Análisis Inteligente

**Procesamiento Digital de Imágenes - IFTS24**

Una aplicación avanzada que integra modelos de difusión de última generación para manipulación de imágenes y análisis inteligente con Gemini 2.0.

## 🚀 Características Principales

### 🖼️ Edición Generativa de Imágenes
- **Inpainting**: Eliminar objetos no deseados con relleno inteligente
- **Outpainting**: Extender imágenes más allá de sus bordes
- **Style Transfer**: Transferir estilos artísticos manteniendo contenido
- **Object Removal**: Eliminación precisa de objetos específicos
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
- Inpainting para limpieza de assets

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

> **⚠️ IMPORTANTE**: Esta aplicación ahora usa **HuggingFace Inference API** en lugar de modelos locales.  
> ✅ **Funciona perfectamente en Streamlit Cloud** sin necesidad de GPU ni modelos pesados.  
> 📖 Ver [MIGRACION_CLOUD.md](MIGRACION_CLOUD.md) para detalles técnicos completos.

### Prerrequisitos
- Python 3.8+
- **NO requiere GPU** (usa APIs en la nube)
- ~500 MB RAM
- ~200 MB espacio libre en disco
- Conexión a Internet

### Instalación Rápida

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

3. **Instalar dependencias (solo ~100 MB)**
```bash
pip install -r requirements.txt
```

4. **Configurar API Keys (REQUERIDO)**
```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar .env con tus API keys:
# HUGGINGFACE_API_KEY=hf_tu_token_aqui
# GOOGLE_API_KEY=tu_google_api_key_aqui
```

### Obtener API Keys (Gratis)

**HuggingFace API Key** (para generación de imágenes):
1. Ve a [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Crea un nuevo token (tipo: Read)
3. Copia el token y agrégalo a tu archivo `.env`

**Google Gemini API Key** (para análisis de imágenes):
1. Ve a [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
2. Crea una nueva API key
3. Copia la key y agrégala a tu archivo `.env`

### Despliegue en Streamlit Cloud

Para desplegar en [Streamlit Cloud](https://share.streamlit.io/):

1. **Sube tu código a GitHub**
2. **Conecta con Streamlit Cloud**
3. **Configura los Secrets** en Settings → Secrets:
```toml
HUGGINGFACE_API_KEY = "hf_tu_token_aqui"
GOOGLE_API_KEY = "tu_google_api_key_aqui"
```
4. **Deploy** - ¡La app estará lista en menos de 1 minuto!

📖 **Guía completa de migración**: Ver [MIGRACION_CLOUD.md](MIGRACION_CLOUD.md)

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

#### 🖼️ Inpainting
```python
# Eliminar objetos no deseados
params = {
    'prompt': 'natural background texture',
    'num_inference_steps': 30,
    'guidance_scale': 7.5
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

### 🌐 Procesamiento en la Nube (HuggingFace Inference API)
- **Stable Diffusion Inpainting**: `stabilityai/stable-diffusion-2-inpainting`
- **Stable Diffusion v1.5**: `runwayml/stable-diffusion-v1-5` 
- **Sin descarga de modelos**: Todo se procesa en servidores de HuggingFace
- **Sin GPU local requerida**: Procesamiento 100% en la nube

### 🧠 Análisis Visual (Google Gemini)
- **Gemini 2.0 Flash**: Análisis multimodal avanzado
- **Comprensión espacial**: Detección de objetos y bounding boxes
- **Razonamiento visual**: Descripción y comparación de imágenes
- **Análisis comparativo**: Evaluación automática antes/después

### ⚡ Ventajas de la Arquitectura Cloud
- ✅ **Instalación instantánea**: Solo ~100 MB de dependencias
- ✅ **Sin GPU necesaria**: Funciona en cualquier computadora
- ✅ **Funciona en Streamlit Cloud**: Deploy gratuito y rápido
- ✅ **Mantenimiento mínimo**: Sin actualizaciones de modelos pesados
- ✅ **APIs gratuitas**: Para uso educativo y personal

## 🎯 Casos de Uso Específicos

### Para Concept Artists
```markdown
1. **Expansión de Conceptos**: Usar outpainting para crear worlds más grandes
2. **Unificación de Estilo**: Style transfer para portfolios consistentes  
3. **Limpieza Rápida**: Inpainting para refinar bocetos
```

### Para Desarrolladores de Juegos
```markdown
1. **Backgrounds Dinámicos**: Background replacement para diferentes niveles
2. **Asset Cleanup**: Object removal para limpiar recursos
3. **Environment Design**: Composición para crear escenas complejas
```

## 📊 Parámetros de Rendimiento

### Configuraciones Recomendadas (Cloud API)

**Para Desarrollo Rápido:**
- Steps: 20-25
- Guidance Scale: 6-7
- Resolución: 512x512
- Tiempo: 15-30 segundos (depende de la red)

**Para Calidad Máxima:**
- Steps: 30-40
- Guidance Scale: 8-9
- Resolución: 512x512
- Tiempo: 30-60 segundos (depende de la red)

**Notas sobre Rendimiento:**
- ⏱️ El tiempo depende de la velocidad de Internet y carga del servidor
- 🌐 Procesamiento 100% en la nube (sin uso de recursos locales)
- 📊 HuggingFace puede tener límites de rate (requests por minuto)
- ✅ Considera HuggingFace Pro para mayor velocidad y sin límites

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

**Error: "No se encontró HUGGINGFACE_API_KEY"**
```bash
# Verificar que el archivo .env existe
ls -la .env

# Verificar contenido
cat .env

# Debe contener:
HUGGINGFACE_API_KEY=hf_tu_token_aqui
GOOGLE_API_KEY=tu_google_api_key_aqui
```

**Error: "API request failed with status 503"**
```
Causa: El modelo se está cargando en los servidores de HuggingFace
Solución: Esperar 20-30 segundos y reintentar
La app reintenta automáticamente una vez
```

**Error: "Rate limit exceeded"**
```
Causa: Demasiadas peticiones en poco tiempo
Solución: 
- Esperar unos minutos antes de procesar otra imagen
- Considerar HuggingFace Pro para límites más altos
- Usar API keys diferentes para desarrollo y producción
```

**Procesamiento muy lento**
```
Causa: Conexión lenta o servidor saturado
Solución:
- Verificar conexión a Internet
- Reducir resolución de imagen (automático en la app)
- Reducir num_inference_steps a 20-25
- Intentar en otro momento del día
```

**Error de API Gemini**
```bash
# Configurar API key
export GOOGLE_API_KEY=tu_api_key

# O crear archivo .env
echo "GOOGLE_API_KEY=tu_api_key" > .env

# Verificar que la key es válida en:
# https://makersuite.google.com/app/apikey
```

**La app no inicia en Streamlit Cloud**
```
Verificar:
1. Secrets configurados correctamente en Settings → Secrets
2. requirements.txt sin errores de sintaxis
3. Logs de deployment para ver errores específicos
4. Que el repositorio esté actualizado
```

## 🚀 Próximas Mejoras

- [ ] **Segment Anything Model (SAM)** para máscaras automáticas (vía API)
- [ ] **ControlNet API** para mayor control espacial
- [ ] **Batch Processing** para múltiples imágenes
- [ ] **Export de máscaras** y metadatos
- [ ] **Caché de resultados** para evitar reprocesamiento
- [ ] **Soporte para más modelos** de HuggingFace
- [ ] **Integración con Replicate.com** como alternativa
- [ ] **API REST propia** para integración externa

## 📚 Referencias

### Documentación de APIs
- [HuggingFace Inference API](https://huggingface.co/docs/api-inference/index)
- [Gemini API Documentation](https://ai.google.dev/)
- [Streamlit Cloud Deployment](https://docs.streamlit.io/streamlit-community-cloud)

### Papers Importantes
- "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022)
- "Segment Anything" (Kirillov et al., 2023)

### Recursos Adicionales
- [Stable Diffusion Models](https://stability.ai/)
- [HuggingFace Model Hub](https://huggingface.co/models)
- [Streamlit Documentation](https://docs.streamlit.io/)

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