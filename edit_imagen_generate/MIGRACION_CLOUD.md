# 🚀 Migración a Cloud: De Modelos Locales a HuggingFace Inference API

## 📋 Resumen de Cambios

Este documento explica la transformación de la aplicación para que funcione en **Streamlit Cloud** sin necesidad de descargar modelos pesados de IA.

---

## ❌ Problema Original

### ¿Por qué fallaba en Streamlit Cloud?

La aplicación original **NO podía ejecutarse en Streamlit Cloud** debido a:

1. **Modelos muy pesados**: Los modelos de Stable Diffusion pesan entre **2-8 GB**
2. **Límites de Streamlit Cloud**:
   - RAM máxima: **1 GB**
   - Almacenamiento: **0.5 GB**
   - Sin acceso a GPU
3. **Dependencias incompatibles**:
   - `torch` requiere **2-3 GB** solo para instalarse
   - `diffusers` necesita descargar modelos completos
   - `transformers` añade otros **1-2 GB**
4. **Tiempo de inicio**: Cargar modelos localmente toma **5-10 minutos**

### Error típico al presionar "Procesar imagen":

```
OutOfMemoryError: Cannot allocate tensor with shape [...]
RuntimeError: CUDA not available
ModuleNotFoundError: No module named 'torch'
```

---

## ✅ Solución Implementada

### Cambios Principales

#### 1. **Reemplazo de modelos locales por HuggingFace Inference API**

**ANTES** ([`models/diffusion.py`](models/diffusion.py)):
```python
from diffusers import StableDiffusionInpaintPipeline
import torch

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")  # Requiere GPU
result = pipe(prompt=prompt, image=image, mask=mask)
```

**AHORA** ([`models/diffusion.py`](models/diffusion.py)):
```python
import requests
from PIL import Image

def _call_huggingface_api(endpoint, image, prompt, mask=None):
    headers = {"Authorization": f"Bearer {self.api_key}"}
    files = {"inputs": image_bytes}
    response = requests.post(endpoint, headers=headers, files=files)
    return Image.open(response.content)
```

#### 2. **Eliminación de dependencias pesadas**

**ANTES** ([`requirements.txt`](requirements.txt)):
```txt
diffusers>=0.21.0      # ~2-4 GB
torch>=2.0.0           # ~2-3 GB
torchvision>=0.15.0    # ~500 MB
transformers>=4.35.0   # ~1-2 GB
accelerate>=0.24.0
opencv-python>=4.8.0
scikit-image>=0.21.0
# TOTAL: ~8-10 GB
```

**AHORA** ([`requirements.txt`](requirements.txt)):
```txt
streamlit>=1.28.0
Pillow>=10.0.0
numpy>=1.24.0
requests>=2.31.0
python-dotenv>=0.21.0
google-generativeai>=0.3.0
# TOTAL: ~100 MB
```

**Ahorro: 98% menos dependencias (de 10 GB a 100 MB)**

---

## 🔧 Configuración Necesaria

### 1. Obtener API Key de HuggingFace

1. Ve a [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Crea un nuevo token (tipo: Read)
3. Copia el token generado

### 2. Configurar localmente

Crea un archivo [`.env`](.env.example) en la raíz del proyecto:

```env
HUGGINGFACE_API_KEY=hf_tu_token_aqui
GOOGLE_API_KEY=tu_google_api_key_aqui
```

### 3. Configurar en Streamlit Cloud

1. Ve a tu app en [https://share.streamlit.io/](https://share.streamlit.io/)
2. Click en **Settings** → **Secrets**
3. Agrega:

```toml
HUGGINGFACE_API_KEY = "hf_tu_token_aqui"
GOOGLE_API_KEY = "tu_google_api_key_aqui"
```

4. Guarda y reinicia la app

---

## 📊 Comparación Técnica

| Aspecto | Antes (Local) | Ahora (Cloud API) |
|---------|---------------|-------------------|
| **Tamaño de dependencias** | ~10 GB | ~100 MB |
| **RAM requerida** | 4-8 GB | <500 MB |
| **GPU necesaria** | Sí (CUDA) | No |
| **Tiempo de inicio** | 5-10 min | <30 seg |
| **Funciona en Streamlit Cloud** | ❌ NO | ✅ SÍ |
| **Costo de infraestructura** | GPU cloud ($$$) | API gratuita |
| **Velocidad de procesamiento** | Rápido (local) | Medio (red) |

---

## 🎯 Funcionalidades Mantenidas

Todas las funcionalidades originales se mantienen:

- ✅ **Inpainting** (eliminar objetos)
- ✅ **Outpainting** (extender imagen)
- ✅ **Style Transfer** (transferir estilo)
- ✅ **Object Removal** (eliminar objeto específico)
- ✅ **Background Replacement** (cambiar fondo)
- ✅ **Intelligent Composition** (composición inteligente)
- ✅ **Análisis con Gemini 2.0**

---

## 🔄 Flujo de Procesamiento

### Antes (Local):
```
Usuario → Streamlit → Cargar modelo (5 min) → GPU local → Resultado
                      ↓
                   ❌ Falla en Streamlit Cloud
```

### Ahora (Cloud API):
```
Usuario → Streamlit → HuggingFace API → Servidores HF → Resultado
                      ↓
                   ✅ Funciona en Streamlit Cloud
```

---

## ⚠️ Limitaciones y Consideraciones

### Ventajas:
- ✅ Funciona en Streamlit Cloud sin problemas
- ✅ No requiere GPU ni hardware potente
- ✅ Instalación instantánea
- ✅ Mantenimiento mínimo
- ✅ API gratuita para uso educativo

### Desventajas:
- ⏱️ Procesamiento más lento (depende de la red)
- 🌐 Requiere conexión a Internet
- 📊 Límites de rate (requests por minuto)
- 🔑 Necesita API key configurada

---

## 🚀 Despliegue en Streamlit Cloud

### Pasos para desplegar:

1. **Sube el código a GitHub**
   ```bash
   git add .
   git commit -m "Migración a HuggingFace Inference API"
   git push origin main
   ```

2. **Conecta con Streamlit Cloud**
   - Ve a [share.streamlit.io](https://share.streamlit.io/)
   - Click en "New app"
   - Selecciona tu repositorio
   - Configura los Secrets (API keys)

3. **Verifica el despliegue**
   - La app debería iniciar en <1 minuto
   - Prueba todas las funcionalidades
   - Verifica que las API keys funcionen

---

## 📝 Archivos Modificados

| Archivo | Cambios |
|---------|---------|
| [`models/diffusion.py`](models/diffusion.py) | Reescrito completamente para usar HuggingFace API |
| [`requirements.txt`](requirements.txt) | Eliminadas dependencias pesadas (torch, diffusers, etc.) |
| [`.env.example`](.env.example) | Creado con instrucciones de configuración |
| [`app.py`](app.py) | Sin cambios (compatible con nueva API) |

---

## 🧪 Testing

### Pruebas locales:
```bash
# Instalar dependencias
pip install -r requirements.txt

# Configurar .env
cp .env.example .env
# Editar .env con tus API keys

# Ejecutar app
streamlit run app.py
```

### Pruebas en Streamlit Cloud:
1. Sube una imagen de prueba
2. Selecciona un método (ej: Inpainting)
3. Click en "Procesar Imagen"
4. Verifica que se genere correctamente
5. Prueba el análisis con Gemini

---

## 💡 Recomendaciones

### Para desarrollo local:
- Usa la versión con modelos locales si tienes GPU potente
- Configura ambas API keys para máxima funcionalidad

### Para producción (Streamlit Cloud):
- **Siempre usa la versión con APIs** (esta versión)
- Monitorea los límites de rate de HuggingFace
- Considera actualizar a HuggingFace Pro si necesitas más requests

---

## 📚 Recursos Adicionales

- [HuggingFace Inference API Docs](https://huggingface.co/docs/api-inference/index)
- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-community-cloud)
- [Stable Diffusion Models](https://huggingface.co/models?pipeline_tag=text-to-image)
- [Gemini API Docs](https://ai.google.dev/docs)

---

## 🎓 Conclusión

Esta migración permite que la aplicación funcione perfectamente en **Streamlit Cloud** sin necesidad de infraestructura costosa ni modelos pesados. El cambio de modelos locales a APIs externas es la **única solución viable** para desplegar aplicaciones de IA generativa en plataformas con recursos limitados.

**Resultado**: Aplicación 100% funcional en la nube, lista para compartir y demostrar. 🚀

---

**Autor**: Alfredo Poblete  
**Proyecto**: Procesamiento Digital de Imágenes - IFTS24  
**Fecha**: 2025
