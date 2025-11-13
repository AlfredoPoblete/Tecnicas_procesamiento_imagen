"""
Clasificador de Imágenes con Comparativa de Modelos

Esta aplicación utiliza Gradio para una interfaz de usuario web.
Compara la clasificación de imágenes utilizando:
1. Un modelo personalizado (Keras/Teachable Machine) cargado desde "models/keras_model.h5".
2. Un modelo preentrenado Zero-Shot (CLIP) de Hugging Face.
"""

import gradio as gr
from transformers import pipeline
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import sys

# ============================================
# CONFIGURACIÓN DE RUTAS Y CATEGORÍAS
# ============================================

# Categorías usadas por el modelo CLIP (deben ser descriptivas para el zero-shot)
CATEGORIAS_CLIP = [
    'Hojas sanas',
    'Hojas afectadas por insectos',
    'Hojas con presencia de hongos',
    'Hojas con deficiencia de nutrientes'
]

MODELO_CUSTOM_PATH = "models/keras_model.h5"
ETIQUETAS_CUSTOM_PATH = "models/labels.txt"

# ============================================
# CARGAR MODELOS
# ============================================

# Solución para compatibilidad de Teachable Machine con Keras
# Esto es necesario si el modelo personalizado fue exportado desde Teachable Machine.
def DepthwiseConv2D_personalizada(
    kernel_size, strides=(1, 1), padding="valid", depth_multiplier=1,
    data_format=None, dilation_rate=(1, 1), activation=None,
    use_bias=True, depthwise_initializer="glorot_uniform",
    bias_initializer="zeros", depthwise_regularizer=None,
    bias_regularizer=None, activity_regularizer=None,
    depthwise_constraint=None, bias_constraint=None, **kwargs
):
    kwargs.pop("groups", None)
    return tf.keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size, strides=strides, padding=padding,
        depth_multiplier=depth_multiplier, data_format=data_format,
        dilation_rate=dilation_rate, activation=activation,
        use_bias=use_bias, depthwise_initializer=depthwise_initializer,
        bias_initializer=bias_initializer,
        depthwise_regularizer=depthwise_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        depthwise_constraint=depthwise_constraint,
        bias_constraint=bias_constraint, **kwargs
    )

modelo_custom = None
etiquetas_custom = []

try:
    print(f"Cargando modelo personalizado desde: {MODELO_CUSTOM_PATH}...")
    # Cargar modelo Keras
    modelo_custom = load_model(
        MODELO_CUSTOM_PATH,
        compile=False,
        custom_objects={"DepthwiseConv2D": DepthwiseConv2D_personalizada}
    )

    # Cargar etiquetas
    with open(ETIQUETAS_CUSTOM_PATH, "r", encoding="utf-8") as f:
        # Se asume que las etiquetas pueden venir con el índice: '0 Hojas sanas'
        etiquetas_custom = [line.strip().split(" ", 1)[-1] for line in f.readlines() if line.strip()]

    print(f"Modelo personalizado cargado correctamente con {len(etiquetas_custom)} clases.")
except FileNotFoundError:
    print(f"ADVERTENCIA: Archivos de modelo personalizado no encontrados en {MODELO_CUSTOM_PATH} o {ETIQUETAS_CUSTOM_PATH}.")
    print("La clasificación con el modelo personalizado no estará disponible.")
except Exception as e:
    print(f"ERROR al cargar modelo personalizado: {e}")
    
# Cargar modelo preentrenado (CLIP)
try:
    print("Cargando modelo CLIP (Preentrenado)...")
    modelo_clip = pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32"
    )
    print("Modelo CLIP cargado correctamente.")
except Exception as e:
    print(f"ERROR al cargar modelo CLIP: {e}")
    sys.exit(1) # Detener si el modelo principal no carga

# ============================================
# FUNCIONES DE PROCESAMIENTO
# ============================================

def preprocesar_para_teachable(imagen):
    """
    Preprocesa la imagen para el formato esperado por un modelo entrenado 
    con Teachable Machine (224x224, normalización [-1, 1]).
    """
    if imagen.mode != 'RGB':
        imagen = imagen.convert('RGB')
        
    # Redimensionar a 224x224
    imagen_redim = imagen.resize((224, 224))
    
    # Convertir a array (float32)
    array_imagen = np.asarray(imagen_redim, dtype=np.float32)
    
    # Añadir dimensión de batch (1, 224, 224, 3)
    array_imagen = array_imagen.reshape(1, 224, 224, 3)
    
    # Normalizar [-1, 1]
    array_imagen = (array_imagen / 127.5) - 1
    
    return array_imagen

def clasificar_con_custom(imagen):
    """
    Clasifica la imagen utilizando el modelo Keras personalizado.
    """
    if imagen is None:
        return {"Error": 1.0}
    
    if modelo_custom is None or not etiquetas_custom:
        return {"Error: Modelo no cargado o etiquetas faltantes.": 1.0}

    try:
        # Preprocesar
        img_procesada = preprocesar_para_teachable(imagen)
        
        # Predecir
        prediccion = modelo_custom.predict(img_procesada, verbose=0)
        
        # Formatear resultados
        resultados = {
            etiquetas_custom[i]: float(prediccion[0][i])
            for i in range(len(etiquetas_custom))
        }
        
        # Ordenar por probabilidad
        resultados = dict(
            sorted(resultados.items(), key=lambda x: x[1], reverse=True)
        )
        
        return resultados
    
    except Exception as e:
        print(f"Error en clasificación con modelo personalizado: {e}")
        return {"Error": 1.0}

def clasificar_con_clip(imagen):
    """
    Clasifica la imagen utilizando el modelo CLIP zero-shot.
    """
    if imagen is None:
        return {"Error": 1.0}
    
    try:
        # Ejecutar pipeline de CLIP
        resultados = modelo_clip(imagen, candidate_labels=CATEGORIAS_CLIP)
        
        # Formatear resultados
        return {
            resultado['label']: float(resultado['score']) 
            for resultado in resultados
        }
    except Exception as e:
        print(f"Error en clasificación con CLIP: {e}")
        return {"Error": 1.0}

# Función unificada para que un solo botón active ambos modelos (opcional, pero útil)
def clasificar_ambos(imagen):
    """Clasifica la imagen con ambos modelos simultáneamente."""
    return clasificar_con_custom(imagen), clasificar_con_clip(imagen)

# ============================================
# INTERFAZ GRADIO CON COMPARACIÓN
# ============================================

# Se usa el número de clases máximo para el Label.
num_top_custom = len(etiquetas_custom) if etiquetas_custom else 5
num_top_clip = len(CATEGORIAS_CLIP)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # Clasificador de Imágenes de Hojas - Comparativa 
    
    Aplicación Mínimo Viable (MVP) para la materia Procesamiento Digital de Imágenes.
    Compara el rendimiento de dos estrategias de clasificación.
    
    **Instrucciones**:
    1. Sube una imagen de una hoja de planta o usa tu cámara.
    2. Haz clic en "Clasificar Ambos Modelos" para ver los resultados.
    """)
    
    with gr.Row():
        imagen_input = gr.Image(
            type="pil",
            label="1. Imagen a clasificar",
            sources=["upload", "webcam"],
            height=300
        )

    # Botón principal para ejecutar ambos
    boton_clasificar_ambos = gr.Button(
        "Clasificar Ambos Modelos",
        variant="primary",
        size="lg"
    )

    with gr.Row():
        with gr.Column():
            resultado_custom = gr.Label(
                label=f"2. Resultado: Modelo Personalizado ({num_top_custom} clases)",
                num_top_classes=num_top_custom
            )
        
        with gr.Column():
            resultado_clip = gr.Label(
                label=f"3. Resultado: CLIP (Zero-Shot) ({num_top_clip} categorías)",
                num_top_classes=num_top_clip
            )
            
    # Ejemplos (Asegúrate de agregar rutas válidas si las tienes en tu repo)
    gr.Examples(
        examples=[
              "assets/Sana.jpg",
              "assets/Nutrientes.jpg",
              "assets/Insectos.jpg",
              "assets/hongos.jpg",
        ],
        inputs=imagen_input,
        label="Ejemplos",
        cache_examples=False # Desactivar cache para desarrollo
    )

    gr.Markdown("""
    ---
    **Información Técnica**
    
    - **Modelo Personalizado (Keras)**: Entrenado específicamente para esta tarea con `models/keras_model.h5`.
    - **Modelo CLIP (OpenAI)**: Modelo general preentrenado, realiza clasificación zero-shot con las categorías definidas.
    - **Framework**: Hugging Face / TensorFlow / Gradio
    """)
    
    # Conectar función única al botón
    boton_clasificar_ambos.click(
        fn=clasificar_ambos,
        inputs=imagen_input,
        outputs=[resultado_custom, resultado_clip]
    )

# ============================================
# LANZAR APLICACIÓN
# ============================================

if __name__ == "__main__":
    demo.launch()