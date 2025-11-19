"""
Configuración centralizada del proyecto MVP
"""

import os
from typing import Dict, Any

class Config:
    """Configuración de la aplicación"""
    
    # Configuración de la aplicación
    APP_NAME = "MVP Edición Generativa de Imágenes"
    VERSION = "1.0.0"
    DESCRIPTION = "Aplicación avanzada que integra modelos de difusión y análisis inteligente"
    
    # Configuración OPTIMIZADA para Streamlit - Modelos más ligeros
    MODEL_CONFIG = {
        'inpainting': {
            'model_name': 'runwayml/stable-diffusion-inpainting',
            'default_steps': 20,  # Reducido para velocidad
            'default_guidance': 6.5,  # Reducido para velocidad
            'api_model': 'diffusers/stable-diffusion-inpainting-0.1'
        },
        'img2img': {
            'model_name': 'runwayml/stable-diffusion-v1-5',
            'default_steps': 15,  # Muy reducido para velocidad
            'default_guidance': 6.0,  # Reducido para velocidad
            'api_model': 'runwayml/stable-diffusion-v1-5'
        },
        'style_transfer': {
            'model_name': 'prompthero/openjourney',
            'default_steps': 12,  # Muy reducido para velocidad
            'default_guidance': 5.5,  # Reducido para velocidad
            'api_model': 'prompthero/openjourney'
        }
    }
    
    IMAGE_CONFIG = {
        'max_size': 256, 'min_size': 64,  # Más agresivo para Streamlit
        'allowed_formats': ['JPEG', 'JPG', 'PNG'],
        'quality': 90,  # Ligeramente reducido para velocidad
        'streamlit_max_size': 256  # Límite específico para Streamlit
    }
    
    ANALYSIS_CONFIG = {
        'model': 'gemini-2.0-flash', 'max_tokens': 1024,  # Reducido para velocidad
        'temperature': 0.3, 'topK': 20, 'topP': 0.8,  # Más conservador para velocidad
        'timeout': 30  # Timeout más corto para Streamlit
    }
    
    UI_CONFIG = {
        'page_title': 'Edición Generativa de Imágenes',
        'page_icon': '🎨', 'layout': 'wide',
        'initial_sidebar_state': 'expanded',
        'show_optimization_options': True,
        'show_model_status': True,
        'enable_performance_mode': True
    }
    
    PERFORMANCE_CONFIG = {
        'batch_size': 1, 'num_workers': 0, 'use_amp': True,
        'torch_dtype': 'float16' if os.getenv('CUDA_VISIBLE_DEVICES') else 'float32',
        'enable_cudnn_benchmark': True, 'allow_tf32': True,
        'max_memory_mb': 6000, 'low_memory_mode': True,  # Más conservador para Streamlit
        'aggressive_optimization': True,
        'enable_lazy_loading': True
    }
    
    FILE_CONFIG = {
        'temp_dir': 'temp', 'output_dir': 'output',
        'cache_dir': '.cache', 'max_file_size': 10 * 1024 * 1024
    }
    
    @classmethod
    def get_model_config(cls, model_type: str) -> Dict[str, Any]:
        """Obtener configuración de modelo específico"""
        return cls.MODEL_CONFIG.get(model_type, {})
    
    @classmethod
    def get_device(cls) -> str:
        """Obtener dispositivo de procesamiento"""
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def is_gpu_available(cls) -> bool:
        """Verificar si GPU está disponible"""
        import torch
        return torch.cuda.is_available()
    
    @classmethod
    def get_api_key(cls, service: str = 'gemini') -> str:
        """Obtener API key del servicio"""
        env_var = f"{service.upper()}_API_KEY"
        return os.getenv(env_var, '')

# Configuraciones predefinidas OPTIMIZADAS para diferentes casos de uso
PRESET_CONFIGS = {
    'development': {
        'num_inference_steps': 15,  # Reducido para velocidad
        'guidance_scale': 6.5,      # Reducido para velocidad
        'image_size': 256,          # Mantener para velocidad
        'batch_size': 1,
        'memory_efficient': True    # Optimización de memoria
    },
    'production': {
        'num_inference_steps': 30,  # Reducido de 50
        'guidance_scale': 7.5,      # Reducido de 8.0
        'image_size': 512,
        'batch_size': 1,
        'memory_efficient': True
    },
    'quality': {
        'num_inference_steps': 45,  # Reducido de 75
        'guidance_scale': 8.0,      # Reducido de 8.5
        'image_size': 512,
        'batch_size': 1,
        'memory_efficient': False   # Máxima calidad
    }
}