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
    
    # Configuración simplificada y consolidada
    MODEL_CONFIG = {
        'inpainting': {
            'model_name': 'runwayml/stable-diffusion-inpainting',
            'default_steps': 25,
            'default_guidance': 7.0
        },
        'img2img': {
            'model_name': 'runwayml/stable-diffusion-v1-5',
            'default_steps': 20,
            'default_guidance': 6.5
        },
        'upscale': {
            'model_name': 'stabilityai/stable-diffusion-x4-upscaler',
            'default_steps': 25,
            'default_guidance': 7.0
        }
    }
    
    IMAGE_CONFIG = {
        'max_size': 512, 'min_size': 64,
        'allowed_formats': ['JPEG', 'JPG', 'PNG'],
        'quality': 95
    }
    
    ANALYSIS_CONFIG = {
        'model': 'gemini-2.0-flash', 'max_tokens': 2048,
        'temperature': 0.5, 'topK': 32, 'topP': 1
    }
    
    UI_CONFIG = {
        'page_title': 'Edición Generativa de Imágenes',
        'page_icon': '🎨', 'layout': 'wide',
        'initial_sidebar_state': 'expanded'
    }
    
    PERFORMANCE_CONFIG = {
        'batch_size': 1, 'num_workers': 0, 'use_amp': True,
        'torch_dtype': 'float16' if os.getenv('CUDA_VISIBLE_DEVICES') else 'float32',
        'enable_cudnn_benchmark': True, 'allow_tf32': True,
        'max_memory_mb': 8000, 'low_memory_mode': True
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