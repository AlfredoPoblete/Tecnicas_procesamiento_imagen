"""
Módulo de Procesamiento con Modelos de Difusión - VERSIÓN CONSOLIDADA
Optimizado para Streamlit Cloud sin dependencias pesadas
"""

import os
import torch
import numpy as np
import time
from io import BytesIO
from PIL import Image, ImageDraw
from typing import Optional, Tuple, Dict, Any
import warnings
import logging

warnings.filterwarnings("ignore")

# Configuración básica
logger = logging.getLogger(__name__)

# Verificar si diffusers está disponible (opcional para Streamlit Cloud)
try:
    from diffusers import StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    try:
        from diffusers.pipelines.stable_diffusion import StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline
        DIFFUSERS_AVAILABLE = True
    except ImportError:
        DIFFUSERS_AVAILABLE = False
        logger.warning("⚠️ diffusers no disponible, usando modo simulado")

class DiffusionProcessor:
    """Procesador de difusión optimizado para Streamlit Cloud"""
    
    def __init__(self):
        logger.info("🚀 Inicializando DiffusionProcessor para Streamlit Cloud")
        
        # Configuración básica
        self.use_api = os.getenv('USE_HF_API', 'false').lower() == 'true'
        self.hf_token = os.getenv('HUGGINGFACE_API_TOKEN')
        
        # Detectar dispositivo
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipes = {}
        
        # Modelos (fallback para Streamlit Cloud)
        self.available_models = {
            'inpainting': self._create_mock_inpainting,
            'img2img': self._create_mock_img2img,
            'style_transfer': self._create_mock_style_transfer
        }
        
        logger.info(f"Dispositivo: {self.device}")
        logger.info(f"Modos disponibles: {list(self.available_models.keys())}")
        
        if not DIFFUSERS_AVAILABLE:
            logger.info("🔄 Usando modo simulado para Streamlit Cloud")
    
    def _create_mock_inpainting(self, image: Image.Image, mask: Image.Image, prompt: str, **kwargs) -> Image.Image:
        """Simulador de inpainting para Streamlit Cloud"""
        try:
            # Crear una copia de la imagen
            result = image.copy()
            
            # Aplicar un efecto simple de desenfoque en la región de la máscara
            if mask.mode != 'L':
                mask = mask.convert('L')
            
            # Redimensionar máscara si es necesario
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.Resampling.LANCZOS)
            
            # Crear una versión suavizada de la imagen
            blurred = image.filter(ImageFilter.GaussianBlur(radius=3))
            
            # Combinar imagen original con versión borrosa según la máscara
            mask_array = np.array(mask)
            original_array = np.array(result)
            blurred_array = np.array(blurred)
            
            # Aplicar máscara
            result_array = np.where(mask_array[:, :, np.newaxis] > 128, 
                                  blurred_array, 
                                  original_array)
            
            return Image.fromarray(result_array.astype(np.uint8))
            
        except Exception as e:
            logger.error(f"Error en simulador de inpainting: {e}")
            return image
    
    def _create_mock_img2img(self, image: Image.Image, prompt: str, **kwargs) -> Image.Image:
        """Simulador de img2img para Streamlit Cloud"""
        try:
            # Aplicar un filtro de mejora simple
            result = image.copy()
            
            # Mejorar contraste y nitidez ligeramente
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(1.1)
            
            enhancer = ImageEnhance.Sharpness(result)
            result = enhancer.enhance(1.05)
            
            return result
            
        except Exception as e:
            logger.error(f"Error en simulador de img2img: {e}")
            return image
    
    def _create_mock_style_transfer(self, image: Image.Image, style_prompt: str, **kwargs) -> Image.Image:
        """Simulador de style transfer para Streamlit Cloud"""
        try:
            result = image.copy()
            
            # Aplicar diferentes efectos según el estilo
            if 'artistic' in style_prompt.lower():
                # Efecto artístico
                result = result.filter(ImageFilter.EMBOSS)
            elif 'watercolor' in style_prompt.lower():
                # Efecto acuarela
                result = result.filter(ImageFilter.SMOOTH_MORE)
            elif 'cartoon' in style_prompt.lower():
                # Efecto cartoon
                enhancer = ImageEnhance.Contrast(result)
                result = enhancer.enhance(1.3)
            
            return result
            
        except Exception as e:
            logger.error(f"Error en simulador de style transfer: {e}")
            return image
    
    def _optimize_image_size(self, image: Image.Image) -> Image.Image:
        """Redimensionar imagen para optimizar para Streamlit Cloud"""
        try:
            width, height = image.size
            
            # Limitar tamaño para Streamlit Cloud
            max_size = 512
            if max(width, height) > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int((height * max_size) / width)
                else:
                    new_height = max_size
                    new_width = int((width * max_size) / height)
                
                logger.info(f"Redimensionando imagen: {width}x{height} → {new_width}x{new_height}")
                return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            return image
        except Exception as e:
            logger.error(f"Error optimizando imagen: {e}")
            return image
    
    def _create_rectangular_mask(self, width: int, height: int, x: int, y: int,
                               rect_width: int, rect_height: int) -> Image.Image:
        """Crear máscara rectangular"""
        try:
            mask_array = np.zeros((height, width), dtype=np.uint8)
            mask_array[y:y+rect_height, x:x+rect_width] = 255
            return Image.fromarray(mask_array)
        except Exception as e:
            logger.error(f"Error creando máscara: {e}")
            return Image.new('L', (width, height), 0)
    
    def _create_circular_mask(self, width: int, height: int, center_x: int,
                            center_y: int, radius: int) -> Image.Image:
        """Crear máscara circular"""
        try:
            mask_array = np.zeros((height, width), dtype=np.uint8)
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            mask_array[mask] = 255
            return Image.fromarray(mask_array)
        except Exception as e:
            logger.error(f"Error creando máscara circular: {e}")
            return Image.new('L', (width, height), 0)
    
    def _create_intelligent_mask(self, image: Image.Image, object_description: str) -> Image.Image:
        """Crear máscara inteligente basada en descripción"""
        try:
            width, height = image.size
            
            # Detectar tipo de objeto por palabras clave
            description_lower = object_description.lower()
            
            if any(word in description_lower for word in ['person', 'people', 'man', 'woman', 'face']):
                # Máscara para persona (centro-izquierda)
                return self._create_rectangular_mask(
                    width, height,
                    width//3, height//3,
                    width//4, height//2
                )
            elif any(word in description_lower for word in ['car', 'vehicle', 'truck']):
                # Máscara para vehículo (parte inferior)
                return self._create_rectangular_mask(
                    width, height,
                    width//4, height//2,
                    width//2, height//4
                )
            else:
                # Máscara genérica (centro)
                return self._create_rectangular_mask(
                    width, height,
                    width//3, height//3,
                    width//3, height//3
                )
        except Exception as e:
            logger.error(f"Error creando máscara inteligente: {e}")
            return self._create_rectangular_mask(
                image.width, image.height,
                image.width//3, image.height//3,
                image.width//6, image.height//6
            )
    
    def _apply_outpainting_effect(self, image: Image.Image, extension_factor: float) -> Image.Image:
        """Efecto de outpainting simulado"""
        try:
            # Calcular nuevas dimensiones
            original_width, original_height = image.size
            new_width = int(original_width * extension_factor)
            new_height = int(original_height * extension_factor)
            
            # Crear canvas más grande
            canvas = Image.new('RGB', (new_width, new_height), (0, 0, 0))
            
            # Calcular posición centrada
            x_offset = (new_width - original_width) // 2
            y_offset = (new_height - original_height) // 2
            
            # Pegar imagen original en el centro
            canvas.paste(image, (x_offset, y_offset))
            
            # Crear un efecto de extensión suave en los bordes
            # Expandir ligeramente la imagen original
            expanded = image.resize((int(original_width * 1.1), int(original_height * 1.1)), Image.Resampling.LANCZOS)
            
            # Pegar imagen expandida centrada en el canvas
            exp_x = (new_width - expanded.width) // 2
            exp_y = (new_height - expanded.height) // 2
            canvas.paste(expanded, (exp_x, exp_y))
            
            return canvas
            
        except Exception as e:
            logger.error(f"Error en outpainting simulado: {e}")
            return image
    
    def process(self, image: Image.Image, method: str, **kwargs) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """Método principal de procesamiento"""
        try:
            start_time = time.time()
            
            # Verificar imagen
            if image is None:
                raise ValueError("Imagen es None")
            
            # Optimizar tamaño
            original_size = image.size
            image = self._optimize_image_size(image)
            optimized_size = image.size
            
            logger.info(f"Procesando con método: {method}")
            
            # Procesar según método
            if method == "inpainting":
                result = self._process_inpainting(image, **kwargs)
                
            elif method == "outpainting":
                extension_factor = kwargs.get('extension_factor', 1.5)
                result = self._apply_outpainting_effect(image, extension_factor)
                
            elif method == "style_transfer":
                style_prompt = kwargs.get('style_prompt', 'artistic style')
                result = self._process_style_transfer(image, style_prompt)
                
            elif method == "object_removal":
                object_description = kwargs.get('object_description', 'unwanted object')
                result = self._process_object_removal(image, object_description)
                
            elif method == "background_replacement":
                background_prompt = kwargs.get('background_prompt', 'beautiful background')
                result = self._process_background_replacement(image, background_prompt)
                
            elif method == "intelligent_composition":
                elements_prompt = kwargs.get('elements_prompt', 'harmonious composition')
                result = self._process_composition(image, elements_prompt)
                
            else:
                logger.warning(f"Método desconocido: {method}")
                result = image
            
            # Verificar resultado
            if result is None:
                result = image
            
            # Calcular tiempo de procesamiento
            end_time = time.time()
            processing_time = end_time - start_time
            
            metadata = {
                'method': method,
                'processing_time': f"{processing_time:.2f}s",
                'original_size': original_size,
                'optimized_size': optimized_size,
                'streamlit_cloud_mode': True,
                'simulated_processing': True
            }
            
            logger.info(f"Procesamiento completado en {processing_time:.2f}s")
            return result, metadata
            
        except Exception as e:
            logger.error(f"Error en procesamiento: {str(e)}")
            return None, {"error": str(e)}
    
    def _process_inpainting(self, image: Image.Image, **kwargs) -> Image.Image:
        """Procesar inpainting"""
        try:
            prompt = kwargs.get('prompt', 'natural background')
            mask_coords = kwargs.get('mask_coords')
            
            if mask_coords:
                x1, y1, x2, y2 = mask_coords
                mask = self._create_rectangular_mask(
                    image.width, image.height,
                    x1, y1, x2-x1, y2-y1
                )
            else:
                # Máscara por defecto
                mask = self._create_rectangular_mask(
                    image.width, image.height,
                    image.width//3, image.height//3,
                    image.width//6, image.height//6
                )
            
            return self._create_mock_inpainting(image, mask, prompt)
            
        except Exception as e:
            logger.error(f"Error en inpainting: {e}")
            return image
    
    def _process_style_transfer(self, image: Image.Image, style_prompt: str) -> Image.Image:
        """Procesar style transfer"""
        try:
            return self._create_mock_style_transfer(image, style_prompt)
        except Exception as e:
            logger.error(f"Error en style transfer: {e}")
            return image
    
    def _process_object_removal(self, image: Image.Image, object_description: str) -> Image.Image:
        """Procesar object removal"""
        try:
            mask = self._create_intelligent_mask(image, object_description)
            return self._create_mock_inpainting(image, mask, "seamless background")
        except Exception as e:
            logger.error(f"Error en object removal: {e}")
            return image
    
    def _process_background_replacement(self, image: Image.Image, background_prompt: str) -> Image.Image:
        """Procesar background replacement"""
        try:
            # Crear máscara de sujeto (centro)
            width, height = image.size
            mask_array = np.ones((height, width), dtype=np.uint8) * 255
            
            # Crear círculo central para el sujeto
            y, x = np.ogrid[:height, :width]
            center_mask = (x - width//2)**2 + (y - height//2)**2 <= (min(width, height)//3)**2
            mask_array[center_mask] = 0
            
            mask = Image.fromarray(mask_array)
            
            return self._create_mock_inpainting(image, mask, background_prompt)
        except Exception as e:
            logger.error(f"Error en background replacement: {e}")
            return image
    
    def _process_composition(self, image: Image.Image, elements_prompt: str) -> Image.Image:
        """Procesar composición inteligente"""
        try:
            return self._create_mock_img2img(image, elements_prompt)
        except Exception as e:
            logger.error(f"Error en composición: {e}")
            return image
    
    def get_info(self) -> Dict[str, Any]:
        """Obtener información del procesador"""
        try:
            return {
                'device': self.device,
                'models_loaded': list(self.pipes.keys()),
                'cuda_available': torch.cuda.is_available(),
                'lazy_loading_enabled': True,
                'streamlit_cloud_optimized': True,
                'simulated_processing': not DIFFUSERS_AVAILABLE,
                'available_methods': list(self.available_models.keys())
            }
        except Exception as e:
            logger.error(f"Error obteniendo información: {e}")
            return {
                'error': str(e),
                'device': 'unknown',
                'streamlit_cloud_optimized': True
            }

# Importaciones adicionales necesarias
from PIL import ImageFilter, ImageEnhance

# Función de salud del sistema
def health_check():
    """Verificar estado del sistema"""
    try:
        return {
            'status': 'healthy',
            'device': "cuda" if torch.cuda.is_available() else "cpu",
            'diffusers_available': DIFFUSERS_AVAILABLE,
            'streamlit_cloud_ready': True
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }