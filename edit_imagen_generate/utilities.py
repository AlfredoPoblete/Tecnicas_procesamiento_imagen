"""
Utilidades consolidadas para la aplicación de edición de imágenes
Incluye UIHelper e ImageProcessor optimizados para Streamlit Cloud
"""

import streamlit as st
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
import io
import base64
import time
import logging

logger = logging.getLogger(__name__)

class UIHelper:
    """Helper para componentes de UI reutilizables"""
    
    def __init__(self):
        self.processing_methods = {
            "Inpainting (Eliminar objetos)": {
                "key": "inpainting",
                "description": "Elimina y rellena objetos no deseados",
                "icon": "🖼️"
            },
            "Outpainting (Extender imagen)": {
                "key": "outpainting", 
                "description": "Extiende la imagen más allá de sus bordes",
                "icon": "🔄"
            },
            "Style Transfer (Transferir estilo)": {
                "key": "style_transfer",
                "description": "Aplica estilos artísticos a la imagen",
                "icon": "🎭"
            },
            "Object Removal (Eliminar objeto específico)": {
                "key": "object_removal",
                "description": "Elimina objetos específicos con precisión",
                "icon": "🗑️"
            },
            "Background Replacement (Cambiar fondo)": {
                "key": "background_replacement", 
                "description": "Reemplaza el fondo manteniendo el sujeto",
                "icon": "🖼️"
            },
            "Composición Inteligente (Combinar elementos)": {
                "key": "intelligent_composition",
                "description": "Combina elementos de múltiples imágenes",
                "icon": "🧩"
            }
        }
    
    def get_processing_params(self, method: str) -> Dict[str, Any]:
        """Obtener parámetros para el método de procesamiento seleccionado"""
        try:
            params = {}
            
            if "Inpainting" in method:
                params = self._get_inpainting_params()
            elif "Outpainting" in method:
                params = self._get_outpainting_params()
            elif "Style Transfer" in method:
                params = self._get_style_transfer_params()
            elif "Object Removal" in method:
                params = self._get_object_removal_params()
            elif "Background Replacement" in method:
                params = self._get_background_replacement_params()
            elif "Composición Inteligente" in method:
                params = self._get_composition_params()
            else:
                params = self._get_default_params()
            
            return params
            
        except Exception as e:
            logger.error(f"Error obteniendo parámetros para {method}: {e}")
            return self._get_default_params()
    
    def _get_inpainting_params(self) -> Dict[str, Any]:
        """Parámetros para inpainting"""
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                x = st.number_input("Posición X", min_value=0, max_value=512, value=200)
                y = st.number_input("Posición Y", min_value=0, max_value=512, value=200)
            
            with col2:
                width = st.number_input("Ancho", min_value=10, max_value=512, value=100)
                height = st.number_input("Alto", min_value=10, max_value=512, value=100)
            
            prompt = st.text_input(
                "Prompt para el relleno", 
                value="natural background texture",
                help="Describe qué quieres que aparezca en el área eliminada"
            )
            
            return {
                'prompt': prompt,
                'mask_coords': (x, y, x + width, y + height),
                'num_inference_steps': st.slider(
                    "Pasos de procesamiento", 
                    min_value=15, max_value=50, value=25,
                    help="Más pasos = mejor calidad pero más lento"
                ),
                'guidance_scale': st.slider(
                    "Control de adherencia", 
                    min_value=5.0, max_value=12.0, value=7.5, step=0.5
                )
            }
        except Exception as e:
            logger.error(f"Error en parámetros de inpainting: {e}")
            return self._get_default_params()
    
    def _get_outpainting_params(self) -> Dict[str, Any]:
        """Parámetros para outpainting"""
        try:
            return {
                'extension_factor': st.slider(
                    "Factor de extensión",
                    min_value=1.2, max_value=2.5, value=1.5, step=0.1,
                    help="1.5 = 50% más grande"
                ),
                'prompt': st.text_input(
                    "Prompt para la extensión",
                    value="seamless natural extension, matching the existing scene"
                ),
                'num_inference_steps': st.slider(
                    "Pasos de procesamiento",
                    min_value=20, max_value=60, value=35
                ),
                'guidance_scale': st.slider(
                    "Control de adherencia",
                    min_value=6.0, max_value=12.0, value=8.0, step=0.5
                )
            }
        except Exception as e:
            logger.error(f"Error en parámetros de outpainting: {e}")
            return self._get_default_params()
    
    def _get_style_transfer_params(self) -> Dict[str, Any]:
        """Parámetros para style transfer"""
        try:
            style_options = [
                "artistic painting style",
                "watercolor painting",
                "oil painting effect",
                "cartoon/anime style",
                "sketch/pencil drawing",
                "vintage sepia style",
                "modern digital art",
                "impressionist style"
            ]
            
            return {
                'style_prompt': st.selectbox(
                    "Estilo artístico", 
                    style_options,
                    index=0
                ),
                'strength': st.slider(
                    "Intensidad del estilo", 
                    min_value=0.1, max_value=1.0, value=0.6, step=0.1
                ),
                'num_inference_steps': st.slider(
                    "Pasos de procesamiento", 
                    min_value=15, max_value=50, value=25
                ),
                'guidance_scale': st.slider(
                    "Control de adherencia", 
                    min_value=5.0, max_value=12.0, value=7.5, step=0.5
                )
            }
        except Exception as e:
            logger.error(f"Error en parámetros de style transfer: {e}")
            return self._get_default_params()
    
    def _get_object_removal_params(self) -> Dict[str, Any]:
        """Parámetros para object removal"""
        try:
            return {
                'object_description': st.text_input(
                    "Objeto a eliminar",
                    value="unwanted object",
                    help="Describe específicamente el objeto que quieres eliminar"
                ),
                'context_prompt': st.text_input(
                    "Contexto del fondo",
                    value="natural seamless background"
                ),
                'num_inference_steps': st.slider(
                    "Pasos de procesamiento",
                    min_value=20, max_value=60, value=35
                ),
                'guidance_scale': st.slider(
                    "Control de adherencia",
                    min_value=7.0, max_value=12.0, value=9.0, step=0.5
                )
            }
        except Exception as e:
            logger.error(f"Error en parámetros de object removal: {e}")
            return self._get_default_params()
    
    def _get_background_replacement_params(self) -> Dict[str, Any]:
        """Parámetros para background replacement"""
        try:
            background_options = [
                "beautiful sunset landscape",
                "modern city skyline", 
                "serene forest scene",
                "ocean beach setting",
                "mountain vista",
                "studio photography backdrop",
                "abstract colorful background"
            ]
            
            return {
                'background_prompt': st.selectbox(
                    "Tipo de fondo", 
                    background_options,
                    index=0
                ),
                'num_inference_steps': st.slider(
                    "Pasos de procesamiento", 
                    min_value=20, max_value=60, value=35
                ),
                'guidance_scale': st.slider(
                    "Control de adherencia", 
                    min_value=6.0, max_value=12.0, value=8.5, step=0.5
                )
            }
        except Exception as e:
            logger.error(f"Error en parámetros de background replacement: {e}")
            return self._get_default_params()
    
    def _get_composition_params(self) -> Dict[str, Any]:
        """Parámetros para composición inteligente"""
        try:
            composition_options = [
                "harmonious artistic composition",
                "balanced visual elements",
                "creative collage style",
                "seamless blending",
                "layered depth composition"
            ]
            
            return {
                'elements_prompt': st.selectbox(
                    "Estilo de composición", 
                    composition_options,
                    index=0
                ),
                'strength': st.slider(
                    "Intensidad de la composición", 
                    min_value=0.2, max_value=1.0, value=0.5, step=0.1
                ),
                'num_inference_steps': st.slider(
                    "Pasos de procesamiento", 
                    min_value=20, max_value=60, value=30
                ),
                'guidance_scale': st.slider(
                    "Control de adherencia", 
                    min_value=6.0, max_value=12.0, value=8.0, step=0.5
                )
            }
        except Exception as e:
            logger.error(f"Error en parámetros de composición: {e}")
            return self._get_default_params()
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Parámetros por defecto seguros"""
        return {
            'num_inference_steps': 25,
            'guidance_scale': 7.5,
            'prompt': 'natural background'
        }

class ImageProcessor:
    """Procesador de utilidades para imágenes"""
    
    def __init__(self):
        pass
    
    def resize_image(self, image: Image.Image, max_size: int = 512) -> Image.Image:
        """Redimensionar imagen manteniendo proporción"""
        try:
            if image is None:
                return None
                
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            logger.error(f"Error redimensionando imagen: {e}")
            return image
    
    def create_circular_mask(self, image: Image.Image, center: Tuple[int, int], radius: int) -> Image.Image:
        """Crear máscara circular"""
        try:
            if image is None:
                return None
                
            width, height = image.size
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)
            
            bbox = [center[0] - radius, center[1] - radius, 
                    center[0] + radius, center[1] + radius]
            draw.ellipse(bbox, fill=255)
            return mask
        except Exception as e:
            logger.error(f"Error creando máscara circular: {e}")
            return Image.new('L', (image.width, image.height), 0)
    
    def enhance_image(self, image: Image.Image, brightness: float = 1.0,
                     contrast: float = 1.0, sharpness: float = 1.0) -> Image.Image:
        """Mejorar imagen ajustando parámetros básicos"""
        try:
            if image is None:
                return None
                
            enhanced = image.copy()
            
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(enhanced)
                enhanced = enhancer.enhance(brightness)
            
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(contrast)
            
            if sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(enhanced)
                enhanced = enhancer.enhance(sharpness)
            
            return enhanced
        except Exception as e:
            logger.error(f"Error mejorando imagen: {e}")
            return image
    
    def calculate_image_metrics(self, image: Image.Image) -> Dict[str, Any]:
        """Calcular métricas básicas de la imagen"""
        try:
            if image is None:
                return {}
                
            img_array = np.array(image.convert('L'))
            
            # Métricas básicas
            mean_brightness = float(np.mean(img_array))
            std_brightness = float(np.std(img_array))
            min_brightness = float(np.min(img_array))
            max_brightness = float(np.max(img_array))
            
            # Contraste
            contrast = max_brightness - min_brightness
            
            # Entropía
            hist, _ = np.histogram(img_array, bins=256, range=(0, 256))
            hist = hist / np.sum(hist)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log2(hist))
            
            return {
                'width': image.width,
                'height': image.height,
                'mode': image.mode,
                'mean_brightness': round(mean_brightness, 2),
                'std_brightness': round(std_brightness, 2),
                'contrast': round(contrast, 2),
                'entropy': round(entropy, 2),
                'aspect_ratio': round(image.width / image.height, 2)
            }
        except Exception as e:
            logger.error(f"Error calculando métricas: {e}")
            return {}
    
    def compare_images(self, img1: Image.Image, img2: Image.Image) -> Dict[str, Any]:
        """Comparar dos imágenes"""
        try:
            if img1 is None or img2 is None:
                return {}
                
            # Convertir a arrays numpy
            arr1 = np.array(img1.convert('RGB'))
            arr2 = np.array(img2.convert('RGB'))
            
            # Redimensionar si es necesario
            if arr1.shape != arr2.shape:
                target_size = (min(arr1.shape[1], arr2.shape[1]), 
                              min(arr1.shape[0], arr2.shape[0]))
                img1_resized = img1.resize(target_size, Image.Resampling.LANCZOS)
                img2_resized = img2.resize(target_size, Image.Resampling.LANCZOS)
                arr1 = np.array(img1_resized.convert('RGB'))
                arr2 = np.array(img2_resized.convert('RGB'))
            
            # Calcular diferencias
            diff = np.abs(arr1.astype(np.float32) - arr2.astype(np.float32))
            total_diff = np.mean(diff)
            
            # Similitud
            similarity = max(0, 1 - (total_diff / 255))
            
            return {
                'similarity': round(similarity, 3),
                'mean_difference': round(total_diff, 2),
                'psnr': 20 * np.log10(255 / np.sqrt(np.mean((arr1 - arr2) ** 2)))
            }
        except Exception as e:
            logger.error(f"Error comparando imágenes: {e}")
            return {}
    
    def create_sample_masks(self, image: Image.Image) -> Dict[str, Image.Image]:
        """Crear máscaras de ejemplo"""
        try:
            if image is None:
                return {}
                
            width, height = image.size
            masks = {}
            
            # Máscara central circular
            center_mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(center_mask)
            radius = min(width, height) // 4
            bbox = [width//2 - radius, height//2 - radius, 
                    width//2 + radius, height//2 + radius]
            draw.ellipse(bbox, fill=255)
            masks['circular_center'] = center_mask
            
            # Máscara cuadrada
            square_mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(square_mask)
            size = min(width, height) // 3
            draw.rectangle([0, 0, size, size], fill=255)
            masks['square_corner'] = square_mask
            
            return masks
        except Exception as e:
            logger.error(f"Error creando máscaras: {e}")
            return {}
    
    def image_to_base64(self, image: Image.Image, format: str = 'PNG') -> str:
        """Convertir imagen a base64"""
        try:
            if image is None:
                return ""
                
            buffer = io.BytesIO()
            image.save(buffer, format=format)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error convirtiendo a base64: {e}")
            return ""
    
    def validate_image(self, image: Image.Image) -> Tuple[bool, str]:
        """Validar imagen"""
        try:
            if image is None:
                return False, "La imagen es None"
            
            # Verificar tamaño
            if image.width < 64 or image.height < 64:
                return False, "La imagen es demasiado pequeña"
            
            if image.width > 2048 or image.height > 2048:
                return False, "La imagen es demasiado grande"
            
            return True, "Imagen válida"
        except Exception as e:
            return False, f"Error validando: {str(e)}"
    
    def get_recommended_params(self, image: Image.Image, method: str) -> Dict[str, Any]:
        """Obtener parámetros recomendados"""
        try:
            if image is None:
                return {}
                
            width, height = image.size
            area = width * height
            
            base_params = {
                'num_inference_steps': 25,
                'guidance_scale': 7.5
            }
            
            if method == 'inpainting':
                if area > 512*512:
                    base_params['num_inference_steps'] = 35
                    base_params['guidance_scale'] = 8.0
                else:
                    base_params['num_inference_steps'] = 20
                    base_params['guidance_scale'] = 7.0
            
            elif method == 'style_transfer':
                base_params['strength'] = 0.6
                base_params['num_inference_steps'] = 30
            
            elif method == 'outpainting':
                base_params['num_inference_steps'] = 40
                base_params['guidance_scale'] = 8.5
            
            return base_params
        except Exception as e:
            logger.error(f"Error obteniendo parámetros recomendados: {e}")
            return {}

# Funciones de utilidad adicionales
def create_thumbnail(image: Image.Image, size: Tuple[int, int] = (160, 160)) -> Optional[Image.Image]:
    """Crear miniatura segura"""
    try:
        if image is None:
            return None
            
        thumb = image.copy()
        thumb.thumbnail(size, Image.Resampling.LANCZOS)
        
        canvas = Image.new('RGB', size, (255, 255, 255))
        x_off = (size[0] - thumb.size[0]) // 2
        y_off = (size[1] - thumb.size[1]) // 2
        canvas.paste(thumb, (x_off, y_off))
        
        return canvas
    except Exception as e:
        logger.error(f"Error creando miniatura: {e}")
        return None

def health_check() -> Dict[str, Any]:
    """Verificar estado del sistema de utilidades"""
    try:
        return {
            'status': 'healthy',
            'timestamp': time.time(),
            'ui_helper_ready': True,
            'image_processor_ready': True
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }