"""
Utilidades de Procesamiento de Imágenes - VERSIÓN ROBUSTA
Funciones auxiliares para manipulación y preparación de imágenes con manejo de errores

Integrado con conceptos del notebook de difusión
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from typing import Optional, Tuple, Dict, Any, List
import io
import base64
import logging
from functools import wraps
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def robust_execution(fallback_result=None):
    """Decorador para ejecución robusta con fallback"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error en {func.__name__}: {str(e)}")
                return fallback_result
        return wrapper
    return decorator

class ImageProcessorRobust:
    """Procesador de utilidades para imágenes con manejo robusto de errores"""
    
    def __init__(self):
        logger.info("🚀 Inicializando ImageProcessorRobust")
    
    @robust_execution(fallback_result=None)
    def resize_image(self, image: Image.Image, max_size: int = 512, 
                    maintain_aspect: bool = True) -> Optional[Image.Image]:
        """Redimensionar imagen manteniendo proporción"""
        try:
            if image is None:
                return None
                
            if maintain_aspect:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                return image
            else:
                return image.resize((max_size, max_size), Image.Resampling.LANCZOS)
        except Exception as e:
            logger.error(f"Error redimensionando imagen: {e}")
            return None
    
    @robust_execution(fallback_result=None)
    def create_mask_from_bbox(self, image: Image.Image, bbox: Tuple[int, int, int, int]) -> Optional[Image.Image]:
        """Crear máscara desde bounding box (x, y, width, height)"""
        try:
            if image is None:
                return None
                
            width, height = image.size
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle(bbox, fill=255)
            return mask
        except Exception as e:
            logger.error(f"Error creando máscara desde bbox: {e}")
            return None
    
    @robust_execution(fallback_result=None)
    def create_circular_mask(self, image: Image.Image, center: Tuple[int, int], 
                           radius: int) -> Optional[Image.Image]:
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
            return None
    
    @robust_execution(fallback_result=None)
    def create_brush_mask(self, image: Image.Image, points: List[Tuple[int, int]], 
                         brush_size: int = 20) -> Optional[Image.Image]:
        """Crear máscara de pincel libre"""
        try:
            if image is None or not points:
                return None
                
            width, height = image.size
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)
            
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                draw.line([(x1, y1), (x2, y2)], fill=255, width=brush_size)
            
            # Suavizar máscara
            mask = mask.filter(ImageFilter.GaussianBlur(radius=brush_size//4))
            return mask
        except Exception as e:
            logger.error(f"Error creando máscara de pincel: {e}")
            return None
    
    @robust_execution(fallback_result=None)
    def enhance_image(self, image: Image.Image, 
                     brightness: float = 1.0,
                     contrast: float = 1.0,
                     sharpness: float = 1.0) -> Optional[Image.Image]:
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
            return None
    
    @robust_execution(fallback_result={})
    def calculate_image_metrics(self, image: Image.Image) -> Dict[str, Any]:
        """Calcular métricas básicas de la imagen"""
        try:
            if image is None:
                return {}
                
            img_array = np.array(image.convert('L'))  # Convertir a escala de grises
            
            # Métricas básicas
            mean_brightness = float(np.mean(img_array))
            std_brightness = float(np.std(img_array))
            min_brightness = float(np.min(img_array))
            max_brightness = float(np.max(img_array))
            
            # Calcular contraste
            contrast = max_brightness - min_brightness
            
            # Calcular entropía
            hist, _ = np.histogram(img_array, bins=256, range=(0, 256))
            hist = hist / np.sum(hist)  # Normalizar
            hist = hist[hist > 0]  # Remover ceros para evitar log(0)
            entropy = -np.sum(hist * np.log2(hist))
            
            return {
                'width': image.width,
                'height': image.height,
                'mode': image.mode,
                'mean_brightness': round(mean_brightness, 2),
                'std_brightness': round(std_brightness, 2),
                'min_brightness': round(min_brightness, 2),
                'max_brightness': round(max_brightness, 2),
                'contrast': round(contrast, 2),
                'entropy': round(entropy, 2),
                'aspect_ratio': round(image.width / image.height, 2)
            }
        except Exception as e:
            logger.error(f"Error calculando métricas: {e}")
            return {}
    
    @robust_execution(fallback_result={})
    def compare_images(self, img1: Image.Image, img2: Image.Image) -> Dict[str, Any]:
        """Comparar dos imágenes y calcular similitudes/diferencias"""
        try:
            if img1 is None or img2 is None:
                return {}
                
            # Convertir a arrays numpy
            arr1 = np.array(img1.convert('RGB'))
            arr2 = np.array(img2.convert('RGB'))
            
            # Redimensionar si es necesario para comparación
            if arr1.shape != arr2.shape:
                target_size = (min(arr1.shape[1], arr2.shape[1]), 
                              min(arr1.shape[0], arr2.shape[0]))
                img1_resized = img1.resize(target_size, Image.Resampling.LANCZOS)
                img2_resized = img2.resize(target_size, Image.Resampling.LANCZOS)
                arr1 = np.array(img1_resized.convert('RGB'))
                arr2 = np.array(img2_resized.convert('RGB'))
            
            # Calcular diferencias por canal
            diff = np.abs(arr1.astype(np.float32) - arr2.astype(np.float32))
            mean_diff = np.mean(diff, axis=(0, 1))
            total_diff = np.mean(diff)
            
            # Calcular similitud
            similarity = max(0, 1 - (total_diff / 255))
            
            # Calcular PSNR aproximado
            mse = np.mean((arr1.astype(np.float32) - arr2.astype(np.float32)) ** 2)
            if mse > 0:
                psnr = 20 * np.log10(255 / np.sqrt(mse))
            else:
                psnr = float('inf')
            
            return {
                'similarity': round(similarity, 3),
                'mean_difference': {
                    'red': round(mean_diff[0], 2),
                    'green': round(mean_diff[1], 2),
                    'blue': round(mean_diff[2], 2),
                    'total': round(total_diff, 2)
                },
                'psnr': round(psnr, 2) if psnr != float('inf') else 'inf',
                'mse': round(mse, 2),
                'image1_size': (arr1.shape[1], arr1.shape[0]),
                'image2_size': (arr2.shape[1], arr2.shape[0])
            }
        except Exception as e:
            logger.error(f"Error comparando imágenes: {e}")
            return {}
    
    @robust_execution(fallback_result={})
    def create_sample_masks(self, image: Image.Image) -> Dict[str, Image.Image]:
        """Crear máscaras de ejemplo para diferentes técnicas"""
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
            
            # Máscara cuadrada en esquina
            square_mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(square_mask)
            size = min(width, height) // 3
            draw.rectangle([0, 0, size, size], fill=255)
            masks['square_corner'] = square_mask
            
            # Máscara de franja vertical
            stripe_mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(stripe_mask)
            stripe_width = width // 8
            draw.rectangle([width//2 - stripe_width//2, 0, 
                           width//2 + stripe_width//2, height], fill=255)
            masks['vertical_stripe'] = stripe_mask
            
            # Máscara de franja horizontal
            horizontal_mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(horizontal_mask)
            stripe_height = height // 8
            draw.rectangle([0, height//2 - stripe_height//2, 
                           width, height//2 + stripe_height//2], fill=255)
            masks['horizontal_stripe'] = horizontal_mask
            
            return masks
        except Exception as e:
            logger.error(f"Error creando máscaras de ejemplo: {e}")
            return {}
    
    @robust_execution(fallback_result="")
    def image_to_base64(self, image: Image.Image, format: str = 'PNG') -> str:
        """Convertir imagen a base64 string"""
        try:
            if image is None:
                return ""
                
            buffer = io.BytesIO()
            image.save(buffer, format=format)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error convirtiendo a base64: {e}")
            return ""
    
    @robust_execution(fallback_result=None)
    def base64_to_image(self, base64_string: str) -> Optional[Image.Image]:
        """Convertir base64 string a imagen"""
        try:
            if not base64_string:
                return None
                
            image_data = base64.b64decode(base64_string)
            return Image.open(io.BytesIO(image_data))
        except Exception as e:
            logger.error(f"Error convirtiendo desde base64: {e}")
            return None
    
    @robust_execution(fallback_result=(False, "Error procesando imagen"))
    def validate_image(self, image: Image.Image) -> Tuple[bool, str]:
        """Validar si la imagen es adecuada para procesamiento"""
        try:
            if image is None:
                return False, "La imagen es None"
            
            # Verificar tamaño mínimo
            if image.width < 64 or image.height < 64:
                return False, "La imagen es demasiado pequeña (mínimo 64x64 píxeles)"
            
            # Verificar tamaño máximo
            if image.width > 2048 or image.height > 2048:
                return False, "La imagen es demasiado grande (máximo 2048x2048 píxeles)"
            
            # Verificar que no sea completamente transparente
            if image.mode == 'RGBA':
                alpha_channel = np.array(image.split()[-1])
                if np.mean(alpha_channel) < 10:
                    return False, "La imagen es demasiado transparente"
            
            return True, "Imagen válida para procesamiento"
        except Exception as e:
            logger.error(f"Error validando imagen: {e}")
            return False, f"Error validando imagen: {str(e)}"
    
    @robust_execution(fallback_result={})
    def get_recommended_params(self, image: Image.Image, method: str) -> Dict[str, Any]:
        """Obtener parámetros recomendados según imagen y método"""
        try:
            if image is None:
                return {}
                
            width, height = image.size
            area = width * height
            
            base_params = {
                'num_inference_steps': 30,
                'guidance_scale': 7.5
            }
            
            if method == 'inpainting':
                if area > 512*512:
                    base_params['num_inference_steps'] = 40
                    base_params['guidance_scale'] = 8.0
                else:
                    base_params['num_inference_steps'] = 25
                    base_params['guidance_scale'] = 7.0
            
            elif method == 'style_transfer':
                base_params['strength'] = 0.6
                base_params['num_inference_steps'] = 35
            
            elif method == 'outpainting':
                base_params['num_inference_steps'] = 50
                base_params['guidance_scale'] = 8.5
            
            return base_params
        except Exception as e:
            logger.error(f"Error obteniendo parámetros recomendados: {e}")
            return {}
    
    @robust_execution(fallback_result=None)
    def create_safe_thumbnail(self, image: Image.Image, size: Tuple[int, int] = (160, 160)) -> Optional[Image.Image]:
        """Crear miniatura segura con manejo de errores"""
        try:
            if image is None:
                return None
                
            # Crear copia y redimensionar
            thumb = image.copy()
            thumb.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Crear canvas uniforme
            canvas = Image.new('RGB', size, (255, 255, 255))
            
            # Centrar la miniatura en el canvas
            x_off = (size[0] - thumb.size[0]) // 2
            y_off = (size[1] - thumb.size[1]) // 2
            canvas.paste(thumb, (x_off, y_off))
            
            return canvas
        except Exception as e:
            logger.error(f"Error creando miniatura: {e}")
            return None
    
    def health_check(self) -> Dict[str, Any]:
        """Verificar el estado de salud del procesador"""
        try:
            return {
                'status': 'healthy',
                'timestamp': time.time(),
                'processor_ready': True,
                'capabilities': [
                    'resize_image',
                    'create_mask_from_bbox',
                    'create_circular_mask',
                    'create_brush_mask',
                    'enhance_image',
                    'calculate_image_metrics',
                    'compare_images',
                    'create_sample_masks',
                    'image_to_base64',
                    'base64_to_image',
                    'validate_image',
                    'get_recommended_params'
                ]
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': time.time()
            }

# Alias para compatibilidad
ImageProcessor = ImageProcessorRobust