"""
Módulo de Procesamiento con Modelos de Difusión - VERSIÓN OPTIMIZADA STREAMLIT
Usa exclusivamente Hugging Face Inference API para evitar problemas de memoria en la nube
"""

import os
import base64
import io
import time
import requests
import torch
import numpy as np
from PIL import Image, ImageDraw
from typing import Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings("ignore")

class DiffusionProcessor:
    """Procesador principal para modelos de difusión - EXCLUSIVAMENTE API REMOTA"""
    
    def __init__(self):
        # FORZAR modo API remota para Streamlit Cloud
        self.use_hf_api = True
        self.hf_token = os.getenv('HUGGINGFACE_API_TOKEN') or os.getenv('HUGGINGFACEHUB_API_TOKEN') or os.getenv('HF_TOKEN')
        
        if not self.hf_token:
            raise RuntimeError("HUGGINGFACE_API_TOKEN es obligatorio para Streamlit Cloud")
        
        self.device = "cpu"  # No importa ya que usamos API remota
        self.pipes = {}  # No necesitamos pipelines locales
        
        # Configuración de modelos (solo nombres para API remota)
        # NOTE: hemos eliminado la configuración específica de "inpainting" para
        # evitar cualquier intento de carga/descarga local de modelos de inpainting.
        # Las operaciones basadas en imagen-a-imagen usan el modelo `img2img`.
        self.model_configs = {
            'img2img': {
                'model_name': 'runwayml/stable-diffusion-v1-5'
            },
            'upscale': {
                'model_name': 'stabilityai/stable-diffusion-x4-upscaler'
            }
        }
        
        print("🚀 DiffusionProcessor OPTIMIZADO para Streamlit Cloud")
        print("✅ Usando EXCLUSIVAMENTE Hugging Face Inference API")
        print("✅ Sin descarga de modelos locales - Sin problemas de memoria")
        
    def _call_hf_api(self, model_name: str, payload: dict, timeout: int = 120):
        """Llamar al endpoint de Inference API de Hugging Face.

        Envía JSON con 'inputs' y 'parameters' según lo requiera el modelo.
        Si la respuesta es binaria (imagen), la retornamos como bytes.
        """
        url = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {
            'Authorization': f'Bearer {self.hf_token}',
            'Accept': 'application/json'
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except Exception as e:
            raise RuntimeError(f"Error llamando a Hugging Face API: {e}")

        if resp.status_code != 200:
            # Intentar leer mensaje de error
            try:
                err = resp.json()
            except Exception:
                err = resp.text
            raise RuntimeError(f"Hugging Face API error ({resp.status_code}): {err}")

        # Si la respuesta es JSON, devolver el JSON; si es binaria (imagen), devolver bytes
        content_type = resp.headers.get('Content-Type', '')
        if content_type.startswith('application/json'):
            return resp.json()
        else:
            return resp.content

    def _parse_hf_image_response(self, resp) -> Image.Image:
        """Intentar extraer una imagen PIL de la respuesta de HF (bytes o JSON con base64)."""
        try:
            # Si ya son bytes (image/png), cargar directamente
            if isinstance(resp, (bytes, bytearray)):
                return Image.open(io.BytesIO(resp)).convert('RGB')

            # Si es JSON, buscar cadenas base64 en varias ubicaciones comunes
            if isinstance(resp, dict):
                # formatos posibles: {'image': '<b64>'} o {'images': ['<b64>']}
                if 'image' in resp and isinstance(resp['image'], str):
                    return Image.open(io.BytesIO(base64.b64decode(resp['image']))).convert('RGB')
                if 'images' in resp and isinstance(resp['images'], list) and len(resp['images']) > 0:
                    first = resp['images'][0]
                    if isinstance(first, str):
                        return Image.open(io.BytesIO(base64.b64decode(first))).convert('RGB')
                    if isinstance(first, dict) and 'image' in first and isinstance(first['image'], str):
                        return Image.open(io.BytesIO(base64.b64decode(first['image']))).convert('RGB')

            if isinstance(resp, list) and len(resp) > 0:
                first = resp[0]
                if isinstance(first, str):
                    return Image.open(io.BytesIO(base64.b64decode(first))).convert('RGB')
                if isinstance(first, dict):
                    for k in ['image', 'generated_image', 'image_base64']:
                        if k in first and isinstance(first[k], str):
                            return Image.open(io.BytesIO(base64.b64decode(first[k]))).convert('RGB')

        except Exception:
            pass

        raise RuntimeError('No se pudo interpretar la respuesta de la API de Hugging Face como imagen')
    
    def _optimize_image_size(self, image: Image.Image) -> Image.Image:
        """Redimensionar imagen para optimizar velocidad de procesamiento"""
        width, height = image.size
        
        # Si la imagen es muy grande, redimensionar para mejorar velocidad
        if max(width, height) > 512:
            # Mantener aspecto ratio
            if width > height:
                new_width = 512
                new_height = int((height * 512) / width)
            else:
                new_height = 512
                new_width = int((width * 512) / height)
            
            print(f"Redimensionando imagen de {width}x{height} a {new_width}x{new_height} para optimizar velocidad")
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Si la imagen es muy pequeña, mantener para preservar calidad
        elif min(width, height) < 256:
            print(f"Imagen pequeña ({width}x{height}) - manteniendo tamaño para preservar calidad")
            return image
        
        return image
    
    def create_rectangular_mask(self, width: int, height: int, x: int, y: int,
                               rect_width: int, rect_height: int) -> Image.Image:
        """Crear máscara rectangular"""
        mask_array = np.zeros((height, width), dtype=np.uint8)
        mask_array[y:y+rect_height, x:x+rect_width] = 255
        return Image.fromarray(mask_array)
    
    def create_circular_mask(self, width: int, height: int, center_x: int,
                           center_y: int, radius: int) -> Image.Image:
        """Crear máscara circular"""
        mask_array = np.zeros((height, width), dtype=np.uint8)
        y, x = np.ogrid[:height, :width]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        mask_array[mask] = 255
        return Image.fromarray(mask_array)
    
    def inpainting(self, image: Image.Image, mask: Image.Image,
                  prompt: str, **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """Función deshabilitada - usar outpainting o object_removal como alternativa"""
        raise Exception("Inpainting no está disponible. Usa 'Outpainting (Extender imagen)' o 'Object Removal (Eliminar objeto específico)' como alternativas.")
    
    def outpainting(self, image: Image.Image, extension_factor: float = 1.5,
                   prompt: str = "extended natural landscape", **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """Realizar outpainting (extender imagen) - EXCLUSIVAMENTE API REMOTA"""
        try:
            # Usar el modelo img2img (no inpainting) para operaciones de outpainting
            model_name = self.model_configs['img2img']['model_name']
            
            # Obtener dimensiones originales
            original_width, original_height = image.size
            
            # Calcular nuevas dimensiones
            new_width = int(original_width * extension_factor)
            new_height = int(original_height * extension_factor)
            
            print(f"🖼️ Outpainting: {original_width}x{original_height} → {new_width}x{new_height}")
            
            # Crear canvas más grande con fondo negro (para la máscara)
            extended_canvas = Image.new('RGB', (new_width, new_height), (0, 0, 0))
            
            # Calcular posición para centrar la imagen original
            x_offset = (new_width - original_width) // 2
            y_offset = (new_height - original_height) // 2
            
            # Pegar imagen original en el centro del canvas
            extended_canvas.paste(image, (x_offset, y_offset))
            
            # Crear máscara para las áreas a extender (bordes)
            mask_array = np.zeros((new_height, new_width), dtype=np.uint8)
            
            # Área central (imagen original) = 0 (no procesar)
            # Áreas de los bordes = 255 (procesar)
            
            # Superior
            mask_array[:y_offset, :] = 255
            # Inferior
            mask_array[y_offset + original_height:, :] = 255
            # Izquierda
            mask_array[y_offset:y_offset + original_height, :x_offset] = 255
            # Derecha
            mask_array[y_offset:y_offset + original_height, x_offset + original_width:] = 255
            
            # Crear máscara PIL
            mask = Image.fromarray(mask_array)
            
            print(f"✅ Máscara de outpainting creada: {mask.size}")
            print(f"📍 Imagen centrada en: ({x_offset}, {y_offset})")
            print(f"🎯 Áreas a extender: {np.sum(mask_array == 255)} píxeles")
            
            num_inference_steps = kwargs.get('num_inference_steps', 40)
            guidance_scale = kwargs.get('guidance_scale', 8.0)
            
            # Codificar imágenes a base64
            buf_img = io.BytesIO()
            extended_canvas.save(buf_img, format='PNG')
            img_b64 = base64.b64encode(buf_img.getvalue()).decode('utf-8')

            buf_mask = io.BytesIO()
            mask.save(buf_mask, format='PNG')
            mask_b64 = base64.b64encode(buf_mask.getvalue()).decode('utf-8')

            payload = {
                'inputs': {
                    'prompt': prompt,
                    'image': img_b64,
                    'mask_image': mask_b64
                },
                'parameters': {
                    'num_inference_steps': num_inference_steps,
                    'guidance_scale': guidance_scale
                }
            }

            resp = self._call_hf_api(model_name, payload)
            result = self._parse_hf_image_response(resp)
            
            metadata = {
                'method': 'outpainting',
                'prompt': prompt,
                'steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'device': 'hf_api',
                'optimized': True,
                'original_size': (original_width, original_height),
                'extended_size': (new_width, new_height),
                'extension_factor': extension_factor,
                'center_offset': (x_offset, y_offset),
                'api_mode': 'huggingface_remote'
            }
            
            print(f"✅ Outpainting completado: {result.size}")
            return result, metadata
            
        except Exception as e:
            raise Exception(f"Error en outpainting: {str(e)}")
    
    def style_transfer(self, image: Image.Image, style_prompt: str,
                      **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """Transferir estilo artístico - EXCLUSIVAMENTE API REMOTA"""
        try:
            model_name = self.model_configs['img2img']['model_name']
            
            strength = kwargs.get('strength', 0.5)
            num_inference_steps = kwargs.get('num_inference_steps', 20)
            guidance_scale = kwargs.get('guidance_scale', 6.5)
            
            # Codificar imagen a base64
            buf_img = io.BytesIO()
            image.save(buf_img, format='PNG')
            img_b64 = base64.b64encode(buf_img.getvalue()).decode('utf-8')

            payload = {
                'inputs': {
                    'prompt': style_prompt,
                    'image': img_b64
                },
                'parameters': {
                    'strength': strength,
                    'num_inference_steps': num_inference_steps,
                    'guidance_scale': guidance_scale
                }
            }

            resp = self._call_hf_api(model_name, payload)
            result = self._parse_hf_image_response(resp)
            
            metadata = {
                'method': 'style_transfer',
                'prompt': style_prompt,
                'strength': strength,
                'steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'device': 'hf_api',
                'optimized': True,
                'api_mode': 'huggingface_remote'
            }
            
            return result, metadata
            
        except Exception as e:
            raise Exception(f"Error en style transfer: {str(e)}")
    
    def object_removal(self, image: Image.Image, mask: Image.Image,
                      context_prompt: str, **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """Eliminar objetos específicos con detección inteligente - EXCLUSIVAMENTE API REMOTA"""
        try:
            # Para eliminación de objetos usamos el modelo img2img vía API remota
            model_name = self.model_configs['img2img']['model_name']
            
            # Parámetros optimizados para eliminación inteligente
            num_inference_steps = kwargs.get('num_inference_steps', 45)
            guidance_scale = kwargs.get('guidance_scale', 9.0)
            
            # Prompts mejorados para eliminación inteligente
            enhanced_prompt = f"remove object and fill with {context_prompt}, seamless natural background"
            
            # Codificar imágenes a base64
            buf_img = io.BytesIO()
            image.save(buf_img, format='PNG')
            img_b64 = base64.b64encode(buf_img.getvalue()).decode('utf-8')

            buf_mask = io.BytesIO()
            mask.save(buf_mask, format='PNG')
            mask_b64 = base64.b64encode(buf_mask.getvalue()).decode('utf-8')

            payload = {
                'inputs': {
                    'prompt': enhanced_prompt,
                    'image': img_b64,
                    'mask_image': mask_b64
                },
                'parameters': {
                    'num_inference_steps': num_inference_steps,
                    'guidance_scale': guidance_scale
                }
            }

            resp = self._call_hf_api(model_name, payload)
            result = self._parse_hf_image_response(resp)
            
            metadata = {
                'method': 'object_removal',
                'context_prompt': context_prompt,
                'enhanced_prompt': enhanced_prompt,
                'steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'device': 'hf_api',
                'optimized': True,
                'intelligent_detection': True,
                'api_mode': 'huggingface_remote'
            }
            
            return result, metadata
            
        except Exception as e:
            raise Exception(f"Error en object removal: {str(e)}")
    
    def _create_intelligent_mask(self, image: Image.Image, object_description: str,
                               context_prompt: str) -> Image.Image:
        """Crear máscara inteligente basada en descripción del objeto"""
        width, height = image.size
        
        # Palabras clave para diferentes tipos de objetos
        keywords = {
            'person': ['person', 'people', 'man', 'woman', 'boy', 'girl', 'face', 'body', 'head'],
            'vehicle': ['car', 'truck', 'bus', 'motorcycle', 'bike', 'vehicle', 'automobile'],
            'animal': ['dog', 'cat', 'bird', 'horse', 'cow', 'animal', 'pet'],
            'building': ['house', 'building', 'structure', 'tower', 'wall', 'fence'],
            'object': ['object', 'item', 'thing', 'sign', 'banner', 'post', 'pole'],
            'tree': ['tree', 'plant', 'bush', 'vegetation', 'branch'],
            'furniture': ['chair', 'table', 'bench', 'seat', 'furniture']
        }
        
        # Detectar tipo de objeto
        detected_type = 'object'
        description_lower = object_description.lower()
        
        for obj_type, type_keywords in keywords.items():
            if any(keyword in description_lower for keyword in type_keywords):
                detected_type = obj_type
                break
        
        # Crear máscara según el tipo detectado
        mask_array = np.zeros((height, width), dtype=np.uint8)
        
        if detected_type == 'person':
            # Máscara centrada verticalmente (para personas)
            center_x, center_y = width // 3, height // 2
            rect_width, rect_height = width // 3, height // 2
            mask_array[center_y-rect_height//2:center_y+rect_height//2,
                      center_x-rect_width//2:center_x+rect_width//2] = 255
            
        elif detected_type == 'vehicle':
            # Máscara en la parte inferior (para vehículos)
            rect_width, rect_height = width // 2, height // 4
            center_x, center_y = width // 2, int(height * 0.8)
            mask_array[center_y-rect_height//2:center_y+rect_height//2,
                      center_x-rect_width//2:center_x+rect_width//2] = 255
            
        elif detected_type == 'building':
            # Máscara amplia (para edificios)
            rect_width, rect_height = width // 2, height
            center_x, center_y = width // 3, height // 2
            mask_array[center_y-rect_height//2:center_y+rect_height//2,
                      center_x-rect_width//2:center_x+rect_width//2] = 255
            
        else:
            # Máscara genérica en el centro-izquierda
            rect_width, rect_height = width // 3, height // 3
            center_x, center_y = width // 3, height // 2
            mask_array[center_y-rect_height//2:center_y+rect_height//2,
                      center_x-rect_width//2:center_x+rect_width//2] = 255
        
        return Image.fromarray(mask_array)
    
    def background_replacement(self, image: Image.Image, mask: Image.Image,
                             background_prompt: str, **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """Reemplazar fondo manteniendo sujeto principal - EXCLUSIVAMENTE API REMOTA"""
        try:
            # Para reemplazo de fondo usamos el modelo img2img vía API remota
            model_name = self.model_configs['img2img']['model_name']
            
            num_inference_steps = kwargs.get('num_inference_steps', 45)
            guidance_scale = kwargs.get('guidance_scale', 8.5)
            
            # Prompts específicos para reemplazo de fondo
            enhanced_prompt = f"replace background with {background_prompt}, keep subject unchanged, professional photography"
            
            # Codificar imágenes a base64
            buf_img = io.BytesIO()
            image.save(buf_img, format='PNG')
            img_b64 = base64.b64encode(buf_img.getvalue()).decode('utf-8')

            buf_mask = io.BytesIO()
            mask.save(buf_mask, format='PNG')
            mask_b64 = base64.b64encode(buf_mask.getvalue()).decode('utf-8')

            payload = {
                'inputs': {
                    'prompt': enhanced_prompt,
                    'image': img_b64,
                    'mask_image': mask_b64
                },
                'parameters': {
                    'num_inference_steps': num_inference_steps,
                    'guidance_scale': guidance_scale
                }
            }

            resp = self._call_hf_api(model_name, payload)
            result = self._parse_hf_image_response(resp)
            
            metadata = {
                'method': 'background_replacement',
                'background_prompt': background_prompt,
                'enhanced_prompt': enhanced_prompt,
                'steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'device': 'hf_api',
                'optimized': True,
                'api_mode': 'huggingface_remote'
            }
            
            return result, metadata
            
        except Exception as e:
            raise Exception(f"Error en background replacement: {str(e)}")
    
    def intelligent_composition(self, image: Image.Image, elements_prompt: str,
                              **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """Composición inteligente de elementos - EXCLUSIVAMENTE API REMOTA"""
        try:
            model_name = self.model_configs['img2img']['model_name']
            
            strength = kwargs.get('strength', 0.4)
            num_inference_steps = kwargs.get('num_inference_steps', 25)
            guidance_scale = kwargs.get('guidance_scale', 7.0)
            
            # Prompts específicos para composición inteligente
            enhanced_prompt = f"{elements_prompt}, harmonious blend, professional composition, high quality"
            
            # Codificar imagen a base64
            buf_img = io.BytesIO()
            image.save(buf_img, format='PNG')
            img_b64 = base64.b64encode(buf_img.getvalue()).decode('utf-8')

            payload = {
                'inputs': {
                    'prompt': enhanced_prompt,
                    'image': img_b64
                },
                'parameters': {
                    'strength': strength,
                    'num_inference_steps': num_inference_steps,
                    'guidance_scale': guidance_scale
                }
            }

            resp = self._call_hf_api(model_name, payload)
            result = self._parse_hf_image_response(resp)
            
            metadata = {
                'method': 'intelligent_composition',
                'prompt': elements_prompt,
                'enhanced_prompt': enhanced_prompt,
                'strength': strength,
                'steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'device': 'hf_api',
                'optimized': True,
                'api_mode': 'huggingface_remote'
            }
            
            return result, metadata
            
        except Exception as e:
            raise Exception(f"Error en intelligent composition: {str(e)}")
    
    def process(self, image: Image.Image, method: str, **kwargs) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """Método principal de procesamiento - EXCLUSIVAMENTE API REMOTA"""
        try:
            start_time = time.time()
            
            # Optimización: Redimensionar imagen para mejorar velocidad
            original_size = image.size
            image = self._optimize_image_size(image)
            optimized_size = image.size
            
            print(f"Iniciando procesamiento con método: {method}")
            print(f"Tamaño original: {original_size}, Optimizado: {optimized_size}")
            
            # Remover parámetros que no son argumentos del método específico
            filtered_kwargs = {k: v for k, v in kwargs.items()
                             if k not in ['prompt', 'style_prompt', 'context_prompt', 'background_prompt', 'elements_prompt']}
            
            # Procesar según método (cada uno usa exclusivamente API remota)
            if method == "inpainting":
                raise Exception("Inpainting no está disponible. Usa 'Outpainting' o 'Object Removal' como alternativas.")
                
            elif method == "outpainting":
                extension_factor = kwargs.get('extension_factor', 1.5)
                prompt = kwargs.get('prompt', 'extended natural landscape seamlessly')
                # Filtrar extension_factor para evitar conflicto
                filtered_outpaint_kwargs = {k: v for k, v in filtered_kwargs.items()
                                          if k not in ['extension_factor', 'prompt']}
                result, metadata = self.outpainting(image, extension_factor, prompt, **filtered_outpaint_kwargs)
                
            elif method == "style_transfer":
                style_prompt = kwargs.get('style_prompt', 'artistic style painting')
                result, metadata = self.style_transfer(image, style_prompt, **filtered_kwargs)
                
            elif method == "object_removal":
                # Detección inteligente del objeto
                object_description = kwargs.get('object_description', 'unwanted object')
                context_prompt = kwargs.get('context_prompt', 'natural seamless background')
                mask = self._create_intelligent_mask(image, object_description, context_prompt)
                result, metadata = self.object_removal(image, mask, context_prompt, **filtered_kwargs)
                
            elif method == "background_replacement":
                mask = self._create_optimized_mask(image, kwargs, "background_replacement")
                background_prompt = kwargs.get('background_prompt', 'beautiful background')
                result, metadata = self.background_replacement(image, mask, background_prompt, **filtered_kwargs)
                
            elif method == "intelligent_composition":
                elements_prompt = kwargs.get('elements_prompt', 'harmonious composition')
                result, metadata = self.intelligent_composition(image, elements_prompt, **filtered_kwargs)
                
            else:
                raise ValueError(f"Método no soportado: {method}")
            
            # Añadir métricas de optimización
            end_time = time.time()
            processing_time = end_time - start_time
            
            metadata.update({
                'processing_time': f"{processing_time:.2f}s",
                'original_size': original_size,
                'optimized_size': optimized_size,
                'memory_optimized': original_size != optimized_size,
                'api_mode': 'huggingface_remote'
            })
            
            print(f"Procesamiento completado en {processing_time:.2f}s")
            return result, metadata
                
        except Exception as e:
            print(f"Error procesando imagen: {str(e)}")
            return None, {"error": str(e)}
    
    def _create_optimized_mask(self, image: Image.Image, kwargs: Dict, method_type: str) -> Image.Image:
        """Crear máscara optimizada según el tamaño de la imagen actual"""
        width, height = image.size
        mask = kwargs.get('mask')
        
        if mask is None:
            if 'mask_coords' in kwargs:
                x1, y1, x2, y2 = kwargs['mask_coords']
                # Escalar coordenadas según el nuevo tamaño
                scale_x = width / 512
                scale_y = height / 512
                mask = self.create_rectangular_mask(
                    width, height, 
                    int(x1 * scale_x), int(y1 * scale_y),
                    int((x2-x1) * scale_x), int((y2-y1) * scale_y)
                )
            else:
                # Crear máscara por defecto según el método
                if method_type == "outpainting":
                    mask_array = np.zeros((height, width), dtype=np.uint8)
                    # Extender proporcionalmente
                    border_size = min(width, height) // 5
                    mask_array[:border_size, :] = 255
                    mask_array[-border_size:, :] = 255
                    mask_array[:, :border_size] = 255
                    mask_array[:, -border_size:] = 255
                    mask = Image.fromarray(mask_array)
                elif method_type == "object_removal":
                    mask = self.create_rectangular_mask(width, height,
                        width//2, height//2, width//8, height//8)
                elif method_type == "background_replacement":
                    mask_array = np.ones((height, width), dtype=np.uint8) * 255
                    # Círculo central para el sujeto
                    y, x = np.ogrid[:height, :width]
                    center_mask = (x - width//2)**2 + (y - height//2)**2 <= (min(width, height)//3)**2
                    mask_array[center_mask] = 0
                    mask = Image.fromarray(mask_array)
                else:
                    # Máscara genérica para otros casos
                    mask = self.create_rectangular_mask(width, height,
                        width//3, height//3, width//6, height//6)
        
        return mask
    
    def get_info(self) -> Dict[str, Any]:
        """Obtener información sobre el procesador"""
        return {
            'device': 'huggingface_api',
            'api_mode': 'remote_only',
            'models_available': list(self.model_configs.keys()),
            'optimized_for_streamlit': True,
            'memory_usage': 'minimal',
            'gpu_required': False
        }
