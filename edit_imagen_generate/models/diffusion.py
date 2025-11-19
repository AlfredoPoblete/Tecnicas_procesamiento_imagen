"""
Módulo de Procesamiento con HuggingFace Inference API - VERSIÓN CLOUD OPTIMIZADA
Diseñado para funcionar en Streamlit Cloud sin modelos locales pesados
Actualizado para usar el nuevo HuggingFace Inference Router
"""

import requests
import numpy as np
import time
import os
import base64
from PIL import Image
from typing import Optional, Tuple, Dict, Any
import io
import warnings
warnings.filterwarnings("ignore")

class DiffusionProcessor:
    """Procesador usando HuggingFace Inference Router - Sin modelos locales"""
    
    def __init__(self):
        """Inicializar procesador con API de HuggingFace"""
        # Detectar API key con prioridad: HUGGINGFACE_API_TOKEN > HF_API_TOKEN > HUGGINGFACE_API_KEY
        self.api_key = (
            os.getenv('HUGGINGFACE_API_TOKEN') or 
            os.getenv('HF_API_TOKEN') or 
            os.getenv('HUGGINGFACE_API_KEY') or 
            ''
        )
        
        # Nuevo endpoint de HuggingFace Inference Router
        self.base_url = 'https://router.huggingface.co/hf-inference/models'
        
        # Modelos disponibles
        self.models = {
            'inpainting': 'stabilityai/stable-diffusion-2-inpainting',
            'img2img': 'runwayml/stable-diffusion-v1-5',
            'style_transfer': 'runwayml/stable-diffusion-v1-5'
        }
        
        self.device = "cloud"  # Indicar que se ejecuta en la nube
        
        print(f"✅ DiffusionProcessor inicializado - Modo: Cloud API")
        print(f"🌐 Usando HuggingFace Inference Router")
        
        if not self.api_key:
            print("⚠️ ERROR: No se encontró ninguna API key de HuggingFace.")
            print("   Configura una de estas variables de entorno:")
            print("   - HUGGINGFACE_API_TOKEN (recomendada)")
            print("   - HF_API_TOKEN")
            print("   - HUGGINGFACE_API_KEY")
        else:
            print(f"✅ API Key detectada: {self.api_key[:10]}...")
    
    def _get_endpoint(self, model_key: str) -> str:
        """Construir endpoint completo para el modelo"""
        model_name = self.models.get(model_key, self.models['img2img'])
        return f"{self.base_url}/{model_name}"
    
    def _parse_hf_response_to_image(self, response: requests.Response) -> Optional[Image.Image]:
        """Parsear respuesta de HuggingFace a imagen PIL
        
        Maneja 3 formatos posibles:
        1. image/* directamente (bytes)
        2. JSON con base64
        3. JSON con "outputs" o "images"
        """
        try:
            content_type = response.headers.get('Content-Type', '')
            
            # Caso 1: Imagen directa (image/png, image/jpeg, etc.)
            if 'image/' in content_type:
                print(f"✅ Respuesta: imagen directa ({content_type})")
                return Image.open(io.BytesIO(response.content))
            
            # Caso 2 y 3: JSON
            elif 'application/json' in content_type:
                data = response.json()
                print(f"✅ Respuesta: JSON")
                
                # Caso 2a: Base64 directo
                if isinstance(data, str):
                    img_data = base64.b64decode(data)
                    return Image.open(io.BytesIO(img_data))
                
                # Caso 2b: JSON con campo "image" o "output"
                if isinstance(data, dict):
                    # Buscar campo de imagen
                    for key in ['image', 'output', 'generated_image', 'result']:
                        if key in data:
                            img_str = data[key]
                            img_data = base64.b64decode(img_str)
                            return Image.open(io.BytesIO(img_data))
                    
                    # Caso 3: Array de outputs
                    if 'outputs' in data or 'images' in data:
                        outputs = data.get('outputs') or data.get('images')
                        if outputs and len(outputs) > 0:
                            img_str = outputs[0]
                            if isinstance(img_str, str):
                                img_data = base64.b64decode(img_str)
                                return Image.open(io.BytesIO(img_data))
                
                print(f"⚠️ Formato JSON no reconocido: {list(data.keys())}")
                return None
            
            # Caso fallback: intentar como imagen directa
            else:
                print(f"⚠️ Content-Type desconocido: {content_type}, intentando como imagen...")
                return Image.open(io.BytesIO(response.content))
                
        except Exception as e:
            print(f"❌ Error parseando respuesta: {str(e)}")
            return None
    
    def _call_huggingface_api(self, model_key: str, image: Image.Image, 
                             prompt: str, mask: Optional[Image.Image] = None,
                             **params) -> Optional[Image.Image]:
        """Llamar a la API de HuggingFace Inference Router
        
        Formato correcto según documentación:
        - Headers: Authorization: Bearer <token>
        - Files: multipart/form-data con "file"
        - Data: JSON con "inputs" y "parameters"
        """
        try:
            # Construir endpoint
            endpoint = self._get_endpoint(model_key)
            
            # Headers
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Convertir imagen a bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # Preparar files (multipart/form-data)
            files = {
                "file": ("image.png", img_byte_arr, "image/png")
            }
            
            # Si hay máscara, incluirla
            if mask is not None:
                mask_byte_arr = io.BytesIO()
                mask.save(mask_byte_arr, format='PNG')
                mask_byte_arr.seek(0)
                files["mask"] = ("mask.png", mask_byte_arr, "image/png")
            
            # Preparar data (JSON)
            data = {
                "inputs": prompt,
                "parameters": {
                    "num_inference_steps": params.get('num_inference_steps', 25),
                    "guidance_scale": params.get('guidance_scale', 7.5)
                }
            }
            
            # Agregar parámetros adicionales si existen
            if 'strength' in params:
                data["parameters"]["strength"] = params['strength']
            
            # Realizar petición POST
            print(f"🌐 Enviando petición a: {endpoint}")
            print(f"📝 Prompt: {prompt[:50]}...")
            
            response = requests.post(
                endpoint,
                headers=headers,
                files=files,
                json=data,
                timeout=120  # 2 minutos de timeout
            )
            
            # Manejar respuestas
            if response.status_code == 200:
                result_image = self._parse_hf_response_to_image(response)
                if result_image:
                    print(f"✅ Imagen generada exitosamente")
                    return result_image
                else:
                    print(f"❌ No se pudo parsear la respuesta")
                    return None
                    
            elif response.status_code == 503:
                print(f"⏳ Modelo cargándose en HuggingFace, reintentando en 20s...")
                time.sleep(20)
                
                # Reintentar una vez
                response = requests.post(
                    endpoint, 
                    headers=headers, 
                    files=files, 
                    json=data, 
                    timeout=120
                )
                
                if response.status_code == 200:
                    result_image = self._parse_hf_response_to_image(response)
                    return result_image
                else:
                    print(f"❌ Error después de reintentar: {response.status_code}")
                    print(f"Respuesta: {response.text}")
                    return None
                    
            elif response.status_code == 410:
                print(f"❌ Error 410: Endpoint obsoleto")
                print(f"   El endpoint antiguo ya no está soportado")
                print(f"   Usando nuevo router: {endpoint}")
                return None
                
            elif response.status_code == 404:
                print(f"❌ Error 404: Modelo no encontrado")
                print(f"   Modelo: {self.models.get(model_key)}")
                return None
                
            elif response.status_code == 500:
                print(f"❌ Error 500: Error interno del servidor")
                print(f"   Intenta de nuevo en unos minutos")
                return None
                
            else:
                print(f"❌ Error en API: {response.status_code}")
                print(f"Respuesta: {response.text[:200]}")
                return None
                
        except Exception as e:
            print(f"❌ Error llamando a HuggingFace API: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _optimize_image_size(self, image: Image.Image, max_size: int = 512) -> Image.Image:
        """Redimensionar imagen para optimizar velocidad y costos de API"""
        width, height = image.size
        
        # Si la imagen es muy grande, redimensionar
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int((height * max_size) / width)
            else:
                new_height = max_size
                new_width = int((width * max_size) / height)
            
            # Asegurar que las dimensiones sean múltiplos de 8 (requerimiento de Stable Diffusion)
            new_width = (new_width // 8) * 8
            new_height = (new_height // 8) * 8
            
            print(f"📐 Redimensionando: {width}x{height} → {new_width}x{new_height}")
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
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
        """Realizar inpainting usando HuggingFace API"""
        try:
            # Optimizar tamaño de imagen
            image = self._optimize_image_size(image)
            mask = mask.resize(image.size, Image.Resampling.LANCZOS)
            
            # Parámetros
            num_inference_steps = kwargs.get('num_inference_steps', 25)
            guidance_scale = kwargs.get('guidance_scale', 7.0)
            
            # Llamar a la API
            result = self._call_huggingface_api(
                'inpainting',
                image,
                prompt,
                mask=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            
            if result is None:
                raise Exception("No se pudo generar la imagen con la API")
            
            metadata = {
                'method': 'inpainting',
                'prompt': prompt,
                'steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'device': 'cloud',
                'api': 'huggingface-router'
            }
            
            return result, metadata
            
        except Exception as e:
            raise Exception(f"Error en inpainting: {str(e)}")
    
    def outpainting(self, image: Image.Image, extension_factor: float = 1.5,
                   prompt: str = "extended natural landscape", **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """Realizar outpainting usando inpainting API"""
        try:
            # Optimizar imagen original
            image = self._optimize_image_size(image)
            original_width, original_height = image.size
            
            # Calcular nuevas dimensiones (múltiplos de 8)
            new_width = int(original_width * extension_factor)
            new_height = int(original_height * extension_factor)
            new_width = (new_width // 8) * 8
            new_height = (new_height // 8) * 8
            
            print(f"🖼️ Outpainting: {original_width}x{original_height} → {new_width}x{new_height}")
            
            # Crear canvas extendido
            extended_canvas = Image.new('RGB', (new_width, new_height), (128, 128, 128))
            
            # Centrar imagen original
            x_offset = (new_width - original_width) // 2
            y_offset = (new_height - original_height) // 2
            extended_canvas.paste(image, (x_offset, y_offset))
            
            # Crear máscara para bordes
            mask_array = np.zeros((new_height, new_width), dtype=np.uint8)
            mask_array[:y_offset, :] = 255
            mask_array[y_offset + original_height:, :] = 255
            mask_array[y_offset:y_offset + original_height, :x_offset] = 255
            mask_array[y_offset:y_offset + original_height, x_offset + original_width:] = 255
            mask = Image.fromarray(mask_array)
            
            # Llamar a la API
            num_inference_steps = kwargs.get('num_inference_steps', 30)
            guidance_scale = kwargs.get('guidance_scale', 8.0)
            
            result = self._call_huggingface_api(
                'inpainting',
                extended_canvas,
                prompt,
                mask=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            
            if result is None:
                raise Exception("No se pudo generar la imagen con la API")
            
            metadata = {
                'method': 'outpainting',
                'prompt': prompt,
                'steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'device': 'cloud',
                'api': 'huggingface-router',
                'original_size': (original_width, original_height),
                'extended_size': (new_width, new_height)
            }
            
            return result, metadata
            
        except Exception as e:
            raise Exception(f"Error en outpainting: {str(e)}")
    
    def style_transfer(self, image: Image.Image, style_prompt: str,
                      **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """Transferir estilo usando img2img API"""
        try:
            # Optimizar imagen
            image = self._optimize_image_size(image)
            
            strength = kwargs.get('strength', 0.5)
            num_inference_steps = kwargs.get('num_inference_steps', 20)
            guidance_scale = kwargs.get('guidance_scale', 7.0)
            
            # Para style transfer, usar el endpoint de img2img
            result = self._call_huggingface_api(
                'style_transfer',
                image,
                style_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength
            )
            
            if result is None:
                raise Exception("No se pudo generar la imagen con la API")
            
            metadata = {
                'method': 'style_transfer',
                'prompt': style_prompt,
                'strength': strength,
                'steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'device': 'cloud',
                'api': 'huggingface-router'
            }
            
            return result, metadata
            
        except Exception as e:
            raise Exception(f"Error en style transfer: {str(e)}")
    
    def object_removal(self, image: Image.Image, mask: Image.Image,
                      context_prompt: str, **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """Eliminar objetos usando inpainting API"""
        try:
            # Optimizar imagen y máscara
            image = self._optimize_image_size(image)
            mask = mask.resize(image.size, Image.Resampling.LANCZOS)
            
            num_inference_steps = kwargs.get('num_inference_steps', 30)
            guidance_scale = kwargs.get('guidance_scale', 9.0)
            
            # Prompt mejorado para eliminación
            enhanced_prompt = f"remove object and fill with {context_prompt}, seamless natural background"
            
            result = self._call_huggingface_api(
                'inpainting',
                image,
                enhanced_prompt,
                mask=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            
            if result is None:
                raise Exception("No se pudo generar la imagen con la API")
            
            metadata = {
                'method': 'object_removal',
                'context_prompt': context_prompt,
                'enhanced_prompt': enhanced_prompt,
                'steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'device': 'cloud',
                'api': 'huggingface-router'
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
            center_x, center_y = width // 3, height // 2
            rect_width, rect_height = width // 3, height // 2
            mask_array[center_y-rect_height//2:center_y+rect_height//2,
                      center_x-rect_width//2:center_x+rect_width//2] = 255
        elif detected_type == 'vehicle':
            rect_width, rect_height = width // 2, height // 4
            center_x, center_y = width // 2, int(height * 0.8)
            mask_array[center_y-rect_height//2:center_y+rect_height//2,
                      center_x-rect_width//2:center_x+rect_width//2] = 255
        elif detected_type == 'building':
            rect_width, rect_height = width // 2, height
            center_x, center_y = width // 3, height // 2
            mask_array[center_y-rect_height//2:center_y+rect_height//2,
                      center_x-rect_width//2:center_x+rect_width//2] = 255
        else:
            rect_width, rect_height = width // 3, height // 3
            center_x, center_y = width // 3, height // 2
            mask_array[center_y-rect_height//2:center_y+rect_height//2,
                      center_x-rect_width//2:center_x+rect_width//2] = 255
        
        return Image.fromarray(mask_array)
    
    def background_replacement(self, image: Image.Image, mask: Image.Image,
                             background_prompt: str, **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """Reemplazar fondo usando inpainting API"""
        try:
            return self.inpainting(image, mask, background_prompt, **kwargs)
        except Exception as e:
            raise Exception(f"Error en background replacement: {str(e)}")
    
    def intelligent_composition(self, image: Image.Image, elements_prompt: str,
                              **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """Composición inteligente usando img2img API"""
        try:
            # Optimizar imagen
            image = self._optimize_image_size(image)
            
            strength = kwargs.get('strength', 0.4)
            num_inference_steps = kwargs.get('num_inference_steps', 25)
            guidance_scale = kwargs.get('guidance_scale', 7.0)
            
            result = self._call_huggingface_api(
                'img2img',
                image,
                elements_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength
            )
            
            if result is None:
                raise Exception("No se pudo generar la imagen con la API")
            
            metadata = {
                'method': 'intelligent_composition',
                'prompt': elements_prompt,
                'strength': strength,
                'steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'device': 'cloud',
                'api': 'huggingface-router'
            }
            
            return result, metadata
            
        except Exception as e:
            raise Exception(f"Error en intelligent composition: {str(e)}")
    
    def process(self, image: Image.Image, method: str, **kwargs) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """Método principal de procesamiento usando APIs"""
        try:
            start_time = time.time()
            
            original_size = image.size
            print(f"🚀 Iniciando procesamiento: {method}")
            print(f"📐 Tamaño original: {original_size}")
            
            # Filtrar kwargs
            filtered_kwargs = {k: v for k, v in kwargs.items()
                             if k not in ['prompt', 'style_prompt', 'context_prompt', 'background_prompt', 'elements_prompt']}
            
            # Procesar según método
            if method == "inpainting":
                mask = self._create_optimized_mask(image, kwargs, "inpainting")
                prompt = kwargs.get('prompt', 'natural background')
                result, metadata = self.inpainting(image, mask, prompt, **filtered_kwargs)
                
            elif method == "outpainting":
                extension_factor = kwargs.get('extension_factor', 1.5)
                prompt = kwargs.get('prompt', 'extended natural landscape seamlessly')
                filtered_outpaint_kwargs = {k: v for k, v in filtered_kwargs.items()
                                          if k not in ['extension_factor', 'prompt']}
                result, metadata = self.outpainting(image, extension_factor, prompt, **filtered_outpaint_kwargs)
                
            elif method == "style_transfer":
                style_prompt = kwargs.get('style_prompt', 'artistic style painting')
                result, metadata = self.style_transfer(image, style_prompt, **filtered_kwargs)
                
            elif method == "object_removal":
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
            
            # Añadir métricas
            end_time = time.time()
            processing_time = end_time - start_time
            
            metadata.update({
                'processing_time': f"{processing_time:.2f}s",
                'original_size': original_size,
                'cloud_processing': True,
                'no_local_models': True
            })
            
            print(f"✅ Procesamiento completado en {processing_time:.2f}s")
            return result, metadata
                
        except Exception as e:
            print(f"❌ Error procesando imagen: {str(e)}")
            return None, {"error": str(e)}
    
    def _create_optimized_mask(self, image: Image.Image, kwargs: Dict, method_type: str) -> Image.Image:
        """Crear máscara optimizada según el método"""
        width, height = image.size
        mask = kwargs.get('mask')
        
        if mask is None:
            if 'mask_coords' in kwargs:
                x1, y1, x2, y2 = kwargs['mask_coords']
                mask = self.create_rectangular_mask(
                    width, height, x1, y1, x2-x1, y2-y1
                )
            else:
                # Máscaras por defecto
                if method_type == "inpainting":
                    mask = self.create_rectangular_mask(width, height, 
                        width//3, height//3, width//6, height//6)
                elif method_type == "outpainting":
                    mask_array = np.zeros((height, width), dtype=np.uint8)
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
                    y, x = np.ogrid[:height, :width]
                    center_mask = (x - width//2)**2 + (y - height//2)**2 <= (min(width, height)//3)**2
                    mask_array[center_mask] = 0
                    mask = Image.fromarray(mask_array)
        
        return mask
    
    def get_info(self) -> Dict[str, Any]:
        """Obtener información del procesador"""
        return {
            'device': 'cloud',
            'api': 'huggingface-router',
            'models_loaded': 'none (cloud-based)',
            'cuda_available': False,
            'local_models': False,
            'api_key_configured': bool(self.api_key)
        }
