"""
Módulo de Procesamiento con Modelos de Difusión - VERSIÓN OPTIMIZADA PARA STREAMLIT
"""

import os
import torch
import numpy as np
import time
import requests
import json
import base64
from io import BytesIO
from PIL import Image, ImageDraw
from typing import Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# Imports de diffusers actualizados
try:
    from diffusers import StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline
except ImportError:
    # Fallback para versiones antiguas
    from diffusers.pipelines.stable_diffusion import StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline

class DiffusionProcessor:
    """Procesador principal para modelos de difusión OPTIMIZADO para Streamlit"""
    
    def __init__(self):
        # Verificar si usar API de Hugging Face
        self.use_hf_api = os.getenv('USE_HF_API', 'false').lower() == 'true'
        self.hf_token = os.getenv('HUGGINGFACE_API_TOKEN')
        
        # Configurar optimizaciones de GPU primero
        self._setup_gpu_optimizations()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipes = {}  # Diccionario vacío - Carga lazy
        
        # Configuración de modelos OPTIMIZADA para Streamlit
        if self.use_hf_api and self.hf_token:
            # Modelos ligeros vía API de Hugging Face
            self.model_configs = {
                'inpainting': {
                    'model_name': 'diffusers/stable-diffusion-inpainting-0.1',
                    'pipeline_class': None,  # Se usará API REST
                    'api_endpoint': f'https://api-inference.huggingface.co/models/diffusers/stable-diffusion-inpainting-0.1'
                },
                'img2img': {
                    'model_name': 'runwayml/stable-diffusion-v1-5',
                    'pipeline_class': StableDiffusionImg2ImgPipeline,
                    'api_endpoint': f'https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5'
                },
                'style_transfer': {
                    'model_name': 'prompthero/openjourney',
                    'pipeline_class': StableDiffusionImg2ImgPipeline,
                    'api_endpoint': f'https://api-inference.huggingface.co/models/prompthero/openjourney'
                }
            }
            print("🚀 Usando API de Hugging Face para modelos ligeros")
        else:
            # Modelos locales optimizados (carga lazy)
            self.model_configs = {
                'inpainting': {
                    'model_name': 'runwayml/stable-diffusion-inpainting',
                    'pipeline_class': StableDiffusionInpaintPipeline,
                    'optimized': True,
                    'variant': 'fp16'  # Usar variante de precisión reducida
                },
                'img2img': {
                    'model_name': 'runwayml/stable-diffusion-v1-5',
                    'pipeline_class': StableDiffusionImg2ImgPipeline,
                    'optimized': True,
                    'variant': 'fp16'
                },
                'style_transfer': {
                    'model_name': 'prompthero/openjourney',
                    'pipeline_class': StableDiffusionImg2ImgPipeline,
                    'optimized': True,
                    'variant': 'fp16'
                }
            }
            print("⚡ Usando modelos locales optimizados")
        
        print(f"DiffusionProcessor inicializado - Dispositivo: {self.device}")
        if self.device == 'cuda':
            print("GPU disponible - Optimizaciones activadas")
        else:
            print("CPU solamente - Modo de bajo consumo activado")
        
    def _setup_gpu_optimizations(self):
        """Configurar optimizaciones de GPU para mayor velocidad"""
        try:
            import torch
            if torch.cuda.is_available():
                # Optimizaciones CUDA
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.set_float32_matmul_precision('high')
                print("Optimizaciones de GPU configuradas")
        except Exception as e:
            print(f"Error configurando optimizaciones GPU: {e}")
        
    def _get_device_dtype(self):
        """Obtener dtype optimizado según el dispositivo"""
        import torch
        if self.device == "cuda":
            return torch.float16  # Más rápido en GPU
        else:
            return torch.float32
        
    def _load_single_model(self, model_key: str):
        """Cargar un modelo específico bajo demanda"""
        if model_key not in self.model_configs:
            raise ValueError(f"Modelo no configurado: {model_key}")
            
        try:
            config = self.model_configs[model_key]
            
            # Si usar API de Hugging Face, no cargar modelo local
            if self.use_hf_api and 'api_endpoint' in config:
                print(f"Modelo {model_key} usará API de Hugging Face")
                return None  # No necesitamos cargar el modelo localmente
            
            model_name = config['model_name']
            pipeline_class = config['pipeline_class']
            
            print(f"Cargando modelo {model_key}: {model_name}")
            
            # Configuración optimizada para modelos locales
            load_kwargs = {
                'torch_dtype': self._get_device_dtype(),
                'safety_checker': None,
                'resume_download': True,
            }
            
            # Usar variante optimizada si está disponible
            if 'variant' in config:
                load_kwargs['variant'] = config['variant']
            
            pipe = pipeline_class.from_pretrained(
                model_name,
                **load_kwargs
            )
            
            # Mover al dispositivo de forma segura
            if hasattr(torch.nn.Module, 'to_empty'):
                try:
                    pipe = pipe.to_empty(device=self.device)
                except:
                    pipe = pipe.to(self.device)
            else:
                pipe = pipe.to(self.device)
                
            print(f"Modelo {model_key} cargado exitosamente")
            return pipe
            
        except Exception as e:
            print(f"Error cargando modelo {model_key}: {str(e)}")
            raise
    
    def _get_model(self, model_key: str):
        """Obtener modelo - cargar si no existe (carga lazy)"""
        if model_key not in self.pipes:
            self.pipes[model_key] = self._load_single_model(model_key)
        return self.pipes[model_key]
    
    def _optimize_image_size(self, image: Image.Image) -> Image.Image:
        """Redimensionar imagen para optimizar velocidad de procesamiento"""
        width, height = image.size
        
        # Si la imagen es muy grande, redimensionar para mejorar velocidad
        if max(width, height) > 256:
            # Mantener aspecto ratio
            if width > height:
                new_width = 256
                new_height = int((height * 256) / width)
            else:
                new_height = 256
                new_width = int((width * 256) / height)
            
            print(f"Redimensionando imagen de {width}x{height} a {new_width}x{new_height} para optimizar velocidad en Streamlit")
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Si la imagen es muy pequeña, mantener para preservar calidad
        elif min(width, height) < 128:
            print(f"Imagen pequena ({width}x{height}) - manteniendo tamano para preservar calidad")
            return image
        
        return image
    
    def _encode_image_for_hf_api(self, image: Image.Image) -> str:
        """Codificar imagen para envío a API de Hugging Face"""
        buffer = BytesIO()
        # Redimensionar primero para reducir tamaño
        image_resized = self._optimize_image_size(image)
        image_resized.save(buffer, format='PNG')
        image_data = buffer.getvalue()
        return base64.b64encode(image_data).decode('utf-8')
    
    def _encode_mask_for_hf_api(self, mask: Image.Image) -> str:
        """Codificar máscara para envío a API de Hugging Face"""
        buffer = BytesIO()
        mask.save(buffer, format='PNG')
        image_data = buffer.getvalue()
        return base64.b64encode(image_data).decode('utf-8')
    
    def _call_hf_api(self, model_key: str, payload: Dict[str, Any]) -> Optional[Image.Image]:
        """Llamar a la API de Hugging Face para procesamiento"""
        if not self.use_hf_api or not self.hf_token:
            return None
            
        try:
            config = self.model_configs.get(model_key, {})
            if 'api_endpoint' not in config:
                return None
                
            endpoint = config['api_endpoint']
            
            # Headers para Hugging Face
            headers = {
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json",
            }
            
            print(f"Llamando API de Hugging Face: {endpoint}")
            
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=120  # Timeout más largo para modelos
            )
            
            if response.status_code == 200:
                # La respuesta es una imagen
                image = Image.open(BytesIO(response.content))
                return image.convert('RGB')
            else:
                print(f"Error en API de Hugging Face: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"Error llamando API de Hugging Face: {str(e)}")
            return None
    
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
        """Realizar inpainting en la imagen con soporte para API de Hugging Face"""
        try:
            # Parámetros por defecto optimizados para Streamlit
            num_inference_steps = kwargs.get('num_inference_steps', 20)  # Más agresivo para velocidad
            guidance_scale = kwargs.get('guidance_scale', 6.5)  # Reducido para velocidad
            
            # Intentar usar API de Hugging Face primero si está configurada
            if self.use_hf_api and self.hf_token:
                config = self.model_configs.get('inpainting', {})
                if 'api_endpoint' in config:
                    print("🚀 Usando API de Hugging Face para inpainting")
                    
                    # Preparar payload para API
                    payload = {
                        "inputs": {
                            "prompt": prompt,
                            "image": self._encode_image_for_hf_api(image),
                            "mask_image": self._encode_mask_for_hf_api(mask)
                        },
                        "parameters": {
                            "num_inference_steps": min(num_inference_steps, 20),  # Límite API
                            "guidance_scale": guidance_scale
                        }
                    }
                    
                    result = self._call_hf_api('inpainting', payload)
                    if result:
                        metadata = {
                            'method': 'inpainting',
                            'prompt': prompt,
                            'steps': num_inference_steps,
                            'guidance_scale': guidance_scale,
                            'device': 'huggingface_api',
                            'optimized': True,
                            'api_processed': True
                        }
                        return result, metadata
            
            # Fallback a modelo local
            pipe = self._get_model('inpainting')  # Carga lazy
            
            result = pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
            
            metadata = {
                'method': 'inpainting',
                'prompt': prompt,
                'steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'device': self.device,
                'optimized': True
            }
            
            return result, metadata
            
        except Exception as e:
            raise Exception(f"Error en inpainting: {str(e)}")
    
    def outpainting(self, image: Image.Image, extension_factor: float = 1.5,
                   prompt: str = "extended natural landscape", **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """Realizar outpainting (extender imagen) - VERSIÓN CORREGIDA"""
        try:
            pipe = self._get_model('inpainting')  # Usa inpainting para outpainting
            
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
            
            # Aplicar inpainting para extender la imagen
            result = pipe(
                prompt=prompt,
                image=extended_canvas,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
            
            metadata = {
                'method': 'outpainting',
                'prompt': prompt,
                'steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'device': self.device,
                'optimized': True,
                'original_size': (original_width, original_height),
                'extended_size': (new_width, new_height),
                'extension_factor': extension_factor,
                'center_offset': (x_offset, y_offset)
            }
            
            print(f"✅ Outpainting completado: {result.size}")
            return result, metadata
            
        except Exception as e:
            raise Exception(f"Error en outpainting: {str(e)}")
    
    def style_transfer(self, image: Image.Image, style_prompt: str,
                      **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """Transferir estilo artístico con soporte para API de Hugging Face"""
        try:
            # Parámetros optimizados para velocidad en Streamlit
            strength = kwargs.get('strength', 0.4)  # Más agresivo para velocidad
            num_inference_steps = kwargs.get('num_inference_steps', 15)  # Muy reducido
            guidance_scale = kwargs.get('guidance_scale', 6.0)  # Reducido para velocidad
            
            # Intentar usar API de Hugging Face primero si está configurada
            if self.use_hf_api and self.hf_token:
                config = self.model_configs.get('style_transfer', {})
                if 'api_endpoint' in config:
                    print("🎨 Usando API de Hugging Face para style transfer")
                    
                    # Preparar payload para API
                    payload = {
                        "inputs": {
                            "prompt": style_prompt,
                            "image": self._encode_image_for_hf_api(image)
                        },
                        "parameters": {
                            "num_inference_steps": min(num_inference_steps, 15),  # Límite API
                            "guidance_scale": guidance_scale,
                            "strength": strength
                        }
                    }
                    
                    result = self._call_hf_api('style_transfer', payload)
                    if result:
                        metadata = {
                            'method': 'style_transfer',
                            'prompt': style_prompt,
                            'strength': strength,
                            'steps': num_inference_steps,
                            'guidance_scale': guidance_scale,
                            'device': 'huggingface_api',
                            'optimized': True,
                            'api_processed': True
                        }
                        return result, metadata
            
            # Fallback a modelo local
            pipe = self._get_model('img2img')  # Carga lazy
            
            result = pipe(
                prompt=style_prompt,
                image=image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
            
            metadata = {
                'method': 'style_transfer',
                'prompt': style_prompt,
                'strength': strength,
                'steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'device': self.device,
                'optimized': True
            }
            
            return result, metadata
            
        except Exception as e:
            raise Exception(f"Error en style transfer: {str(e)}")
    
    def object_removal(self, image: Image.Image, mask: Image.Image,
                      context_prompt: str, **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """Eliminar objetos específicos con detección inteligente"""
        try:
            pipe = self._get_model('inpainting')  # Carga lazy
            
            # Parámetros optimizados para eliminación inteligente
            num_inference_steps = kwargs.get('num_inference_steps', 45)  # Más pasos para mejor calidad
            guidance_scale = kwargs.get('guidance_scale', 9.0)  # Mayor adherencia al contexto
            
            # Prompts mejorados para eliminación inteligente
            enhanced_prompt = f"remove object and fill with {context_prompt}, seamless natural background"
            
            result = pipe(
                prompt=enhanced_prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
            
            metadata = {
                'method': 'object_removal',
                'context_prompt': context_prompt,
                'enhanced_prompt': enhanced_prompt,
                'steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'device': self.device,
                'optimized': True,
                'intelligent_detection': True
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
        """Reemplazar fondo manteniendo sujeto principal"""
        try:
            return self.inpainting(image, mask, background_prompt, **kwargs)
            
        except Exception as e:
            raise Exception(f"Error en background replacement: {str(e)}")
    
    def intelligent_composition(self, image: Image.Image, elements_prompt: str,
                              **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """Composición inteligente de elementos con optimizaciones para Streamlit"""
        try:
            # Parámetros muy optimizados para velocidad
            strength = kwargs.get('strength', 0.3)  # Muy reducido para velocidad
            num_inference_steps = kwargs.get('num_inference_steps', 15)  # Muy reducido
            guidance_scale = kwargs.get('guidance_scale', 6.5)  # Reducido para velocidad
            
            # Intentar usar API de Hugging Face primero si está configurada
            if self.use_hf_api and self.hf_token:
                config = self.model_configs.get('img2img', {})
                if 'api_endpoint' in config:
                    print("🧩 Usando API de Hugging Face para composición inteligente")
                    
                    # Preparar payload para API
                    payload = {
                        "inputs": {
                            "prompt": elements_prompt,
                            "image": self._encode_image_for_hf_api(image)
                        },
                        "parameters": {
                            "num_inference_steps": min(num_inference_steps, 15),  # Límite API
                            "guidance_scale": guidance_scale,
                            "strength": strength
                        }
                    }
                    
                    result = self._call_hf_api('img2img', payload)
                    if result:
                        metadata = {
                            'method': 'intelligent_composition',
                            'prompt': elements_prompt,
                            'strength': strength,
                            'steps': num_inference_steps,
                            'guidance_scale': guidance_scale,
                            'device': 'huggingface_api',
                            'optimized': True,
                            'api_processed': True
                        }
                        return result, metadata
            
            # Fallback a modelo local
            pipe = self._get_model('img2img')  # Carga lazy
            
            result = pipe(
                prompt=elements_prompt,
                image=image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
            
            metadata = {
                'method': 'intelligent_composition',
                'prompt': elements_prompt,
                'strength': strength,
                'steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'device': self.device,
                'optimized': True
            }
            
            return result, metadata
            
        except Exception as e:
            raise Exception(f"Error en intelligent composition: {str(e)}")
    
    def process(self, image: Image.Image, method: str, **kwargs) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """Método principal de procesamiento con optimizaciones"""
        try:
            start_time = time.time()
            
            # Optimización: Redimensionar imagen para mejorar velocidad
            original_size = image.size
            image = self._optimize_image_size(image)
            optimized_size = image.size
            
            print(f"Iniciando procesamiento con metodo: {method}")
            print(f"Tamano original: {original_size}, Optimizado: {optimized_size}")
            
            # Remover parámetros que no son argumentos del método específico
            filtered_kwargs = {k: v for k, v in kwargs.items()
                             if k not in ['prompt', 'style_prompt', 'context_prompt', 'background_prompt', 'elements_prompt']}
            
            # Procesar según método (cada uno usa carga lazy de modelos)
            if method == "inpainting":
                mask = self._create_optimized_mask(image, kwargs, "inpainting")
                prompt = kwargs.get('prompt', 'natural background')
                result, metadata = self.inpainting(image, mask, prompt, **filtered_kwargs)
                
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
                raise ValueError(f"Metodo no soportado: {method}")
            
            # Añadir métricas de optimización
            end_time = time.time()
            processing_time = end_time - start_time
            
            metadata.update({
                'processing_time': f"{processing_time:.2f}s",
                'original_size': original_size,
                'optimized_size': optimized_size,
                'memory_optimized': original_size != optimized_size,
                'lazy_loading': True
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
                if method_type == "inpainting":
                    mask = self.create_rectangular_mask(width, height, 
                        width//3, height//3, width//6, height//6)
                elif method_type == "outpainting":
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
        
        return mask
    
    def _create_object_removal_mask(self, image: Image.Image, kwargs: Dict) -> Image.Image:
        """Crear máscara específica para eliminación de objetos - método consolidado"""
        object_description = kwargs.get('object_description', 'object')
        context_prompt = kwargs.get('context_prompt', 'natural background')
        return self._create_intelligent_mask(image, object_description, context_prompt)
    
    def get_info(self) -> Dict[str, Any]:
        """Obtener información sobre los modelos cargados"""
        return {
            'device': self.device,
            'models_loaded': list(self.pipes.keys()),
            'cuda_available': torch.cuda.is_available(),
            'lazy_loading_enabled': True,
            'gpu_optimizations': self.device == 'cuda'
        }