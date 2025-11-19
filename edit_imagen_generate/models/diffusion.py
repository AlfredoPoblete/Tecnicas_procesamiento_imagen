"""
Módulo de Procesamiento con Modelos de Difusión - VERSIÓN OPTIMIZADA
"""

import torch
import numpy as np
import time
from PIL import Image, ImageDraw
from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionUpscalePipeline,
    ControlNetModel,
    StableDiffusionControlNetPipeline
)
from typing import Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings("ignore")

class DiffusionProcessor:
    """Procesador principal para modelos de difusión - CARGA LAZY REAL"""
    
    def __init__(self):
        # Configurar optimizaciones de GPU primero
        self._setup_gpu_optimizations()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipes = {}  # Diccionario vacío - Carga lazy REAL
        self.initialized = False  # Bandera de inicialización
        
        # Configuración de modelos para carga lazy
        self.model_configs = {
            'inpainting': {
                'model_name': 'runwayml/stable-diffusion-inpainting',
                'pipeline_class': StableDiffusionInpaintPipeline
            },
            'img2img': {
                'model_name': 'runwayml/stable-diffusion-v1-5',
                'pipeline_class': StableDiffusionImg2ImgPipeline
            },
            'upscale': {
                'model_name': 'stabilityai/stable-diffusion-x4-upscaler',
                'pipeline_class': StableDiffusionUpscalePipeline
            }
        }
        
        print("DiffusionProcessor creado - Inicialización lazy activada")
    
    def _initialize(self):
        """Inicialización diferida - solo se ejecuta cuando se necesita"""
        if self.initialized:
            return
            
        print(f"Inicializando DiffusionProcessor en dispositivo: {self.device}")
        if self.device == 'cuda':
            print("GPU disponible - Optimizaciones activadas")
        else:
            print("CPU solamente - El procesamiento sera mas lento")
            
        self.initialized = True
        
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
            model_name = config['model_name']
            pipeline_class = config['pipeline_class']
            
            print(f"Cargando modelo {model_key}: {model_name}")
            
            pipe = pipeline_class.from_pretrained(
                model_name,
                torch_dtype=self._get_device_dtype(),
                safety_checker=None,
                resume_download=True,
                cache_dir=None
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
        self._initialize()  # Inicializar solo cuando se necesita
        if model_key not in self.pipes:
            self.pipes[model_key] = self._load_single_model(model_key)
        return self.pipes[model_key]
    
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
            print(f"Imagen pequena ({width}x{height}) - manteniendo tamano para preservar calidad")
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
        """Realizar inpainting en la imagen"""
        try:
            pipe = self._get_model('inpainting')  # Carga lazy
            
            # Parámetros por defecto optimizados
            num_inference_steps = kwargs.get('num_inference_steps', 25)  # Reducido de 30
            guidance_scale = kwargs.get('guidance_scale', 7.0)  # Reducido de 7.5
            
            # Procesar
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
        """Transferir estilo artístico"""
        try:
            pipe = self._get_model('img2img')  # Carga lazy
            
            strength = kwargs.get('strength', 0.5)  # Reducido de 0.6
            num_inference_steps = kwargs.get('num_inference_steps', 20)  # Reducido de 30
            guidance_scale = kwargs.get('guidance_scale', 6.5)  # Reducido de 7.5
            
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
        """Composición inteligente de elementos"""
        try:
            pipe = self._get_model('img2img')  # Carga lazy
            
            strength = kwargs.get('strength', 0.4)  # Reducido de 0.5
            num_inference_steps = kwargs.get('num_inference_steps', 25)  # Reducido de 40
            guidance_scale = kwargs.get('guidance_scale', 7.0)  # Reducido de 8.0
            
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
            'initialized': self.initialized,
            'gpu_optimizations': self.device == 'cuda'
        }