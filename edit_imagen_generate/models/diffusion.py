"""
Módulo de Procesamiento con Modelos de Difusión - VERSIÓN OPTIMIZADA SIN INPAINTING
"""

import os
import base64
import io
import time
import requests
import torch
import numpy as np
from PIL import Image, ImageDraw
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale import StableDiffusionUpscalePipeline
from typing import Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings("ignore")

class DiffusionProcessor:
    """Procesador principal para modelos de difusión OPTIMIZADO SIN INPAINTING"""
    
    def __init__(self):
        # Configurar optimizaciones de GPU primero
        self._setup_gpu_optimizations()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipes = {}  # Diccionario vacío - Carga lazy
        # Modo remoto por Hugging Face Inference API (evita descargar pesos pesados)
        # Activar con la variable de entorno `USE_HF_API=true` y variable de token `HUGGINGFACE_API_TOKEN`
        use_hf = os.getenv('USE_HF_API', '').lower() in ['1', 'true', 'yes']
        self.use_hf_api = use_hf
        self.hf_token = os.getenv('HUGGINGFACE_API_TOKEN') or os.getenv('HUGGINGFACEHUB_API_TOKEN') or os.getenv('HF_TOKEN')
        
        # Configuración de modelos para carga lazy (SIN INPAINTING)
        self.model_configs = {
            'img2img': {
                'model_name': 'runwayml/stable-diffusion-v1-5', 
                'pipeline_class': StableDiffusionImg2ImgPipeline
            },
            'upscale': {
                'model_name': 'stabilityai/stable-diffusion-x4-upscaler',
                'pipeline_class': StableDiffusionUpscalePipeline
            }
        }
        
        print(f"DiffusionProcessor inicializado - Dispositivo: {self.device} (SIN INPAINTING)")
        if self.device == 'cuda':
            print("GPU disponible - Optimizaciones activadas")
        else:
            print("CPU solamente - El procesamiento sera mas lento")
        
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

            # Si está configurado para usar la API remota, no descargamos pesos
            if self.use_hf_api:
                if not self.hf_token:
                    raise RuntimeError("USE_HF_API está activado pero no se encontró HUGGINGFACE_API_TOKEN en el entorno")
                print(f"Usando Hugging Face Inference API para: {model_name}")
                # Guardamos una referencia ligera indicando modo remoto
                return {'remote': True, 'model_name': model_name, 'token': self.hf_token}

            # Carga local vía diffusers
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

            print(f"Modelo {model_key} cargado exitosamente (local)")
            return pipe
            
        except Exception as e:
            print(f"Error cargando modelo {model_key}: {str(e)}")
            raise
    
    def _get_model(self, model_key: str):
        """Obtener modelo - cargar si no existe (carga lazy)"""
        if model_key not in self.pipes:
            self.pipes[model_key] = self._load_single_model(model_key)
        return self.pipes[model_key]

    def _call_hf_api(self, model_name: str, payload: dict, timeout: int = 120):
        """Llamar al endpoint de Inference API de Hugging Face.

        Envía JSON con 'inputs' y 'parameters' según lo requiera el modelo.
        Si la respuesta es binaria (imagen), la retornamos como bytes.
        """
        # Usar el router de HF Inference (nuevo endpoint) para mayor compatibilidad
        url_primary = f"https://router.huggingface.co/hf-inference/models/{model_name}"
        url_fallback = f"https://api-inference.huggingface.co/models/{model_name}"
        url = url_primary
        headers = {
            'Authorization': f'Bearer {self.hf_token}',
            'Accept': 'application/json'
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 410:
                # antiguo endpoint deshabilitado, intentar fallback
                resp = requests.post(url_fallback, headers=headers, json=payload, timeout=timeout)
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
    
    def outpainting(self, image: Image.Image, extension_factor: float = 1.5,
                   prompt: str = "extended natural landscape seamlessly", **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """Realizar outpainting (extender imagen) - VERSIÓN OPTIMIZADA"""
        try:
            # Usar img2img como fallback para outpainting cuando inpainting fue removido
            pipe = self._get_model('img2img')
            
            # Obtener dimensiones originales
            original_width, original_height = image.size
            
            # Calcular nuevas dimensiones
            new_width = int(original_width * extension_factor)
            new_height = int(original_height * extension_factor)
            
            print(f"🖼️ Outpainting optimizado: {original_width}x{original_height} → {new_width}x{new_height}")
            
            # Crear canvas más grande con fondo negro (para la máscara)
            extended_canvas = Image.new('RGB', (new_width, new_height), (0, 0, 0))
            
            # Calcular posición para centrar la imagen original
            x_offset = (new_width - original_width) // 2
            y_offset = (new_height - original_height) // 2
            
            # Pegar imagen original en el centro del canvas
            extended_canvas.paste(image, (x_offset, y_offset))
            
            # Parámetros optimizados para outpainting usando img2img
            num_inference_steps = kwargs.get('num_inference_steps', 50)
            guidance_scale = kwargs.get('guidance_scale', 8.5)
            strength = kwargs.get('strength', 0.7)

            # Aplicar img2img para generar extensión del canvas
            if isinstance(pipe, dict) and pipe.get('remote'):
                buf_img = io.BytesIO()
                extended_canvas.save(buf_img, format='PNG')
                img_b64 = base64.b64encode(buf_img.getvalue()).decode('utf-8')

                payload = {
                    'inputs': {
                        'prompt': prompt,
                        'image': img_b64
                    },
                    'parameters': {
                        'strength': strength,
                        'num_inference_steps': num_inference_steps,
                        'guidance_scale': guidance_scale
                    }
                }

                resp = self._call_hf_api(pipe['model_name'], payload)
                result = self._parse_hf_image_response(resp)
            else:
                result = pipe(
                    prompt=prompt,
                    image=extended_canvas,
                    strength=strength,
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
        """Transferir estilo artístico - VERSIÓN OPTIMIZADA"""
        try:
            pipe = self._get_model('img2img')  # Carga lazy
            
            # Parámetros optimizados para style transfer
            strength = kwargs.get('strength', 0.6)
            num_inference_steps = kwargs.get('num_inference_steps', 35)  # Aumentado para mejor calidad
            guidance_scale = kwargs.get('guidance_scale', 7.5)
            
            if isinstance(pipe, dict) and pipe.get('remote'):
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

                resp = self._call_hf_api(pipe['model_name'], payload)
                result = self._parse_hf_image_response(resp)
            else:
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
    
    # Object removal / intelligent mask helpers were removed per requirement
    
    def background_replacement(self, image: Image.Image, mask: Image.Image,
                             background_prompt: str, **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """Reemplazar fondo manteniendo sujeto principal - VERSIÓN OPTIMIZADA"""
        try:
            # Parámetros optimizados para background replacement
            num_inference_steps = kwargs.get('num_inference_steps', 45)
            guidance_scale = kwargs.get('guidance_scale', 8.5)
            
            # Enhanced prompt for better background replacement
            enhanced_prompt = f"replace background with {background_prompt}, maintain subject, high quality seamless integration"
            # Usar img2img como aproximación para reemplazo de fondo (sin inpainting)
            pipe = self._get_model('img2img')
            strength = kwargs.get('strength', 0.7)

            if isinstance(pipe, dict) and pipe.get('remote'):
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

                resp = self._call_hf_api(pipe['model_name'], payload)
                result = self._parse_hf_image_response(resp)
            else:
                result = pipe(
                    prompt=enhanced_prompt,
                    image=image,
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                ).images[0]
            
            # Crear metadata para el resultado
            metadata = {
                'method': 'background_replacement',
                'background_prompt': background_prompt,
                'enhanced_prompt': enhanced_prompt,
                'steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'device': self.device,
                'optimized': True
            }

            return result, metadata
            
        except Exception as e:
            raise Exception(f"Error en background replacement: {str(e)}")
    
    def intelligent_composition(self, image: Image.Image, elements_prompt: str,
                              **kwargs) -> Tuple[Image.Image, Dict[str, Any]]:
        """Composición inteligente de elementos - VERSIÓN OPTIMIZADA"""
        try:
            pipe = self._get_model('img2img')  # Carga lazy
            
            # Parámetros optimizados para composición
            strength = kwargs.get('strength', 0.5)
            num_inference_steps = kwargs.get('num_inference_steps', 40)  # Aumentado para mejor calidad
            guidance_scale = kwargs.get('guidance_scale', 8.0)
            
            if isinstance(pipe, dict) and pipe.get('remote'):
                buf_img = io.BytesIO()
                image.save(buf_img, format='PNG')
                img_b64 = base64.b64encode(buf_img.getvalue()).decode('utf-8')

                payload = {
                    'inputs': {
                        'prompt': elements_prompt,
                        'image': img_b64
                    },
                    'parameters': {
                        'strength': strength,
                        'num_inference_steps': num_inference_steps,
                        'guidance_scale': guidance_scale
                    }
                }

                resp = self._call_hf_api(pipe['model_name'], payload)
                result = self._parse_hf_image_response(resp)
            else:
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
        """Método principal de procesamiento con optimizaciones (SIN INPAINTING) - VERSIÓN CORREGIDA"""
        original_size = None
        optimized_size = None
        
        try:
            start_time = time.time()
            
            # Validar entrada
            if image is None:
                raise ValueError("La imagen no puede ser None")
            
            if not method or method.strip() == "":
                raise ValueError("El método de procesamiento no puede estar vacío")
            
            # Optimización: Redimensionar imagen para mejorar velocidad
            original_size = image.size
            image = self._optimize_image_size(image)
            optimized_size = image.size
            
            print(f"🚀 Iniciando procesamiento con método: {method} (SIN INPAINTING)")
            print(f"📐 Tamaño original: {original_size}, Optimizado: {optimized_size}")
            
            # Validar método antes de continuar
            supported_methods = ["outpainting", "style_transfer", "background_replacement", "intelligent_composition"]
            if method not in supported_methods:
                raise ValueError(f"Método no soportado: {method}. Métodos disponibles: {supported_methods}")
            
            # Remover parámetros que no son argumentos del método específico
            filtered_kwargs = {k: v for k, v in kwargs.items()
                             if k not in ['prompt', 'style_prompt', 'context_prompt', 'background_prompt', 'elements_prompt']}
            
            # Procesar según método (SIN INPAINTING - cada uno usa carga lazy de modelos)
            result = None
            metadata = {}
            
            if method == "outpainting":
                extension_factor = kwargs.get('extension_factor', 1.5)
                prompt = kwargs.get('prompt', 'extended natural landscape seamlessly')
                # Filtrar extension_factor para evitar conflicto
                filtered_outpaint_kwargs = {k: v for k, v in filtered_kwargs.items()
                                          if k not in ['extension_factor', 'prompt']}
                result, metadata = self.outpainting(image, extension_factor, prompt, **filtered_outpaint_kwargs)
                
            elif method == "style_transfer":
                style_prompt = kwargs.get('style_prompt', 'artistic style painting')
                result, metadata = self.style_transfer(image, style_prompt, **filtered_kwargs)
                
            elif method == "background_replacement":
                # Llamar a background_replacement que usa img2img internamente
                background_prompt = kwargs.get('background_prompt', 'beautiful background')
                result, metadata = self.background_replacement(image, background_prompt, **filtered_kwargs)
                
            elif method == "intelligent_composition":
                elements_prompt = kwargs.get('elements_prompt', 'harmonious composition')
                result, metadata = self.intelligent_composition(image, elements_prompt, **filtered_kwargs)
            
            # Validar que tenemos un resultado válido
            if result is None:
                raise Exception(f"No se generó resultado para el método: {method}")
                
            # Añadir métricas de optimización
            end_time = time.time()
            processing_time = end_time - start_time
            
            metadata.update({
                'processing_time': f"{processing_time:.2f}s",
                'original_size': original_size,
                'optimized_size': optimized_size,
                'memory_optimized': original_size != optimized_size,
                'lazy_loading': True,
                'no_inpainting': True,  # Flag para indicar versión sin inpainting
                'method_processed': method
            })
            
            print(f"✅ Procesamiento completado en {processing_time:.2f}s (SIN INPAINTING)")
            return result, metadata
                
        except Exception as e:
            error_msg = f"Error procesando imagen con método {method}: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Fallback local: generar una versión simple simulada del resultado
            try:
                print("🔄 Intentando fallback local...")
                
                # Usar valores por defecto si no se definieron
                if original_size is None:
                    original_size = (512, 512)
                if optimized_size is None:
                    optimized_size = (512, 512)
                
                # Asegurar que tenemos una imagen válida
                if image is None:
                    image = Image.new('RGB', original_size, (128, 128, 128))
                
                fallback_img, fallback_meta = self._local_fallback(image, method, kwargs)
                fallback_meta.update({
                    'processing_time': '0.00s', 
                    'original_size': original_size, 
                    'optimized_size': optimized_size, 
                    'fallback_used': True,
                    'original_error': str(e),
                    'method_processed': method
                })
                print(f"✅ Fallback completado para método: {method}")
                return fallback_img, fallback_meta
            except Exception as e2:
                print(f"❌ Error en fallback local: {str(e2)}")
                return None, {
                    "error": error_msg, 
                    "fallback_error": str(e2),
                    "method": method,
                    "original_size": original_size or 'unknown',
                    "optimized_size": optimized_size or 'unknown'
                }
    
    def _create_optimized_mask(self, image: Image.Image, kwargs: Dict, method_type: str) -> Image.Image:
        """Crear máscara optimizada según el tamaño de la imagen actual (SIN INPAINTTING)"""
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
                # Crear máscara por defecto según el método (SIN INPAINTING)
                if method_type == "outpainting":
                    mask_array = np.zeros((height, width), dtype=np.uint8)
                    # Extender proporcionalmente
                    border_size = min(width, height) // 5
                    mask_array[:border_size, :] = 255
                    mask_array[-border_size:, :] = 255
                    mask_array[:, :border_size] = 255
                    mask_array[:, -border_size:] = 255
                    mask = Image.fromarray(mask_array)
                
                elif method_type == "background_replacement":
                    mask_array = np.ones((height, width), dtype=np.uint8) * 255
                    # Círculo central para el sujeto
                    y, x = np.ogrid[:height, :width]
                    center_mask = (x - width//2)**2 + (y - height//2)**2 <= (min(width, height)//3)**2
                    mask_array[center_mask] = 0
                    mask = Image.fromarray(mask_array)
        
        return mask

    def _local_fallback(self, image: Image.Image, method: str, kwargs: Dict) -> Tuple[Image.Image, Dict[str, Any]]:
        """Generar una versión simulada del resultado localmente cuando falla la API remota.
        Esto permite pruebas E2E sin dependencias externas pesadas.
        """
        try:
            img = image.copy().convert('RGB')
            from PIL import ImageFilter, ImageOps, ImageEnhance

            if method == 'outpainting':
                # Simplemente crear un canvas más grande y pegar la imagen en el centro
                w, h = img.size
                new_w, new_h = int(w * 1.3), int(h * 1.3)
                canvas = Image.new('RGB', (new_w, new_h), (int(120), int(120), int(120)))
                x = (new_w - w) // 2
                y = (new_h - h) // 2
                canvas.paste(img, (x, y))
                # Suavizar bordes
                canvas = canvas.filter(ImageFilter.GaussianBlur(radius=1))
                meta = {'method': 'outpainting', 'fallback': True}
                return canvas, meta

            if method == 'style_transfer':
                # Aplicar un cambio de color simple
                enhancer = ImageEnhance.Color(img)
                img2 = enhancer.enhance(1.5)
                img2 = img2.filter(ImageFilter.DETAIL)
                meta = {'method': 'style_transfer', 'fallback': True}
                return img2, meta

            if method == 'background_replacement':
                # Reemplazar bordes con color del prompt aproximado
                w, h = img.size
                background = Image.new('RGB', (w, h), (250, 200, 150))
                # Pegar sujeto centrado
                background.paste(img, (0, 0), None)
                meta = {'method': 'background_replacement', 'fallback': True}
                return background, meta

            if method == 'intelligent_composition':
                # Combinar con una versión suavizada y superponer
                overlay = img.copy().filter(ImageFilter.GaussianBlur(radius=3)).convert('RGBA')
                base = img.convert('RGBA')
                overlay.putalpha(100)
                composed = Image.alpha_composite(base, overlay).convert('RGB')
                meta = {'method': 'intelligent_composition', 'fallback': True}
                return composed, meta

            # Default: ligero sharpen
            img_default = img.filter(ImageFilter.SHARPEN)
            return img_default, {'method': method, 'fallback': True}
        except Exception as e:
            raise
    
    def _create_object_removal_mask(self, image: Image.Image, kwargs: Dict) -> Image.Image:
        """Crear máscara específica para eliminación de objetos - método consolidado"""
        object_description = kwargs.get('object_description', 'object')
        context_prompt = kwargs.get('context_prompt', 'natural background')
        # Funcionalidad de object removal/inpainting eliminada; devolver máscara rectangular por defecto
        return self.create_rectangular_mask(image.size[0], image.size[1], image.size[0]//2, image.size[1]//2, image.size[0]//8, image.size[1]//8)
    
    def get_info(self) -> Dict[str, Any]:
        """Obtener información sobre los modelos cargados"""
        return {
            'device': self.device,
            'models_loaded': list(self.pipes.keys()),
            'cuda_available': torch.cuda.is_available(),
            'lazy_loading_enabled': True,
            'gpu_optimizations': self.device == 'cuda',
            'inpainting_disabled': True  # Flag para indicar que inpainting está deshabilitado
        }