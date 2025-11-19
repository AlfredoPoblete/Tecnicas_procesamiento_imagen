"""
DiffusionProcessor – Versión TOTALMENTE corregida y funcional
Compatible con Streamlit Cloud + HuggingFace Inference Router
"""

import requests
import numpy as np
import time
import os
import base64
import json
from PIL import Image
from typing import Optional, Tuple, Dict, Any
import io
import warnings
import traceback

# Para logs visibles en Streamlit
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

warnings.filterwarnings("ignore")


class DiffusionProcessor:
    """Procesador usando HUGGINGFACE ROUTER – Sin modelos locales"""

    def __init__(self):
        # Detección de API keys
        self.api_key = (
            os.getenv('HUGGINGFACE_API_TOKEN')
            or os.getenv('HF_API_TOKEN')
            or os.getenv('HUGGINGFACE_API_KEY')
            or ''
        )

        # HuggingFace Inference Router (obligatorio)
        self.base_url = "https://router.huggingface.co/hf-inference/models"

        # Modelos
        self.models = {
            "inpainting": "stabilityai/stable-diffusion-2-inpainting",
            "img2img": "runwayml/stable-diffusion-v1-5",
            "style_transfer": "runwayml/stable-diffusion-v1-5"
        }

        print("✅ DiffusionProcessor inicializado (cloud mode)")
        if not self.api_key:
            print("❌ ERROR: No hay API KEY configurada")
        else:
            print(f"🔑 API Key encontrada: {self.api_key[:8]}...")

    # ------------------------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------------------------
    def _log(self, msg: str):
        print(msg)
        if STREAMLIT_AVAILABLE:
            st.write(msg)

    # ------------------------------------------------------------------------
    # ENDPOINT
    # ------------------------------------------------------------------------
    def _get_endpoint(self, model_key: str):
        model_name = self.models.get(model_key, self.models["img2img"])
        return f"{self.base_url}/{model_name}"

    # ------------------------------------------------------------------------
    # PARSEAR RESPUESTA DE HF
    # ------------------------------------------------------------------------
    def _parse_hf_response_to_image(self, response: requests.Response):
        """Maneja:
        - Imagen directa
        - JSON con base64
        - JSON con images / outputs
        """

        content_type = response.headers.get("Content-Type", "")

        # Caso 1: imagen directa
        if "image/" in content_type:
            return Image.open(io.BytesIO(response.content))

        # Caso 2: JSON
        try:
            data = response.json()
        except:
            return None

        # Base64 en string
        if isinstance(data, str):
            try:
                decoded = base64.b64decode(data)
                return Image.open(io.BytesIO(decoded))
            except:
                return None

        # Diccionario normal
        if isinstance(data, dict):
            # Campos comunes
            if "image" in data:
                decoded = base64.b64decode(data["image"])
                return Image.open(io.BytesIO(decoded))

            if "generated_image" in data:
                decoded = base64.b64decode(data["generated_image"])
                return Image.open(io.BytesIO(decoded))

            # Arrays
            if "images" in data and len(data["images"]) > 0:
                decoded = base64.b64decode(data["images"][0])
                return Image.open(io.BytesIO(decoded))

            if "outputs" in data and len(data["outputs"]) > 0:
                decoded = base64.b64decode(data["outputs"][0])
                return Image.open(io.BytesIO(decoded))

        return None

    # ------------------------------------------------------------------------
    # LLAMAR A HF ROUTER (FORMATO MULTIPART CORRECTO)
    # ------------------------------------------------------------------------
    def _call_huggingface_api(self, model_key: str, image: Image.Image, prompt: str, mask=None, **params):
        endpoint = self._get_endpoint(model_key)

        self._log(f"🌐 Enviando petición a: {endpoint}")
        self._log(f"📝 Prompt: {prompt}")

        if not self.api_key:
            return None

        # Convertir imagen a bytes
        img_buf = io.BytesIO()
        image.save(img_buf, format="PNG")
        img_buf.seek(0)

        files = {
            "file": ("image.png", img_buf, "image/png"),
            # Campo necesario: JSON dentro del multipart
            "data": (None, json.dumps({
                "inputs": prompt,
                "parameters": params
            }), "application/json")
        }

        # Si hay máscara, agregar
        if mask is not None:
            mask_buf = io.BytesIO()
            mask.save(mask_buf, format="PNG")
            mask_buf.seek(0)
            files["mask"] = ("mask.png", mask_buf, "image/png")

        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                files=files,
                timeout=180
            )

            self._log(f"📡 Status: {response.status_code}")

            # Manejar casos
            if response.status_code == 200:
                img = self._parse_hf_response_to_image(response)
                return img

            return None

        except Exception as e:
            self._log(f"❌ Error en request: {e}")
            traceback.print_exc()
            return None

    # ------------------------------------------------------------------------
    # OPTIMIZAR TAMAÑO
    # ------------------------------------------------------------------------
    def _optimize_image_size(self, img: Image.Image, max_size=512):
        w, h = img.size
        if max(w, h) <= max_size:
            return img

        if w > h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)

        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8

        self._log(f"📐 Redimensionando: {w}x{h} → {new_w}x{new_h}")
        return img.resize((new_w, new_h), Image.LANCZOS)

    # ------------------------------------------------------------------------
    # MÉTODOS DE PROCESAMIENTO
    # ------------------------------------------------------------------------
    def inpainting(self, image, mask, prompt, **params):
        img = self._optimize_image_size(image)
        mask = mask.resize(img.size)
        result = self._call_huggingface_api("inpainting", img, prompt, mask, **params)
        return result

    def outpainting(self, image, prompt, extension_factor=1.4, **params):
        img = self._optimize_image_size(image)
        w, h = img.size

        new_w = int(w * extension_factor)
        new_h = int(h * extension_factor)
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8

        canvas = Image.new("RGB", (new_w, new_h), (128, 128, 128))
        x = (new_w - w) // 2
        y = (new_h - h) // 2
        canvas.paste(img, (x, y))

        # Máscara: todo lo que no es la imagen original
        mask = Image.new("L", (new_w, new_h), 255)
        mask.paste(0, (x, y, x+w, y+h))

        result = self._call_huggingface_api("inpainting", canvas, prompt, mask, **params)
        return result

    def style_transfer(self, image, prompt, **params):
        img = self._optimize_image_size(image)
        result = self._call_huggingface_api("style_transfer", img, prompt, None, **params)
        return result

    def object_removal(self, image, prompt, mask, **params):
        img = self._optimize_image_size(image)
        mask = mask.resize(img.size)
        result = self._call_huggingface_api("inpainting", img, prompt, mask, **params)
        return result

    def background_replacement(self, image, prompt, mask, **params):
        return self.object_removal(image, prompt, mask, **params)

    def intelligent_composition(self, image, prompt, **params):
        img = self._optimize_image_size(image)
        result = self._call_huggingface_api("img2img", img, prompt, None, **params)
        return result

    # ------------------------------------------------------------------------
    # MÉTODO PRINCIPAL - YA NO SE ROMPE
    # ------------------------------------------------------------------------
    def process(self, image: Image.Image, method: str, **kwargs):
        self._log(f"🚀 Procesando método: {method}")

        # prompts correctos según método
        prompts = {
            "inpainting": kwargs.get("prompt", "natural background"),
            "outpainting": kwargs.get("prompt", "extend scene naturally"),
            "style_transfer": kwargs.get("prompt", "artistic painting"),
            "object_removal": kwargs.get("prompt", "remove object and fill naturally"),
            "background_replacement": kwargs.get("prompt", "new background"),
            "intelligent_composition": kwargs.get("prompt", "harmonious composition")
        }

        prompt = prompts.get(method, "artistic image")

        # Parámetros numéricos
        params = {
            "num_inference_steps": kwargs.get("steps", 25),
            "guidance_scale": kwargs.get("guidance_scale", 7.5),
            "strength": kwargs.get("strength", 0.7)
        }

        # Seleccionar método
        if method == "inpainting":
            mask = kwargs.get("mask")
            result = self.inpainting(image, mask or Image.new("L", image.size, 255), prompt, **params)

        elif method == "outpainting":
            result = self.outpainting(image, prompt, **params)

        elif method == "style_transfer":
            result = self.style_transfer(image, prompt, **params)

        elif method == "object_removal":
            mask = kwargs.get("mask") or Image.new("L", image.size, 255)
            result = self.object_removal(image, prompt, mask, **params)

        elif method == "background_replacement":
            mask = kwargs.get("mask") or Image.new("L", image.size, 255)
            result = self.background_replacement(image, prompt, mask, **params)

        elif method == "intelligent_composition":
            result = self.intelligent_composition(image, prompt, **params)

        else:
            return None, {"error": f"Método desconocido: {method}"}

        # Final
        if result is None:
            return None, {"error": "HF no generó imagen"}

        return result, {"status": "success", "method": method, "prompt": prompt, "params": params}

