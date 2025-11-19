"""
Lightweight DiffusionProcessor para Streamlit Cloud
Usa HuggingFace Inference API en lugar de modelos locales (diffusers / torch).
Mantiene la interfaz:
    dp = DiffusionProcessor()
    result_image, metadata = dp.process(image=image, method='inpainting', **kwargs)
"""

import os
import io
import time
import base64
import json
from typing import Tuple, Dict, Any, Optional
from PIL import Image
import requests

HF_API_URL_TEMPLATE = "https://api-inference.huggingface.co/models/{model}"
# Mapear método -> modelo HF
METHOD_TO_HF_MODEL = {
    "inpainting": "stabilityai/stable-diffusion-2-inpainting",
    "outpainting": "stabilityai/stable-diffusion-2-inpainting",
    "style_transfer": "hakurei/waifu-diffusion",
    "object_removal": "stabilityai/stable-diffusion-2-inpainting",
    "background_replacement": "stabilityai/stable-diffusion-2-inpainting",
    "img2img": "runwayml/stable-diffusion-v1-5",
    "upscale": "stabilityai/stable-diffusion-x4-upscaler"
}

REQUEST_TIMEOUT = 60


class DiffusionProcessor:
    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token or os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN")
        if not self.hf_token:
            print("⚠️ WARNING: No HF_API_TOKEN encontrado en el entorno.")
        self.session = requests.Session()
        if self.hf_token:
            self.session.headers.update({"Authorization": f"Bearer {self.hf_token}"})

    def _image_to_bytes(self, image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        return buf

    def _post_to_hf(self, model: str, payload: dict = None, files: dict = None) -> requests.Response:
        url = HF_API_URL_TEMPLATE.format(model=model)
        headers = {}

        try:
            if files:
                resp = self.session.post(url, data=payload or {}, files=files, timeout=REQUEST_TIMEOUT)
            else:
                headers["Content-Type"] = "application/json"
                resp = self.session.post(url, json=payload or {}, headers=headers, timeout=REQUEST_TIMEOUT)
            return resp
        except requests.RequestException as e:
            raise RuntimeError(f"Error en la petición a HuggingFace: {e}")

    def _parse_hf_response_to_image(self, resp: requests.Response) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        metadata = {"status_code": resp.status_code, "headers": dict(resp.headers)}

        if resp.status_code != 200:
            try:
                metadata["error"] = resp.json().get("error", resp.text)
            except:
                metadata["error"] = resp.text
            return None, metadata

        content_type = resp.headers.get("content-type", "")

        if content_type.startswith("image/"):
            try:
                return Image.open(io.BytesIO(resp.content)).convert("RGB"), metadata
            except Exception as e:
                metadata["error"] = str(e)
                return None, metadata

        try:
            j = resp.json()
            for key in ("image", "images", "generated_image", "image_base64", "result"):
                if key in j:
                    val = j[key]
                    if isinstance(val, list) and val:
                        v = val[0]
                        if isinstance(v, str):
                            if v.startswith("http"):
                                down = self.session.get(v, timeout=REQUEST_TIMEOUT)
                                return Image.open(io.BytesIO(down.content)).convert("RGB"), metadata
                            else:
                                b = base64.b64decode(v)
                                return Image.open(io.BytesIO(b)).convert("RGB"), metadata
                    elif isinstance(val, str):
                        if val.startswith("http"):
                            down = self.session.get(val, timeout=REQUEST_TIMEOUT)
                            return Image.open(io.BytesIO(down.content)).convert("RGB"), metadata
                        else:
                            b = base64.b64decode(val)
                            return Image.open(io.BytesIO(b)).convert("RGB"), metadata

            metadata["error"] = "Respuesta JSON sin imagen reconocible."
            metadata["response_json"] = j
            return None, metadata

        except:
            metadata["error"] = "Respuesta no JSON y no image/*"
            return None, metadata

    def process(self, image: Image.Image, method: str = "inpainting", **kwargs):
        start = time.time()
        method_key = method.lower().replace(" ", "_")
        model = METHOD_TO_HF_MODEL.get(method_key, METHOD_TO_HF_MODEL["inpainting"])

        metadata = {
            "method": method_key,
            "model": model,
            "timestamp": time.time(),
        }

        if not self.hf_token:
            metadata["error"] = "HF_API_TOKEN no configurado."
            return None, metadata

        prompt = (
            kwargs.get("prompt")
            or kwargs.get("style_prompt")
            or kwargs.get("object_description")
            or ""
        )

        parameters = {
            "num_inference_steps": kwargs.get("num_inference_steps"),
            "guidance_scale": kwargs.get("guidance_scale"),
            "strength": kwargs.get("strength"),
        }
        parameters = {k: v for k, v in parameters.items() if v is not None}

        img_buf = self._image_to_bytes(image)

        files = {"image": ("image.png", img_buf, "image/png")}
        payload = {"inputs": prompt, "options": json.dumps({"wait_for_model": True})}
        if parameters:
            payload["parameters"] = json.dumps(parameters)

        try:
            resp = self._post_to_hf(model, payload=payload, files=files)
            img, m = self._parse_hf_response_to_image(resp)
            metadata.update(m)
            metadata["time_s"] = round(time.time() - start, 2)
            return img, metadata
        except Exception as e:
            metadata["exception"] = str(e)
            metadata["time_s"] = round(time.time() - start, 2)
            return None, metadata
