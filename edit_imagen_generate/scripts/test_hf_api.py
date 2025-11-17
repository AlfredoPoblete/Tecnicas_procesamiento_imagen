#!/usr/bin/env python3
"""Script de prueba para invocar la Inference API de Hugging Face

Genera una imagen de prueba y una máscara, instancia `DiffusionProcessor`
y ejecuta `inpainting`. Guarda el resultado en `output/test_hf_result.png`.

Uso:
  En PowerShell (en la carpeta raíz del proyecto):
    $env:USE_HF_API = "true"
    $env:HUGGINGFACE_API_TOKEN = "hf_TU_TOKEN_AQUI"
    python -m pip install -r requirements.txt
    python scripts/test_hf_api.py

No pegues el token en el chat; configúralo sólo en tu terminal o Secrets.
"""
import os
import sys
import traceback
# Asegurar que la raíz del proyecto esté en sys.path para poder importar `models`
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PIL import Image, ImageDraw

try:
    from models.diffusion import DiffusionProcessor
except Exception as e:
    print("Error importando models.diffusion:", e)
    raise


def make_test_image(size=512):
    img = Image.new("RGB", (size, size), (120, 140, 160))
    draw = ImageDraw.Draw(img)
    # Dibujar un rectángulo central para tener contenido
    draw.rectangle([size//4, size//4, size*3//4, size*3//4], fill=(200, 200, 200))
    return img


def make_test_mask(size=512):
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    # Área a inpaint: rectángulo central
    draw.rectangle([size//3, size//3, size*2//3, size*2//3], fill=255)
    return mask


def main():
    use_hf = os.getenv("USE_HF_API", "").lower() in ("1", "true", "yes")
    token = os.getenv("HUGGINGFACE_API_TOKEN", "")
    print("USE_HF_API:", use_hf)
    print("HUGGINGFACE_API_TOKEN set:", bool(token))

    proc = DiffusionProcessor()

    img = make_test_image()
    mask = make_test_mask()

    try:
        print("Lanzando outpainting de prueba (fallback img2img)...")
        result, meta = proc.outpainting(img, extension_factor=1.5, prompt="A scenic natural background, photorealistic, high detail")
        out_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "test_hf_result.png")
        result.save(out_path)
        print("Resultado guardado en:", out_path)
        print("Metadata:")
        print(meta)
    except Exception as e:
        print("Error durante la inferencia:", e)
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
