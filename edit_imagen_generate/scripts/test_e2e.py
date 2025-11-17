#!/usr/bin/env python3
"""Prueba E2E programática: simula carga de imagen -> procesado -> análisis

Crea una imagen de prueba, corre `process` con los métodos soportados y
lanza el analizador comparativo. Guarda resultados en `output/` y muestra
resúmenes por consola.
"""
import os
import sys
from PIL import Image, ImageDraw

# Asegurar ruta del proyecto
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.diffusion import DiffusionProcessor
from models.analysis import GeminiAnalyzer


def make_test_image(size=512):
    img = Image.new("RGB", (size, size), (120, 140, 160))
    draw = ImageDraw.Draw(img)
    draw.rectangle([size//4, size//4, size*3//4, size*3//4], fill=(200, 200, 200))
    return img


def ensure_output_dir():
    out = os.path.join(project_root, 'output')
    os.makedirs(out, exist_ok=True)
    return out


def run_e2e():
    print('Instanciando procesador y analizador...')
    proc = DiffusionProcessor()
    analyzer = GeminiAnalyzer()

    original = make_test_image(320)
    out_dir = ensure_output_dir()
    original_path = os.path.join(out_dir, 'e2e_original.png')
    original.save(original_path)
    print(f'Imagen original guardada en: {original_path}')

    methods = [
        ('outpainting', {'extension_factor': 1.3, 'prompt': 'Extend the scene with a natural continuation', 'num_inference_steps': 20}),
        ('style_transfer', {'style_prompt': 'watercolor painting', 'strength': 0.6, 'num_inference_steps': 20}),
        ('background_replacement', {'background_prompt': 'sunset beach', 'num_inference_steps': 20}),
        ('intelligent_composition', {'elements_prompt': 'harmonious artistic composition', 'strength': 0.5, 'num_inference_steps': 20})
    ]

    results = {}

    for method_key, params in methods:
        print('\n---')
        print(f'Procesando método: {method_key} con params: {params}')
        try:
            result_img, metadata = proc.process(original.copy(), method_key, **params)
            if result_img is None:
                print(f'Error: resultado nulo para {method_key} - metadata: {metadata}')
                results[method_key] = {'error': metadata}
                continue

            fname = f'e2e_{method_key}.png'
            path = os.path.join(out_dir, fname)
            result_img.save(path)
            print(f'Resultado guardado en: {path}')
            print('Metadata:', metadata)

            # Ejecutar análisis comparativo (usará mock si no hay API key)
            analysis = analyzer.analyze(result_img, analysis_type='comparison_analysis', original_image=original)
            print('Análisis (resumen):')
            if isinstance(analysis, dict):
                # Priorizar campos existentes
                text = analysis.get('brief_analysis') or analysis.get('changes_description') or str(analysis)
                print(text if text else analysis)
            else:
                print(str(analysis))

            results[method_key] = {'metadata': metadata, 'analysis': analysis, 'path': path}

        except Exception as e:
            print(f'Excepción durante procesamiento de {method_key}: {e}')
            results[method_key] = {'error': str(e)}

    # Resumen final
    print('\n=== Resumen E2E ===')
    for k, v in results.items():
        print(f'- {k}:', 'OK' if 'path' in v else f'ERROR: {v.get("error")}')

    return results


if __name__ == '__main__':
    run_e2e()
