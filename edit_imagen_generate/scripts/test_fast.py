#!/usr/bin/env python3
"""Prueba ultra-rapida sin cargar modelos"""
import sys
import os
from PIL import Image

# Asegurar ruta del proyecto
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_fallback_only():
    """Probar solo el fallback local"""
    print("Probando solo fallback local...")
    
    try:
        # Importar solo lo necesario
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Crear imagen simple
        test_img = Image.new('RGB', (200, 200), (100, 150, 200))
        print(f"Imagen de prueba creada: {test_img.size}")
        
        # Importar y crear procesador sin inicializar modelos
        from models.diffusion import DiffusionProcessor
        
        # Crear instancia sin cargar modelos
        processor = DiffusionProcessor()
        processor.pipes = {}  # Forzar empty para evitar carga
        print("Procesador creado (sin modelos cargados)")
        
        # Probar fallback directamente
        print("Probando fallback local...")
        result, metadata = processor._local_fallback(test_img, 'outpainting', {})
        
        if result is not None:
            print(f"Fallback exitoso! Tamano: {result.size}")
            print(f"Metadata: {metadata}")
            return True
        else:
            print("Error en fallback")
            return False
            
    except Exception as e:
        print(f"Error en prueba: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_fallback_only()
    if success:
        print("EXITO: El fallback local funciona correctamente!")
    else:
        print("FALLO: Hay problemas con el fallback local.")