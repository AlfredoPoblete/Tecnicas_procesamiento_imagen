#!/usr/bin/env python3
"""Prueba simple de funcionalidad básica"""
import sys
import os
from PIL import Image

# Asegurar ruta del proyecto
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_basic_functionality():
    """Probar funcionalidad básica sin modelos pesados"""
    print("Probando funcionalidad basica...")
    
    try:
        from models.diffusion import DiffusionProcessor
        
        # Crear imagen simple
        test_img = Image.new('RGB', (200, 200), (100, 150, 200))
        print(f"Imagen de prueba creada: {test_img.size}")
        
        # Crear procesador
        processor = DiffusionProcessor()
        print("Procesador creado exitosamente")
        
        # Probar método de fallback local
        print("Probando método de fallback local...")
        result, metadata = processor._local_fallback(test_img, 'outpainting', {})
        
        if result is not None:
            print(f"Fallback exitoso! Tamano: {result.size}")
            print(f"Metadata: {metadata}")
        else:
            print("Error en fallback")
            
        # Probar método principal
        print("Probando método principal...")
        result2, metadata2 = processor.process(test_img, 'outpainting', extension_factor=1.2)
        
        if result2 is not None:
            print(f"Procesamiento exitoso! Tamano: {result2.size}")
            print(f"Metadata: {metadata2}")
            return True
        else:
            print("Error en procesamiento principal")
            return False
            
    except Exception as e:
        print(f"Error en prueba: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_basic_functionality()
    if success:
        print("Prueba exitosa! La aplicacion deberia funcionar.")
    else:
        print("Hay problemas con la aplicacion.")