"""
Utilidades de Interfaz de Usuario - Versión Optimizada
Componentes reutilizables para la interfaz Streamlit

Implementa el diseño específico solicitado SIN función Inpainting
"""

import streamlit as st
from typing import Dict, Any, List, Optional
from PIL import Image
import numpy as np

class UIHelper:
    """Helper para componentes de UI reutilizables - Versión optimizada sin Inpainting"""
    
    def __init__(self):
        self.processing_methods = {
            "Outpainting (Extender imagen)": {
                "key": "outpainting", 
                "description": "Extiende la imagen más allá de sus bordes",
                "icon": "🔄",
                "use_case": "Expandir fondos y escenarios"
            },
            "Style Transfer (Transferir estilo)": {
                "key": "style_transfer",
                "description": "Aplica estilos artísticos a la imagen",
                "icon": "🎭",
                "use_case": "Cambiar estilo visual manteniendo contenido"
            },
            "Background Replacement (Cambiar fondo)": {
                "key": "background_replacement", 
                "description": "Reemplaza el fondo manteniendo el sujeto",
                "icon": "🖼️",
                "use_case": "Cambiar contextos manteniendo protagonista"
            },
            "Composición Inteligente (Combinar elementos)": {
                "key": "intelligent_composition",
                "description": "Combina elementos de múltiples imágenes",
                "icon": "🧩",
                "use_case": "Crear composiciones complejas"
            }
        }
    
    def get_processing_params(self, method: str) -> Dict[str, Any]:
        """Obtener parámetros optimizados para el método de procesamiento seleccionado"""
        params = {}
        
        if "Outpainting" in method:
            params.update({
                'extension_factor': st.slider(
                    "Factor de extensión",
                    min_value=1.2, max_value=3.0, value=1.5, step=0.1,
                    help="1.5 = 50% más grande, 2.0 = 2x más grande"
                ),
                'prompt': st.text_input(
                    "Prompt para la extensión",
                    value="seamless natural extension, matching the existing scene",
                    help="Describe coherentemente cómo debe verse la extensión"
                ),
                'num_inference_steps': st.slider(
                    "Pasos de procesamiento",
                    min_value=30, max_value=120, value=50,  # Aumentado para mejor calidad
                    help="Outpainting requiere más pasos para mejor calidad"
                ),
                'guidance_scale': st.slider(
                    "Control de adherencia",
                    min_value=6.0, max_value=15.0, value=8.5, step=0.5
                )
            })
            
        elif "Style Transfer" in method:
            style_options = [
                "artistic painting style",
                "watercolor painting",
                "oil painting effect",
                "cartoon/anime style",
                "sketch/pencil drawing",
                "vintage sepia style",
                "modern digital art",
                "impressionist style",
                "cyberpunk neon style",
                "renaissance painting",
                "minimalist design"
            ]
            
            params.update({
                'style_prompt': st.selectbox(
                    "Estilo artístico", 
                    style_options,
                    index=0,
                    help="Selecciona el estilo que quieres aplicar"
                ),
                'strength': st.slider(
                    "Intensidad del estilo", 
                    min_value=0.1, max_value=1.0, value=0.6, step=0.1,
                    help="Qué tan fuerte aplicar el nuevo estilo"
                ),
                'num_inference_steps': st.slider(
                    "Pasos de procesamiento", 
                    min_value=25, max_value=80, value=35
                ),
                'guidance_scale': st.slider(
                    "Control de adherencia", 
                    min_value=5.0, max_value=15.0, value=7.5, step=0.5
                )
            })
            
        # Note: Object Removal / Inpainting fue eliminado por requerimiento.
            
        elif "Background Replacement" in method:
            background_options = [
                "beautiful sunset landscape",
                "modern city skyline", 
                "serene forest scene",
                "ocean beach setting",
                "mountain vista",
                "studio photography backdrop",
                "abstract colorful background",
                "cozy indoor environment",
                "space nebula",
                "futuristic cyberpunk city",
                "ancient castle",
                "tropical paradise"
            ]
            
            params.update({
                'background_prompt': st.selectbox(
                    "Tipo de fondo", 
                    background_options,
                    index=0,
                    help="Selecciona el nuevo fondo deseado"
                ),
                'num_inference_steps': st.slider(
                    "Pasos de procesamiento", 
                    min_value=30, max_value=100, value=45
                ),
                'guidance_scale': st.slider(
                    "Control de adherencia", 
                    min_value=6.0, max_value=15.0, value=8.5, step=0.5
                )
            })
            
        elif "Composición Inteligente" in method:
            composition_options = [
                "harmonious artistic composition",
                "balanced visual elements",
                "creative collage style",
                "seamless blending",
                "layered depth composition",
                "abstract creative arrangement",
                "cinematic scene composition",
                "surreal artistic composition"
            ]
            
            params.update({
                'elements_prompt': st.selectbox(
                    "Estilo de composición", 
                    composition_options,
                    index=0,
                    help="Cómo deben combinarse los elementos"
                ),
                'strength': st.slider(
                    "Intensidad de la composición", 
                    min_value=0.2, max_value=1.0, value=0.5, step=0.1
                ),
                'num_inference_steps': st.slider(
                    "Pasos de procesamiento", 
                    min_value=30, max_value=100, value=40
                ),
                'guidance_scale': st.slider(
                    "Control de adherencia", 
                    min_value=6.0, max_value=15.0, value=8.0, step=0.5
                )
            })
        
        return params
    
    def render_method_info(self, method: str):
        """Renderizar información del método seleccionado"""
        if method in self.processing_methods:
            info = self.processing_methods[method]
            
            st.markdown(f"### {info['icon']} {method}")
            st.write(info['description'])
            st.write(f"**Caso de uso:** {info['use_case']}")
            
            # Mostrar consejos específicos (sin Inpainting)
            if "Outpainting" in method:
                st.info("""
                **Consejos para outpainting:**
                - Describe coherentemente el contexto extendido
                - Esta técnica requiere más procesamiento
                - Ideal para escenarios y fondos amplios
                - Usa factores de extensión moderados (1.2-1.8)
                """)
            elif "Style Transfer" in method:
                st.info("""
                **Consejos para style transfer:**
                - Ajusta la intensidad según el efecto deseado
                - Estilos sutiles mantienen más del original
                - Experimenta con diferentes estilos artísticos
                - Para máximo impacto usa strength > 0.7
                """)
            elif "Object Removal" in method:
                st.info("""
                **Consejos para object removal:**
                - Selecciona el tipo de objeto específico
                - Especifica el contexto del fondo deseado
                - Detección automática de la ubicación
                - Usa guidance scale alto (8.0-9.5) para mejor adherencia
                """)
            elif "Background Replacement" in method:
                st.info("""
                **Consejos para background replacement:**
                - Selecciona fondos compatibles con el sujeto
                - Ajusta la iluminación si es necesario
                - Considera la perspectiva y escala
                - Usa prompts descriptivos del nuevo ambiente
                """)
            elif "Composición" in method:
                st.info("""
                **Consejos para composición inteligente:**
                - Define claramente qué elementos combinar
                - Usa prompts que sugieran armonía visual
                - Strength controla la intensidad de cambios
                - Experimenta con diferentes estilos de composición
                """)
    
    def render_processing_status(self, status_info: Dict[str, Any]):
        """Renderizar estado del procesamiento"""
        if 'device' in status_info:
            device = status_info['device']
            if device == 'cuda':
                st.success(f"🚀 GPU disponible - Procesamiento acelerado")
            else:
                st.warning("⚠️ CPU solamente - El procesamiento será más lento")
        
        if 'models_loaded' in status_info:
            models = status_info['models_loaded']
            st.write(f"**Modelos cargados:** {', '.join(models)}")
    
    def create_comparison_view(self, original: Image.Image, processed: Image.Image, 
                             title: str = "Comparación Original vs Procesada"):
        """Crear vista de comparación lado a lado"""
        st.subheader(f"📊 {title}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🖼️ Imagen Original**")
            st.image(original, use_column_width=True)
            
        with col2:
            st.markdown("**✨ Imagen Procesada**")
            st.image(processed, use_column_width=True)
    
    def render_analysis_results(self, analysis: Dict[str, Any]):
        """Renderizar resultados del análisis"""
        if 'error' in analysis:
            st.error(f"Error en análisis: {analysis['error']}")
            return
        
        # Descripción de cambios
        if 'changes_description' in analysis:
            st.markdown("### 🔍 Cambios Detectados")
            st.write(analysis['changes_description'])
        
        # Métricas de calidad
        if 'quality_metrics' in analysis:
            st.markdown("### 📈 Métricas de Calidad")
            metrics = analysis['quality_metrics']
            for metric, value in metrics.items():
                st.write(f"**{metric}:** {value}")
        
        # Comparación detallada
        if 'comparison' in analysis:
            st.markdown("### 🔬 Comparación Detallada")
            st.write(analysis['comparison'])
        
        # Recomendaciones
        if 'recommendations' in analysis:
            st.markdown("### 💡 Recomendaciones")
            for i, rec in enumerate(analysis['recommendations'], 1):
                st.write(f"{i}. {rec}")
    
    def render_user_guide(self):
        """Renderizar guía de usuario específica"""
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎮 Guía para Desarrolladores de Videojuegos")
        
        st.sidebar.markdown("""
        ### 🎯 Casos de uso específicos:
        
        **Para Concept Artists:**
        - **Outpainting**: Expandir bocetos para mundos más grandes
        - **Style Transfer**: Unificar estilos en portfolios
        - **Composición**: Crear environments complejos
        
        **Para Desarrolladores de Juegos:**
        - **Background Replacement**: Cambiar contextos de sprites
        - **Object Removal**: Limpiar assets de recursos no deseados
        - **Style Transfer**: Adaptar assets a diferentes estilos
        
        ### ⚙️ Configuraciones recomendadas:
        
        **Para velocidad (desarrollo rápido):**
        - Steps: 20-30
        - Guidance Scale: 6-7
        
        **Para calidad máxima (assets finales):**
        - Steps: 50-80
        - Guidance Scale: 8-10
        
        **Para estilo artístico (concept art):**
        - Strength: 0.7-0.9
        - Guidance Scale: 9-12
        """)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 Consejos Técnicos")
        
        st.sidebar.markdown("""
        **Optimización de rendimiento:**
        - GPU: 10-30 segundos por imagen
        - CPU: 1-5 minutos por imagen
        - Imágenes 512x512: velocidad óptima
        
        **Mejores prácticas:**
        1. Comienza con parámetros por defecto
        2. Ajusta un parámetro a la vez
        3. Guarda versiones para comparar
        4. Usa análisis automático para validar
        
        **Resolución de problemas:**
        - Imagen borrosa: Aumentar steps
        - No sigue el prompt: Aumentar guidance scale
        - Artefactos: Reducir strength (img2img)
        - Muy lento: Reducir resolución
        """)
    
    def get_method_tips(self, method: str) -> List[str]:
        """Obtener consejos específicos para cada método (sin Inpainting)"""
        tips = {
            "Outpainting": [
                "Describe coherentemente el contexto extendido",
                "Usa prompts que incluyan la continuación natural",
                "Esta técnica requiere más procesamiento (50+ steps)",
                "Ideal para expandir landscapes y fondos",
                "Factores de extensión moderados dan mejores resultados"
            ],
            "Style Transfer": [
                "Ajusta strength según el efecto deseado",
                "Estilos sutiles mantienen más del original",
                "Experimenta con diferentes estilos artísticos",
                "Para máximo impacto usa strength > 0.7",
                "Combina guidance scale alto con strength medio-alto"
            ],
            "Object Removal": [
                "Selecciona el tipo específico de objeto",
                "Usa términos claros (person, carro, edificio, etc.)",
                "Especifica el contexto del fondo deseado",
                "Detección automática de la ubicación del objeto",
                "Usa guidance scale alto (8.0-9.5) para mejor adherencia",
                "Más pasos (40-50) mejoran la integración natural"
            ],
            "Background Replacement": [
                "Selecciona fondos compatibles con el sujeto",
                "Ajusta la iluminación si es necesario",
                "Considera la perspectiva y escala",
                "Usa prompts descriptivos del nuevo ambiente",
                "Backgrounds más simples suelen funcionar mejor"
            ],
            "Composición": [
                "Define claramente qué elementos combinar",
                "Usa prompts que sugieran armonía visual",
                "Strength controla la intensidad de cambios",
                "Experimenta con diferentes estilos de composición",
                "Composición sutil preserva mejor el contenido original"
            ]
        }
        
        for method_key, method_tips in tips.items():
            if method_key in method:
                return method_tips
        
        return ["Usa parámetros moderados como punto de partida"]

    def get_optimized_default_params(self, method: str) -> Dict[str, Any]:
        """Obtener parámetros optimizados por defecto para cada método"""
        defaults = {
            "Outpainting": {
                'extension_factor': 1.5,
                'prompt': 'seamless natural extension, matching the existing scene',
                'num_inference_steps': 50,
                'guidance_scale': 8.5
            },
            "Style Transfer": {
                'style_prompt': 'artistic painting style',
                'strength': 0.6,
                'num_inference_steps': 35,
                'guidance_scale': 7.5
            },
            # Object Removal eliminado
            "Background Replacement": {
                'background_prompt': 'beautiful sunset landscape',
                'num_inference_steps': 45,
                'guidance_scale': 8.5
            },
            "Composición Inteligente": {
                'elements_prompt': 'harmonious artistic composition',
                'strength': 0.5,
                'num_inference_steps': 40,
                'guidance_scale': 8.0
            }
        }
        
        for method_key, method_defaults in defaults.items():
            if method_key in method:
                return method_defaults.copy()
        
        return {}  # Parámetros vacíos si no se encuentra el método