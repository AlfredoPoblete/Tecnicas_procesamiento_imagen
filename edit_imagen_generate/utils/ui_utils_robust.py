"""
Utilidades de Interfaz de Usuario - VERSIÓN ROBUSTA
Componentes reutilizables para la interfaz Streamlit con manejo de errores

Implementa el diseño específico solicitado con recuperación automática
"""

import streamlit as st
from typing import Dict, Any, List, Optional
from PIL import Image
import numpy as np
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UIHelperRobust:
    """Helper para componentes de UI reutilizables con manejo robusto de errores"""
    
    def __init__(self):
        logger.info("🚀 Inicializando UIHelperRobust")
        self.processing_methods = {}
        self._initialize_methods()
    
    def _initialize_methods(self):
        """Inicializar métodos con manejo de errores"""
        try:
            self.processing_methods = {
                "Inpainting (Eliminar objetos)": {
                    "key": "inpainting",
                    "description": "Elimina y rellena objetos no deseados",
                    "icon": "🖼️",
                    "use_case": "Remover elementos indeseados de fotos"
                },
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
                "Object Removal (Eliminar objeto específico)": {
                    "key": "object_removal",
                    "description": "Elimina objetos específicos con precisión",
                    "icon": "🗑️",
                    "use_case": "Limpieza precisa de elementos"
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
        except Exception as e:
            logger.error(f"Error inicializando métodos: {e}")
            self.processing_methods = {}
    
    def get_processing_params(self, method: str) -> Dict[str, Any]:
        """Obtener parámetros para el método de procesamiento seleccionado"""
        try:
            if not method:
                return {}
            
            params = {}
            
            # Manejar errores con try-catch para cada método
            if "Inpainting" in method:
                params = self._get_inpainting_params()
            elif "Outpainting" in method:
                params = self._get_outpainting_params()
            elif "Style Transfer" in method:
                params = self._get_style_transfer_params()
            elif "Object Removal" in method:
                params = self._get_object_removal_params()
            elif "Background Replacement" in method:
                params = self._get_background_replacement_params()
            elif "Composición Inteligente" in method:
                params = self._get_composition_params()
            else:
                logger.warning(f"Método desconocido: {method}")
                return self._get_default_params()
            
            return params
            
        except Exception as e:
            logger.error(f"Error obteniendo parámetros para {method}: {e}")
            return self._get_default_params()
    
    def _get_inpainting_params(self) -> Dict[str, Any]:
        """Obtener parámetros para inpainting"""
        try:
            params = {}
            
            # Prompt para el relleno
            try:
                params['prompt'] = st.text_input(
                    "Prompt para el relleno", 
                    value="natural background texture",
                    help="Describe qué quieres que aparezca en el área eliminada"
                )
            except Exception as e:
                logger.error(f"Error en prompt de inpainting: {e}")
                params['prompt'] = "natural background texture"
            
            # Pasos de procesamiento
            try:
                params['num_inference_steps'] = st.slider(
                    "Pasos de procesamiento", 
                    min_value=20, max_value=100, value=30,
                    help="Más pasos = mejor calidad pero más lento"
                )
            except Exception as e:
                logger.error(f"Error en slider steps inpainting: {e}")
                params['num_inference_steps'] = 30
            
            # Guidance scale
            try:
                params['guidance_scale'] = st.slider(
                    "Control de adherencia al prompt", 
                    min_value=5.0, max_value=15.0, value=7.5, step=0.5,
                    help="Qué tan fuerte seguir la descripción"
                )
            except Exception as e:
                logger.error(f"Error en slider guidance inpainting: {e}")
                params['guidance_scale'] = 7.5
            
            # Crear máscara interactiva
            try:
                st.subheader("🎯 Configuración de Máscara")
                col1, col2 = st.columns(2)
                
                with col1:
                    x = st.number_input("Posición X", min_value=0, max_value=512, value=200)
                    y = st.number_input("Posición Y", min_value=0, max_value=512, value=200)
                
                with col2:
                    width = st.number_input("Ancho", min_value=10, max_value=512, value=100)
                    height = st.number_input("Alto", min_value=10, max_value=512, value=100)
                
                # Crear máscara
                params['mask_coords'] = (x, y, x + width, y + height)
                
            except Exception as e:
                logger.error(f"Error configurando máscara inpainting: {e}")
                # Parámetros por defecto para la máscara
                params['mask_coords'] = (200, 200, 300, 300)
            
            return params
            
        except Exception as e:
            logger.error(f"Error en _get_inpainting_params: {e}")
            return self._get_default_params()
    
    def _get_outpainting_params(self) -> Dict[str, Any]:
        """Obtener parámetros para outpainting"""
        try:
            params = {}
            
            # Factor de extensión
            try:
                params['extension_factor'] = st.slider(
                    "Factor de extensión",
                    min_value=1.2, max_value=3.0, value=1.5, step=0.1,
                    help="1.5 = 50% más grande, 2.0 = 2x más grande"
                )
            except Exception as e:
                logger.error(f"Error en slider extension factor: {e}")
                params['extension_factor'] = 1.5
            
            # Prompt para la extensión
            try:
                params['prompt'] = st.text_input(
                    "Prompt para la extensión",
                    value="seamless natural extension, matching the existing scene",
                    help="Describe coherentemente cómo debe verse la extensión"
                )
            except Exception as e:
                logger.error(f"Error en prompt outpainting: {e}")
                params['prompt'] = "seamless natural extension, matching the existing scene"
            
            # Pasos de procesamiento
            try:
                params['num_inference_steps'] = st.slider(
                    "Pasos de procesamiento",
                    min_value=30, max_value=150, value=45,
                    help="Outpainting requiere más pasos para mejor calidad"
                )
            except Exception as e:
                logger.error(f"Error en slider steps outpainting: {e}")
                params['num_inference_steps'] = 45
            
            # Guidance scale
            try:
                params['guidance_scale'] = st.slider(
                    "Control de adherencia",
                    min_value=6.0, max_value=15.0, value=8.5, step=0.5
                )
            except Exception as e:
                logger.error(f"Error en slider guidance outpainting: {e}")
                params['guidance_scale'] = 8.5
            
            return params
            
        except Exception as e:
            logger.error(f"Error en _get_outpainting_params: {e}")
            return self._get_default_params()
    
    def _get_style_transfer_params(self) -> Dict[str, Any]:
        """Obtener parámetros para style transfer"""
        try:
            params = {}
            
            # Opciones de estilo
            style_options = [
                "artistic painting style",
                "watercolor painting",
                "oil painting effect",
                "cartoon/anime style",
                "sketch/pencil drawing",
                "vintage sepia style",
                "modern digital art",
                "impressionist style"
            ]
            
            # Estilo artístico
            try:
                params['style_prompt'] = st.selectbox(
                    "Estilo artístico", 
                    style_options,
                    index=0,
                    help="Selecciona el estilo que quieres aplicar"
                )
            except Exception as e:
                logger.error(f"Error en selectbox style: {e}")
                params['style_prompt'] = style_options[0]
            
            # Intensidad del estilo
            try:
                params['strength'] = st.slider(
                    "Intensidad del estilo", 
                    min_value=0.1, max_value=1.0, value=0.6, step=0.1,
                    help="Qué tan fuerte aplicar el nuevo estilo"
                )
            except Exception as e:
                logger.error(f"Error en slider strength style: {e}")
                params['strength'] = 0.6
            
            # Pasos de procesamiento
            try:
                params['num_inference_steps'] = st.slider(
                    "Pasos de procesamiento", 
                    min_value=25, max_value=80, value=35
                )
            except Exception as e:
                logger.error(f"Error en slider steps style: {e}")
                params['num_inference_steps'] = 35
            
            # Guidance scale
            try:
                params['guidance_scale'] = st.slider(
                    "Control de adherencia", 
                    min_value=5.0, max_value=15.0, value=7.5, step=0.5
                )
            except Exception as e:
                logger.error(f"Error en slider guidance style: {e}")
                params['guidance_scale'] = 7.5
            
            return params
            
        except Exception as e:
            logger.error(f"Error en _get_style_transfer_params: {e}")
            return self._get_default_params()
    
    def _get_object_removal_params(self) -> Dict[str, Any]:
        """Obtener parámetros para object removal"""
        try:
            params = {}
            
            # Objeto a eliminar
            try:
                params['object_description'] = st.text_input(
                    "Objeto a eliminar",
                    value="unwanted object",
                    help="Describe específicamente el objeto que quieres eliminar (ej: 'persona en la izquierda', 'carro rojo', 'señal de tránsito')"
                )
            except Exception as e:
                logger.error(f"Error en text input object description: {e}")
                params['object_description'] = "unwanted object"
            
            # Contexto del fondo
            try:
                params['context_prompt'] = st.text_input(
                    "Contexto del fondo",
                    value="natural seamless background",
                    help="Describe el fondo natural que debe aparecer"
                )
            except Exception as e:
                logger.error(f"Error en text input context prompt: {e}")
                params['context_prompt'] = "natural seamless background"
            
            # Pasos de procesamiento
            try:
                params['num_inference_steps'] = st.slider(
                    "Pasos de procesamiento",
                    min_value=30, max_value=80, value=45
                )
            except Exception as e:
                logger.error(f"Error en slider steps object removal: {e}")
                params['num_inference_steps'] = 45
            
            # Guidance scale
            try:
                params['guidance_scale'] = st.slider(
                    "Control de adherencia",
                    min_value=7.0, max_value=15.0, value=9.0, step=0.5
                )
            except Exception as e:
                logger.error(f"Error en slider guidance object removal: {e}")
                params['guidance_scale'] = 9.0
            
            # Información para el usuario
            try:
                st.info("""
                **Object Removal Inteligente:**
                1. Describe el objeto que quieres eliminar
                2. Especifica cómo debe verse el fondo
                3. El sistema detectará automáticamente el objeto y lo eliminará
                4. No necesitas especificar posiciones manualmente
                """)
            except Exception as e:
                logger.error(f"Error mostrando info object removal: {e}")
            
            return params
            
        except Exception as e:
            logger.error(f"Error en _get_object_removal_params: {e}")
            return self._get_default_params()
    
    def _get_background_replacement_params(self) -> Dict[str, Any]:
        """Obtener parámetros para background replacement"""
        try:
            params = {}
            
            # Opciones de fondo
            background_options = [
                "beautiful sunset landscape",
                "modern city skyline", 
                "serene forest scene",
                "ocean beach setting",
                "mountain vista",
                "studio photography backdrop",
                "abstract colorful background",
                "cozy indoor environment"
            ]
            
            # Tipo de fondo
            try:
                params['background_prompt'] = st.selectbox(
                    "Tipo de fondo", 
                    background_options,
                    index=0,
                    help="Selecciona el nuevo fondo deseado"
                )
            except Exception as e:
                logger.error(f"Error en selectbox background: {e}")
                params['background_prompt'] = background_options[0]
            
            # Pasos de procesamiento
            try:
                params['num_inference_steps'] = st.slider(
                    "Pasos de procesamiento", 
                    min_value=30, max_value=100, value=45
                )
            except Exception as e:
                logger.error(f"Error en slider steps background: {e}")
                params['num_inference_steps'] = 45
            
            # Guidance scale
            try:
                params['guidance_scale'] = st.slider(
                    "Control de adherencia", 
                    min_value=6.0, max_value=15.0, value=8.5, step=0.5
                )
            except Exception as e:
                logger.error(f"Error en slider guidance background: {e}")
                params['guidance_scale'] = 8.5
            
            return params
            
        except Exception as e:
            logger.error(f"Error en _get_background_replacement_params: {e}")
            return self._get_default_params()
    
    def _get_composition_params(self) -> Dict[str, Any]:
        """Obtener parámetros para composición inteligente"""
        try:
            params = {}
            
            # Opciones de composición
            composition_options = [
                "harmonious artistic composition",
                "balanced visual elements",
                "creative collage style",
                "seamless blending",
                "layered depth composition",
                "abstract creative arrangement"
            ]
            
            # Estilo de composición
            try:
                params['elements_prompt'] = st.selectbox(
                    "Estilo de composición", 
                    composition_options,
                    index=0,
                    help="Cómo deben combinarse los elementos"
                )
            except Exception as e:
                logger.error(f"Error en selectbox composition: {e}")
                params['elements_prompt'] = composition_options[0]
            
            # Intensidad de la composición
            try:
                params['strength'] = st.slider(
                    "Intensidad de la composición", 
                    min_value=0.2, max_value=1.0, value=0.5, step=0.1
                )
            except Exception as e:
                logger.error(f"Error en slider strength composition: {e}")
                params['strength'] = 0.5
            
            # Pasos de procesamiento
            try:
                params['num_inference_steps'] = st.slider(
                    "Pasos de procesamiento", 
                    min_value=30, max_value=100, value=40
                )
            except Exception as e:
                logger.error(f"Error en slider steps composition: {e}")
                params['num_inference_steps'] = 40
            
            # Guidance scale
            try:
                params['guidance_scale'] = st.slider(
                    "Control de adherencia", 
                    min_value=6.0, max_value=15.0, value=8.0, step=0.5
                )
            except Exception as e:
                logger.error(f"Error en slider guidance composition: {e}")
                params['guidance_scale'] = 8.0
            
            return params
            
        except Exception as e:
            logger.error(f"Error en _get_composition_params: {e}")
            return self._get_default_params()
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Obtener parámetros por defecto seguros"""
        return {
            'num_inference_steps': 30,
            'guidance_scale': 7.5,
            'prompt': 'natural background'
        }
    
    def render_method_info(self, method: str):
        """Renderizar información del método seleccionado"""
        try:
            if method in self.processing_methods:
                info = self.processing_methods[method]
                
                st.markdown(f"### {info['icon']} {method}")
                st.write(info['description'])
                st.write(f"**Caso de uso:** {info['use_case']}")
                
                # Mostrar consejos específicos
                if "Inpainting" in method:
                    st.info("""
                    **Consejos para mejores resultados:**
                    - Usa prompts descriptivos del contexto
                    - Ajusta la máscara para cubrir exactamente el área
                    - Más pasos mejoran la integración natural
                    """)
                elif "Outpainting" in method:
                    st.info("""
                    **Consejos para outpainting:**
                    - Describe coherentemente el contexto extendido
                    - Esta técnica requiere más procesamiento
                    - Ideal para escenarios y fondos amplios
                    """)
                elif "Style Transfer" in method:
                    st.info("""
                    **Consejos para style transfer:**
                    - Ajusta la intensidad según el efecto deseado
                    - Experimenta con diferentes estilos
                    - Estilos más sutiles mantienen más del original
                    """)
        except Exception as e:
            logger.error(f"Error renderizando información del método {method}: {e}")
            st.error(f"Error cargando información del método")
    
    def render_processing_status(self, status_info: Dict[str, Any]):
        """Renderizar estado del procesamiento"""
        try:
            if 'device' in status_info:
                device = status_info['device']
                if device == 'cuda':
                    st.success(f"🚀 GPU disponible - Procesamiento acelerado")
                else:
                    st.warning("⚠️ CPU solamente - El procesamiento será más lento")
            
            if 'models_loaded' in status_info:
                models = status_info['models_loaded']
                st.write(f"**Modelos cargados:** {', '.join(models)}")
        except Exception as e:
            logger.error(f"Error renderizando estado: {e}")
            st.error("Error mostrando estado del procesamiento")
    
    def create_comparison_view(self, original: Image.Image, processed: Image.Image, 
                             title: str = "Comparación Original vs Procesada"):
        """Crear vista de comparación lado a lado"""
        try:
            st.subheader(f"📊 {title}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🖼️ Imagen Original**")
                st.image(original, use_column_width=True)
                
            with col2:
                st.markdown("**✨ Imagen Procesada**")
                st.image(processed, use_column_width=True)
        except Exception as e:
            logger.error(f"Error creando vista de comparación: {e}")
            st.error("Error creando vista de comparación")
    
    def render_analysis_results(self, analysis: Dict[str, Any]):
        """Renderizar resultados del análisis"""
        try:
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
        except Exception as e:
            logger.error(f"Error renderizando resultados del análisis: {e}")
            st.error("Error mostrando resultados del análisis")
    
    def render_user_guide(self):
        """Renderizar guía de usuario específica"""
        try:
            st.sidebar.markdown("---")
            st.sidebar.subheader("🎮 Guía para Desarrolladores de Videojuegos")
            
            st.sidebar.markdown("""
            ### 🎯 Casos de uso específicos:
            
            **Para Concept Artists:**
            - **Outpainting**: Expandir bocetos para mundos más grandes
            - **Style Transfer**: Unificar estilos en portfolios
            - **Inpainting**: Limpiar sketches rápidamente
            
            **Para Desarrolladores de Juegos:**
            - **Background Replacement**: Cambiar contextos de sprites
            - **Object Removal**: Limpiar assets de recursos no deseados
            - **Composición**: Crear environments complejos
            
            ### ⚙️ Configuraciones recomendadas:
            
            **Para velocidad (desarrollo rápido):**
            - Steps: 20-25
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
        except Exception as e:
            logger.error(f"Error renderizando guía de usuario: {e}")
            st.sidebar.error("Error cargando guía de usuario")
    
    def get_method_tips(self, method: str) -> List[str]:
        """Obtener consejos específicos para cada método"""
        try:
            tips = {
                "Inpainting": [
                    "Usa prompts descriptivos del contexto local",
                    "Ajusta la máscara para cubrir exactamente el área",
                    "Más pasos mejoran la integración natural",
                    "Experimenta con diferentes fondos"
                ],
                "Outpainting": [
                    "Describe coherentemente el contexto extendido",
                    "Usa prompts que incluyan la continuación natural",
                    "Esta técnica requiere más procesamiento",
                    "Ideal para expandir landscapes y fondos"
                ],
                "Style Transfer": [
                    "Ajusta strength según el efecto deseado",
                    "Estilos sutiles mantienen más del original",
                    "Experimenta con diferentes estilos artísticos",
                    "Para máximo impacto usa strength > 0.7"
                ],
                "Object Removal": [
                    "Describe específicamente el objeto que quieres eliminar",
                    "Usa términos claros (persona, carro, edificio, etc.)",
                    "Especifica el contexto del fondo deseado",
                    "Detección automática de la ubicación del objeto",
                    "Usa guidance scale alto (8.0-9.5) para mejor adherencia",
                    "Más pasos (40-50) mejoran la integración natural"
                ],
                "Background Replacement": [
                    "Selecciona fondos compatibles con el sujeto",
                    "Ajusta la iluminación si es necesario",
                    "Considera la perspectiva y escala",
                    "Usa prompts descriptivos del nuevo ambiente"
                ],
                "Composición": [
                    "Define claramente qué elementos combinar",
                    "Usa prompts que sugieran armonía visual",
                    "Strength controla la intensidad de cambios",
                    "Experimenta con diferentes estilos de composición"
                ]
            }
            
            for method_key, method_tips in tips.items():
                if method_key in method:
                    return method_tips
            
            return ["Usa parámetros moderados como punto de partida"]
            
        except Exception as e:
            logger.error(f"Error obteniendo consejos para {method}: {e}")
            return ["Error obteniendo consejos"]
    
    def health_check(self) -> Dict[str, Any]:
        """Verificar el estado de salud del UI Helper"""
        try:
            return {
                'status': 'healthy',
                'timestamp': time.time(),
                'methods_initialized': len(self.processing_methods),
                'capabilities': list(self.processing_methods.keys()) if self.processing_methods else [],
                'robust_mode': True
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': time.time()
            }

# Alias para compatibilidad
UIHelper = UIHelperRobust