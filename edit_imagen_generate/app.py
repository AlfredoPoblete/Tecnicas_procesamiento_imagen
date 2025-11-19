"""
🎨 Aplicación de Edición Generativa de Imágenes - VERSIÓN CONSOLIDADA
Tema oscuro optimizado para Streamlit Cloud
Compatible con Streamlit en la web sin problemas
"""

import streamlit as st
import os
from PIL import Image
from typing import Optional, Tuple, Dict, Any
import time
import warnings
from datetime import datetime

# Configuración de warnings
warnings.filterwarnings("ignore")

# Configuración de página
st.set_page_config(
    page_title="Edición Generativa de Imágenes",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar variables de entorno
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except:
    pass

# Importar módulos consolidados
from diffusion import DiffusionProcessor
from utils import UIHelper, ImageProcessor

# CSS optimizado para tema oscuro
def load_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        :root {
            --primary-bg: #1E0C2B;
            --sidebar-start: #2B1A55;
            --sidebar-end: #482880;
            --card-bg: rgba(255, 255, 255, 0.05);
            --border: #7E57C2;
            --purple: #BB86FC;
            --blue: #03DAC6;
            --dark-purple: #5E35B1;
            --text-primary: #EDE7F6;
            --text-secondary: #B39DDB;
            --shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        body {
            background-color: var(--primary-bg);
            color: var(--text-primary);
            font-family: 'Poppins', sans-serif;
        }
        
        /* Sidebar */
        .css-1d391kg { background: linear-gradient(180deg, var(--sidebar-start), var(--sidebar-end)); }
        
        /* Componentes */
        .stButton > button, .stSelectbox > div > div {
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-primary);
        }
        
        .stButton > button {
            background-color: var(--dark-purple);
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: var(--shadow);
        }
        
        .stButton > button:hover {
            background-color: var(--purple);
            transform: translateY(-2px);
        }
        
        /* Header */
        .main-header {
            background: linear-gradient(90deg, var(--purple), var(--blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        /* Cards */
        .stContainer {
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: var(--shadow);
        }
        
        /* File uploader */
        .css-1cpxqw2 {
            background-color: var(--card-bg);
            border: 2px dashed var(--purple);
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
        }
        
        /* Spinner */
        .stSpinner > div { border-top-color: var(--purple); }
        
        /* Success/Error */
        .stSuccess { background-color: rgba(76, 175, 80, 0.2); color: #4CAF50; }
        .stError { background-color: rgba(244, 67, 54, 0.2); color: #f44336; }
        </style>
        """,
        unsafe_allow_html=True
    )

class ImageEditingApp:
    """Aplicación principal optimizada para Streamlit Cloud"""
    
    def __init__(self):
        # Estado de carga lazy
        self.diffusion_processor = None
        self.ui_helper = None
        self.image_processor = None
        
        # Configurar variables de estado
        if 'analysis_results' not in st.session_state:
            st.session_state['analysis_results'] = {}
        
        # Configuración optimizada
        st.session_state['streamlit_optimized'] = True
        st.session_state['api_mode'] = False
        
        print("🚀 ImageEditingApp inicializado para Streamlit Cloud")
    
    def _ensure_diffusion_processor(self):
        """Carga lazy del procesador de difusión"""
        if self.diffusion_processor is None:
            with st.spinner("🔄 Cargando modelo de difusión..."):
                self.diffusion_processor = DiffusionProcessor()
            st.success("✅ Modelo cargado correctamente")
    
    def _ensure_ui_helper(self):
        """Carga lazy del helper de UI"""
        if self.ui_helper is None:
            self.ui_helper = UIHelper()
    
    def _ensure_image_processor(self):
        """Carga lazy del procesador de imágenes"""
        if self.image_processor is None:
            self.image_processor = ImageProcessor()
    
    def load_image(self, uploaded_file) -> Optional[Image.Image]:
        """Cargar imagen con validación robusta"""
        try:
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                return image.convert('RGB')
            return None
        except Exception as e:
            st.error(f"Error al cargar la imagen: {str(e)}")
            return None
    
    def process_image(self, image: Image.Image, method: str, **kwargs) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """Procesar imagen con manejo robusto de errores"""
        try:
            # Asegurar que el procesador esté cargado
            self._ensure_diffusion_processor()
            
            # Configurar mensajes de progreso
            method_messages = {
                "inpainting": "🎯 Aplicando inpainting...",
                "outpainting": "🔄 Realizando outpainting...",
                "style_transfer": "🎨 Transferiendo estilo...",
                "object_removal": "🗑️ Eliminando objeto...",
                "background_replacement": "🖼️ Cambiando fondo...",
                "intelligent_composition": "🧩 Creando composición..."
            }
            
            message = method_messages.get(method, f"🚀 Procesando imagen...")
            
            with st.spinner(message):
                result, metadata = self.diffusion_processor.process(
                    image=image,
                    method=method,
                    **kwargs
                )
                
                # Verificar resultado
                if result is None:
                    st.error("❌ Error en el procesamiento")
                    return None, {}
                
                # Mostrar información de procesamiento
                if 'processing_time' in metadata:
                    st.success(f"✅ Completado en {metadata['processing_time']}")
                
                return result, metadata
                
        except Exception as e:
            st.error(f"❌ Error procesando imagen: {str(e)}")
            return None, {"error": str(e)}
    
    def analyze_image(self, image: Image.Image) -> Dict[str, Any]:
        """Análisis simplificado de imagen"""
        try:
            self._ensure_image_processor()
            
            # Análisis básico usando métricas de imagen
            metrics = self.image_processor.calculate_image_metrics(image)
            
            # Crear análisis simplificado
            analysis = {
                "success": True,
                "analysis": f"Imagen de {metrics.get('width', 'N/A')}x{metrics.get('height', 'N/A')} píxeles",
                "quality": "Buena calidad" if metrics.get('contrast', 0) > 100 else "Calidad media",
                "brightness": f"{metrics.get('mean_brightness', 0):.1f}",
                "metrics": metrics
            }
            
            return analysis
            
        except Exception as e:
            return {"error": f"Error en análisis: {str(e)}"}
    
    def render_sidebar(self):
        """Renderizar panel lateral"""
        with st.sidebar:
            st.markdown('<h2 style="color: white;">📚 Guía de Uso</h2>', unsafe_allow_html=True)
            
            st.markdown(
                """
                <div style="color: white; line-height: 1.6;">
                <h4>Pasos para usar:</h4>
                <ol>
                    <li><strong>Subí tu imagen</strong> en JPG o PNG</li>
                    <li><strong>Seleccioná el método</strong> de procesamiento</li>
                    <li><strong>Ajustá los parámetros</strong> según necesidades</li>
                    <li><strong>Procesá</strong> la imagen</li>
                    <li><strong>Analizá</strong> los resultados</li>
                </ol>
                
                <h4>⚙️ Métodos disponibles:</h4>
                <ul>
                    <li><strong>Inpainting:</strong> Eliminar objetos</li>
                    <li><strong>Outpainting:</strong> Extender imagen</li>
                    <li><strong>Style Transfer:</strong> Cambiar estilo</li>
                    <li><strong>Object Removal:</strong> Limpiar elementos</li>
                </ul>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Estado del sistema
            st.markdown('<hr style="border-color: #7E57C2;">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: white;">⚡ Estado del Sistema</h3>', unsafe_allow_html=True)
            
            if self.diffusion_processor is not None:
                info = self.diffusion_processor.get_info()
                if info.get('device') == 'cuda':
                    st.success("🚀 GPU disponible")
                else:
                    st.info("💻 Modo CPU")
                
                st.info("✅ Sistema optimizado para Streamlit Cloud")
            else:
                st.info("💻 Modelos se cargarán bajo demanda")
    
    def render_header(self):
        """Renderizar encabezado principal"""
        st.markdown(
            '<h1 class="main-header">🎨 Edición Generativa de Imágenes</h1>',
            unsafe_allow_html=True
        )
        
        st.markdown(
            """
            <div style="text-align: center; color: var(--text-secondary); font-size: 1.1rem; margin-bottom: 2rem;">
                Plataforma avanzada de edición con IA que integra técnicas de difusión para:
                <span style="color: var(--purple); font-weight: 500;">Inpainting</span>,
                <span style="color: var(--purple); font-weight: 500;">Style Transfer</span>,
                <span style="color: var(--purple); font-weight: 500;">Object Removal</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def render_upload_section(self):
        """Sección de carga de imagen"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<h3 class="section-header">📁 Carga de Imagen</h3>', unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Selecciona una imagen para comenzar", 
                type=['jpg', 'jpeg', 'png'],
                help="Formatos soportados: JPG, JPEG, PNG",
                key="file_uploader_main"
            )
            
            if uploaded_file:
                image = self.load_image(uploaded_file)
                if image:
                    st.session_state['original_image'] = image
                    st.session_state['uploaded_file'] = uploaded_file
                    st.success("✅ Imagen cargada exitosamente")
        
        with col2:
            if 'original_image' in st.session_state:
                image = st.session_state['original_image']
                st.markdown('<h3 class="section-header">📐 Info de la Imagen</h3>', unsafe_allow_html=True)
                
                st.markdown(
                    f"""
                    <div style="background: rgba(187, 134, 252, 0.1); padding: 1rem; border-radius: 8px;">
                        <strong>📐 Información:</strong><br>
                        • Tamaño: {image.size[0]} × {image.size[1]} píxeles<br>
                        • Formato: {image.format or 'RGB'}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    def render_processing_section(self):
        """Sección de procesamiento"""
        if 'original_image' not in st.session_state:
            st.info("👆 Sube una imagen para comenzar")
            return
        
        st.markdown('<h3 class="section-header">🎯 Procesamiento</h3>', unsafe_allow_html=True)
        
        # Asegurar que el UI helper esté cargado
        self._ensure_ui_helper()
        
        # Selector de método
        col1, col2 = st.columns([2, 1])
        
        with col1:
            processing_method = st.selectbox(
                "Selecciona el método de procesamiento",
                [
                    "Inpainting (Eliminar objetos)",
                    "Outpainting (Extender imagen)", 
                    "Style Transfer (Transferir estilo)",
                    "Object Removal (Eliminar objeto específico)",
                    "Background Replacement (Cambiar fondo)",
                    "Composición Inteligente (Combinar elementos)"
                ],
                key="method_selector"
            )
            
            # Obtener parámetros para el método seleccionado
            params = self.ui_helper.get_processing_params(processing_method)
            
            # Botón de procesamiento
            if st.button("🚀 Procesar Imagen", key="process_button", type="primary"):
                # Mapear método
                method_mapping = {
                    "inpainting": "inpainting",
                    "outpainting": "outpainting", 
                    "style transfer": "style_transfer",
                    "object removal": "object_removal",
                    "background replacement": "background_replacement",
                    "composición inteligente": "intelligent_composition"
                }
                
                method_key = processing_method.split(' (')[0].lower()
                if method_key in method_mapping:
                    method_key = method_mapping[method_key]
                else:
                    method_key = method_key.replace(' ', '_')
                
                # Procesar imagen
                result, metadata = self.process_image(
                    st.session_state['original_image'], 
                    method_key, 
                    **params
                )
                
                if result:
                    st.session_state['processed_image'] = result
                    st.session_state['processing_metadata'] = metadata
                    st.success("✅ Imagen procesada exitosamente")
        
        with col2:
            # Mostrar imagen original
            if 'original_image' in st.session_state:
                st.image(st.session_state['original_image'], width=400, caption="Imagen Original")
    
    def render_comparison_section(self):
        """Sección de comparación"""
        if 'original_image' not in st.session_state:
            return
        
        st.markdown('<h3 class="section-header">📊 Comparación</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🖼️ Original**")
            st.image(st.session_state['original_image'], width=400)
        
        with col2:
            if 'processed_image' in st.session_state:
                st.markdown("**✨ Procesada**")
                st.image(st.session_state['processed_image'], width=400)
            else:
                st.markdown(
                    '<div style="background: var(--card-bg); border: 2px dashed var(--card-border); border-radius: 8px; padding: 2rem; text-align: center; color: var(--text-secondary);">Procesa una imagen para ver la comparación</div>',
                    unsafe_allow_html=True
                )
    
    def render_analysis_section(self):
        """Sección de análisis"""
        if 'processed_image' not in st.session_state:
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("🔍 Analizar Resultados", key="analysis_button", type="primary"):
                analysis = self.analyze_image(st.session_state['processed_image'])
                st.session_state['analysis'] = analysis
                st.success("✅ Análisis completado")
        
        with col2:
            if 'analysis' in st.session_state:
                analysis = st.session_state['analysis']
                
                if 'error' in analysis:
                    st.error(f"Error en análisis: {analysis['error']}")
                else:
                    st.markdown("### 🔍 Análisis de Resultados")
                    st.write(analysis.get('analysis', ''))
                    
                    if 'quality' in analysis:
                        st.write(f"**Calidad:** {analysis['quality']}")
                    
                    if 'metrics' in analysis:
                        metrics = analysis['metrics']
                        st.write(f"**Brillo promedio:** {metrics.get('brightness', 'N/A')}")
                        st.write(f"**Contraste:** {metrics.get('contrast', 'N/A')}")
    
    def render_footer(self):
        """Renderizar footer"""
        st.markdown(
            """
            <div style="text-align: center; margin-top: 3rem; padding: 2rem; border-top: 1px solid var(--card-border);">
                <div style="color: var(--text-secondary);">
                    💻 Procesamiento Digital de Imágenes - IFTS24<br>
                    🚀 Aplicación Optimizada para Streamlit Cloud<br>
                    ⚡ Alfredo Poblete - 2025
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def run(self):
        """Ejecutar aplicación"""
        # Cargar CSS
        load_css()
        
        # Renderizar componentes
        self.render_sidebar()
        self.render_header()
        
        # Contenedor principal
        with st.container():
            self.render_upload_section()
            self.render_processing_section()
            self.render_comparison_section()
            self.render_analysis_section()
        
        # Footer
        self.render_footer()

# Función principal
def main():
    """Función principal de la aplicación"""
    try:
        app = ImageEditingApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Error crítico en la aplicación: {str(e)}")
        st.info("💡 Si el problema persiste, recarga la página o contacta al soporte")

if __name__ == "__main__":
    main()