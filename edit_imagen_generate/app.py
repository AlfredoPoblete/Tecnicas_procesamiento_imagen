"""
Aplicación de Edición Generativa de Imágenes - Versión Final Simplificada
Tema oscuro con interfaz optimizada, carga lazy de modelos y API Key desde .env

Procesamiento Digital de Imágenes - IFTS24
"""

import streamlit as st
import os
from PIL import Image
from typing import Optional, Tuple, Dict, Any
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# Cargar variables de entorno usando python-dotenv (más robusto y estándar)
try:
    from dotenv import load_dotenv
    # load_dotenv carga variables desde .env en el directorio actual
    # override=False evita sobreescribir variables ya definidas en el sistema
    load_dotenv(override=False)
    print("✅ .env cargado con python-dotenv (si existe)")
except Exception as e:
    # No fallamos la ejecución si python-dotenv no está instalado
    print(f"⚠️ No se pudo cargar python-dotenv: {e}")

# Importar módulos del proyecto
from models.diffusion import DiffusionProcessor
from models.analysis import GeminiAnalyzer
from utils.image_utils import ImageProcessor
from utils.ui_utils import UIHelper

def load_css():
    """Cargar estilos CSS optimizados para el tema oscuro moderno"""
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
        
        /* Sidebar con gradiente */
        .css-1d391kg { background: linear-gradient(180deg, var(--sidebar-start), var(--sidebar-end)); }
        .css-1v3fvcr { background-color: transparent; }
        .css-1v3fvcr .sidebar-content { background: transparent; color: white; }
        
        /* Componentes base */
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
            box-shadow: 0 12px 40px rgba(126, 87, 194, 0.4);
        }
        
        /* File uploader */
        .css-1cpxqw2 {
            background-color: var(--card-bg);
            border: 2px dashed var(--purple);
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .css-1cpxqw2:hover {
            border-color: var(--purple);
            background-color: rgba(187, 134, 252, 0.1);
        }
        
        /* Sliders */
        .stSlider > div > div > div { background-color: var(--purple); }
        .stSlider > div > div { background-color: var(--card-bg); }
        
        /* Contenedores */
        .stContainer {
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: var(--shadow);
        }
        
        /* Títulos */
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
        
        .section-header {
            color: var(--purple);
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        
        /* Otros elementos */
        .stImage { border-radius: 8px; box-shadow: var(--shadow); }
        .stSpinner > div { border-top-color: var(--purple); }
        .stAlert { border-radius: 8px; border: none; }
        .stSuccess { background-color: rgba(76, 175, 80, 0.2); color: #4CAF50; }
        .stError { background-color: rgba(244, 67, 54, 0.2); color: #f44336; }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        
        /* Chips y texto */
        .tech-chip {
            background: rgba(187, 134, 252, 0.2);
            color: var(--purple);
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            display: inline-block;
            margin: 0.25rem;
            border: 1px solid var(--purple);
        }
        
        /* Animaciones */
        .fade-in { animation: fadeIn 0.5s ease-in; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        
        /* Responsive */
        @media (max-width: 768px) {
            .main-header { font-size: 2rem; }
            .block-container { padding-top: 1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def configure_page():
    """Configurar página Streamlit"""
    st.set_page_config(
        page_title="Edición Generativa de Imágenes",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Cargar CSS personalizado
    load_css()

class ImageEditingApp:
    """Aplicación principal para edición generativa de imágenes - Versión OPTIMIZADA para Streamlit"""
    
    def __init__(self):
        # Configuración de optimizaciones
        self.streamlit_mode = os.getenv('STREAMLIT_APP', 'false').lower() == 'true'
        self.use_hf_api = os.getenv('USE_HF_API', 'false').lower() == 'true'
        
        # Inicializar con indicador de carga
        with st.spinner("🚀 Inicializando optimizaciones para Streamlit..."):
            # Usar el modelo optimizado
            self.diffusion_processor = DiffusionProcessor()
            self.analyzer = GeminiAnalyzer()
            self.image_processor = ImageProcessor()
            self.ui_helper = UIHelper()
        
        # Configurar variables de estado
        if 'analysis_results' not in st.session_state:
            st.session_state['analysis_results'] = {}
        
        # Configurar estado de optimización
        st.session_state['optimization_enabled'] = True
        st.session_state['lightweight_mode'] = True
        st.session_state['api_mode'] = self.use_hf_api
    
    def resize_images_to_same_size(self, img1: Image.Image, img2: Image.Image, target_width: int = 400) -> Tuple[Image.Image, Image.Image]:
        """Redimensionar dos imágenes para que tengan exactamente el mismo tamaño manteniendo proporciones"""
        # Obtener dimensiones originales
        w1, h1 = img1.size
        w2, h2 = img2.size
        
        # Calcular el factor de escala para cada imagen basado en el ancho objetivo
        scale1 = target_width / w1
        new_h1 = int(h1 * scale1)
        
        scale2 = target_width / w2
        new_h2 = int(h2 * scale2)
        
        # Redimensionar ambas imágenes
        img1_resized = img1.resize((target_width, new_h1), Image.Resampling.LANCZOS)
        img2_resized = img2.resize((target_width, new_h2), Image.Resampling.LANCZOS)
        
        # Crear un canvas con la altura máxima y centrar cada imagen
        max_height = max(new_h1, new_h2)
        
        # Canvas para imagen 1 (centrada verticalmente)
        canvas1 = Image.new('RGB', (target_width, max_height), (255, 255, 255))
        y_offset1 = (max_height - new_h1) // 2
        canvas1.paste(img1_resized, (0, y_offset1))
        
        # Canvas para imagen 2 (centrada verticalmente)
        canvas2 = Image.new('RGB', (target_width, max_height), (255, 255, 255))
        y_offset2 = (max_height - new_h2) // 2
        canvas2.paste(img2_resized, (0, y_offset2))
        
        return canvas1, canvas2
        
    def load_image(self, uploaded_file) -> Optional[Image.Image]:
        """Cargar imagen desde archivo subido"""
        try:
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                return image.convert('RGB')
            return None
        except Exception as e:
            st.error(f"Error al cargar la imagen: {str(e)}")
            return None
    
    def process_image(self, image: Image.Image, method: str, **kwargs) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """Procesar imagen usando modelos de difusión optimizados para Streamlit"""
        try:
            # Mostrar progreso específico para cada método
            method_descriptions = {
                "inpainting": "🎯 Inpainting: Eliminando y rellenando objetos...",
                "outpainting": "🔄 Outpainting: Extendiendo imagen...",
                "style_transfer": "🎨 Style Transfer: Aplicando estilo artístico...",
                "object_removal": "🗑️ Object Removal: Eliminando objetos...",
                "background_replacement": "🖼️ Background: Cambiando fondo...",
                "intelligent_composition": "🧩 Composición: Creando composición inteligente..."
            }
            
            description = method_descriptions.get(method, f'🚀 Procesando imagen con {method}...')
            
            # Spinner con mensaje específico
            with st.spinner(description):
                # Verificar si usar API de Hugging Face
                if hasattr(self, 'use_hf_api') and self.use_hf_api:
                    st.info("🌐 Usando API de Hugging Face (sin carga de modelos locales)")
                
                # Mostrar configuración optimizada
                if method in ['inpainting', 'outpainting']:
                    steps = kwargs.get('num_inference_steps', 20)
                    st.info(f"⚡ Modo optimizado: {steps} pasos (reducido para velocidad)")
                
                result, metadata = self.diffusion_processor.process(
                    image=image,
                    method=method,
                    **kwargs
                )
                
                # Mostrar métricas de optimización
                if result and metadata:
                    processing_time = metadata.get('processing_time', 'N/A')
                    device = metadata.get('device', 'unknown')
                    
                    st.success(f"✅ Procesamiento completado en {processing_time}")
                    if 'api_processed' in metadata:
                        st.info("🌐 Procesado vía API - Sin consumo de memoria local")
                    elif device == 'cuda':
                        st.info("🚀 Acelerado por GPU")
                    else:
                        st.info("💻 Procesado en CPU - Modo de bajo consumo")
            
            return result, metadata
        except Exception as e:
            st.error(f"Error en el procesamiento: {str(e)}")
            # Mostrar sugerencia de optimización
            st.warning("💡 Tip: Si el error persiste, intenta reducir el tamaño de la imagen")
            return None, {}
    
    def analyze_image(self, image: Image.Image, analysis_type: str = "comparison_analysis") -> Dict[str, Any]:
        """Analizar imagen usando Gemini 2.0 con análisis comparativo entre original y procesada"""
        try:
            # Verificar si hay API key configurada
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                st.error("❌ No se encontró API key de Gemini configurada")
                return {"error": "API key no configurada"}
            
            # Obtener imagen original si existe
            original_image = st.session_state.get('original_image')
            
            # Obtener información de procesamiento si está disponible
            processing_info = st.session_state.get('processing_metadata', {})
            method = processing_info.get('method', 'Desconocida')
            steps = processing_info.get('steps', 'N/A')
            guidance_scale = processing_info.get('guidance_scale', 'N/A')
            
            # Crear descripción específica basada en la técnica aplicada
            method_descriptions = {
                "inpainting": "técnica de inpainting (eliminación y relleno inteligente de objetos)",
                "outpainting": "técnica de outpainting (extensión de imagen más allá de sus bordes)",
                "style_transfer": "técnica de transferencia de estilo artístico",
                "object_removal": "técnica de eliminación específica de objetos",
                "background_replacement": "técnica de reemplazo de fondo manteniendo el sujeto",
                "intelligent_composition": "técnica de composición inteligente combinando elementos"
            }
            
            method_description = method_descriptions.get(method, "técnica de procesamiento de imágenes con IA")
            
            with st.spinner('🧠 Analizando imagen'):
                # Si hay imagen original, usar análisis comparativo
                if original_image and analysis_type == "comparison_analysis":
                    analysis = self.analyzer.analyze(
                        image=image,
                        analysis_type="comparison_brief",
                        original_image=original_image
                    )
                else:
                    # Usar análisis estándar si no hay imagen original
                    analysis = self.analyzer.analyze(
                        image=image,
                        analysis_type=analysis_type
                    )
            
            # Añadir información específica sobre el procesamiento al análisis
            if 'error' not in analysis:
                analysis['processing_details'] = {
                    'tecnica_aplicada': method_description,
                    'pasos_procesamiento': steps,
                    'guidance_scale': guidance_scale,
                    'device_used': processing_info.get('device', 'CPU/GPU'),
                    'optimized': processing_info.get('optimized', False),
                    'processing_time': processing_info.get('processing_time', 'N/A')
                }
                
                # Si es análisis comparativo, agregar información específica
                if 'brief_analysis' in analysis:
                    analysis['analysis_type'] = 'comparative_brief'
                    analysis['comparison_available'] = True
                else:
                    analysis['comparison_available'] = False
            
            return analysis
        except Exception as e:
            st.error(f"Error en el análisis: {str(e)}")
            return {"error": str(e)}
    
    def render_sidebar(self):
        """Renderizar panel lateral con instrucciones y tecnologías"""
        with st.sidebar:
            st.markdown('<h2 style="color: white;">📚 Instrucciones de Uso</h2>', unsafe_allow_html=True)
            
            st.markdown(
                """
                <div style="color: white; line-height: 1.6;">
                <h4>Pasos para usar la aplicación:</h4>
                <ol>
                    <li><strong>Prepará tu imagen</strong> en formato JPG o PNG</li>
                    <li><strong>Subí la imagen</strong> usando el botón de carga</li>
                    <li><strong>Elegí el método</strong> de procesamiento</li>
                    <li><strong>Ajustá los parámetros</strong> según el resultado deseado</li>
                    <li><strong>Procesá la imagen</strong> y analizá los resultados</li>
                    <li><strong>Iterá</strong> si es necesario para mejorar el resultado</li>
                </ol>
                
                <h4>⚙️ Parámetros importantes:</h4>
                <ul>
                    <li><strong>Steps:</strong> Más pasos = mejor calidad pero más lento</li>
                    <li><strong>Guidance Scale:</strong> Controla adherencia al prompt</li>
                    <li><strong>Strength:</strong> Intensidad de la transformación</li>
                </ul>
                
                <h4>🔑 API Configuration:</h4>
                <p>La API key de Gemini se configura automáticamente desde el archivo .env</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            st.markdown('<hr style="border-color: #7E57C2; margin: 2rem 0;">', unsafe_allow_html=True)
            
            # Verificar estado de la API key
            api_key = os.getenv('GOOGLE_API_KEY')
            if api_key:
                st.success("✅ API Key de Gemini configurada")
            else:
                st.error("❌ API Key de Gemini no configurada en .env")
            
            st.markdown('<hr style="border-color: #7E57C2; margin: 2rem 0;">', unsafe_allow_html=True)
            
            # Información de rendimiento OPTIMIZADA para Streamlit
            st.markdown('<h3 style="color: white;">⚡ Estado del Sistema Optimizado</h3>', unsafe_allow_html=True)
            
            # Verificar modo de API
            if hasattr(self, 'use_hf_api') and self.use_hf_api:
                st.success("🚀 API de Hugging Face habilitada - Modelos ligeros")
                st.info("⚡ Procesamiento vía cloud - Sin carga local")
            else:
                if hasattr(self.diffusion_processor, 'get_info'):
                    info = self.diffusion_processor.get_info()
                    device = info.get('device', 'unknown')
                    
                    if device == 'cuda':
                        st.success("🚀 GPU disponible - Procesamiento acelerado")
                    else:
                        st.warning("🖥️ CPU solamente - Modo de bajo consumo activado")
                
                lazy_loading = info.get('lazy_loading_enabled', False)
                if lazy_loading:
                    st.info("⚡ Carga lazy habilitada - Inicialización rápida")
            
            # Mostrar optimizaciones específicas para Streamlit
            st.success("🎯 Modo optimizado para Streamlit habilitado")
            st.info(f"📐 Resolución máxima: 256px (reducida para velocidad)")
            st.info(f"⚙️ Parámetros optimizados: pasos reducidos")
            
            if 'optimization_enabled' in st.session_state:
                st.success("✅ Optimizaciones Streamlit: ACTIVAS")
            
            st.markdown('<hr style="border-color: #7E57C2; margin: 2rem 0;">', unsafe_allow_html=True)
            
            # Tecnologías utilizadas
            st.markdown('<h3 style="color: white;">🔧 Tecnologías Utilizadas</h3>', unsafe_allow_html=True)
            
            technologies = [
                ("Modelos de Difusión", "Stable Diffusion, ControlNet"),
                ("Análisis Visual", "Gemini 2.0"),
                ("Interfaz", "Streamlit"),
                ("Procesamiento", "PIL, NumPy"),
                ("Optimización", "Carga Lazy, GPU Acelerada")
            ]
            
            for tech_name, tech_desc in technologies:
                st.markdown(f"""
                <div style="margin-bottom: 1rem;">
                    <span class="tech-chip">{tech_name}</span>
                    <div style="color: #B39DDB; font-size: 0.85rem; margin-top: 0.25rem;">
                        {tech_desc}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    def render_header(self):
        """Renderizar encabezado principal con capacidades"""
        st.markdown(
            '<h1 class="main-header fade-in">🎨 Edición Generativa de Imágenes</h1>',
            unsafe_allow_html=True
        )
        
        st.markdown(
            """
            <div style="text-align: center; color: var(--text-secondary); font-size: 1.1rem; margin-bottom: 2rem;">
                Plataforma avanzada que integra las últimas tecnologías de IA para ofrecer capacidades
                de edición generativa: <span style="color: var(--accent-purple); font-weight: 500;">Inpainting</span>,
                <span style="color: var(--accent-purple); font-weight: 500;">Style Transfer</span>,
                <span style="color: var(--accent-purple); font-weight: 500;">Object Removal</span>
                y análisis inteligente con Gemini 2.0
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Capacidades principales
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 2rem;">
                <h3 style="color: var(--accent-purple); margin-bottom: 1rem;">Capacidades principales:</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; max-width: 800px; margin: 0 auto;">
                    <div style="background: var(--card-bg); padding: 1rem; border-radius: 8px; border: 1px solid var(--card-border);">
                        <div style="color: var(--accent-purple); font-size: 1.5rem;">🖼️</div>
                        <strong>Inpainting:</strong> Eliminar y rellenar objetos de forma inteligente
                    </div>
                    <div style="background: var(--card-bg); padding: 1rem; border-radius: 8px; border: 1px solid var(--card-border);">
                        <div style="color: var(--accent-purple); font-size: 1.5rem;">🔄</div>
                        <strong>Outpainting:</strong> Extender imágenes más allá de sus bordes
                    </div>
                    <div style="background: var(--card-bg); padding: 1rem; border-radius: 8px; border: 1px solid var(--card-border);">
                        <div style="color: var(--accent-purple); font-size: 1.5rem;">🎭</div>
                        <strong>Style Transfer:</strong> Transferir estilos artísticos
                    </div>
                    <div style="background: var(--card-bg); padding: 1rem; border-radius: 8px; border: 1px solid var(--card-border);">
                        <div style="color: var(--accent-purple); font-size: 1.5rem;">🗑️</div>
                        <strong>Object Removal:</strong> Eliminar objetos no deseados
                    </div>
                    <div style="background: var(--card-bg); padding: 1rem; border-radius: 8px; border: 1px solid var(--card-border);">
                        <div style="color: var(--accent-purple); font-size: 1.5rem;">🖼️</div>
                        <strong>Background Replacement:</strong> Cambiar fondos manteniendo sujetos
                    </div>
                    <div style="background: var(--card-bg); padding: 1rem; border-radius: 8px; border: 1px solid var(--card-border);">
                        <div style="color: var(--accent-purple); font-size: 1.5rem;">🧠</div>
                        <strong>Análisis Inteligente:</strong> Análisis visual con Gemini 2.0
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def render_upload_section(self):
        """Renderizar sección de carga de imagen"""
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
                    st.session_state['uploaded_file'] = uploaded_file  # Guardar para obtener formato
                    st.success("✅ Imagen cargada exitosamente")
            
            # Galería de ejemplos debajo del uploader
            examples_dir = os.path.join('assets', 'ejemplos')
            if os.path.isdir(examples_dir):
                st.markdown("<div style='margin-top:1rem'><strong>🖼️ Ejemplos rápidos:</strong></div>", unsafe_allow_html=True)
                example_files = [f for f in os.listdir(examples_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if example_files:
                    # Limitar columnas a máximo 4 para buena visualización
                    cols = st.columns(min(len(example_files), 4))
                    for idx, fname in enumerate(example_files):
                        col = cols[idx % len(cols)]
                        with col:
                            try:
                                img_path = os.path.join(examples_dir, fname)
                                img = Image.open(img_path)
                                # Crear miniatura uniforme
                                thumb = img.copy()
                                thumb.thumbnail((160,160), Image.Resampling.LANCZOS)
                                canvas = Image.new('RGB', (160,160), (255,255,255))
                                x_off = (160 - thumb.size[0]) // 2
                                y_off = (160 - thumb.size[1]) // 2
                                canvas.paste(thumb, (x_off, y_off))
                                st.image(canvas, caption=fname, width=160)
                                if st.button(f"Usar {fname}", key=f"example_{idx}"):
                                    st.session_state['original_image'] = img.convert('RGB')
                                    st.session_state['uploaded_file'] = None
                                    st.success(f"✅ Ejemplo '{fname}' seleccionado")
                            except Exception as e:
                                st.error(f"Error leyendo {fname}: {e}")
        
        with col2:
            st.markdown('<h3 class="section-header">📐 Información de la imagen</h3>', unsafe_allow_html=True)
            
            if 'original_image' in st.session_state:
                image = st.session_state['original_image']
                # Obtener formato del archivo original
                uploaded_file = st.session_state.get('uploaded_file')
                file_format = uploaded_file.name.split('.')[-1].upper() if uploaded_file else 'RGB'
                
                st.markdown(
                    f"""
                    <div style="background: rgba(187, 134, 252, 0.1); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                        <strong>📐 Información de la imagen:</strong><br>
                        • Tamaño: {image.size[0]} × {image.size[1]} píxeles<br>
                        • Formato: {file_format}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    def render_processing_section(self):
        """Renderizar sección de procesamiento"""
        if 'original_image' not in st.session_state:
            return
        
        st.markdown('<h3 class="section-header">🎯 Procesamiento</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Selector de método
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
            
            # Parámetros según el método
            params = self.ui_helper.get_processing_params(processing_method)
            
            # Botón de procesamiento
            if st.button("🚀 Procesar Imagen", key="process_button", type="primary"):
                # Mapear métodos
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
                
                result, metadata = self.process_image(st.session_state['original_image'], method_key, **params)
                
                if result:
                    st.session_state['processed_image'] = result
                    st.session_state['processing_metadata'] = metadata
                    st.success("✅ Imagen procesada exitosamente")
                    
                    # Mostrar métricas si están disponibles
                    if 'processing_time' in metadata:
                        st.info(f"⏱️ Tiempo de procesamiento: {metadata['processing_time']}")
                
        with col2:
            # Imagen original (sin texto adicional)
            st.image(st.session_state['original_image'], width=500)
    
    def render_comparison_section(self):
        """Renderizar sección de comparación con redimensionamiento automático"""
        if 'original_image' not in st.session_state:
            return
        
        st.markdown('<h3 class="section-header">📊 Comparación Original vs Procesada</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Imagen original sin título
            st.image(st.session_state['original_image'], width=400)
        
        with col2:
            if 'processed_image' in st.session_state:
                # Redimensionar automáticamente para que ambas imágenes tengan el mismo tamaño
                original = st.session_state['original_image']
                processed = st.session_state['processed_image']
                
                # Redimensionar ambas imágenes al mismo tamaño
                original_resized, processed_resized = self.resize_images_to_same_size(original, processed, 400)
                
                # Mostrar la imagen procesada redimensionada
                st.image(processed_resized, width=400)
            else:
                st.markdown(
                    '<div style="background: var(--card-bg); border: 2px dashed var(--card-border); border-radius: 8px; padding: 2rem; text-align: center; color: var(--text-secondary);">Procesa una imagen para ver la comparación</div>',
                    unsafe_allow_html=True
                )
    
    def render_analysis_section(self):
        """Renderizar sección de análisis simplificada con Gemini 2.0"""
        if 'processed_image' not in st.session_state:
            return
        
        # Solo el botón (sin input de API key)
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("🔍 Análisis de resultados", key="analysis_button", type="primary"):
                # Verificar API key antes de proceder
                api_key = os.getenv('GOOGLE_API_KEY')
                if not api_key:
                    st.error("❌ API Key de Gemini no configurada en archivo .env")
                else:
                    try:
                        # Obtener imágenes
                        original = st.session_state.get('original_image')
                        processed = st.session_state.get('processed_image')

                        if not original or not processed:
                            st.error("❌ No hay imágenes disponibles para analizar")
                            return

                        # Información del procesamiento aplicado
                        processing_info = st.session_state.get('processing_metadata', {})
                        method = processing_info.get('method', 'Desconocida')
                        
                        # Prompt detallado basado en el método aplicado
                        method_descriptions = {
                            "inpainting": "eliminación y relleno inteligente de objetos",
                            "outpainting": "extensión de imagen más allá de sus bordes",
                            "style_transfer": "transferencia de estilo artístico",
                            "object_removal": "eliminación específica de objetos",
                            "background_replacement": "reemplazo de fondo manteniendo el sujeto",
                            "intelligent_composition": "composición inteligente combinando elementos"
                        }
                        
                        method_desc = method_descriptions.get(method, "procesamiento de imágenes con IA")
                        
                        # Prompt estructurado para obtener análisis detallado
                        prompt = f"""
                        Analiza estas dos imágenes: la primera es la ORIGINAL y la segunda es la PROCESADA con técnica de {method_desc}.
                        
                        Proporciona un análisis comparativo detallado en los siguientes puntos:
                        
                        1. **Cambios principales**: Resume los cambios observados entre ambas imágenes
                        2. **Calidad del procesamiento**: Evalúa la calidad de la transformación (precisión, bordes, coherencia)
                        3. **Artefactos o problemas**: Identifica artefactos, distorsiones o problemas visuales si los hay
                        4. **Fidelidad**: Evalúa cómo de fiel o realista es el resultado
                        5. **Recomendaciones**: Sugiere mejoras o ajustes para próximos intentos
                        
                        Sé específico y técnico en tu análisis. Usa observaciones concretas sobre colores, texturas, formas, detalles finos, etc.
                        """

                        # Crear copias y reducir resolución segura para envío (máx 1024)
                        orig_copy = original.copy()
                        proc_copy = processed.copy()
                        orig_copy.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
                        proc_copy.thumbnail([1024, 1024], Image.Resampling.LANCZOS)

                        # Usar el método del analyzer basado en Gemini2_espacial.ipynb
                        with st.spinner('🧠 Analizando con Gemini 2.0'):
                            try:
                                # Llamar al método mejorado que usa el SDK oficial
                                result = self.analyzer.analyze_comparison_with_genai_sdk(
                                    orig_copy, proc_copy, prompt
                                )
                                
                                if result and result.get('success'):
                                    # Agregar información de procesamiento
                                    result['processing_details'] = processing_info
                                    st.session_state['processed_analysis'] = result
                                    st.success("✅ Análisis completado")
                                else:
                                    st.error("❌ La respuesta de Gemini no fue válida")
                                    
                            except Exception as e:
                                print(f"Error con analyze_comparison_with_genai_sdk: {e}")
                                import traceback
                                traceback.print_exc()
                                st.error(f"❌ Error al conectar con Gemini: {str(e)}")

                    except Exception as e:
                        import traceback
                        print(traceback.format_exc())
                        st.error(f"❌ Excepción durante el análisis: {str(e)}")
        
        # Mostrar resultados del análisis (solo texto del análisis de Gemini)
        if 'processed_analysis' in st.session_state:
            analysis = st.session_state['processed_analysis']

            # Priorizar 'brief_analysis', si no existe usar 'changes_description' o todo el dict como fallback
            analysis_text = analysis.get('brief_analysis') or analysis.get('changes_description') or str(analysis)

            if analysis_text:
                # Mostrar texto crudo recibido de Gemini (sin tabs ni encabezados extra)
                st.markdown(analysis_text)
            else:
                st.write(analysis)

            # Botón simple para descargar el texto completo del análisis
            export_text = f"ANÁLISIS GEMINI\n\n{analysis_text}\n"
            st.download_button(
                label="📥 Descargar análisis",
                data=export_text,
                file_name=f"analisis_gemini_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="download_analysis"
            )
    
    def run(self):
        """Ejecutar aplicación principal"""
        configure_page()
        
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
        st.markdown(
            """
            <div style="text-align: center; margin-top: 3rem; padding: 2rem; border-top: 1px solid var(--card-border);">
                <div style="color: var(--text-secondary);">
                    💻 Procesamiento Digital de Imágenes - IFTS24<br>
                    ⚡ Alfredo Poblete - 2025
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def main():
    """Función principal"""
    app = ImageEditingApp()
    app.run()

if __name__ == "__main__":
    main()