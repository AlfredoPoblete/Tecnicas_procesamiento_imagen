import streamlit as st
import os
from PIL import Image
from typing import Optional, Tuple, Dict, Any
import time
from datetime import datetime

# ========== CARGA .ENV ==========
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
    print("✅ .env cargado (si existe)")
except Exception as e:
    print("⚠️ No se pudo cargar dotenv:", e)

# ========== IMPORTAR MÓDULOS PROYECTO ==========
from models.diffusion import DiffusionProcessor
from models.analysis import GeminiAnalyzer
from utils.image_utils import ImageProcessor
from utils.ui_utils import UIHelper

# ---------------------------------------------------
# CSS
# ---------------------------------------------------
def load_css():
    st.markdown("""
        <style>
        body { background-color: #1E0C2B; color: #EDE7F6; font-family: 'Poppins', sans-serif; }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .main-header { font-size: 2.5rem; font-weight:700; text-align:center;
            background: linear-gradient(90deg, #BB86FC, #03DAC6);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
        .section-header { color:#BB86FC; font-size:1.5rem; font-weight:600; margin-bottom:1rem; }
        .tech-chip { background:rgba(187,134,252,0.2); color:#BB86FC; padding:5px 10px;
            border-radius:20px; border:1px solid #BB86FC; font-size:.8rem; display:inline-block; }
        .stImage img { border-radius:8px; }
        </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------
# CONFIGURAR PÁGINA
# ---------------------------------------------------
def configure_page():
    st.set_page_config(
        page_title="Edición Generativa",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    load_css()


# ---------------------------------------------------
# APLICACIÓN PRINCIPAL
# ---------------------------------------------------
class ImageEditingApp:

    def __init__(self):
        # instancias
        self.diffusion = DiffusionProcessor()
        self.analyzer = GeminiAnalyzer()
        self.imgproc = ImageProcessor()
        self.ui = UIHelper()

        # inicializar session_state
        defaults = {
            "selected_method": "inpainting",
            "prompt": "",
            "strength": 0.7,
            "steps": 30,
            "guidance_scale": 7.5,
            "analysis_results": {},
            "processed_image": None
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    # ------------ CARGA DE IMAGEN ---------------
    def load_image(self, file) -> Optional[Image.Image]:
        try:
            if file:
                return Image.open(file).convert("RGB")
        except:
            st.error("No se pudo abrir la imagen.")
        return None

    # ------------ PROCESAMIENTO HF ---------------
    def run_processing(self, image: Image.Image, method: str, params: dict):
        st.info("⏳ Enviando imagen a HuggingFace...")
        result, meta = self.diffusion.process(
            image=image,
            method=method,
            prompt=params.get("prompt", ""),
            strength=params.get("strength", None),
            guidance_scale=params.get("guidance_scale", None),
            num_inference_steps=params.get("steps", None),
        )
        return result, meta

    # ---------------------------------------------------
    # SIDEBAR
    # ---------------------------------------------------
    def render_sidebar(self):
        with st.sidebar:
            st.markdown("## 📚 Instrucciones")
            st.write("""
            1. Subí tu imagen  
            2. Elegí un método  
            3. Ajustá parámetros  
            4. Procesá la imagen  
            5. Analizá con Gemini (opcional)
            """)

            st.markdown("---")
            if os.getenv("GOOGLE_API_KEY"):
                st.success("API Key de Gemini configurada")
            else:
                st.error("Falta GOOGLE_API_KEY en .env")

            st.markdown("---")
            st.markdown("## 🔧 Tecnologías")
            tech = ["Stable Diffusion (via HuggingFace API)", "Gemini 2.0", "Streamlit", "PIL", "Optimización en la nube"]
            for t in tech:
                st.markdown(f"<div class='tech-chip'>{t}</div>", unsafe_allow_html=True)

    # ---------------------------------------------------
    # HEADER
    # ---------------------------------------------------
    def render_header(self):
        st.markdown("<h1 class='main-header'>🎨 Edición Generativa de Imágenes</h1>", unsafe_allow_html=True)
        st.write("Plataforma avanzada para modificar imágenes con IA de manera intuitiva.")

    # ---------------------------------------------------
    # SECCIÓN DE CARGA
    # ---------------------------------------------------
    def render_upload_section(self):
        st.markdown("<h3 class='section-header'>📁 Cargar Imagen</h3>", unsafe_allow_html=True)

        uploaded = st.file_uploader("Subí una imagen", type=["jpg", "jpeg", "png"])
        if uploaded:
            img = self.load_image(uploaded)
            st.session_state["original_image"] = img
            st.session_state["uploaded_file"] = uploaded
            st.success("Imagen cargada correctamente")
            st.image(img, width=500)

    # ---------------------------------------------------
    # SECCIÓN DE PROCESAMIENTO  ⭐⭐⭐ REESCRITA COMPLETA ⭐⭐⭐
    # ---------------------------------------------------
    def render_processing_section(self):
        if "original_image" not in st.session_state:
            return
        
        st.markdown("<h3 class='section-header'>🎯 Procesamiento</h3>", unsafe_allow_html=True)

        col1, col2 = st.columns([2,1])
        with col1:

            # ---------------- MÉTODO -------------------
            method_name = st.selectbox(
                "Seleccioná el método",
                [
                    "Inpainting (Eliminar objetos)",
                    "Outpainting (Extender imagen)",
                    "Style Transfer",
                    "Object Removal",
                    "Background Replacement",
                    "Composición Inteligente"
                ]
            )

            # ----- Mapeo interno -----
            map_methods = {
                "inpainting": "inpainting",
                "outpainting": "outpainting",
                "style transfer": "style_transfer",
                "object removal": "object_removal",
                "background replacement": "background_replacement",
                "composición inteligente": "intelligent_composition"
            }
            mk = method_name.split("(")[0].strip().lower()
            mk = map_methods.get(mk, mk.replace(" ", "_"))
            st.session_state["selected_method"] = mk

            # ---------- PARÁMETROS ----------
            params = self.ui.get_processing_params(method_name)

            # ---------- BOTÓN DE PROCESAMIENTO ----------
            if st.button("🚀 Procesar Imagen"):
                img = st.session_state["original_image"]

                result, meta = self.run_processing(img, mk, params)

                if result is None:
                    st.error("❌ No se pudo procesar la imagen")
                    st.json(meta)
                else:
                    st.success("✅ Procesada con éxito")
                    st.image(result, width=500)
                    st.session_state["processed_image"] = result
                    st.session_state["processing_metadata"] = meta

        with col2:
            if "original_image" in st.session_state:
                st.image(st.session_state["original_image"], caption="Original", width=350)

    # ---------------------------------------------------
    # COMPARACIÓN
    # ---------------------------------------------------
    def render_comparison_section(self):
        if "processed_image" not in st.session_state:
            return

        st.markdown("<h3 class='section-header'>📊 Comparación</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.image(st.session_state["original_image"], caption="Original", width=400)

        with col2:
            st.image(st.session_state["processed_image"], caption="Procesada", width=400)

    # ---------------------------------------------------
    # ANÁLISIS GEMINI
    # ---------------------------------------------------
    def render_analysis_section(self):
        if "processed_image" not in st.session_state:
            return

        st.markdown("<h3 class='section-header'>🧠 Análisis (Gemini 2.0)</h3>", unsafe_allow_html=True)

        if st.button("🔍 Analizar resultado"):
            if not os.getenv("GOOGLE_API_KEY"):
                st.error("Falta GOOGLE_API_KEY en .env")
                return
            
            orig = st.session_state["original_image"]
            proc = st.session_state["processed_image"]

            st.info("Analizando con Gemini 2.0...")

            try:
                analysis = self.analyzer.analyze(
                    image=proc,
                    analysis_type="analysis"
                )
                st.session_state["analysis_results"] = analysis
                st.success("Análisis completo")

            except Exception as e:
                st.error(f"Error analizando con Gemini: {e}")

        if st.session_state["analysis_results"]:
            st.write(st.session_state["analysis_results"])

    # ---------------------------------------------------
    # RUN
    # ---------------------------------------------------
    def run(self):
        configure_page()
        self.render_sidebar()
        self.render_header()

        self.render_upload_section()
        self.render_processing_section()
        self.render_comparison_section()
        self.render_analysis_section()

        st.markdown(
            "<hr><center>💻 IFTS 24 — Proyecto Alfredo Poblete (2025)</center>",
            unsafe_allow_html=True
        )


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
def main():
    app = ImageEditingApp()
    app.run()

if __name__ == "__main__":
    main()
