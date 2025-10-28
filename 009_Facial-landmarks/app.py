"""
Aplicación Streamlit para detección de landmarks faciales y análisis de expresiones.
"""
import streamlit as st
from PIL import Image
import cv2
import tempfile
import os
from src.detector import FaceLandmarkDetector
from src.utils import pil_to_cv2, cv2_to_pil, resize_image, calcular_apertura_boca, calcular_apertura_ojos, calcular_inclinacion_cabeza
from src.config import TOTAL_LANDMARKS


# Configuración de la página
st.set_page_config(
    page_title="Detector de Landmarks Faciales",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurar tema oscuro
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stSidebar {
        background-color: #262730;
    }
    .stTextInput, .stTextArea, .stSelectbox, .stRadio, .stFileUploader {
        background-color: #262730;
        color: #fafafa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .metric-container {
        background-color: #262730;
        border-radius: 5px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Título y descripción
st.title("Detector de Landmarks Faciales y Análisis de Expresiones")
st.markdown("""
Esta aplicación detecta **478 puntos clave** en rostros humanos usando MediaPipe y calcula métricas de expresiones faciales.
Subí una imagen o video para analizar las expresiones faciales.
""")

# Sidebar con información y controles
with st.sidebar:
    st.header("Configuración")

    # Selector de estilo de visualización
    estilo_visualizacion = st.selectbox(
        "Estilo de Visualización",
        ["points", "mesh", "contours", "heatmap"],
        format_func=lambda x: {
            "points": "Solo Puntos",
            "mesh": "Puntos + Malla",
            "contours": "Contornos Principales",
            "heatmap": "Heatmap de Densidad"
        }[x]
    )

    st.header("Información")
    st.markdown("""
    ### ¿Qué son los Landmarks?
    Son puntos de referencia que mapean:
    - Ojos (iris, párpados)
    - Nariz (puente, fosas)
    - Boca (labios, comisuras)
    - Contorno facial

    ### Estilos de Visualización
    - **Solo Puntos**: Puntos individuales
    - **Puntos + Malla**: Conexiones triangulares
    - **Contornos**: Solo ojos, boca y rostro
    - **Heatmap**: Densidad de puntos

    ### Métricas de Expresiones
    - **Apertura de boca**: Distancia entre labios
    - **Apertura de ojos**: Distancia entre párpados
    - **Inclinación de cabeza**: Ángulo de rotación

    ### Aplicaciones
    - Filtros AR (Instagram)
    - Análisis de expresiones
    - Animación facial
    - Autenticación biométrica
    """)

    st.divider()
    st.caption("Desarrollado en el Laboratorio 2 - IFTS24")

# Selector de tipo de archivo
tipo_archivo = st.radio(
    "Selecciona el tipo de archivo:",
    ["Imagen", "Video"],
    horizontal=True
)

# Uploader de imagen o video
if tipo_archivo == "Imagen":
    uploaded_file = st.file_uploader(
        "Subí una imagen con un rostro",
        type=["jpg", "jpeg", "png"],
        help="Formatos aceptados: JPG, JPEG, PNG"
    )
else:
    uploaded_file = st.file_uploader(
        "Subí un video con expresiones faciales",
        type=["mp4", "avi", "mov"],
        help="Formatos aceptados: MP4, AVI, MOV"
    )

if uploaded_file is not None:
    if tipo_archivo == "Imagen":
        # Procesamiento de imagen
        imagen_original = Image.open(uploaded_file)

        # Convertir a formato OpenCV
        imagen_cv2 = pil_to_cv2(imagen_original)

        # Redimensionar si es muy grande
        imagen_cv2 = resize_image(imagen_cv2, max_width=800)

        # Columnas para mostrar antes/después
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Imagen Original")
            st.image(cv2_to_pil(imagen_cv2), use_container_width=True)

        # Detectar landmarks
        with st.spinner("Detectando landmarks faciales..."):
            detector = FaceLandmarkDetector()
            imagen_procesada, landmarks, info = detector.detect(imagen_cv2, estilo_visualizacion)
            detector.close()

        with col2:
            st.subheader("Landmarks Detectados")
            st.image(cv2_to_pil(imagen_procesada), use_container_width=True)

        # Mostrar información de detección
        st.divider()

        if info["deteccion_exitosa"]:
            st.success("Detección exitosa")

            # Calcular métricas de expresión
            alto, ancho = imagen_cv2.shape[:2]
            apertura_boca = calcular_apertura_boca(landmarks, alto, ancho)
            apertura_ojos_izq, apertura_ojos_der = calcular_apertura_ojos(landmarks, alto, ancho)
            inclinacion_cabeza = calcular_inclinacion_cabeza(landmarks, alto, ancho)

            # Métricas
            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric("Rostros detectados", info["rostros_detectados"])

            with metric_col2:
                st.metric("Landmarks detectados", f"{info['total_landmarks']}/{TOTAL_LANDMARKS}")

            with metric_col3:
                porcentaje = (info['total_landmarks'] / TOTAL_LANDMARKS) * 100
                st.metric("Precisión", f"{porcentaje:.1f}%")

            # Métricas de expresión
            st.subheader("Análisis de Expresiones")
            expr_col1, expr_col2, expr_col3 = st.columns(3)

            with expr_col1:
                st.metric("Apertura Boca", f"{apertura_boca:.1f}px")

            with expr_col2:
                st.metric("Apertura Ojos", f"I:{apertura_ojos_izq:.1f} D:{apertura_ojos_der:.1f}px")

            with expr_col3:
                st.metric("Inclinación Cabeza", f"{inclinacion_cabeza:.1f}°")
        else:
            st.error("No se detectó ningún rostro en la imagen")
            st.info("""
            **Consejos**:
            - Asegurate de que el rostro esté bien iluminado
            - El rostro debe estar mirando hacia la cámara
            - Probá con una imagen de mayor calidad
            """)

    else:
        # Procesamiento de video
        st.subheader("Procesamiento de Video")

        # Guardar video temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        # Abrir video
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            st.error("Error al abrir el video")
        else:
            # Obtener propiedades del video
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            st.info(f"Video cargado: {total_frames} frames a {fps:.1f} FPS")

            # Selector de frame
            frame_number = st.slider("Selecciona el frame a analizar", 0, total_frames-1, 0)

            # Ir al frame seleccionado
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if ret:
                # Redimensionar frame
                frame = resize_image(frame, max_width=800)

                # Columnas para mostrar frame
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Frame Original")
                    st.image(cv2_to_pil(frame), use_container_width=True)

                # Detectar landmarks en el frame
                with st.spinner("Analizando expresiones..."):
                    detector = FaceLandmarkDetector()
                    frame_procesado, landmarks, info = detector.detect(frame, estilo_visualizacion)
                    detector.close()

                with col2:
                    st.subheader("Análisis de Expresiones")
                    if info["deteccion_exitosa"]:
                        st.image(cv2_to_pil(frame_procesado), use_container_width=True)

                        # Calcular métricas
                        alto, ancho = frame.shape[:2]
                        apertura_boca = calcular_apertura_boca(landmarks, alto, ancho)
                        apertura_ojos_izq, apertura_ojos_der = calcular_apertura_ojos(landmarks, alto, ancho)
                        inclinacion_cabeza = calcular_inclinacion_cabeza(landmarks, alto, ancho)

                        # Mostrar métricas
                        st.metric("Apertura Boca", f"{apertura_boca:.1f}px")
                        st.metric("Apertura Ojos", f"I:{apertura_ojos_izq:.1f} D:{apertura_ojos_der:.1f}px")
                        st.metric("Inclinación Cabeza", f"{inclinacion_cabeza:.1f}°")
                    else:
                        st.warning("No se detectó rostro en este frame")

            cap.release()

        # Limpiar archivo temporal
        os.unlink(video_path)

else:
    # Mensaje de bienvenida
    st.info("Subí una imagen para comenzar la detección")
    
    # Ejemplo visual
    st.markdown("### Ejemplo de Resultado")
    st.image(
        "https://ai.google.dev/static/mediapipe/images/solutions/face_landmarker_keypoints.png?hl=es-419",
        caption="MediaPipe detecta 478 landmarks faciales",
        width=400
    )