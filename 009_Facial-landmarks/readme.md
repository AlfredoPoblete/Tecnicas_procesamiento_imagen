# Detector de Landmarks Faciales

Aplicación web para detectar 478 puntos clave en rostros humanos usando MediaPipe y Streamlit.

## Características

- Detección de 478 landmarks faciales
- Interfaz web interactiva
- Procesamiento en tiempo real
- Visualización antes/después

## Tecnologías

- **MediaPipe**: Detección de landmarks
- **OpenCV**: Procesamiento de imágenes
- **Streamlit**: Framework web
- **Python 3.11+**

App Desplegada en HuggingFace: https://huggingface.co/spaces/AlfredoPoblete/Landmark_Facial

## Instalación Local

```bash
# Clonar repositorio
git clone https://github.com/AlfredoPoblete/009_Facial-landmarks
cd facial-landmarks-app

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación

streamlit run app.py

