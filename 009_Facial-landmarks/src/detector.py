"""
Detector de landmarks faciales usando MediaPipe.
"""
import cv2
import mediapipe as mp
import numpy as np
from .config import FACE_MESH_CONFIG, LANDMARK_COLOR, LANDMARK_RADIUS, LANDMARK_THICKNESS


class FaceLandmarkDetector:
    """
    Clase para detectar y visualizar landmarks faciales.
    """
    
    def __init__(self):
        """Inicializa el detector de MediaPipe."""
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(**FACE_MESH_CONFIG)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def detect(self, image, visualization_style="points"):
        """
        Detecta landmarks faciales en la imagen.

        Args:
            image (numpy.ndarray): Imagen en formato BGR (OpenCV)
            visualization_style (str): Estilo de visualización
                - "points": Solo puntos
                - "mesh": Puntos + malla conectada
                - "contours": Solo contornos principales
                - "heatmap": Heatmap de densidad

        Returns:
            tuple: (imagen_procesada, landmarks, info)
                - imagen_procesada: imagen con landmarks dibujados
                - landmarks: objeto de landmarks de MediaPipe
                - info: diccionario con información de detección
        """
        # Convertir BGR a RGB para MediaPipe
        imagen_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Procesar la imagen
        resultados = self.face_mesh.process(imagen_rgb)

        # Crear copia para dibujar
        imagen_con_puntos = image.copy()

        info = {
            "rostros_detectados": 0,
            "total_landmarks": 0,
            "deteccion_exitosa": False
        }

        # Si se detectaron rostros
        if resultados.multi_face_landmarks:
            info["rostros_detectados"] = len(resultados.multi_face_landmarks)

            # Tomar el primer rostro
            rostro = resultados.multi_face_landmarks[0]
            info["total_landmarks"] = len(rostro.landmark)
            info["deteccion_exitosa"] = True

            # Dibujar landmarks según el estilo seleccionado
            if visualization_style == "points":
                # Solo puntos (estilo original)
                alto, ancho = image.shape[:2]
                for punto in rostro.landmark:
                    coord_x_pixel = int(punto.x * ancho)
                    coord_y_pixel = int(punto.y * alto)

                    cv2.circle(
                        imagen_con_puntos,
                        (coord_x_pixel, coord_y_pixel),
                        LANDMARK_RADIUS,
                        LANDMARK_COLOR,
                        LANDMARK_THICKNESS
                    )

            elif visualization_style == "mesh":
                # Puntos + malla conectada completa en violeta
                self.mp_drawing.draw_landmarks(
                    image=imagen_con_puntos,
                    landmark_list=rostro,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1)
                )

            elif visualization_style == "contours":
                # Solo contornos principales (ojos, boca, rostro)
                self.mp_drawing.draw_landmarks(
                    image=imagen_con_puntos,
                    landmark_list=rostro,
                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )

            elif visualization_style == "heatmap":
                # Heatmap de densidad usando círculos con transparencia
                alto, ancho = image.shape[:2]
                heatmap = np.zeros((alto, ancho, 3), dtype=np.uint8)

                for punto in rostro.landmark:
                    coord_x_pixel = int(punto.x * ancho)
                    coord_y_pixel = int(punto.y * alto)

                    # Crear heatmap con círculos de diferentes intensidades
                    cv2.circle(
                        heatmap,
                        (coord_x_pixel, coord_y_pixel),
                        15,  # Radio mayor para efecto heatmap
                        (0, 0, 255),  # Rojo para puntos calientes
                        -1  # Relleno
                    )

                # Aplicar blur para efecto heatmap
                heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)

                # Superponer heatmap sobre imagen original
                imagen_con_puntos = cv2.addWeighted(imagen_con_puntos, 0.7, heatmap, 0.3, 0)

            return imagen_con_puntos, rostro, info

        # No se detectó rostro
        return imagen_con_puntos, None, info
    
    def close(self):
        """Libera recursos del detector."""
        self.face_mesh.close()