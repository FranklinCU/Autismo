# app_streamlit.py
# Detector de autismo

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import threading
import time
import mediapipe as mp
import joblib

import tensorflow as tf
from tensorflow import keras

# Sección
# CONFIGURACIÓN
# Sección

st.set_page_config(
    page_title="ASD Detector Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS PREMIUM
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    :root {
        --primary: #D4A574;
        --secondary: #E8D5C4;
        --dark: #8B7355;
        --light: #F5EBE0;
        --coral: #E07A5F;
        --green: #81B29A;
    }
    
    .main {
        background: linear-gradient(135deg, #FAF7F2 0%, #F5EBE0 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #D4A574 0%, #8B7355 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: var(--dark);
        font-size: 1.3rem;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    
    .card-2d {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 8px 8px 0px rgba(212, 165, 116, 0.3), 0px 4px 20px rgba(0, 0, 0, 0.1);
        border: 2px solid var(--secondary);
        margin-bottom: 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #D4A574 0%, #C4956B 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 4px 4px 0px rgba(139, 115, 85, 0.3);
    }
    
    .badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        box-shadow: 2px 2px 0px rgba(0,0,0,0.1);
    }
    
    .badge-autista {
        background: linear-gradient(135deg, #E07A5F 0%, #D46A50 100%);
        color: white;
    }
    
    .badge-no-autista {
        background: linear-gradient(135deg, #81B29A 0%, #72A58A 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Sección
# CARGAR SISTEMA
# Sección

@st.cache_resource(show_spinner=False)
def cargar_sistema():
    """Carga detector OpenCV, MediaPipe, modelo híbrido y scaler"""
    
    # 1. Detector de rostros OpenCV
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # 2. MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    
    # 3. Buscar modelo
    posibles_modelos = [
        'mejor_modelo_hibrido.keras',
        'modelo_hibrido_autismo_final.keras',
        'mejor_modelo_hibrido_fase2.keras'
    ]
    
    modelo_path = None
    for nombre in posibles_modelos:
        if os.path.exists(nombre):
            modelo_path = nombre
            break
    
    if not modelo_path:
        raise FileNotFoundError(
            "❌ No se encontró el modelo .keras\n"
            "Archivos necesarios:\n"
            "- mejor_modelo_hibrido.keras\n"
            "- scaler_caracteristicas.pkl"
        )
    
    modelo = keras.models.load_model(modelo_path)
    
    # 4. Cargar scaler
    scaler_path = 'scaler_caracteristicas.pkl'
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"❌ No se encontró {scaler_path}")
    
    scaler = joblib.load(scaler_path)
    
    return detector, face_mesh, modelo, scaler, modelo_path

# Inicializar sistema
try:
    with st.spinner("⏳ Cargando sistema avanzado de IA..."):
        detector, face_mesh, modelo, scaler, modelo_usado = cargar_sistema()
    
    st.success(f"✅ Sistema listo - Modelo: `{os.path.basename(modelo_usado)}`")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("AUC", "86.86%", "+3.3%")
    with col2:
        st.metric("Recall", "89.39%", "Excelente")
    with col3:
        st.metric("Características", "29", "Geométricas")
    
except Exception as e:
    st.error(f"❌ Error al cargar el sistema:\n{e}")
    st.info("""
    **Archivos necesarios en el repositorio:**
    1. `mejor_modelo_hibrido.keras` (modelo entrenado)
    2. `scaler_caracteristicas.pkl` (normalizador)
    
    Descárgalos de Google Colab y súbelos a GitHub.
    """)
    st.stop()

# Sección
# Funciones de extracción
# Sección

def calcular_distancia(p1, p2):
    """Distancia euclidiana"""
    return np.linalg.norm(p1 - p2)

def calcular_angulo(p1, p2, p3):
    """Ángulo entre tres puntos"""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def extraer_caracteristicas_faciales(imagen, face_mesh):
    """
    Extrae 29 características geométricas usando MediaPipe
    """
    
    # Convertir a RGB
    if isinstance(imagen, np.ndarray):
        if len(imagen.shape) == 2:
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)
        elif imagen.shape[2] == 4:
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGRA2RGB)
        elif imagen.shape[2] == 3:
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    else:
        return None
    
    # Procesar con MediaPipe
    results = face_mesh.process(imagen_rgb)
    
    if not results.multi_face_landmarks:
        return None
    
    landmarks = results.multi_face_landmarks[0]
    
    # Convertir a coordenadas
    h, w = imagen.shape[:2]
    puntos = []
    for lm in landmarks.landmark:
        x = lm.x * w
        y = lm.y * h
        z = lm.z * w
        puntos.append([x, y, z])
    
    puntos = np.array(puntos)
    
    # Landmarks clave
    ojo_izq_interior = puntos[133]
    ojo_izq_exterior = puntos[33]
    ojo_der_interior = puntos[362]
    ojo_der_exterior = puntos[263]
    
    ceja_izq_inicio = puntos[70]
    ceja_izq_centro = puntos[105]
    ceja_izq_fin = puntos[107]
    ceja_der_inicio = puntos[300]
    ceja_der_centro = puntos[334]
    ceja_der_fin = puntos[336]
    
    nariz_punta = puntos[1]
    nariz_izq = puntos[48]
    nariz_der = puntos[278]
    
    boca_izq = puntos[61]
    boca_der = puntos[291]
    boca_arriba = puntos[13]
    boca_abajo = puntos[14]
    
    menton = puntos[152]
    frente = puntos[10]
    cara_izq = puntos[234]
    cara_der = puntos[454]
    
    # Extraer 29 características (en orden exacto del entrenamiento)
    caracteristicas = {}
    
    # Distancias
    caracteristicas['distancia_entre_ojos'] = calcular_distancia(ojo_izq_interior, ojo_der_interior)
    caracteristicas['ancho_ojo_izquierdo'] = calcular_distancia(ojo_izq_interior, ojo_izq_exterior)
    caracteristicas['ancho_ojo_derecho'] = calcular_distancia(ojo_der_interior, ojo_der_exterior)
    caracteristicas['ancho_nariz'] = calcular_distancia(nariz_izq, nariz_der)
    caracteristicas['ancho_boca'] = calcular_distancia(boca_izq, boca_der)
    caracteristicas['altura_boca'] = calcular_distancia(boca_arriba, boca_abajo)
    caracteristicas['altura_rostro'] = calcular_distancia(frente, menton)
    caracteristicas['ancho_rostro'] = calcular_distancia(cara_izq, cara_der)
    caracteristicas['distancia_nariz_boca'] = calcular_distancia(nariz_punta, boca_arriba)
    caracteristicas['distancia_ojos_cejas_izq'] = calcular_distancia(ojo_izq_exterior, ceja_izq_centro)
    caracteristicas['distancia_ojos_cejas_der'] = calcular_distancia(ojo_der_exterior, ceja_der_centro)
    
    # Proporciones
    caracteristicas['ratio_altura_ancho_rostro'] = caracteristicas['altura_rostro'] / (caracteristicas['ancho_rostro'] + 1e-8)
    caracteristicas['ratio_ojos_ancho_rostro'] = caracteristicas['distancia_entre_ojos'] / (caracteristicas['ancho_rostro'] + 1e-8)
    caracteristicas['ratio_nariz_ancho_rostro'] = caracteristicas['ancho_nariz'] / (caracteristicas['ancho_rostro'] + 1e-8)
    caracteristicas['ratio_boca_ancho_rostro'] = caracteristicas['ancho_boca'] / (caracteristicas['ancho_rostro'] + 1e-8)
    caracteristicas['ratio_altura_ancho_boca'] = caracteristicas['altura_boca'] / (caracteristicas['ancho_boca'] + 1e-8)
    caracteristicas['ratio_nariz_ojos'] = caracteristicas['ancho_nariz'] / (caracteristicas['distancia_entre_ojos'] + 1e-8)
    
    # Simetría
    caracteristicas['simetria_ancho_ojos'] = 1 - abs(
        caracteristicas['ancho_ojo_izquierdo'] - caracteristicas['ancho_ojo_derecho']
    ) / (caracteristicas['ancho_ojo_izquierdo'] + caracteristicas['ancho_ojo_derecho'] + 1e-8)
    
    caracteristicas['simetria_cejas'] = 1 - abs(
        caracteristicas['distancia_ojos_cejas_izq'] - caracteristicas['distancia_ojos_cejas_der']
    ) / (caracteristicas['distancia_ojos_cejas_izq'] + caracteristicas['distancia_ojos_cejas_der'] + 1e-8)
    
    centro_cara = np.mean(puntos, axis=0)
    dist_ojo_izq = calcular_distancia(ojo_izq_exterior, centro_cara)
    dist_ojo_der = calcular_distancia(ojo_der_exterior, centro_cara)
    
    caracteristicas['simetria_ojos_centro'] = 1 - abs(dist_ojo_izq - dist_ojo_der) / (dist_ojo_izq + dist_ojo_der + 1e-8)
    
    # Ángulos
    caracteristicas['angulo_ceja_izquierda'] = calcular_angulo(ceja_izq_inicio, ceja_izq_centro, ceja_izq_fin)
    caracteristicas['angulo_ceja_derecha'] = calcular_angulo(ceja_der_inicio, ceja_der_centro, ceja_der_fin)
    caracteristicas['angulo_nariz'] = calcular_angulo(nariz_izq, nariz_punta, nariz_der)
    caracteristicas['angulo_boca'] = calcular_angulo(boca_izq, boca_arriba, boca_der)
    
    # Profundidad 3D
    caracteristicas['profundidad_nariz'] = abs(puntos[1][2])
    caracteristicas['profundidad_ojos'] = abs(np.mean([puntos[33][2], puntos[263][2]]))
    caracteristicas['profundidad_boca'] = abs(puntos[13][2])
    
    # Áreas
    caracteristicas['area_rostro'] = caracteristicas['altura_rostro'] * caracteristicas['ancho_rostro']
    caracteristicas['area_tercio_superior'] = caracteristicas['distancia_ojos_cejas_izq'] * caracteristicas['ancho_rostro']
    
    return caracteristicas

def predecir_con_modelo_hibrido(imagen, detector, face_mesh, modelo, scaler):
    """
    Predicción completa con modelo híbrido
    Retorna: img_resultado, rostro_zoom, prediccion, confianza, score, caracteristicas
    """
    
    # 1. Detectar rostro
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    rostros = detector.detectMultiScale(gray, 1.2, 4, minSize=(80, 80))
    
    if len(rostros) == 0:
        return None, None, None, None, None, None
    
    x, y, w, h = sorted(rostros, key=lambda r: r[2] * r[3], reverse=True)[0]
    
    # 2. Recortar con margen
    margen = int(max(w, h) * 0.15)
    x_crop = max(0, x - margen)
    y_crop = max(0, y - margen)
    w_crop = min(w + 2*margen, imagen.shape[1] - x_crop)
    h_crop = min(h + 2*margen, imagen.shape[0] - y_crop)
    
    rostro = imagen[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]
    
    # 3. Extraer características geométricas
    caracteristicas = extraer_caracteristicas_faciales(rostro, face_mesh)
    
    if caracteristicas is None:
        return None, None, None, None, None, None
    
    # 4. Normalizar características
    car_array = np.array([list(caracteristicas.values())])
    car_scaled = scaler.transform(car_array)
    
    # 5. Preparar imagen para CNN
    rostro_resized = cv2.resize(rostro, (224, 224)) / 255.0
    rostro_batch = np.expand_dims(rostro_resized, 0)
    
    # 6. Predecir con modelo híbrido
    try:
        pred = modelo.predict({
            'input_imagen': rostro_batch,
            'input_caracteristicas': car_scaled
        }, verbose=0)[0][0]
        
        pred = float(pred)
        es_autista = pred > 0.5
        prediccion = "AUTISTA" if es_autista else "NO AUTISTA"
        confianza = float(pred if es_autista else 1 - pred)
        
    except Exception as e:
        print(f"Error en predicción: {e}")
        return None, None, None, None, None, None
    
    # 7. Dibujar resultado
    img_resultado = imagen.copy()
    color = (0, 0, 255) if es_autista else (0, 255, 0)
    
    cv2.rectangle(img_resultado, (x, y), (x+w, y+h), color, 4)
    cv2.rectangle(img_resultado, (x, y-60), (x+w, y), color, -1)
    cv2.putText(img_resultado, prediccion, (x+10, y-35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img_resultado, f"Conf: {confianza:.0%}", (x+10, y-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    rostro_zoom = cv2.resize(rostro, (400, 400))
    cv2.rectangle(rostro_zoom, (0, 0), (399, 399), color, 8)
    
    return img_resultado, rostro_zoom, prediccion, confianza, pred, caracteristicas

# Sección
# CLASE VIDEO PROCESSOR
# Sección

