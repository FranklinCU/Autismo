# app_streamlit.py
# DETECTOR DE AUTISMO - MODELO HÍBRIDO CON MEDIAPIPE

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import threading
import time
import mediapipe as mp
import joblib

import tensorflow as tf
from tensorflow import keras

# ============================================
# CONFIGURACIÓN
# ============================================

st.set_page_config(
    page_title="ASD Detector Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS (MISMO DE ANTES)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    :root {
        --primary-beige: #D4A574;
        --secondary-beige: #E8D5C4;
        --dark-beige: #8B7355;
        --light-beige: #F5EBE0;
        --accent-coral: #E07A5F;
        --accent-green: #81B29A;
        --text-dark: #3D3D3D;
        --bg-cream: #FAF7F2;
    }
    .main {
        background: linear-gradient(135deg, #FAF7F2 0%, #F5EBE0 100%);
        font-family: 'Poppins', sans-serif;
    }
    .main-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #D4A574 0%, #8B7355 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: var(--dark-beige);
        font-size: 1.3rem;
        font-weight: 300;
        margin-bottom: 2rem;
        letter-spacing: 1px;
    }
    .card-2d {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 8px 8px 0px rgba(212, 165, 116, 0.3), 0px 4px 20px rgba(0, 0, 0, 0.1);
        border: 2px solid var(--secondary-beige);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    .card-2d:hover {
        transform: translate(-2px, -2px);
        box-shadow: 10px 10px 0px rgba(212, 165, 116, 0.4), 0px 6px 25px rgba(0, 0, 0, 0.15);
    }
    .stButton > button {
        background: linear-gradient(135deg, #D4A574 0%, #C4956B 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        box-shadow: 4px 4px 0px rgba(139, 115, 85, 0.3), 0px 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #C4956B 0%, #B8865F 100%);
        transform: translate(-2px, -2px);
        box-shadow: 6px 6px 0px rgba(139, 115, 85, 0.4), 0px 6px 20px rgba(0, 0, 0, 0.25);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: transparent;
        border-bottom: 3px solid var(--secondary-beige);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 12px 12px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
        color: var(--dark-beige);
        border: 2px solid var(--secondary-beige);
        border-bottom: none;
        box-shadow: 4px 0px 0px rgba(212, 165, 116, 0.2);
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #D4A574 0%, #C4956B 100%);
        color: white !important;
        box-shadow: 4px 4px 0px rgba(139, 115, 85, 0.3), 0px 4px 15px rgba(0, 0, 0, 0.2);
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-beige);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #D4A574 0%, #C4956B 100%);
        border-radius: 10px;
    }
    .stProgress > div > div > div {
        background-color: var(--light-beige);
        border-radius: 10px;
        border: 2px solid var(--secondary-beige);
    }
    .stSuccess {
        background: linear-gradient(135deg, #81B29A 0%, #72A58A 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 4px 4px 0px rgba(129, 178, 154, 0.3);
        font-weight: 600;
    }
    .stError {
        background: linear-gradient(135deg, #E07A5F 0%, #D46A50 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 4px 4px 0px rgba(224, 122, 95, 0.3);
        font-weight: 600;
    }
    .stWarning {
        background: linear-gradient(135deg, #F4A261 0%, #E89451 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 4px 4px 0px rgba(244, 162, 97, 0.3);
        font-weight: 600;
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
    .footer {
        text-align: center;
        padding: 2rem;
        background: white;
        border-radius: 20px;
        margin-top: 3rem;
        box-shadow: 8px 8px 0px rgba(212, 165, 116, 0.2), 0px 4px 15px rgba(0, 0, 0, 0.1);
        border: 2px solid var(--secondary-beige);
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--dark-beige);
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--primary-beige);
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CARGAR SISTEMA (NUEVO)
# ============================================

@st.cache_resource(show_spinner=False)
def cargar_sistema():
    """Carga detector, modelo híbrido, MediaPipe y scaler"""
    
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
    
    # 3. Modelo híbrido
    posibles_nombres = [
        'modelo_hibrido_autismo_final.keras',
        'modelo_hibrido.keras',
        'best_modelo_hibrido.keras'
    ]
    
    modelo_path = None
    for nombre in posibles_nombres:
        if os.path.exists(nombre):
            modelo_path = nombre
            break
    
    if not modelo_path:
        raise FileNotFoundError("No se encontró el modelo híbrido .keras")
    
    modelo = keras.models.load_model(modelo_path)
    
    # 4. Scaler para características
    scaler_path = 'scaler_caracteristicas.pkl'
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("No se encontró scaler_caracteristicas.pkl")
    
    scaler = joblib.load(scaler_path)
    
    return detector, face_mesh, modelo, scaler, modelo_path

try:
    with st.spinner("⏳ Cargando sistema avanzado con IA..."):
        detector, face_mesh, modelo, scaler, modelo_usado = cargar_sistema()
    st.success(f"✅ Sistema listo - Modelo: `{os.path.basename(modelo_usado)}`")
    st.info("🧬 Tecnología: CNN + MediaPipe + 29 características geométricas")
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.warning("""
    **Archivos necesarios:**
    - `modelo_hibrido_autismo_final.keras`
    - `scaler_caracteristicas.pkl`
    
    Descárgalos de Google Colab y súbelos al repositorio.
    """)
    st.stop()

# ============================================
# FUNCIONES DE EXTRACCIÓN (IGUAL QUE COLAB)
# ============================================

def calcular_distancia(p1, p2):
    """Distancia euclidiana entre dos puntos"""
    return np.linalg.norm(p1 - p2)

def calcular_angulo(p1, p2, p3):
    """Ángulo formado por tres puntos"""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def extraer_caracteristicas_faciales(imagen, face_mesh):
    """
    Extrae 29 características geométricas usando MediaPipe
    EXACTAMENTE igual que en el entrenamiento
    """
    
    # Manejar diferentes formatos de imagen
    if isinstance(imagen, np.ndarray):
        if len(imagen.shape) == 2:  # Grayscale
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)
        elif imagen.shape[2] == 4:  # RGBA
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGRA2RGB)
        elif imagen.shape[2] == 3:
            if imagen.dtype == np.uint8:
                imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            else:
                imagen_rgb = imagen
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
    
    # EXTRAER LAS MISMAS 29 CARACTERÍSTICAS
    caracteristicas = {}
    
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
    nariz_puente = puntos[168]
    
    boca_izq = puntos[61]
    boca_der = puntos[291]
    boca_arriba = puntos[13]
    boca_abajo = puntos[14]
    
    menton = puntos[152]
    frente = puntos[10]
    cara_izq = puntos[234]
    cara_der = puntos[454]
    
    # 1. DISTANCIAS
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
    
    # 2. PROPORCIONES
    caracteristicas['ratio_altura_ancho_rostro'] = caracteristicas['altura_rostro'] / (caracteristicas['ancho_rostro'] + 1e-8)
    caracteristicas['ratio_ojos_ancho_rostro'] = caracteristicas['distancia_entre_ojos'] / (caracteristicas['ancho_rostro'] + 1e-8)
    caracteristicas['ratio_nariz_ancho_rostro'] = caracteristicas['ancho_nariz'] / (caracteristicas['ancho_rostro'] + 1e-8)
    caracteristicas['ratio_boca_ancho_rostro'] = caracteristicas['ancho_boca'] / (caracteristicas['ancho_rostro'] + 1e-8)
    caracteristicas['ratio_altura_ancho_boca'] = caracteristicas['altura_boca'] / (caracteristicas['ancho_boca'] + 1e-8)
    caracteristicas['ratio_nariz_ojos'] = caracteristicas['ancho_nariz'] / (caracteristicas['distancia_entre_ojos'] + 1e-8)
    
    # 3. SIMETRÍA
    caracteristicas['simetria_ancho_ojos'] = 1 - abs(
        caracteristicas['ancho_ojo_izquierdo'] - caracteristicas['ancho_ojo_derecho']
    ) / (caracteristicas['ancho_ojo_izquierdo'] + caracteristicas['ancho_ojo_derecho'] + 1e-8)
    
    caracteristicas['simetria_cejas'] = 1 - abs(
        caracteristicas['distancia_ojos_cejas_izq'] - caracteristicas['distancia_ojos_cejas_der']
    ) / (caracteristicas['distancia_ojos_cejas_izq'] + caracteristicas['distancia_ojos_cejas_der'] + 1e-8)
    
    centro_cara = np.mean(puntos, axis=0)
    dist_ojo_izq_centro = calcular_distancia(ojo_izq_exterior, centro_cara)
    dist_ojo_der_centro = calcular_distancia(ojo_der_exterior, centro_cara)
    
    caracteristicas['simetria_ojos_centro'] = 1 - abs(
        dist_ojo_izq_centro - dist_ojo_der_centro
    ) / (dist_ojo_izq_centro + dist_ojo_der_centro + 1e-8)
    
    # 4. ÁNGULOS
    caracteristicas['angulo_ceja_izquierda'] = calcular_angulo(ceja_izq_inicio, ceja_izq_centro, ceja_izq_fin)
    caracteristicas['angulo_ceja_derecha'] = calcular_angulo(ceja_der_inicio, ceja_der_centro, ceja_der_fin)
    caracteristicas['angulo_nariz'] = calcular_angulo(nariz_izq, nariz_punta, nariz_der)
    caracteristicas['angulo_boca'] = calcular_angulo(boca_izq, boca_arriba, boca_der)
    
    # 5. PROFUNDIDAD 3D
    caracteristicas['profundidad_nariz'] = abs(puntos[1][2])
    caracteristicas['profundidad_ojos'] = abs(np.mean([puntos[33][2], puntos[263][2]]))
    caracteristicas['profundidad_boca'] = abs(puntos[13][2])
    
    # 6. ÁREA
    caracteristicas['area_rostro'] = caracteristicas['altura_rostro'] * caracteristicas['ancho_rostro']
    caracteristicas['area_tercio_superior'] = caracteristicas['distancia_ojos_cejas_izq'] * caracteristicas['ancho_rostro']
    
    return caracteristicas

def predecir_con_modelo_hibrido(imagen, detector, face_mesh, modelo, scaler):
    """
    Predicción usando modelo híbrido (CNN + Características)
    Maneja imágenes de CUALQUIER resolución
    """
    
    # 1. DETECTAR ROSTRO (funciona con cualquier resolución)
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    rostros = detector.detectMultiScale(gray, 1.2, 4, minSize=(80, 80))
    
    if len(rostros) == 0:
        return None, None, None, None, None
    
    # Tomar rostro más grande
    x, y, w, h = sorted(rostros, key=lambda r: r[2] * r[3], reverse=True)[0]
    
    # 2. RECORTAR ROSTRO
    margen = int(max(w, h) * 0.15)
    x_crop = max(0, x - margen)
    y_crop = max(0, y - margen)
    w_crop = min(w + 2*margen, imagen.shape[1] - x_crop)
    h_crop = min(h + 2*margen, imagen.shape[0] - y_crop)
    
    rostro_recortado = imagen[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]
    
    # 3. EXTRAER CARACTERÍSTICAS GEOMÉTRICAS
    caracteristicas = extraer_caracteristicas_faciales(rostro_recortado, face_mesh)
    
    if caracteristicas is None:
        return None, None, None, None, None
    
    # Convertir a array y normalizar con scaler
    caracteristicas_array = np.array([list(caracteristicas.values())])
    caracteristicas_scaled = scaler.transform(caracteristicas_array)
    
    # 4. PREPARAR IMAGEN PARA CNN
    rostro_resized = cv2.resize(rostro_recortado, (224, 224)) / 255.0
    rostro_batch = np.expand_dims(rostro_resized, 0)
    
    # 5. PREDECIR CON MODELO HÍBRIDO
    try:
        pred = modelo.predict({
            'input_imagen': rostro_batch,
            'input_caracteristicas': caracteristicas_scaled
        }, verbose=0)[0][0]
        
        pred = float(pred)
        
        es_autista = pred > 0.5
        prediccion = "AUTISTA" if es_autista else "NO AUTISTA"
        confianza = float(pred if es_autista else 1 - pred)
        
    except Exception as e:
        print(f"Error en predicción: {e}")
        return None, None, None, None, None
    
    # 6. DIBUJAR RESULTADO EN IMAGEN ORIGINAL
    img_resultado = imagen.copy()
    color = (0, 0, 255) if es_autista else (0, 255, 0)
    
    cv2.rectangle(img_resultado, (x, y), (x+w, y+h), color, 4)
    cv2.rectangle(img_resultado, (x, y-60), (x+w, y), color, -1)
    cv2.putText(img_resultado, prediccion, (x+10, y-35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img_resultado, f"Conf: {confianza:.0%}", (x+10, y-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 7. ROSTRO ZOOM
    rostro_zoom = cv2.resize(rostro_recortado, (400, 400))
    cv2.rectangle(rostro_zoom, (0, 0), (399, 399), color, 8)
    
    return img_resultado, rostro_zoom, prediccion, confianza, pred, caracteristicas

# ============================================
# CLASE VIDEO PROCESSOR (ACTUALIZADA)
# ============================================

class VideoProcessorOptimizado(VideoProcessorBase):
    def __init__(self):
        self.detector = detector
        self.face_mesh = face_mesh
        self.modelo = modelo
        self.scaler = scaler
        
        self.lock = threading.Lock()
        self.prediccion = None
        self.confianza = 0
        self.score = 0
        self.bbox = None
        self.caracteristicas = None
        
        self.frame_count = 0
        self.fps_count = 0
        self.fps = 0
        self.last_time = time.time()
        
        self.PROCESS_EVERY = 15  # Procesar cada 15 frames (más lento por MediaPipe)
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        self.frame_count += 1
        self.fps_count += 1
        
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps = self.fps_count
            self.fps_count = 0
            self.last_time = current_time
        
        if self.frame_count % self.PROCESS_EVERY == 0:
            resultado = predecir_con_modelo_hibrido(img, self.detector, self.face_mesh, self.modelo, self.scaler)
            
            with self.lock:
                if resultado[0] is not None:
                    # Obtener bbox
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    rostros = self.detector.detectMultiScale(gray, 1.2, 4, minSize=(80, 80))
                    if len(rostros) > 0:
                        self.bbox = sorted(rostros, key=lambda r: r[2] * r[3], reverse=True)[0]
                        self.prediccion = resultado[2]
                        self.confianza = resultado[3]
                        self.score = resultado[4]
                        self.caracteristicas = resultado[5]
        
        with self.lock:
            if self.bbox is not None and self.prediccion is not None:
                x, y, w, h = self.bbox
                
                color = (0, 0, 255) if self.prediccion == "AUTISTA" else (0, 255, 0)
                
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
                cv2.rectangle(img, (x, y-50), (x+w, y), color, -1)
                
                cv2.putText(img, self.prediccion, (x+5, y-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img, f"{self.confianza:.0%}", (x+5, y-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(img, "Analizando...", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.putText(img, f"FPS: {self.fps}", (img.shape[1]-100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def get_resultado(self):
        with self.lock:
            return self.prediccion, self.confianza, self.score, self.caracteristicas

# ============================================
# INTERFAZ
# ============================================

st.markdown('<h1 class="main-title">🧠 ASD Detector Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Análisis Avanzado con CNN + MediaPipe</p>', unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📹 Tiempo Real", "📷 Capturar Foto", "📁 Subir Imagen"])

# ============================================
# TAB 1: TIEMPO REAL
# ============================================

with tab1:
    st.markdown('<p class="section-header">Detección en Tiempo Real</p>', unsafe_allow_html=True)
    
    st.info("💡 Video fluido. Análisis cada 0.5 seg con 29 características geométricas extraídas.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card-2d">', unsafe_allow_html=True)
        st.markdown("#### 📹 Video con Análisis Híbrido")
        
        webrtc_ctx = webrtc_streamer(
            key="video-hibrido",
            video_processor_factory=VideoProcessorOptimizado,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 30}
                },
                "audio": False
            },
            async_processing=True,
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card-2d">', unsafe_allow_html=True)
        st.markdown("#### 📊 Resultado")
        
        resultado_placeholder = st.empty()
        confianza_placeholder = st.empty()
        prob_placeholder = st.empty()
        caract_placeholder = st.empty()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    if webrtc_ctx.video_processor:
        pred, conf, score, caract = webrtc_ctx.video_processor.get_resultado()
        
        if pred:
            with resultado_placeholder:
                badge = "badge-autista" if pred == "AUTISTA" else "badge-no-autista"
                emoji = "⚠️" if pred == "AUTISTA" else "✅"
                st.markdown(f'<h3><span class="badge {badge}">{emoji} {pred}</span></h3>', unsafe_allow_html=True)
            
            with confianza_placeholder:
                st.metric("Confianza", f"{conf:.1%}")
            
            with prob_placeholder:
                st.markdown("**Probabilidades:**")
                st.progress(float(score), text=f"🔴 Autista: {score:.1%}")
                st.progress(float(1-score), text=f"🟢 No Autista: {(1-score):.1%}")
            
            if caract:
                with caract_placeholder:
                    with st.expander("🔬 Ver Características Detectadas"):
                        col_c1, col_c2 = st.columns(2)
                        items = list(caract.items())
                        mid = len(items) // 2
                        
                        with col_c1:
                            for key, val in items[:mid]:
                                st.text(f"{key}: {val:.2f}")
                        
                        with col_c2:
                            for key, val in items[mid:]:
                                st.text(f"{key}: {val:.2f}")
        else:
            with resultado_placeholder:
                st.info("⏳ Analizando...")

# ============================================
# TAB 2: CAPTURAR FOTO
# ============================================

with tab2:
    st.markdown('<p class="section-header">Captura Instantánea</p>', unsafe_allow_html=True)
    
    st.info("💡 Análisis completo con extracción de 29 características geométricas faciales.")
    
    foto = st.camera_input("📸 Toma tu foto")
    
    if foto:
        img = np.array(Image.open(foto))
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        with st.spinner("🧠 Analizando con modelo híbrido..."):
            resultado = predecir_con_modelo_hibrido(img_bgr, detector, face_mesh, modelo, scaler)
        
        if resultado[0] is not None:
            img_res, zoom, pred, conf, score, caract = resultado
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="card-2d"><h4>1️⃣ Original</h4>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card-2d"><h4>2️⃣ Rostro</h4>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="card-2d"><h4>3️⃣ Resultado</h4>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB))
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                st.markdown('<div class="card-2d">', unsafe_allow_html=True)
                badge = "badge-autista" if pred == "AUTISTA" else "badge-no-autista"
                emoji = "⚠️" if pred == "AUTISTA" else "✅"
                st.markdown(f'<h1><span class="badge {badge}">{emoji} {pred}</span></h1>', unsafe_allow_html=True)
                st.metric("Confianza", f"{conf:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_r2:
                st.markdown('<div class="card-2d"><h4>📊 Probabilidades</h4>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("🔴 Autista", f"{score:.1%}")
                    st.progress(float(score))
                with c2:
                    st.metric("🟢 No Autista", f"{(1-score):.1%}")
                    st.progress(float(1-score))
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Mostrar características
            st.markdown("---")
            st.markdown("### 🔬 Características Geométricas Extraídas")
            
            with st.expander("Ver las 29 características analizadas"):
                col_a, col_b = st.columns(2)
                items = list(caract.items())
                mid = len(items) // 2
                
                with col_a:
                    for key, val in items[:mid]:
                        st.text(f"• {key}: {val:.2f}")
                
                with col_b:
                    for key, val in items[mid:]:
                        st.text(f"• {key}: {val:.2f}")
        else:
            st.error("❌ No se detectó rostro")

# ============================================
# TAB 3: SUBIR IMAGEN
# ============================================

with tab3:
    st.markdown('<p class="section-header">Análisis de Imagen</p>', unsafe_allow_html=True)
    
    st.info("💡 Soporta imágenes de CUALQUIER resolución (hasta 4K). El sistema las procesará automáticamente.")
    
    archivo = st.file_uploader("Selecciona una imagen", type=['jpg', 'jpeg', 'png'])
    
    if archivo:
        img = np.array(Image.open(archivo))
        
        # Mostrar resolución original
        st.caption(f"📐 Resolución original: {img.shape[1]}x{img.shape[0]} pixels")
        
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        with st.spinner("🧠 Analizando imagen de alta resolución..."):
            resultado = predecir_con_modelo_hibrido(img_bgr, detector, face_mesh, modelo, scaler)
        
        if resultado[0] is not None:
            img_res, zoom, pred, conf, score, caract = resultado
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="card-2d"><h4>1️⃣ Original</h4>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card-2d"><h4>2️⃣ Rostro Ampliado</h4>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="card-2d"><h4>3️⃣ Resultado</h4>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB))
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                st.markdown('<div class="card-2d">', unsafe_allow_html=True)
                badge = "badge-autista" if pred == "AUTISTA" else "badge-no-autista"
                emoji = "⚠️" if pred == "AUTISTA" else "✅"
                st.markdown(f'<h1><span class="badge {badge}">{emoji} {pred}</span></h1>', unsafe_allow_html=True)
                st.metric("Confianza", f"{conf:.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_r2:
                st.markdown('<div class="card-2d"><h4>📊 Probabilidades</h4>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("🔴 Autista", f"{score:.1%}")
                    st.progress(float(score))
                with c2:
                    st.metric("🟢 No Autista", f"{(1-score):.1%}")
                    st.progress(float(1-score))
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Características
            st.markdown("---")
            st.markdown("### 🔬 Análisis Detallado de Características")
            
            with st.expander("Ver las 29 características geométricas"):
                col_a, col_b = st.columns(2)
                items = list(caract.items())
                mid = len(items) // 2
                
                with col_a:
                    for key, val in items[:mid]:
                        st.text(f"• {key}: {val:.2f}")
                
                with col_b:
                    for key, val in items[mid:]:
                        st.text(f"• {key}: {val:.2f}")
        else:
            st.error("❌ No se detectó rostro en la imagen")

# ============================================
# FOOTER
# ============================================

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    with st.expander("ℹ️ Tecnología Avanzada"):
        st.markdown("""
        ### 🧬 Modelo Híbrido
        
        **Componentes:**
        1. **CNN (MobileNetV2)** - Análisis de imagen completa
        2. **MediaPipe Face Mesh** - 468 landmarks faciales en 3D
        3. **29 Características Geométricas** - Medidas precisas
        4. **Normalización** - Escalado con StandardScaler
        
        **Características Extraídas:**
        - Distancias entre puntos faciales
        - Proporciones y ratios
        - Simetría facial
        - Ángulos de cejas, nariz, boca
        - Profundidad 3D
        - Áreas faciales
        
        **Métricas del Modelo:**
        - Accuracy: 80.08%
        - AUC: 86.83%
        - Precision: 78.26%
        - Recall: 83.08%
        """)

with col2:
    with st.expander("⚠️ Advertencia Médica"):
        st.warning("""
        **Sistema de apoyo diagnóstico**
        
        Este sistema tiene 80% de precisión y es solo una 
        herramienta de APOYO. NO reemplaza el diagnóstico 
        médico profesional.
        
        **Limitaciones:**
        - 1 de cada 5 predicciones puede ser incorrecta
        - No detecta niveles de severidad
        - No evalúa comportamiento
        
        **Consulte siempre con:**
        - Psicólogos especializados en TEA
        - Neurólogos pediatras
        - Equipos multidisciplinarios
        
        El diagnóstico del autismo requiere evaluación 
        clínica completa y multidimensional.
        """)

st.markdown("""
<div class="footer">
    <h3 style="color: var(--primary-beige);">🧠 ASD Detector Pro</h3>
    <p style="color: var(--dark-beige);">Sistema Híbrido de Análisis Facial</p>
    <p style="color: var(--text-dark); font-size: 0.85rem;">
        TensorFlow • MediaPipe • OpenCV • WebRTC • Streamlit
    </p>
    <p style="color: #999; font-size: 0.75rem;">
        Accuracy: 80.08% | AUC: 86.83% | v2.0 - 2025
    </p>
</div>
""", unsafe_allow_html=True)
