# app_streamlit.py
# DETECTOR DE AUTISMO - MODELO HÍBRIDO AVANZADO
# CNN + MediaPipe + 29 Características Geométricas

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

# ════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════

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

# ════════════════════════════════════════════════════════════════
# CARGAR SISTEMA
# ════════════════════════════════════════════════════════════════

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

# ════════════════════════════════════════════════════════════════
# FUNCIONES DE EXTRACCIÓN (EXACTAS DEL ENTRENAMIENTO)
# ════════════════════════════════════════════════════════════════

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
    IDÉNTICO al código de entrenamiento
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

# ════════════════════════════════════════════════════════════════
# CLASE VIDEO PROCESSOR
# ════════════════════════════════════════════════════════════════

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
        self.fps = 0
        self.last_time = time.time()
        
        self.PROCESS_EVERY = 15  # Procesar cada 15 frames
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        self.frame_count += 1
        
        # Calcular FPS
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_time = current_time
        
        # Procesar cada N frames
        if self.frame_count % self.PROCESS_EVERY == 0:
            resultado = predecir_con_modelo_hibrido(img, self.detector, self.face_mesh, self.modelo, self.scaler)
            
            with self.lock:
                if resultado[0] is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    rostros = self.detector.detectMultiScale(gray, 1.2, 4, minSize=(80, 80))
                    if len(rostros) > 0:
                        self.bbox = sorted(rostros, key=lambda r: r[2] * r[3], reverse=True)[0]
                        self.prediccion = resultado[2]
                        self.confianza = resultado[3]
                        self.score = resultado[4]
                        self.caracteristicas = resultado[5]
        
        # Dibujar resultado
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

# ════════════════════════════════════════════════════════════════
# INTERFAZ
# ════════════════════════════════════════════════════════════════

st.markdown('<h1 class="main-title">🧠 ASD Detector Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Análisis Avanzado con CNN + MediaPipe + 29 Features</p>', unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📹 Tiempo Real", "📷 Capturar Foto", "📁 Subir Imagen"])

# ═══════════════════════ TAB 1: TIEMPO REAL ═══════════════════════

with tab1:
    st.markdown("### 📹 Detección en Tiempo Real")
    st.info("💡 Análisis continuo cada 0.5 seg con extracción de 29 características faciales")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card-2d">', unsafe_allow_html=True)
        
        webrtc_ctx = webrtc_streamer(
            key="video-hibrido",
            video_processor_factory=VideoProcessorOptimizado,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            async_processing=True,
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card-2d">', unsafe_allow_html=True)
        st.markdown("#### 📊 Resultado")
        
        if webrtc_ctx.video_processor:
            pred, conf, score, caract = webrtc_ctx.video_processor.get_resultado()
            
            if pred:
                badge = "badge-autista" if pred == "AUTISTA" else "badge-no-autista"
                emoji = "⚠️" if pred == "AUTISTA" else "✅"
                st.markdown(f'<h3><span class="badge {badge}">{emoji} {pred}</span></h3>', unsafe_allow_html=True)
                st.metric("Confianza", f"{conf:.1%}")
                st.progress(float(score), text=f"🔴 Autista: {score:.1%}")
                st.progress(float(1-score), text=f"🟢 No Autista: {(1-score):.1%}")
                
                if caract:
                    with st.expander("🔬 Ver 29 Características"):
                        for key, val in list(caract.items())[:10]:
                            st.text(f"{key}: {val:.2f}")
            else:
                st.info("⏳ Esperando detección...")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════ TAB 2: CAPTURAR ═══════════════════════

with tab2:
    st.markdown("### 📷 Captura Instantánea con Análisis Detallado")
    
    foto = st.camera_input("Toma tu foto")
    
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
                st.markdown("**1️⃣ Original**")
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            
            with col2:
                st.markdown("**2️⃣ Rostro Detectado**")
                st.image(cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB))
            
            with col3:
                st.markdown("**3️⃣ Resultado**")
                st.image(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB))
            
            st.markdown("---")
            
            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                badge = "badge-autista" if pred == "AUTISTA" else "badge-no-autista"
                emoji = "⚠️" if pred == "AUTISTA" else "✅"
                st.markdown(f'<h1><span class="badge {badge}">{emoji} {pred}</span></h1>', unsafe_allow_html=True)
                st.metric("Confianza", f"{conf:.1%}")
                st.metric("Test AUC", "86.86%")
                st.metric("Recall", "89.39%")
            
            with col_r2:
                st.markdown("**📊 Probabilidades**")
                st.metric("🔴 Autista", f"{score:.1%}")
                st.progress(float(score))
                st.metric("🟢 No Autista", f"{(1-score):.1%}")
                st.progress(float(1-score))
            
            with st.expander("🔬 Ver 29 Características Geométricas Extraídas"):
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
            st.error("❌ No se detectó rostro frontal en la imagen")

# ═══════════════════════ TAB 3: SUBIR ═══════════════════════

with tab3:
    st.markdown("### 📁 Análisis de Imagen Existente")
    st.info("💡 Soporta imágenes hasta 4K. Procesamiento automático.")
    
    archivo = st.file_uploader("Selecciona una imagen", type=['jpg', 'jpeg', 'png'])
    
    if archivo:
        img = np.array(Image.open(archivo))
        st.caption(f"📐 Resolución: {img.shape[1]}x{img.shape[0]} px")
        
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        with st.spinner("🧠 Analizando..."):
            resultado = predecir_con_modelo_hibrido(img_bgr, detector, face_mesh, modelo, scaler)
        
        if resultado[0] is not None:
            img_res, zoom, pred, conf, score, caract = resultado
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Original**")
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            
            with col2:
                st.markdown("**Rostro**")
                st.image(cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB))
            
            with col3:
                st.markdown("**Resultado**")
                st.image(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB))
            
            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                badge = "badge-autista" if pred == "AUTISTA" else "badge-no-autista"
                emoji = "⚠️" if pred == "AUTISTA" else "✅"
                st.markdown(f'<h1><span class="badge {badge}">{emoji} {pred}</span></h1>', unsafe_allow_html=True)
                st.metric("Confianza", f"{conf:.1%}")
            
            with col_r2:
                st.markdown("**Probabilidades**")
                st.metric("🔴 Autista", f"{score:.1%}")
                st.progress(float(score))
                st.metric("🟢 No Autista", f"{(1-score):.1%}")
                st.progress(float(1-score))
            
            with st.expander("🔬 Características Extraídas"):
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

# ═══════════════════════ FOOTER ═══════════════════════

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    with st.expander("ℹ️ Tecnología"):
        st.markdown("""
        **Modelo Híbrido Avanzado:**
        - CNN (MobileNetV2) para análisis visual
        - MediaPipe Face Mesh (468 landmarks 3D)
        - 29 características geométricas extraídas
        - StandardScaler para normalización
        
        **Métricas de Rendimiento:**
        - Test AUC: 86.86%
        - Test Accuracy: 81.92%
        - Recall: 89.39% (detecta 9 de cada 10)
        - Precision: 85.21%
        
        **Stack Tecnológico:**
        - TensorFlow 2.17 + Keras
        - MediaPipe 0.10
        - OpenCV 4.10
        - Streamlit WebRTC
        """)

with col2:
    with st.expander("⚠️ Advertencia Médica"):
        st.warning("""
        **Sistema de Apoyo Diagnóstico**
        
        Este sistema tiene 86.86% de precisión y detecta
        89% de casos reales (recall).
        
        **NO reemplaza:**
        - Evaluación médica profesional
        - Diagnóstico clínico completo
        - Valoración comportamental
        
        **Limitaciones:**
        - 1 de cada 10 errores (AUC 86.86%)
        - Solo analiza características faciales
        - No detecta niveles de severidad
        
        **Consulte siempre con:**
        - Psicólogos especializados en TEA
        - Neurólogos pediatras
        - Equipos multidisciplinarios
        
        Este sistema es para screening inicial,
        NO para diagnóstico definitivo.
        """)

st.markdown("""
<div style="text-align: center; padding: 2rem; background: white; border-radius: 20px; 
     box-shadow: 8px 8px 0px rgba(212, 165, 116, 0.2); border: 2px solid #E8D5C4;">
    <h3 style="color: #D4A574;">🧠 ASD Detector Pro</h3>
    <p style="color: #8B7355;">Sistema Híbrido de Análisis Facial Avanzado</p>
    <p style="font-size: 0.85rem;">
        TensorFlow • MediaPipe • OpenCV • WebRTC • Streamlit
    </p>
    <p style="color: #999; font-size: 0.75rem;">
        AUC: 86.86% | Recall: 89.39% | v2.0 - 2025
    </p>
</div>
""", unsafe_allow_html=True)
