import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
import zipfile

# =========================
# Configuration Streamlit
# =========================
st.set_page_config(page_title="Détection Faciale Cloud", layout="centered")

# =========================
# Chargement du Cascade
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    st.error("❌ Impossible de charger haarcascade_frontalface_default.xml")
    st.stop()

# =========================
# Dossier de sauvegarde
# =========================
SAVE_DIR = "faces_detected"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# Interface utilisateur
# =========================
st.title("🎯 Détection Faciale via Webcam Navigateur")

st.markdown("""
### Instructions
1. Autorisez l'accès à la webcam.
2. Cliquez sur **Prendre une photo** pour détecter les visages.
3. Ajustez les paramètres si nécessaire.
4. Les visages détectés peuvent être sauvegardés et téléchargés.
""")

# Paramètres utilisateur
color_hex = st.color_picker("🎨 Couleur des rectangles", "#00FF00")
rect_color = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
min_neighbors = st.slider("🔧 minNeighbors", 1, 10, 5)
scale_factor = st.slider("🔍 scaleFactor", 1.05, 1.5, 1.3, 0.05)
save_faces = st.checkbox("💾 Sauvegarder les visages détectés")

# =========================
# Capture image webcam
# =========================
uploaded_file = st.camera_input("📷 Prenez une photo")

if uploaded_file is not None:
    # Convertir l'image en array OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Détection faciale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors
    )

    # Dessiner rectangles et sauvegarder si coché
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), rect_color, 2)
        if save_faces:
            face_img = frame[y:y+h, x:x+w]
            filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, face_img)

    # Afficher l'image avec rectangles
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, channels="RGB", caption=f"Visages détectés: {len(faces)}")

    # Téléchargement ZIP si sauvegarde activée
    if save_faces:
        zip_filename = "faces_detected.zip"
        with zipfile.ZipFile(zip_filename, "w") as zipf:
            for f in os.listdir(SAVE_DIR):
                zipf.write(os.path.join(SAVE_DIR, f))
        with open(zip_filename, "rb") as f:
            st.download_button(
                label="⬇ Télécharger les visages détectés",
                data=f,
                file_name=zip_filename,
                mime="application/zip"
            )
