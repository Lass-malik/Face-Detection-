# =========================
# IMPORTS
# =========================
import cv2
import streamlit as st
import os
from datetime import datetime

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(
    page_title="Détection Faciale",
    layout="centered"
)

# =========================
# CHARGEMENT DU CASCADE (SÉCURISÉ)
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    st.error("❌ Impossible de charger le fichier haarcascade_frontalface_default.xml")
    st.stop()

# =========================
# DOSSIER DE SAUVEGARDE
# =========================
SAVE_DIR = "faces_detected"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# FONCTION DE DÉTECTION
# =========================
def detect_faces(rect_color, min_neighbors, scale_factor, save_faces):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Webcam non détectée")
        return

    frame_area = st.empty()
    stop = st.button("🛑 Arrêter la détection")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Erreur de lecture caméra")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                rect_color,
                2
            )

            if save_faces:
                face_img = frame[y:y+h, x:x+w]
                filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(
                    f"{SAVE_DIR}/face_{filename}.jpg",
                    face_img
                )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_area.image(frame, channels="RGB")

        if stop:
            break

    cap.release()
    cv2.destroyAllWindows()

# =========================
# INTERFACE STREAMLIT
# =========================
def app():
    st.title("🎯 Application de Détection Faciale")

    st.markdown("""
    ### 📖 Instructions
    1. Autorisez l'accès à la webcam
    2. Ajustez les paramètres de détection
    3. Cliquez sur **Démarrer la détection**
    4. Cliquez sur **Arrêter** pour fermer la caméra
    """)

    st.divider()

    # 🎨 Couleur du rectangle
    color_hex = st.color_picker(
        "🎨 Choisissez la couleur des rectangles",
        "#00FF00"
    )
    rect_color = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))

    # 🎚 Paramètres
    min_neighbors = st.slider(
        "🔧 minNeighbors",
        min_value=1,
        max_value=10,
        value=5
    )

    scale_factor = st.slider(
        "🔍 scaleFactor",
        min_value=1.05,
        max_value=1.5,
        value=1.3,
        step=0.05
    )

    # 💾 Sauvegarde
    save_faces = st.checkbox("💾 Sauvegarder les visages détectés")

    st.divider()

    # ▶ Démarrer
    if st.button("▶ Démarrer la détection"):
        detect_faces(rect_color, min_neighbors, scale_factor, save_faces)

# =========================
# LANCEMENT
# =========================
if __name__ == "__main__":
    app()
