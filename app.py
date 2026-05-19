import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, date
from ultralytics import YOLO
from PIL import Image

# --- CONFIGURATION ---
st.set_page_config(page_title="OviStat Vision Pro", layout="wide")
DB_FILE = "data_ovins_final.csv"
COLONNES = ["Date", "ID", "Race", "Poids_kg", "Taille_cm", "Perimetre_cm", "Note_IA", "Comptage", "Ration_MS"]

# Chargement intelligent du modèle IA
@st.cache_resource
def load_yolo_model():
    # Modèle 'nano' : rapide et léger pour le Cloud
    return YOLO('yolov8n.pt')

model = load_yolo_model()

# 1. INITIALISATION DU FICHIER
if not os.path.exists(DB_FILE):
    df_init = pd.DataFrame(columns=COLONNES)
    df_init.to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')

def load_data():
    if os.path.exists(DB_FILE):
        return pd.read_csv(DB_FILE, sep=';', encoding='utf-8-sig')
    return pd.DataFrame(columns=COLONNES)

data = load_data()

def get_next_id(df):
    if df.empty: return "OVIN-1"
    return f"OVIN-{len(df['ID'].unique()) + 1}"

# --- INTERFACE ---
st.title("🐑 OviStat Vision Pro : Intelligence Augmentée")
# --- STYLE POUR AGRANDIR LA CAMÉRA ---
st.markdown("""
    <style>
    /* Force la vidéo de la caméra à prendre toute la largeur du conteneur */
    div[data-testid="stCameraInput"] video {
        width: 100% !important;
        height: auto !important;
        border-radius: 15px;
        border: 2px solid #1f77b4;
    }
    /* Centre le bouton de capture sous la vidéo */
    div[data-testid="stCameraInput"] button {
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📥 Saisie IA & Ration", "🔍 Historique Complet", "📊 Analyse Comparative"])

# --- ONGLET 1 : SAISIE INTELLIGENTE ---
with tab1:
    st.header("📥 Scan & Enregistrement")
    
    with st.form("form_expert", clear_on_submit=True):
        c1, c2 = st.columns(2)
        
        # Identification
        id_in = c1.text_input("ID Animal", value=get_next_id(data))
        race_in = c1.selectbox("Race", ["Ouled Djellal", "Rembi", "Hamra"])
        
        # Paramètres
        poids_in = c2.number_input("Poids (kg)", min_value=0.0, step=0.5, value=45.0)
        
        st.divider()
        st.write("📸 **Capture Optique & Morphométrie**")
        
        # On donne presque toute la place à la photo (ratio 0.1 / 0.8 / 0.1)
        cv1, c_photo, cv2 = st.columns([0.1, 0.8, 0.1])
        
        with c_photo:
            photo = st.camera_input("Scanner l'animal") 

        
        # Variables IA par défaut
        taille_ia, peri_ia, note_ia, count_ia, alerte = 0.0, 0.0, 0.0, 0, "Non détecté"
        
        if photo:
            # --- ANALYSE RÉELLE PAR YOLO ---
            img = Image.open(photo)
            results = model(img)
            
            # Vérifier si un mouton (classe 18) est présent
            detection_reussie = False
            for r in results:
                for box in r.boxes:
                    if int(box.cls) == 18:
                        detection_reussie = True
                        count_ia += 1
            
            if detection_reussie:
                # Simulation morphométrie (Basée sur la détection réussie)
                taille_ia = 68.0  
                peri_ia = 82.5    
                note_ia = 4.0     
                alerte = "Normal"
                st.success(f"✅ IA : Ovin détecté | Taille {taille_ia}cm | Note {note_ia}/5")
            else:
                st.warning("⚠️ Aucun ovin détecté. L'IA n'a pas pu valider la morphologie.")
                alerte = "Échec détection"
        
        # --- CALCUL RATION ---
        ration_ms = round(poids_in * 0.035, 2)
        st.info(f"🌾 **Ration suggérée : {ration_ms} kg de Matière Sèche / jour**")
        
        st.divider()
        if st.form_submit_button("💾 VALIDER ET ENREGISTRER"):
            if id_in:
                date_now = datetime.now().strftime("%Y-%m-%d")
                nouvelle_ligne = pd.DataFrame([[
                    date_now, id_in, race_in, poids_in, taille_ia, peri_ia, note_ia, count_ia, ration_ms
                ]], columns=COLONNES)
                
                nouvelle_ligne.to_csv(DB_FILE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
                st.success(f"Enregistrement réussi pour {id_in}")
                st.balloons()
                st.rerun()

# --- ONGLET 2 : HISTORIQUE ---
with tab2:
    st.subheader("🔍 Base de données enrichie")
    if not data.empty:
        st.dataframe(data, use_container_width=True)
        csv_data = data.to_csv(sep=';', index=False).encode('utf-8-sig')
        st.download_button("📥 Télécharger Excel (CSV)", csv_data, "donnees_ovins.csv", "text/csv")
    else:
        st.info("Aucune donnée.")

# --- ONGLET 3 : ANALYSE ---
with tab3:
    ids_dispo = data["ID"].unique()
    if len(ids_dispo) >= 2:
        ani1 = st.selectbox("Animal A", ids_dispo, index=0)
        ani2 = st.selectbox("Animal B", ids_dispo, index=1)
        
        df1 = data[data["ID"] == ani1].sort_values("Date")
        df2 = data[data["ID"] == ani2].sort_values("Date")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df1["Date"], df1["Poids_kg"], label=f"{ani1}", marker='o', linewidth=2)
        ax.plot(df2["Date"], df2["Poids_kg"], label=f"{ani2}", marker='s', linestyle='--', linewidth=2)
        ax.set_ylabel("Poids (kg)")
        ax.set_title("Comparaison de Croissance")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("Veuillez enregistrer au moins 2 animaux différents pour comparer.")

# --- MAINTENANCE ---
with st.expander("⚙️ Maintenance"):
    if st.button("🗑️ Vider tout"):
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

