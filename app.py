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
    div[data-testid="stCameraInput"] video {
        width: 100% !important;
        height: auto !important;
        border-radius: 15px;
        border: 2px solid #1f77b4;
    }
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
        # IDENTIFICATION
        c1, c2 = st.columns(2)
        id_in = c1.text_input("🆔 ID Animal", value=get_next_id(data))
        race_in = c2.selectbox("🧬 Race", ["Ouled Djellal", "Hamra" , "Taadmit"])
        
        st.divider()
        
        # CAMERA IA
        st.write("📸 **Capture Optique (Analyse Morphométrique)**")
        cv1, c_photo, cv2 = st.columns([0.1, 0.8, 0.1])
        with c_photo:
            photo = st.camera_input("Scanner l'animal")
        
        # LOGIQUE IA
        taille_ia, peri_ia, note_ia, count_ia = 0.0, 0.0, 0.0, 0
        
        if photo:
            img = Image.open(photo)
            results = model(img)
            for r in results:
                for box in r.boxes:
                    if int(box.cls) == 18: # sheep
                        count_ia = 1
            
            if count_ia > 0:
                taille_ia, peri_ia, note_ia = 68.0, 82.5, 4.0
                st.success(f"✅ Ovin détecté ! Suggestion : {taille_ia}cm / {peri_ia}cm")
            else:
                st.warning("⚠️ Aucun ovin détecté sur l'image.")

        st.divider()

        # SAISIE MANUELLE (Toujours visible et pré-remplie par l'IA)
        st.write("📏 **Vérification des Mensurations**")
        cm1, cm2, cm3, cm4, cm5, cm6,cm7, cm8, cm9, cm10, cm11, cm12,cm13, cm14, cm15, cm16, cm17 = st.columns(17)
           HG = cm1.number_input("Hauteur_G (cm)", min_value=0.0, value=0.0)
           HS = cm2.number_input("Hauteur_S (cm)", min_value=0.0, value=0.0)
           LB = cm3.number_input("Longueur_B (cm)", min_value=0.0, value=0.0)
           LQ = cm4.number_input("Longueur_Q (cm)", min_value=0.0, value=0.0)
           LT = cm5.number_input("Longueur_T (cm)", min_value=0.0, value=0.0)
           LC = cm6.number_input("Longueur_C (cm)", min_value=0.0, value=0.0)
           LH = cm7.number_input("Longueur_H (cm)", min_value=0.0, value=0.0)
           LI = cm8.number_input("Longueur_I (cm)", min_value=0.0, value=0.0)
           LP = cm9.number_input("Longueur_P (cm)", min_value=0.0, value=0.0)
           TP = cm10.number_input("Taille_P (cm)", min_value=0.0, value=0.0)
           Lt = cm11.number_input("Longueur_t (cm)", min_value=0.0, value=0.0)
           LO = cm12.number_input("Longueur_O (cm)", min_value=0.0, value=0.0)
           Lo = cm13.number_input("Longueur_o (cm)", min_value=0.0, value=0.0)
           TC = cm14.number_input("Taille_C (cm)", min_value=0.0, value=0.0)
           LY = cm15.number_input("Longueur_Y (cm)", min_value=0.0, value=0.0)
           TS = cm16.number_input("Taille_S (cm)", min_value=0.0, value=0.0)
           LG = cm17.number_input("Longueur_G (cm)", min_value=0.0, value=0.0)
        
        # CALCUL RATION
        ration_ms = round(poids_f * 0.035, 2)
        st.info(f"🌾 **Ration suggérée : {ration_ms} kg de Matière Sèche / jour**")
        
        st.divider()
        if st.form_submit_button("💾 VALIDER ET ENREGISTRER LA FICHE"):
            if id_in:
                date_now = datetime.now().strftime("%Y-%m-%d")
                nouvelle_ligne = pd.DataFrame([[
                   date_now, id_in, race_in, Hauteur_G, Hauteur_S, Longueur_B, Longueur_Q, Longueur_T, Longueur_C, Longueur_H, Longueur_I, Longueur_P, Taille_P, Longueur_t,
                   Longueur_O, Longueur_o, Taille_C, Longueur_Y, Taille_S, Longueur_G,
                ]], columns=COLONNES)
                
                nouvelle_ligne.to_csv(DB_FILE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
                st.success(f"Enregistrement réussi pour {id_in}")
                st.balloons()
                st.rerun()

# --- ONGLET 2 : HISTORIQUE ---
with tab2:
    st.subheader("🔍 Base de données")
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
        st.subheader("📊 Comparaison")
        ani1 = st.selectbox("Animal A", ids_dispo, index=0)
        ani2 = st.selectbox("Animal B", ids_dispo, index=1)
        
        df1 = data[data["ID"] == ani1].sort_values("Date")
        df2 = data[data["ID"] == ani2].sort_values("Date")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df1["Date"], df1["Poids_kg"], label=f"{ani1}", marker='o')
        ax.plot(df2["Date"], df2["Poids_kg"], label=f"{ani2}", marker='s')
        ax.set_ylabel("Poids (kg)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Enregistrez au moins 2 animaux différents.")

# --- MAINTENANCE ---
with st.expander("⚙️ Maintenance"):
    if st.button("🗑️ Vider la base de données"):
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
        st.rerun()

        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

