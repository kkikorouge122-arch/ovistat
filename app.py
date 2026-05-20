import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, date
from ultralytics import YOLO
from PIL import Image

# --- 1. CONFIGURATION & VARIABLES GLOBALES ---
st.set_page_config(page_title="OviStat Vision Pro", layout="wide")

DB_FILE = "data_ovins_v3.csv" # Version v3 pour garantir une structure propre

COLONNES = [
    "Date", "ID", "Race", "HG", "HS", "LB", "LQ", "LT", "LC", "LH", 
    "LI", "LP", "TP", "Lt_min", "LO", "Lo_min", "TC", "LY", "TS", "LG", "Ration_MS"
]

# --- 2. FONCTIONS TECHNIQUES ---
@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n.pt')

def load_data():
    if os.path.exists(DB_FILE):
        return pd.read_csv(DB_FILE, sep=';', encoding='utf-8-sig', on_bad_lines='skip')
    return pd.DataFrame(columns=COLONNES)

# Initialisation physique du fichier
if not os.path.exists(DB_FILE):
    pd.DataFrame(columns=COLONNES).to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')

model = load_yolo_model()
data = load_data()

def get_next_id(df):
    if df.empty: return "OVIN-1"
    return f"OVIN-{len(df['ID'].unique()) + 1}"

# --- 3. STYLE CSS (CAMÉRA & UI) ---
st.markdown("""
    <style>
    div[data-testid="stCameraInput"] video {
        width: 100% !important;
        height: auto !important;
        border-radius: 15px;
        border: 2px solid #1f77b4;
    }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. INTERFACE PRINCIPALE ---
st.title("🐑 OviStat Vision Pro : Intelligence Augmentée")
st.sidebar.info(f"📁 Base de données : {DB_FILE}\n\n📊 Total mesures : {len(data)}")

tab1, tab2, tab3 = st.tabs(["📥 Saisie & IA", "🔍 Historique", "📊 Analyse"])

# --- ONGLET 1 : SAISIE & VISION ---
with tab1:
    with st.form("form_expert", clear_on_submit=True):
        c1, c2 = st.columns(2)
        id_in = c1.text_input("🆔 ID Animal", value=get_next_id(data))
        race_in = c2.selectbox("🧬 Race", ["Ouled Djellal", "Hamra", "Taadmit"])
        
        st.divider()
        st.write("📸 **Capture Optique & Diagnostic IA**")
        cv1, c_photo, cv2 = st.columns([0.1, 0.8, 0.1])
        with c_photo:
            photo = st.camera_input("Scanner l'animal")
        
        # Valeurs suggérées par défaut
        sugg_val = 0.0
        if photo:
            img = Image.open(photo)
            results = model(img)
            detect = any(int(box.cls) == 18 for r in results for box in r.boxes)
            if detect:
                st.success("✅ Ovin détecté ! Suggestions morphométriques activées.")
                sugg_val = 65.0 # Valeur exemple
            else:
                st.warning("⚠️ Aucun ovin détecté. Saisie manuelle requise.")

        st.divider()
        st.write("📏 **Mensurations Morphométriques (cm)**")
        
        # Grille de saisie organisée
        g1, g2, g3 = st.columns(3)
        hg = g1.number_input("Hauteur_G (HG)", value=sugg_val)
        hs = g2.number_input("Hauteur_S (HS)", value=sugg_val)
        lb = g3.number_input("Longueur_B (LB)", value=sugg_val)
        
        g4, g5, g6 = st.columns(3)
        lq = g4.number_input("Longueur_Q (LQ)", value=0.0)
        lt = g5.number_input("Longueur_T (LT)", value=0.0)
        lc = g6.number_input("Longueur_C (LC)", value=0.0)
        
        # ... (Autres mesures simplifiées pour le code final)
        st.write("🔍 *Autres paramètres*")
        g7, g8, g9 = st.columns(3)
        lh = g7.number_input("Longueur_H (LH)", value=0.0)
        li = g8.number_input("Longueur_I (LI)", value=0.0)
        lp = g9.number_input("Longueur_P (LP)", value=0.0)
        
        # Exemple de ration automatique basée sur HG (faute de poids)
        ration_estimee = round(hg * 0.02, 2) if hg > 0 else 0.0
        st.info(f"🌾 Ration estimée : {ration_estimee} kg MS/j")

        if st.form_submit_button("💾 ENREGISTRER LA FICHE"):
            if id_in:
                new_row = [
                    datetime.now().strftime("%Y-%m-%d"), id_in, race_in, 
                    hg, hs, lb, lq, lt, lc, lh, li, lp, 
                    0, 0, 0, 0, 0, 0, 0, 0, ration_estimee # Remplissage des 20 colonnes
                ]
                pd.DataFrame([new_row], columns=COLONNES).to_csv(DB_FILE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
                st.success("Enregistré !")
                st.rerun()

# --- ONGLET 2 : HISTORIQUE ---
with tab2:
    if not data.empty:
        st.dataframe(data, use_container_width=True)
        st.download_button("📥 Télécharger Excel", data.to_csv(sep=';', index=False).encode('utf-8-sig'), "export_ovins.csv")
    else:
        st.info("Base vide.")

# --- ONGLET 3 : ANALYSE ---
with tab3:
    if len(data["ID"].unique()) >= 2:
        ani1 = st.selectbox("Animal A", data["ID"].unique(), index=0)
        ani2 = st.selectbox("Animal B", data["ID"].unique(), index=1)
        
        val1 = data[data["ID"] == ani1]["HG"].iloc[-1]
        val2 = data[data["ID"] == ani2]["HG"].iloc[-1]
        
        fig, ax = plt.subplots()
        ax.bar([ani1, ani2], [val1, val2], color=['#1f77b4', '#ff7f0e'])
        ax.set_title("Comparaison Hauteur Garrot")
        st.pyplot(fig)
    else:
        st.warning("Enregistrez 2 animaux pour comparer.")

# --- MAINTENANCE ---
with st.expander("⚙️ Maintenance"):
    if st.button("🗑️ Réinitialiser tout"):
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()


