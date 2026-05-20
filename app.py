import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, date
from ultralytics import YOLO
from PIL import Image

# --- 1. CONFIGURATION & DESIGN ---
st.set_page_config(page_title="OviStat Vision Pro", layout="wide")

# CSS pour une interface mobile élégante
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; border-radius: 10px; padding: 10px; }
    div[data-testid="stCameraInput"] video { border: 3px solid #1f77b4; border-radius: 15px; }
    .stMetric { background-color: #ffffff; border-left: 5px solid #1f77b4; padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

DB_FILE = "data_ovinstat_V4.csv" # On passe en V4 pour repartir de zéro

# Les 24 paramètres + métadonnées
COLONNES = [
    "Date", "ID", "Race", "Age", "Poids", 
    "HG", "HS", "LB", "LQ", "LT", "LC", "LH", "LI", "LP", "PP", "TP",
    "Lc_cornes", "LT_tete", "Lt_tete", "LO", "Lo_oreille", "TC", "LY", "TS", "PS", "LG", "LL"
]

# --- 2. FONCTIONS TECHNIQUES ---
@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n.pt')

def load_data():
    if os.path.exists(DB_FILE):
        return pd.read_csv(DB_FILE, sep=';', encoding='utf-8-sig')
    return pd.DataFrame(columns=COLONNES)

model = load_yolo_model()
data = load_data()

# --- 3. INTERFACE ---
st.title("🐑 OviStat IA : Morphométrie Avancée")

tab1, tab2, tab3 = st.tabs(["📥 Saisie Terrain", "🔍 Historique", "📊 Analyse"])

with tab1:
    with st.form("form_expert", clear_on_submit=True):
        # SECTION 1 : IDENTITÉ
        st.subheader("🆔 Identification")
        c1, c2, c3 = st.columns(3)
        id_in = c1.text_input("ID Animal", value=f"OVIN-{len(data)+1}")
        age_in = c2.number_input("Âge (mois)", min_value=0, value=12)
        race_in = c3.selectbox("Race", ["Ouled Djellal", "Rembi", "Hamra", "Taadmit"])

        st.divider()

        # SECTION 2 : VISION IA
        st.subheader("📸 Diagnostic Optique")
        col_cam, col_info = st.columns([2, 1])
        with col_cam:
            photo = st.camera_input("Scanner l'animal")
        with col_info:
            if photo:
                st.success("✅ Animal détecté")
                st.info("💡 L'IA suggère de vérifier les mesures de hauteur et de longueur du tronc.")
            else:
                st.warning("Prenez une photo pour activer l'assistance morphométrique.")

        st.divider()

        # SECTION 3 : MENSURATIONS (Organisées par zones)
        st.subheader("📏 Paramètres Morphométriques")

        with st.expander("🏗️ Hauteurs & Longueurs (Corps)", expanded=True):
            g1, g2, g3, g4 = st.columns(4)
            poids = g1.number_input("Poids (kg)", value=0.0)
            hg = g2.number_input("H. Garrot (HG)", value=0.0)
            hs = g3.number_input("H. Sacrum (HS)", value=0.0)
            lb = g4.number_input("Long. Corps (LB)", value=0.0)
            
            g5, g6, g7, g8 = st.columns(4)
            lq = g5.number_input("Long. Queue (LQ)", value=0.0)
            lt = g6.number_input("Long. Tronc (LT)", value=0.0)
            lc = g7.number_input("Long. Cou (LC)", value=0.0)
            lh = g8.number_input("Long. Bassin (LH)", value=0.0)

        with st.expander("📐 Largeurs & Poitrine"):
            g9, g10, g11, g12 = st.columns(4)
            li = g9.number_input("Larg. Ischions (LI)", value=0.0)
            lp = g10.number_input("Larg. Poitrine (LP)", value=0.0)
            pp = g11.number_input("Prof. Poitrine (PP)", value=0.0)
            tp = g12.number_input("Tour Poitrine (TP)", value=0.0)

        with st.expander("👤 Tête & Oreilles"):
            g13, g14, g15, g16, g17 = st.columns(5)
            l_cornes = g13.number_input("Long. Cornes (Lc)", value=0.0)
            lt_tete = g14.number_input("Long. Tête (LT)", value=0.0)
            lt_tete_larg = g15.number_input("Larg. Tête (Lt)", value=0.0)
            lo = g16.number_input("Long. Oreille (LO)", value=0.0)
            lo_larg = g17.number_input("Larg. Oreille (Lo)", value=0.0)

        with st.expander("🧬 Reproduction & Laine"):
            g18, g19, g20, g21, g22, g23 = st.columns(6)
            tc = g18.number_input("Tour Canon (TC)", value=0.0)
            ly = g19.number_input("Long. Trayons (LY)", value=0.0)
            ts = g20.number_input("Tour Scrotal (TS)", value=0.0)
            ps = g21.number_input("Prof. Scrotale (PS)", value=0.0)
            lg = g22.number_input("Long. Gigot (LG)", value=0.0)
            ll = g23.number_input("Long. Laine (LL)", value=0.0)

        st.divider()

        # BOUTON FINAL
        if st.form_submit_button("💾 ENREGISTRER LA FICHE COMPLÈTE"):
            if id_in:
                date_now = datetime.now().strftime("%Y-%m-%d")
                new_row = [
                    date_now, id_in, race_in, age_in, poids,
                    hg, hs, lb, lq, lt, lc, lh, li, lp, pp, tp,
                    l_cornes, lt_tete, lt_tete_larg, lo, lo_larg, tc, ly, ts, ps, lg, ll
                ]
                pd.DataFrame([new_row], columns=COLONNES).to_csv(DB_FILE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
                st.success(f"Fiche de {id_in} archivée avec succès !")
                st.balloons()
                st.rerun()

with tab2:
    st.subheader("📋 Historique du troupeau")
    if not data.empty:
        st.dataframe(data, use_container_width=True)
        st.download_button("📥 Télécharger Excel (CSV)", data.to_csv(sep=';', index=False).encode('utf-8-sig'), "base_ovistat.csv")
    else:
        st.info("Aucune donnée enregistrée.")

with tab3:
    st.subheader("📈 Analyse de performance")
    if len(data) >= 1:
        param = st.selectbox("Choisir un paramètre à analyser", COLONNES[4:])
        fig, ax = plt.subplots()
        ax.bar(data["ID"], data[param], color='#1f77b4')
        plt.xticks(rotation=45)
        ax.set_ylabel(param)
        st.pyplot(fig)
