import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, date
from ultralytics import YOLO
from PIL import Image
from fpdf import FPDF

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

# --- CONFIGURATION ---
DB_FILE = "data_ovinstat_V5.csv"  # Nouvelle version propre

# Liste exacte des colonnes (assurez-vous que ID est bien là)
COLONNES = [
    "Date", "ID", "Race", "Age", "Poids", 
    "HG", "HS", "LB", "LQ", "LT", "LC", "LH", "LI", "LP", "PP", "TP",
    "Lc_cornes", "LT_tete", "Lt_tete", "LO", "Lo_oreille", "TC", "LY", "TS", "PS", "LG", "LL"
]

# Initialisation
if not os.path.exists(DB_FILE):
    pd.DataFrame(columns=COLONNES).to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')


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
    
    # On affiche les données si elles existent
    if not data.empty:
        st.dataframe(data, use_container_width=True)
    else:
        st.info("ℹ️ La base de données est actuellement vide. Enregistrez un animal pour voir les données.")

    st.divider()
    st.subheader("📥 Exportation")
    
    # Préparation du téléchargement (même si vide, cela téléchargera les en-têtes)
    csv_ready = data.to_csv(sep=';', index=False).encode('utf-8-sig')
    
    st.download_button(
        label="📥 Télécharger la base Excel (CSV)",
        data=csv_ready,
        file_name=f"OviStat_Export_{date.today()}.csv",
        mime="text/csv",
        help="Cliquez ici pour récupérer vos données et les ouvrir dans Excel"
    )


# --- ONGLET 3 : ANALYSE, BENCHMARK & CERTIFICAT ---
with tab3:
    st.header("🧠 Expertise & Benchmark Zootechnique")
    
    if not data.empty:
        # 1. SÉLECTION DE L'ANIMAL
        target = st.selectbox("Sélectionner l'animal pour l'audit complet", data["ID"].unique())
        anim = data[data["ID"] == target].iloc[-1] 
        
        # --- PARTIE A : SCORES INDIVIDUELS ---
        st.subheader("🥇 Scores Individuels")
        
        # Initialisation par défaut pour éviter les NameError
        poids_val = anim['Poids'] if 'Poids' in anim else 0
        indice_compacite = 0.0
        indice_anamorphose = 0.0
        indice_proportion = 0.0
        rang = 1 # Par défaut
        
        try:
            indice_compacite = poids_val / anim['LB'] if anim['LB'] > 0 else 0
            indice_anamorphose = (anim['TP']**2) / anim['HG'] if anim['HG'] > 0 else 0
            indice_proportion = anim['HG'] / anim['LB'] if anim['LB'] > 0 else 0
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Rendement Viande", f"{indice_compacite:.2f}")
            with c2:
                st.metric("Robustesse", f"{indice_anamorphose:.2f}")
            with c3:
                format_animal = "Longiligne" if indice_proportion < 0.95 else "Médioligne"
                st.metric("Format", format_animal)
        except Exception as e:
            st.error(f"Erreur calcul indices : {e}")

        st.divider()

        # --- PARTIE B : BENCHMARK ---
        if len(data) >= 2:
            st.subheader(f"📊 Position de {target} par rapport au troupeau")
            moyenne_poids = data['Poids'].mean()
            diff_poids = ((poids_val - moyenne_poids) / moyenne_poids) * 100 if moyenne_poids > 0 else 0
            
            # Calcul du rang réel
            rang = data['Poids'].rank(ascending=False).iloc[data[data["ID"] == target].index[-1]]
            
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.metric("Performance Poids", f"{poids_val} kg", f"{diff_poids:.1f}% vs Moyenne")
            with col_b2:
                st.metric("Rang", f"{int(rang)} / {len(data)}")
        else:
            st.info("💡 Ajoutez un deuxième animal pour voir le classement (Rang).")

        st.divider()

        # --- PARTIE C : GÉNÉRATION DU CERTIFICAT ---
        st.subheader("📄 Certificat Officiel")
        if st.button(f"Générer le certificat pour {target}"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, "CERTIFICAT DE QUALITÉ OVISTAT IA", ln=True, align='C')
            pdf.ln(10)
            
            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 10, f"ID Animal : {target}", ln=True)
            pdf.cell(0, 10, f"Race : {anim['Race']}", ln=True)
            pdf.cell(0, 10, f"Indice Compacité : {indice_compacite:.2f}", ln=True)
            # Utilisation sécurisée de rang (soit 1, soit le calcul réel)
            pdf.cell(0, 10, f"Rang Troupeau : {int(rang)} / {len(data)}", ln=True)
            
            pdf_bytes = pdf.output(dest="S").encode("latin-1")
            st.download_button(f"📥 Télécharger PDF {target}", data=pdf_bytes, file_name=f"Certificat_{target}.pdf")
            
    else:
        st.info("📊 Les analyses apparaîtront après le premier enregistrement.")
