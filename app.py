import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, date
from ultralytics import YOLO
from PIL import Image
from fpdf import FPDF

# --- 1. CONFIGURATION & DESIGN ---
st.set_page_config(page_title="OviStat Vision Pro", page_icon="🐑", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .block-container { padding-top: 1rem; }
    div[data-testid="stCameraInput"] video { width: 100% !important; object-fit: cover !important; border-radius: 15px; border: 3px solid #1f77b4; }
    div[data-testid="stCameraInput"] button { width: 100% !important; background-color: #1f77b4 !important; color: white !important; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

DB_FILE = "data_ovinstat_V7.csv"
DB_SANTE = "data_sante_ovins.csv"

# Les 28 Colonnes (Date, ID, Race, Age, Poids + 24 Mensurations + Ration)
COLONNES = [
    "Date", "ID", "Race", "Age", "Poids", 
    "HG", "HS", "LB", "LQ", "LT", "LC", "LH", "LI", "LP", "PP", "TP",
    "Lcornes", "LTete", "LtTete", "LO", "Lo", "TC", "LY", "TS", "PS", "LG", "LL", "Ration"
]
COL_SANTE = ["Date", "ID", "Type", "Produit", "Veterinaire", "Prochain_RDV"]

# --- 2. FONCTIONS ---
@st.cache_resource
def load_yolo_model(): return YOLO('yolov8n.pt')

def load_data(file, cols):
    if os.path.exists(file): return pd.read_csv(file, sep=';', encoding='utf-8-sig', on_bad_lines='skip')
    return pd.DataFrame(columns=cols)

for f, c in zip([DB_FILE, DB_SANTE], [COLONNES, COL_SANTE]):
    if not os.path.exists(f): pd.DataFrame(columns=c).to_csv(f, index=False, sep=';', encoding='utf-8-sig')

model = load_yolo_model()
data = load_data(DB_FILE, COLONNES)

def get_next_id(df):
    if df.empty: return "OVIN-1"
    return f"OVIN-{len(df['ID'].unique()) + 1}"

# --- 3. INTERFACE ---
st.title("🐑 OviStat Vision Pro")
tabs = st.tabs(["📥 Saisie IA", "🔍 Historique", "📊 Analyse", "🩺 Santé", "ℹ️ À Propos"])

# --- ONGLET 1 : SAISIE ---
with tabs[0]:
    with st.form("form_global", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        id_in = c1.text_input("ID Animal", value=get_next_id(data))
        age_in = c2.number_input("Âge (mois)", value=12)
        race_in = c3.selectbox("Race", ["Ouled Djellal", "Rembi", "Hamra", "Taadmit"])
        
        st.write("📸 **Scan IA**")
        cv1, c_photo, cv2 = st.columns([0.05, 0.9, 0.05])
        with c_photo: photo = st.camera_input("Scanner")
        
        ia_v = 0.0
        if photo:
            img = Image.open(photo); results = model(img)
            if any(int(box.cls) == 18 for r in results for box in r.boxes):
                st.success("✅ Ovin détecté ! Suggestions activées."); ia_v = 65.0
        
        st.write("📏 **Mensurations Complètes (cm)**")
        with st.expander("Ouvrir les 24 paramètres", expanded=True):
            # Ligne 1
            m1, m2, m3, m4 = st.columns(4)
            poids = m1.number_input("Poids (kg)", value=45.0)
            hg = m2.number_input("HG (Garrot)", value=ia_v)
            hs = m3.number_input("HS (Sacrum)", value=0.0)
            lb = m4.number_input("LB (Corps)", value=0.0)
            # Ligne 2
            m5, m6, m7, m8 = st.columns(4)
            lq = m5.number_input("LQ (Queue)", value=0.0)
            lt = m6.number_input("LT (Tronc)", value=0.0)
            lc = m7.number_input("LC (Cou)", value=0.0)
            lh = m8.number_input("LH (Bassin)", value=0.0)
            # Ligne 3
            m9, m10, m11, m12 = st.columns(4)
            li = m9.number_input("LI (Ischions)", value=0.0)
            lp = m10.number_input("LP (Poitrine)", value=0.0)
            pp = m11.number_input("PP (Prof. Poitrine)", value=0.0)
            tp = m12.number_input("TP (Tour Poitrine)", value=ia_v)
            # Ligne 4
            m13, m14, m15, m16 = st.columns(4)
            lc_cornes = m13.number_input("Lc (Cornes)", value=0.0)
            lt_tete = m14.number_input("LT (Long. Tête)", value=0.0)
            lt_larg = m15.number_input("Lt (Larg. Tête)", value=0.0)
            lo = m16.number_input("LO (Long. Oreille)", value=0.0)
            # Ligne 5
            m17, m18, m19, m20 = st.columns(4)
            lo_larg = m17.number_input("Lo (Larg. Oreille)", value=0.0)
            tc = m18.number_input("TC (Canon)", value=0.0)
            ly = m19.number_input("LY (Tranyon)", value=0.0)
            ts = m20.number_input("TS (Scrotal)", value=0.0)
            # Ligne 6
            m21, m22, m23, m24 = st.columns(4)
            ps = m21.number_input("PS (Prof. Scrotale)", value=0.0)
            lg = m22.number_input("LG (Gigot)", value=0.0)
            ll = m23.number_input("LL (Laine)", value=0.0)
            
        ration = round(poids * 0.035, 2)
        st.info(f"🌾 Ration : {ration} kg MS/j")
        
        if st.form_submit_button("💾 ENREGISTRER LA FICHE"):
            if id_in:
                row = [datetime.now().strftime("%Y-%m-%d"), id_in, race_in, age_in, poids, hg, hs, lb, lq, lt, lc, lh, li, lp, pp, tp, lc_cornes, lt_tete, lt_larg, lo, lo_larg, tc, ly, ts, ps, lg, ll, ration]
                pd.DataFrame([row], columns=COLONNES).to_csv(DB_FILE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
                st.success("✅ Enregistré !"); st.balloons(); st.rerun()

# --- ONGLET 2 : HISTORIQUE ---
with tabs[1]:
    if not data.empty:
        st.dataframe(data, use_container_width=True)
        st.download_button("📥 Export Excel", data.to_csv(sep=';', index=False).encode('utf-8-sig'), "OviStat_Data.csv")

# --- ONGLET 3 : ANALYSE ---
with tabs[2]:
    if not data.empty:
        target = st.selectbox("Sélectionner l'animal", data["ID"].unique())
        anim = data[data["ID"] == target].iloc[-1]
        c1, c2, c3 = st.columns(3)
        idx_c = anim['Poids'] / anim['LB'] if anim['LB'] > 0 else 0
        idx_r = (anim['TP']**2) / anim['HG'] if anim['HG'] > 0 else 0
        c1.metric("Indice Viande", f"{idx_c:.2f}")
        c2.metric("Robustesse", f"{idx_r:.2f}")
        
        if st.button("📄 Générer Certificat PDF"):
            pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, f"CERTIFICAT OVI-STAT : {target}", ln=True, align='C')
            pdf.set_font("Arial", '', 12); pdf.ln(10)
            pdf.cell(0, 10, f"Race: {anim['Race']} | Indice Viande: {idx_c:.2f}", ln=True)
            st.download_button("📥 Télécharger PDF", pdf.output(dest="S").encode("latin-1"), f"{target}.pdf")

# --- ONGLET 5 : À PROPOS ---
with tabs[4]:
    st.header("ℹ️ Manuel & Auteur")
    st.write("**Auteur :** MERABIA KAWTHER | **Version :** 1.3.0")
    if os.path.exists("manuel_ovistat.pdf"):
        with open("manuel_ovistat.pdf", "rb") as f:
            st.download_button("📥 Ouvrir le Manuel PDF", f.read(), "Manuel_OviStat.pdf")
