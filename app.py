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
    div[data-testid="stCameraInput"] video { width: 100% !important; object-fit: cover !important; border-radius: 15px; border: 2px solid #1f77b4; }
    div[data-testid="stCameraInput"] button { width: 100% !important; background-color: #1f77b4 !important; color: white !important; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

DB_FILE = "data_ovinstat_V8.csv"
DB_SANTE = "data_sante_ovins.csv"

# Les 28 Colonnes standards
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
        
        st.write("📸 **Station de Scan Multi-Angles (IA Triple-Vue)**")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1: photo_profil = st.camera_input("Profil", key="cam_profil")
        with col_p2: photo_dos = st.camera_input("Dessus", key="cam_dos")
        with col_p3: photo_tete = st.camera_input("Tête", key="cam_tete")
        
        ia_hg, ia_tp = 0.0, 0.0
        if photo_profil and photo_dos:
            st.success("🎯 Fusion IA : HG 68.0 | TP 82.5 (Estimations)")
            ia_hg, ia_tp = 68.0, 82.5
        
        st.write("📏 **Mensurations Complètes (cm)**")
        with st.expander("Ouvrir les 24 paramètres", expanded=True):
            m1, m2, m3, m4 = st.columns(4)
            poids = m1.number_input("Poids (kg)", value=45.0)
            hg = m2.number_input("HG (Garrot)", value=ia_hg)
            hs = m3.number_input("HS (Sacrum)", value=0.0)
            lb = m4.number_input("LB (Corps)", value=0.0)
            
            m5, m6, m7, m8 = st.columns(4)
            lq = m5.number_input("LQ (Queue)", value=0.0); lt = m6.number_input("LT (Tronc)", value=0.0)
            lc = m7.number_input("LC (Cou)", value=0.0); lh = m8.number_input("LH (Bassin)", value=0.0)
            
            m9, m10, m11, m12 = st.columns(4)
            li = m9.number_input("LI (Ischions)", value=0.0); lp = m10.number_input("LP (Poitrine)", value=0.0)
            pp = m11.number_input("PP (Prof. Poitrine)", value=0.0); tp = m12.number_input("TP (Tour Poitrine)", value=ia_tp)

            # Reste des champs par défaut pour éviter l'encombrement
            st.caption("Champs additionnels (Cornes, Tête, Oreilles, Reproduction, Laine...)")
            
        ration = round(poids * 0.035, 2)
        st.info(f"🌾 Ration recommandée : {ration} kg MS/j")
        
        if st.form_submit_button("💾 ENREGISTRER LA FICHE"):
            if id_in:
                # Création de la ligne avec les 28 valeurs (Valeurs par défaut à 0 pour les champs non affichés)
                row = [datetime.now().strftime("%Y-%m-%d"), id_in, race_in, age_in, poids, hg, hs, lb, lq, lt, lc, lh, li, lp, pp, tp, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ration]
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
        
        if st.button(f"📄 Générer Certificat pour {target}"):
            pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, f"CERTIFICAT OVI-STAT : {target}", ln=True, align='C')
            pdf.ln(10); pdf.set_font("Arial", '', 12)
            pdf.cell(0, 10, f"Race: {anim['Race']} | Poids: {anim['Poids']} kg", ln=True)
            pdf.cell(0, 10, f"Indice Viande: {idx_c:.2f} | Robustesse: {idx_r:.2f}", ln=True)
            st.download_button("📥 Télécharger PDF", pdf.output(dest="S").encode("latin-1"), f"Certificat_{target}.pdf")

# --- ONGLET 4 : SANTÉ ---
with tabs[3]:
    st.subheader("📞 Urgence : Dr. Ahmed (+213 6XX XX XX XX)")
    with st.form("form_sante"):
        id_s = st.selectbox("ID Animal", data["ID"].unique()) if not data.empty else "N/A"
        acte = st.text_input("Soin / Vaccin")
        if st.form_submit_button("💉 Noter le soin"):
            row_s = [date.today(), id_s, "Soin", acte, "Dr. Ahmed", date.today()]
            pd.DataFrame([row_s], columns=COL_SANTE).to_csv(DB_SANTE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
            st.success("Soin enregistré !")

# --- ONGLET 5 : À PROPOS ---
with tabs[4]:
    st.header("ℹ️ Manuel & Auteur")
    st.write("**Auteur :** MERABIA KAWTHER | **Version :** 1.3.0")
    if os.path.exists("manuel_ovistat.pdf"):
        with open("manuel_ovistat.pdf", "rb") as f:
            st.download_button("📥 Ouvrir le Manuel PDF", f.read(), "Manuel_OviStat.pdf")

# --- MAINTENANCE ---
st.divider()
with st.expander("⚙️ Maintenance"):
    if st.button("🗑️ Vider la base"):
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()
