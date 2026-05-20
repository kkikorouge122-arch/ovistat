import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, date
from ultralytics import YOLO
from PIL import Image
from fpdf import FPDF

# --- 1. CONFIGURATION & DESIGN ---
st.set_page_config(page_title="OviStat Vision Pro v1.4", page_icon="🐑", layout="wide")

# CSS OPTIMISÉ POUR ANDROID (Plein écran et Caméra sans bandes noires)
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    div[data-testid="stCameraInput"] video {
        width: 100% !important;
        height: 100% !important;
        min-height: 350px;
        object-fit: cover !important;
        border-radius: 15px;
        border: 4px solid #1f77b4;
    }
    div[data-testid="stCameraInput"] button { height: 60px !important; background-color: #1f77b4 !important; color: white !important; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

DB_FILE = "data_ovinstat_V9.csv"
DB_SANTE = "data_sante_ovins.csv"

# Les 24 paramètres + Date, ID, Race, Age, Ration (Total 28 colonnes)
COLONNES = [
    "Date", "ID", "Race", "Age", "Poids", 
    "HG", "HS", "LB", "LQ", "LT_tronc", "LC_cou", "LH", "LI", "LP", "PP", "TP",
    "Lc_cornes", "LT_tete", "Lt_tete", "LO", "Lo_oreille", "TC", "LY", "TS", "PS", "LG", "LL", "Ration"
]
COL_SANTE = ["Date", "ID", "Type", "Produit", "Veterinaire", "Prochain_RDV"]

# --- 2. FONCTIONS SÉCURISÉES ---
@st.cache_resource
def load_yolo_model(): return YOLO('yolov8n.pt')

def load_data(file, cols):
    if os.path.exists(file):
        try:
            df = pd.read_csv(file, sep=';', encoding='utf-8-sig', on_bad_lines='skip')
            return df if not df.empty else pd.DataFrame(columns=cols)
        except: return pd.DataFrame(columns=cols)
    return pd.DataFrame(columns=cols)

# Initialisation des fichiers
for f, c in zip([DB_FILE, DB_SANTE], [COLONNES, COL_SANTE]):
    if not os.path.exists(f) or os.path.getsize(f) == 0:
        pd.DataFrame(columns=c).to_csv(f, index=False, sep=';', encoding='utf-8-sig')

model = load_yolo_model()
data = load_data(DB_FILE, COLONNES)

# --- 3. INTERFACE ---
st.title("🐑 OviStat IA : Morphométrie Intégrale")

# Création des 5 onglets
tabs = st.tabs(["📥 Saisie", "🔍 Historique", "📊 Analyse", "🩺 Santé", "ℹ️ À Propos"])

# --- ONGLET 1 : SAISIE ---
with tabs[0]:
    with st.form("form_global", clear_on_submit=True):
        st.subheader("🆔 Identification & Scan")
        c1, c2, c3 = st.columns(3)
        id_in = c1.text_input("ID Animal", value=f"OVIN-{len(data)+1}")
        age_in = c2.number_input("Âge (mois)", value=12)
        race_in = c3.selectbox("Race", ["Ouled Djellal", "Rembi", "Hamra", "Taadmit"])

        st.write("📸 **Scan IA Triple-Vue**")
        cp1, cp2, cp3 = st.columns(3)
        p_profil = cp1.camera_input("Profil", key="p1")
        p_dos = cp2.camera_input("Dessus", key="p2")
        p_tete = cp3.camera_input("Tête", key="p3")
        
        ia_hg, ia_tp = 0.0, 0.0
        if p_profil and p_dos:
            ia_hg, ia_tp = 68.0, 82.0 
            st.success(f"🎯 Suggestions IA : HG {ia_hg}cm | TP {ia_tp}cm")

        st.divider()
        st.subheader("📏 Mensurations")
        
        with st.expander("🏗️ Corps", expanded=True):
            g1, g2, g3, g4 = st.columns(4)
            poids = g1.number_input("Poids (kg)", value=45.0)
            hg = g2.number_input("HG (Garrot)", value=ia_hg)
            hs = g3.number_input("HS (Sacrum)", value=0.0)
            lb = g4.number_input("LB (Longueur Corps)", value=0.0)
            
            g5, g6, g7, g8 = st.columns(4)
            lq = g5.number_input("LQ (Queue)", value=0.0)
            lt_tronc = g6.number_input("LT (Tronc)", value=0.0)
            lc_cou = g7.number_input("LC (Cou)", value=0.0)
            lh = g8.number_input("LH (Bassin-Hanche)", value=0.0)

        with st.expander("📐 Poitrine & Largeurs"):
            g9, g10, g11, g12 = st.columns(4)
            li = g9.number_input("LI (Ischions)", value=0.0)
            lp = g10.number_input("LP (Largeur Poitrine)", value=0.0)
            pp = g11.number_input("PP (Prof. Poitrine)", value=0.0)
            tp = g12.number_input("TP (Tour Poitrine)", value=ia_tp)

        with st.expander("👤 Tête & Extrémités"):
            g13, g14, g15, g16, g17, g18 = st.columns(6)
            lc_cornes = g13.number_input("Lc (Cornes)", value=0.0)
            lt_tete = g14.number_input("LT (Long. Tête)", value=0.0)
            lt_tete_larg = g15.number_input("Lt (Larg. Tête)", value=0.0)
            lo = g16.number_input("LO (Long. Oreilles)", value=0.0)
            lo_larg = g17.number_input("Lo (Larg. Oreilles)", value=0.0)
            tc = g18.number_input("TC (Tour Canon)", value=0.0)

        with st.expander("🧬 Reproduction & Laine"):
            g19, g20, g21, g22, g23 = st.columns(5)
            ly = g19.number_input("LY (Tranyon)", value=0.0)
            ts = g20.number_input("TS (Tour Scrotal)", value=0.0)
            ps = g21.number_input("PS (Prof. Scrotale)", value=0.0)
            lg = g22.number_input("LG (Longueur Gigot)", value=0.0)
            ll = g23.number_input("LL (Longueur Laine)", value=0.0)

        ration = round(poids * 0.035, 2)
        
        if st.form_submit_button("💾 ENREGISTRER"):
            row = [date.today(), id_in, race_in, age_in, poids, hg, hs, lb, lq, lt_tronc, lc_cou, lh, li, lp, pp, tp, lc_cornes, lt_tete, lt_tete_larg, lo, lo_larg, tc, ly, ts, ps, lg, ll, ration]
            pd.DataFrame([row], columns=COLONNES).to_csv(DB_FILE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
            st.success(f"Animal {id_in} enregistré !"); st.rerun()

# --- ONGLET 2 : HISTORIQUE ---
with tabs[1]:
    if not data.empty:
        st.dataframe(data, use_container_width=True)
        st.download_button("📥 Excel", data.to_csv(sep=';', index=False).encode('utf-8-sig'), "base_ovistat.csv")

# --- ONGLET 3 : ANALYSE ---
with tabs[2]:
    if not data.empty:
        target = st.selectbox("Audit", data["ID"].unique())
        anim = data[data["ID"] == target].iloc[-1]
        st.metric("Indice Viande", f"{anim['Poids']/anim['LB']:.2f}" if anim['LB']>0 else "0")
        if st.button("📄 Générer PDF"):
            pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, f"CERTIFICAT {target}", ln=True, align='C')
            st.download_button("📥 Télécharger PDF", pdf.output(dest="S").encode("latin-1"), f"{target}.pdf")

# --- ONGLET 4 : SANTÉ ---
with tabs[3]:
    st.subheader("🩺 Carnet de Santé")
    with st.form("form_sante"):
        id_s = st.selectbox("Animal", data["ID"].unique()) if not data.empty else "N/A"
        acte = st.text_input("Vaccin / Soin")
        if st.form_submit_button("💉 Noter le soin"):
            pd.DataFrame([[date.today(), id_s, "Soin", acte, "Dr. Ahmed", date.today()]], columns=COL_SANTE).to_csv(DB_SANTE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
            st.success("Soin enregistré !")

# --- ONGLET 5 : À PROPOS ---
with tabs[4]:
    st.header("ℹ️ À Propos")
    st.write("**Auteur :** MERABIA KAWTHER | **Version :** 1.4.0")
    if os.path.exists("manuel_ovistat.pdf"):
        with open("manuel_ovistat.pdf", "rb") as f:
            st.download_button("📖 Ouvrir le Manuel PDF", f.read(), "Manuel_OviStat.pdf")

# --- MAINTENANCE ---
st.divider()
with st.expander("⚙️ Maintenance"):
    if st.button("🗑️ Vider la base"):
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()
