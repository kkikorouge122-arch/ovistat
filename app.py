import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, date
from ultralytics import YOLO
from PIL import Image
from fpdf import FPDF

# --- 1. CONFIGURATION & VARIABLES ---
st.set_page_config(page_title="OviStat Vision Pro v1.2", layout="wide")

DB_FILE = "data_ovinstat_V5.csv"
DB_SANTE = "data_sante_ovins.csv"

COLONNES = [
    "Date", "ID", "Race", "Age", "Poids", 
    "HG", "HS", "LB", "LQ", "LT", "LC", "LH", "LI", "LP", "PP", "TP",
    "Lc_cornes", "LT_tete", "Lt_tete", "LO", "Lo_oreille", "TC", "LY", "TS", "PS", "LG", "LL", "Ration_MS"
]

COL_SANTE = ["Date", "ID", "Type", "Produit", "Veterinaire", "Prochain_RDV"]

# --- 2. FONCTIONS TECHNIQUES ---
@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n.pt')

def load_data(file, cols):
    if os.path.exists(file):
        return pd.read_csv(file, sep=';', encoding='utf-8-sig', on_bad_lines='skip')
    return pd.DataFrame(columns=cols)

# Initialisation
if not os.path.exists(DB_FILE):
    pd.DataFrame(columns=COLONNES).to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')
if not os.path.exists(DB_SANTE):
    pd.DataFrame(columns=COL_SANTE).to_csv(DB_SANTE, index=False, sep=';', encoding='utf-8-sig')

model = load_yolo_model()
data = load_data(DB_FILE, COLONNES)

def get_next_id(df):
    if df.empty: return "OVIN-1"
    return f"OVIN-{len(df['ID'].unique()) + 1}"

# --- 3. STYLE CSS ---
st.markdown("""
    <style>
    div[data-testid="stCameraInput"] video { width: 100% !important; border-radius: 15px; border: 3px solid #1f77b4; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. INTERFACE ---
st.title("🐑 OviStat Vision Pro")
st.sidebar.caption("Version 1.2.0 | IA Activable")

tabs = st.tabs(["📥 Saisie IA", "🔍 Historique", "📊 Analyse", "🩺 Santé", "ℹ️ À Propos"])

# --- ONGLET 1 : SAISIE ---
with tabs[0]:
    with st.form("form_global", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        id_in = c1.text_input("ID Animal", value=get_next_id(data))
        age_in = c2.number_input("Âge (mois)", value=12)
        race_in = c3.selectbox("Race", ["Ouled Djellal", "Rembi", "Hamra", "Taadmit"])
        
        st.divider()
        st.write("📸 **Scan IA**")
        cv1, c_photo, cv2 = st.columns([0.1, 0.8, 0.1])
        with c_photo: photo = st.camera_input("Scanner")
        
        ia_val = 0.0
        if photo:
            img = Image.open(photo); results = model(img)
            if any(int(box.cls) == 18 for r in results for box in r.boxes):
                st.success("✅ Ovin détecté ! Suggestions appliquées."); ia_val = 65.0
        
        st.write("📏 **Mensurations**")
        with st.expander("Ouvrir les 24 paramètres", expanded=True):
            m1, m2, m3 = st.columns(3)
            poids = m1.number_input("Poids (kg)", value=45.0)
            hg = m2.number_input("HG (Garrot)", value=ia_val)
            tp = m3.number_input("TP (Poitrine)", value=ia_val)
            lb = m1.number_input("LB (Corps)", value=ia_val)
            # Les autres champs simplifiés pour l'exemple
            ts = m2.number_input("TS (Scrotal)", value=0.0)
            
        ration = round(poids * 0.035, 2)
        st.info(f"🌾 Ration : {ration} kg MS/j")
        
        if st.form_submit_button("💾 ENREGISTRER"):
            row = [datetime.now().strftime("%Y-%m-%d"), id_in, race_in, age_in, poids, hg, 0, lb, 0, 0, 0, 0, 0, 0, 0, tp, 0, 0, 0, 0, 0, 0, 0, ts, 0, 0, 0, ration]
            pd.DataFrame([row], columns=COLONNES).to_csv(DB_FILE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
            st.rerun()

# --- ONGLET 2 : HISTORIQUE ---
with tabs[1]:
    if not data.empty:
        st.dataframe(data)
        st.download_button("📥 Export Excel", data.to_csv(sep=';', index=False).encode('utf-8-sig'), "base.csv")

# --- ONGLET 3 : ANALYSE ---
with tabs[2]:
    if not data.empty:
        target = st.selectbox("Sélection", data["ID"].unique())
        anim = data[data["ID"] == target].iloc[-1]
        c1, c2 = st.columns(2)
        idx_c = anim['Poids'] / anim['LB'] if anim['LB'] > 0 else 0
        c1.metric("Indice Viande", f"{idx_c:.2f}")
        
        if st.button("📄 Générer Certificat"):
            pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, f"CERTIFICAT : {target}", ln=True, align='C')
            st.download_button("📥 Télécharger PDF", pdf.output(dest="S").encode("latin-1"), f"{target}.pdf")

# --- ONGLET 4 : SANTÉ ---
with tabs[3]:
    st.subheader("📞 Urgence : Dr. Ahmed (+213 6XX XX XX XX)")
    with st.form("form_sante"):
        id_s = st.selectbox("ID", data["ID"].unique()) if not data.empty else "N/A"
        acte = st.text_input("Acte / Vaccin")
        if st.form_submit_button("💉 Noter le soin"):
            row_s = [date.today(), id_s, "Soin", acte, "Dr. Ahmed", date.today()]
            pd.DataFrame([row_s], columns=COL_SANTE).to_csv(DB_SANTE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')

# --- ONGLET 5 : À PROPOS ---
with tabs[4]:
    st.header("ℹ️ Manuel & Auteur")
    st.write("**Auteur :** [MERABIA KAWTHER] | **Version :** 1.2.0")
    with st.expander("📖 Guide Rapide"):
        st.write("1. Scannez l'animal en profil.\n2. Validez les mesures suggérées.\n3. Consultez les indices en Analyse.")
with tabs[5]:
    st.subheader("📖 Manuel d'Utilisation Complet")
        # Vérifier si le fichier existe sur le serveur
    if os.path.exists("manuel_ovistat.pdf"):
        with open("manuel_ovistat.pdf", "rb") as f:
            pdf_data = f.read()
        
        st.download_button(
            label="📥 Cliquer ici pour ouvrir le Manuel Complet (PDF)",
            data=pdf_data,
            file_name="Manuel_Utilisation_OviStat.pdf",
            mime="application/pdf",
            help="Téléchargez le guide illustré pour apprendre à utiliser l'IA et les mesures."
        )
        st.success("Le manuel est disponible ! Téléchargez-le pour une lecture hors-ligne à la ferme.")
    else:
        st.warning("⚠️ Le fichier 'manuel_ovistat.pdf' est introuvable sur le serveur. Veuillez l'ajouter à votre GitHub.")

# --- MAINTENANCE ---
with st.expander("⚙️ Maintenance"):
    st.warning("Attention : Action irréversible")
    if st.button("🗑️ Vider la base de données"):
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
            st.success("Fichier supprimé !")
            st.rerun()
    st.info(f"📍 Serveur : {os.getcwd()}")

