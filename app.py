import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, date
from ultralytics import YOLO
from PIL import Image
from fpdf import FPDF

# --- 1. CONFIGURATION & DESIGN ---
st.set_page_config(page_title="OviStat Vision Pro v1.5", page_icon="🐑", layout="wide")

st.markdown("""
    <style>
    .block-container { padding-top: 1rem; }
    div[data-testid="stCameraInput"] video { width: 100% !important; object-fit: cover !important; border-radius: 15px; border: 4px solid #1f77b4; }
    div[data-testid="stCameraInput"] button { height: 60px !important; background-color: #1f77b4 !important; color: white !important; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

DB_FILE = "data_ovinstat_V10.csv"
DB_SANTE = "data_sante_ovins.csv"

COLONNES = [
    "Date", "ID", "Race", "Age", "Poids", 
    "HG", "HS", "LB", "LQ", "LT_tronc", "LC_cou", "LH", "LI", "LP", "PP", "TP",
    "Lc_cornes", "LT_tete", "Lt_tete", "LO", "Lo_oreille", "TC", "LY", "TS", "PS", "LG", "LL", "Ration"
]
COL_SANTE = ["Date", "ID", "Type", "Produit", "Veterinaire", "Prochain_RDV"]

# --- 2. FONCTIONS ---
@st.cache_resource
def load_yolo_model(): return YOLO('yolov8n.pt')

def load_data(file, cols):
    if os.path.exists(file):
        try:
            df = pd.read_csv(file, sep=';', encoding='utf-8-sig', on_bad_lines='skip')
            return df if not df.empty else pd.DataFrame(columns=cols)
        except: return pd.DataFrame(columns=cols)
    return pd.DataFrame(columns=cols)

for f, c in zip([DB_FILE, DB_SANTE], [COLONNES, COL_SANTE]):
    if not os.path.exists(f) or os.path.getsize(f) == 0:
        pd.DataFrame(columns=c).to_csv(f, index=False, sep=';', encoding='utf-8-sig')

model = load_yolo_model()
data = load_data(DB_FILE, COLONNES)

# --- 3. INTERFACE ---
st.title("🐑 OviStat IA : Scan Séquentiel")

tabs = st.tabs(["📥 Saisie Terrain", "🔍 Historique", "📊 Analyse", "🩺 Santé", "ℹ️ À Propos"])

# --- ONGLET 1 : SAISIE (Système Unique Caméra) ---
with tabs[0]:
    # Gestion du cycle des photos via st.session_state
    if 'step' not in st.session_state: st.session_state.step = 1
    if 'temp_data' not in st.session_state: st.session_state.temp_data = {}

    with st.form("form_global", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        id_in = c1.text_input("ID Animal", value=f"OVIN-{len(data)+1}")
        age_in = c2.number_input("Âge (mois)", value=12)
        race_in = c3.selectbox("Race", ["Ouled Djellal", "Rembi", "Hamra", "Taadmit"])

        st.divider()
        
        # BARRE DE PROGRESSION DU SCAN
        steps = ["📸 Profil", "📏 Dessus", "🐏 Tête", "✅ Terminé"]
        st.progress(st.session_state.step / 3)
        st.write(f"**Étape actuelle : {steps[st.session_state.step-1]}**")
        
        photo = st.camera_input("Scanner l'angle demandé")
        
        ia_hg, ia_tp = 0.0, 0.0
        
        if photo:
            if st.session_state.step == 1:
                st.session_state.temp_data['profil'] = photo
                st.success("Photo Profil enregistrée ! Passez à la suivante.")
            elif st.session_state.step == 2:
                st.session_state.temp_data['dos'] = photo
                st.success("Photo Dos enregistrée !")
            elif st.session_state.step == 3:
                st.session_state.temp_data['tete'] = photo
                st.success("Toutes les photos sont prêtes !")
            
            # Bouton pour passer à la photo suivante
            if st.session_state.step < 3:
                if st.form_submit_button("➡️ Passer à la photo suivante"):
                    st.session_state.step += 1
                    st.rerun()

        # LOGIQUE IA (Une fois les photos prises)
        if 'profil' in st.session_state.temp_data and 'dos' in st.session_state.temp_data:
            ia_hg, ia_tp = 68.0, 82.0 # Simulation IA
            st.info(f"🎯 Suggestions IA : HG {ia_hg}cm | TP {ia_tp}cm")

        st.divider()
        st.subheader("📏 Mensurations finales")
        with st.expander("Saisie des 24 paramètres"):
            g1, g2, g3 = st.columns(3)
            poids = g1.number_input("Poids (kg)", value=45.0)
            hg = g2.number_input("HG (Garrot)", value=ia_hg)
            tp = g3.number_input("TP (Tour Poitrine)", value=ia_tp)
            # Ajoutez ici les autres g4, g5... pour LB, LT, etc.
        
        if st.form_submit_button("💾 ENREGISTRER LA FICHE DÉFINITIVE"):
            if id_in:
                # Création de la ligne (28 colonnes)
                row = [date.today(), id_in, race_in, age_in, poids, hg, 0, 0, 0, 0, 0, 0, 0, 0, 0, tp, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, round(poids*0.035,2)]
                pd.DataFrame([row], columns=COLONNES).to_csv(DB_FILE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
                # Reset pour l'animal suivant
                st.session_state.step = 1
                st.session_state.temp_data = {}
                st.success("Animal enregistré !"); st.rerun()


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
    st.write("**Auteur :** MERABIA KAWTHER | **Version :** 1.5.0")
    if st.button("🔄 Réinitialiser le cycle de scan"):
        st.session_state.step = 1
        st.session_state.temp_data = {}
        st.rerun()

# --- MAINTENANCE ---
st.sidebar.divider()
with st.sidebar.expander("⚙️ Maintenance"):
    if st.button("🗑️ Vider la base"):
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

