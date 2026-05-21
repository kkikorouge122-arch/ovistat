import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, date
from ultralytics import YOLO
from PIL import Image
from fpdf import FPDF

# --- 1. CONFIGURATION & DESIGN ---
st.set_page_config(page_title="OviStat Vision Pro v1.7", page_icon="🐑", layout="wide")

st.markdown("""
    <style>
    .block-container { padding-top: 1rem; }
    div[data-testid="stCameraInput"] video { width: 100% !important; object-fit: cover !important; border-radius: 15px; border: 3px solid #1f77b4; }
    div[data-testid="stCameraInput"] button { height: 60px !important; background-color: #1f77b4 !important; color: white !important; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

DB_FILE = "data_ovinstat_V12.csv"
DB_SANTE = "data_sante_ovins.csv"

# LISTE COMPLÈTE DES 30 COLONNES
COLONNES = [
    "Date", "ID", "Race", "Age", "Poids", 
    "HG", "HS", "LB", "LQ", "LT_tronc", "LC_cou", "LH", "LI", "LP", "PP", "TP",
    "Lc_cornes", "LT_tete", "Lt_tete", "LO", "Lo_oreille", "TC", "LY", "TS", "PS", "LG", "LL", 
    "Wilaya", "Commune", "Ration"
]
COL_SANTE = ["Date", "ID", "Type", "Produit", "Veterinaire", "Prochain_RDV"]

# --- 2. FONCTIONS ---
@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n.pt')

@st.cache_data
def get_algeria_geo():
    wilayas = [
        "01-Adrar", "02-Chlef", "03-Laghouat", "04-Oum El Bouaghi", "05-Batna", "06-Béjaïa", "07-Biskra", "08-Béchar", "09-Blida", "10-Bouira",
        "11-Tamanrasset", "12-Tébessa", "13-Tlemcen", "14-Tiaret", "15-Tizi Ouzou", "16-Alger", "17-Djelfa", "18-Jijel", "19-Sétif", "20-Saïda",
        "21-Skikda", "22-Sidi Bel Abbès", "23-Annaba", "24-Guelma", "25-Constantine", "26-Médéa", "27-Mostaganem", "28-M'Sila", "29-Mascara", "30-Ouargla",
        "31-Oran", "32-El Bayadh", "33-Illizi", "34-Bordj Bou Arreridj", "35-Boumerès", "36-El Tarf", "37-Tindouf", "38-Tissemsilt", "39-El Oued", "40-Khenchela",
        "41-Souk Ahras", "42-Tipaza", "43-Mila", "44-Aïn Defla", "45-Naâma", "46-Aïn Témouchent", "47-Ghardaïa", "48-Relizane", "49-El M'Ghair", "50-El Meniaa",
        "51-Ouled Djellal", "52-Bordj Baji Mokhtar", "53-Béni Abbès", "54-Timimoun", "55-Touggourt", "56-Djanet", "57-In Salah", "58-In Guezzam"
    ]
    if os.path.exists("algeria_geo.csv"):
        try:
            df_geo = pd.read_csv("algeria_geo.csv", sep=";", encoding='utf-8')
            if len(df_geo.columns) < 2:
                df_geo = pd.read_csv("algeria_geo.csv", sep=",", encoding='utf-8')
        except:
            df_geo = pd.DataFrame(columns=["wilaya_name", "commune_name"])
    else:
        df_geo = pd.DataFrame(columns=["wilaya_name", "commune_name"])
    return wilayas, df_geo

def load_data(file, cols):
    if os.path.exists(file):
        try:
            df = pd.read_csv(file, sep=';', encoding='utf-8-sig', on_bad_lines='skip')
            return df if not df.empty else pd.DataFrame(columns=cols)
        except: return pd.DataFrame(columns=cols)
    return pd.DataFrame(columns=cols)

# Initialisation
for f, c in zip([DB_FILE, DB_SANTE], [COLONNES, COL_SANTE]):
    if not os.path.exists(f) or os.path.getsize(f) == 0:
        pd.DataFrame(columns=c).to_csv(f, index=False, sep=';', encoding='utf-8-sig')

model = load_yolo_model()
data = load_data(DB_FILE, COLONNES)
list_wilayas, df_communes = get_algeria_geo()

# --- 3. INTERFACE ---
st.title("🐑 OviStat IA : Expert Algérie")
tabs = st.tabs(["📥 Saisie", "🔍 Historique", "📊 Analyse", "🩺 Santé", "ℹ️ À Propos"])

# --- ONGLET 1 : SAISIE ---
with tabs[0]:
    if 'step' not in st.session_state: st.session_state.step = 1
    
    with st.form("form_global", clear_on_submit=False):
        st.subheader("📍 Localisation de l'étude")
        col_w, col_c = st.columns(2)
        wilaya_sel = col_w.selectbox("Wilaya", list_wilayas)
        w_clean = wilaya_sel.split("-")[-1].strip()
        
        # Filtrage des communes
        if not df_communes.empty:
            mask = df_communes['wilaya_name'].str.strip().str.lower() == w_clean.lower()
            communes_possibles = df_communes[mask]['commune_name'].unique().tolist()
        else:
            communes_possibles = []

        commune_sel = col_c.selectbox("Commune", sorted(communes_possibles) if communes_possibles else ["Saisie manuelle nécessaire"])

        st.divider()
        st.subheader("🆔 Identification")
        c1, c2, c3 = st.columns(3)
        id_in = c1.text_input("ID / Boucle (Vide = Auto-ID)")
        age_in = c2.number_input("Âge (mois)", value=12)
        race_in = c3.selectbox("Race", ["Ouled Djellal", "Rembi", "Hamra", "Taadmit"])
        
        st.divider()
        st.write(f"📸 **Étape {st.session_state.step}/3 :** " + ["Profil", "Dessus", "Tête"][st.session_state.step-1])
        photo = st.camera_input("Capturer")
        
        ia_hg, ia_tp = 0.0, 0.0
        if photo:
            if st.session_state.step < 3:
                if st.form_submit_button(f"➡️ Valider l'étape {st.session_state.step}"):
                    st.session_state.step += 1
                    st.rerun()
            else:
                st.success("✅ Photos prêtes !")
                ia_hg, ia_tp = 68.5, 84.0

        st.divider()
        with st.expander("📏 Mensurations (24 Paramètres)", expanded=True):
            g1, g2, g3, g4 = st.columns(4)
            poids = g1.number_input("Poids (kg)", value=45.0)
            hg = g2.number_input("Hauteur Garrot (HG)", value=ia_hg)
            hs = g3.number_input("Hauteur Sacrum (HS)", value=0.0)
            lb = g4.number_input("Long. Corps (LB)", value=0.0)
            
            g5, g6, g7, g8 = st.columns(4)
            lq = g5.number_input("Long. Queue (LQ)", value=0.0)
            lt_t = g6.number_input("Long. Tronc (LT)", value=0.0)
            lc_c = g7.number_input("Long. Cou (LC)", value=0.0)
            lh = g8.number_input("Long. Bassin (LH)", value=0.0)

            g9, g10, g11, g12 = st.columns(4)
            li = g9.number_input("Larg. Ischions (LI)", value=0.0)
            lp = g10.number_input("Larg. Poitrine (LP)", value=0.0)
            pp = g11.number_input("Prof. Poitrine (PP)", value=0.0)
            tp = g12.number_input("Tour Poitrine (TP)", value=ia_tp)

            st.caption("Paramètres tête et extrémités")
            g13, g14, g15, g16, g17, g18 = st.columns(6)
            lc_cornes = g13.number_input("Lc (Cornes)", value=0.0)
            lt_tete = g14.number_input("LT (Tête)", value=0.0)
            lt_la = g15.number_input("Lt (Tête)", value=0.0)
            lo_lo = g16.number_input("LO (Oreilles)", value=0.0)
            lo_la = g17.number_input("Lo (Oreilles)", value=0.0)
            tc = g18.number_input("TC (Canon)", value=0.0)

            st.caption("Reproduction et Laine")
            g19, g20, g21, g22, g23 = st.columns(5)
            ly = g19.number_input("LY (Trayon)", value=0.0)
            ts = g20.number_input("TS (Scrotal)", value=0.0)
            ps = g21.number_input("PS (Scrotal)", value=0.0)
            lg = g22.number_input("LG (Gigot)", value=0.0)
            ll = g23.number_input("LL (Laine)", value=0.0)

        ration_val = round(poids * 0.035, 2)
        st.info(f"🌾 Ration suggérée : {ration_val} kg MS/jour")

        if st.form_submit_button("💾 ENREGISTRER LA FICHE COMPLÈTE"):
            id_final = id_in if id_in else f"TEMP-{datetime.now().strftime('%d%H%M%S')}"
            row = [date.today(), id_final, race_in, age_in, poids, hg, hs, lb, lq, lt_t, lc_c, lh, li, lp, pp, tp, lc_cornes, lt_tete, lt_la, lo_lo, lo_la, tc, ly, ts, ps, lg, ll, wilaya_sel, commune_sel, ration_val]
            pd.DataFrame([row], columns=COLONNES).to_csv(DB_FILE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
            st.session_state.step = 1
            st.success(f"✅ Animal {id_final} enregistré !")
            st.rerun()

# --- ONGLET 2 : HISTORIQUE ---
with tabs[1]:
    st.subheader("📋 Base de données")
    if not data.empty:
        st.dataframe(data, use_container_width=True)
        st.download_button("📥 Export Excel", data.to_csv(sep=';', index=False).encode('utf-8-sig'), "base_ovins.csv")

# --- ONGLET 3 : ANALYSE ---
with tabs[2]:
    if not data.empty:
        target = st.selectbox("Sélectionner pour audit", data["ID"].unique())
        anim = data[data["ID"] == target].iloc[-1]
        try:
            ic = anim['Poids'] / anim['LB'] if anim['LB'] > 0 else 0
            st.metric("Indice Viande", f"{ic:.2f}")
            if st.button("📄 Certificat PDF"):
                pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", 'B', 16)
                pdf.cell(200, 10, f"CERTIFICAT OVI-STAT : {target}", ln=True, align='C')
                st.download_button("📥 Télécharger", pdf.output(dest="S").encode("latin-1"), f"Certificat_{target}.pdf")
        except Exception as e: st.error(f"Erreur indices : {e}")

# --- ONGLET 4 : SANTÉ ---
with tabs[3]:
    st.subheader("🩺 Suivi Sanitaire")
    with st.form("form_sante"):
        id_s = st.selectbox("Animal", data["ID"].unique()) if not data.empty else "N/A"
        acte = st.text_input("Soin / Vaccin")
        if st.form_submit_button("💉 Enregistrer le soin"):
            row_s = [date.today(), id_s, "Soin", acte, "Dr. Ahmed", date.today()]
            pd.DataFrame([row_s], columns=COL_SANTE).to_csv(DB_SANTE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
            st.success("Soin enregistré !")

# --- ONGLET 5 : À PROPOS ---
with tabs[4]:
    st.header("ℹ️ À Propos")
    st.write(f"**Auteur :** MERABIA KAWTHER | **Version :** 1.7.5")
    if st.button("🔄 Reset Cycle Scan"):
        st.session_state.step = 1
        st.rerun()

# --- MAINTENANCE ---
st.divider()
with st.expander("⚙️ Maintenance"):
    if st.button("🗑️ Vider la base"):
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()
