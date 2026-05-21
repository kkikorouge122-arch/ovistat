import streamlit as st
import pandas as pd
import os
from datetime import datetime, date
from ultralytics import YOLO
from PIL import Image
from fpdf import FPDF

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="OviStat Vision Pro v2.0", page_icon="🐑", layout="wide")

st.markdown("""
    <style>
    .block-container { padding-top: 1rem; }
    div[data-testid="stCameraInput"] video { width: 100% !important; object-fit: cover !important; border-radius: 15px; border: 3px solid #1f77b4; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

DB_FILE = "data_ovinstat_V15.csv"
COLONNES = [
    "Date", "ID", "Race", "Age", "Poids", 
    "HG", "HS", "LB", "LQ", "LT_tronc", "LC_cou", "LH", "LI", "LP", "PP", "TP",
    "Lc_cornes", "LT_tete", "Lt_tete", "LO", "Lo_oreille", "TC", "LY", "TS", "PS", "LG", "LL", 
    "Wilaya", "Commune", "Ration"
]

# --- 2. FONCTIONS ---
@st.cache_resource
def load_yolo_model(): return YOLO('yolov8n.pt')

@st.cache_data
def get_algeria_geo():
    wilayas = ["01-Adrar", "02-Chlef", "03-Laghouat", "16-Alger", "17-Djelfa", "51-Ouled Djellal", "58-In Guezzam"] # Liste abrégée pour l'exemple
    if os.path.exists("algeria_geo.csv"):
        df_geo = pd.read_csv("algeria_geo.csv", sep=";", encoding='utf-8')
    else:
        df_geo = pd.DataFrame({"wilaya_name": ["Djelfa"], "commune_name": ["Djelfa"]})
    return wilayas, df_geo

def load_data():
    if os.path.exists(DB_FILE):
        return pd.read_csv(DB_FILE, sep=';', encoding='utf-8-sig', on_bad_lines='skip')
    return pd.DataFrame(columns=COLONNES)

# Initialisation
if not os.path.exists(DB_FILE):
    pd.DataFrame(columns=COLONNES).to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')

model = load_yolo_model()
list_wilayas, df_communes = get_algeria_geo()

# --- 3. INTERFACE ---
st.title("🐑 OviStat IA : Version 2.0")
tabs = st.tabs(["📥 Saisie", "🔍 Historique & Modif", "📊 Analyse", "🩺 Santé", "ℹ️ À Propos"])

# --- ONGLET 1 : SAISIE ---
with tabs[0]:
    if 'step' not in st.session_state: st.session_state.step = 1
    with st.form("form_global"):
        st.subheader("📍 Localisation & Identité")
        cw, cc = st.columns(2)
        wilaya_sel = cw.selectbox("Wilaya", list_wilayas)
        w_clean = wilaya_sel.split("-")[-1].strip()
        communes_possibles = df_communes[df_communes['wilaya_name'] == w_clean]['commune_name'].tolist()
        commune_sel = cc.selectbox("Commune", communes_possibles if communes_possibles else ["Saisir..."])

        c1, c2, c3 = st.columns(3)
        id_in = c1.text_input("ID Animal")
        age_in = c2.number_input("Âge (mois)", value=12)
        race_in = c3.selectbox("Race", ["Ouled Djellal", "Rembi", "Hamra", "Taadmit"])

        st.write(f"📸 **Scan Étape {st.session_state.step}/3**")
        photo = st.camera_input("Capturer")
        if photo and st.session_state.step < 3:
            if st.form_submit_button(f"Continuer vers étape {st.session_state.step+1}"):
                st.session_state.step += 1
                st.rerun()

        st.divider()
        st.subheader("📏 Mensurations (4 Thématiques)")
        
        with st.expander("1️⃣ Dimensions Corporelles (HG, HS, LB, LQ, LT, LC, LH)", expanded=True):
            g1, g2, g3, g4 = st.columns(4)
            poids = g1.number_input("Poids (kg)", value=0.0)
            hg = g2.number_input("Hauteur Garrot (HG)", value=0.0)
            hs = g3.number_input("Hauteur Sacrum (HS)", value=0.0)
            lb = g4.number_input("Long. Totale (LB)", value=0.0)
            lq = g1.number_input("Long. Queue (LQ)", value=0.0)
            lt_t = g2.number_input("Long. Tronc (LT)", value=0.0)
            lc_c = g3.number_input("Long. Cou (LC)", value=0.0)
            lh = g4.number_input("Long. Bassin (LH)", value=0.0)

        with st.expander("2️⃣ Poitrine & Largeurs (LI, LP, PP, TP)"):
            g5, g6, g7, g8 = st.columns(4)
            li = g5.number_input("Larg. Ischions (LI)", value=0.0)
            lp = g6.number_input("Larg. Poitrine (LP)", value=0.0)
            pp = g7.number_input("Prof. Poitrine (PP)", value=0.0)
            tp = g8.number_input("Tour Poitrine (TP)", value=0.0)

        with st.expander("3️⃣ Tête & Oreilles (Lc, LT, Lt, LO, Lo, TC)"):
            g9, g10, g11 = st.columns(3)
            lc_cornes = g9.number_input("Long. Cornes (Lc)", value=0.0)
            lt_te = g10.number_input("Long. Tête (LTête)", value=0.0)
            lt_la = g11.number_input("Larg. Tête (LtTete)", value=0.0)
            lo_lo = g9.number_input("Long. Oreille (LO)", value=0.0)
            lo_la = g10.number_input("Larg. Oreille (Lo)", value=0.0)
            tc = g11.number_input("Tour Canon (TC)", value=0.0)

        with st.expander("4️⃣ Reproduction & Laine (LY, TS, PS, LG, LL)"):
            g12, g13, g14 = st.columns(3)
            ly = g12.number_input("Long. Trayons (LY)", value=0.0)
            ts = g13.number_input("Tour Scrotal (TS)", value=0.0)
            ps = g14.number_input("Prof. Scrotale (PS)", value=0.0)
            lg = g12.number_input("Long. Gigot (LG)", value=0.0)
            ll = g13.number_input("Long. Laine (LL)", value=0.0)

        if st.form_submit_button("💾 ENREGISTRER TOUT"):
            id_f = id_in if id_in else f"TEMP-{datetime.now().strftime('%H%M%S')}"
            row = [date.today(), id_f, race_in, age_in, poids, hg, hs, lb, lq, lt_t, lc_c, lh, li, lp, pp, tp, lc_cornes, lt_te, lt_la, lo_lo, lo_la, tc, ly, ts, ps, lg, ll, wilaya_sel, commune_sel, round(poids*0.035,2)]
            pd.DataFrame([row], columns=COLONNES).to_csv(DB_FILE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
            st.session_state.step = 1
            st.success("✅ Fiche sauvegardée !"); st.rerun()

# --- ONGLET 2 : HISTORIQUE & MODIF TOTALE ---
with tabs[1]:
    data = load_data()
    if not data.empty:
        st.dataframe(data, use_container_width=True)
        st.divider()
        id_m = st.selectbox("Choisir l'ID pour modification totale", data["ID"].unique())
        idx = data[data["ID"] == id_m].index[-1]
        
        with st.expander(f"📝 Modifier TOUTES les données de {id_m}"):
            with st.form("edit_all"):
                # On recrée dynamiquement tous les champs pour la ligne sélectionnée
                new_vals = {}
                c_edit1, c_edit2 = st.columns(2)
                for i, col in enumerate(COLONNES[2:-1]): # On modifie de Race à Commune
                    target_col = c_edit1 if i % 2 == 0 else c_edit2
                    if col in ["Race", "Wilaya", "Commune"]:
                        new_vals[col] = target_col.text_input(f"{col}", value=str(data.at[idx, col]))
                    else:
                        new_vals[col] = target_col.number_input(f"{col}", value=float(data.at[idx, col]))
                
                if st.form_submit_button("💾 Appliquer la modification totale"):
                    for k, v in new_vals.items():
                        data.at[idx, k] = v
                    data.to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')
                    st.success("Mise à jour réussie !"); st.rerun()
        
        if st.button(f"❌ Supprimer {id_m}"):
            data[data["ID"] != id_m].to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')
            st.rerun()

# --- ONGLET 3 : ANALYSE ---
with tabs[2]:
    if not data.empty:
        target = st.selectbox("Audit", data["ID"].unique())
        anim = data[data["ID"] == target].iloc[-1]
        try:
            ic = anim['Poids'] / anim['LB'] if anim['LB'] > 0 else 0
            st.metric("Indice Viande", f"{ic:.2f}")
            if st.button("📄 Certificat PDF"):
                pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", 'B', 16)
                pdf.cell(200, 10, f"CERTIFICAT : {target}", ln=True, align='C')
                st.download_button("📥 Télécharger", pdf.output(dest="S").encode("latin-1"), f"{target}.pdf")
        except: st.error("Données insuffisantes")

# --- ONGLET 4 : SANTÉ ---
with tabs[3]:
    st.subheader("🩺 Suivi Sanitaire")
    with st.form("form_sante"):
        id_s = st.selectbox("Animal", data["ID"].unique()) if not data.empty else "N/A"
        acte = st.text_input("Vaccin / Soin")
        if st.form_submit_button("💉 Noter"):
            pd.DataFrame([[date.today(), id_s, "Soin", acte, "Dr. Ahmed", date.today()]], columns=COL_SANTE).to_csv(DB_SANTE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')

# --- ONGLET 5 : À PROPOS ---
with tabs[4]:
    st.header("ℹ️ À Propos")
    st.write("**Auteur :** MERABIA KAWTHER | **Version :** 1.9.0")
    if os.path.exists("manuel_ovistat.pdf"):
        with open("manuel_ovistat.pdf", "rb") as f:
            st.download_button("📖 Ouvrir le Manuel PDF", f.read(), "Manuel_OviStat.pdf")
    if st.button("🔄 Reset Scan"): st.session_state.step = 1; st.rerun()

# --- MAINTENANCE ---
with st.sidebar.expander("⚙️ Maintenance"):
    if st.button("🗑️ Vider base"):
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()
