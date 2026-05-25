import streamlit as st
import pandas as pd
import os
from datetime import datetime, date
from ultralytics import YOLO
from PIL import Image
from fpdf import FPDF

# --- 1. CONFIGURATION & SÉCURITÉ ---
st.set_page_config(page_title="OviStat Vision Pro v2.3", page_icon="🐑", layout="wide")

if 'auth' not in st.session_state: st.session_state.auth = False
if 'step' not in st.session_state: st.session_state.step = 1
if 'last_photo' not in st.session_state: st.session_state.last_photo = None

def login():
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.image("https://flaticon.com", width=100)
        st.title("🛡️ Accès Sécurisé ENSV")
        pwd = st.text_input("🔑 Code d'accès chercheur :", type="password")
        if st.button("DÉVERROUILLER L'INTERFACE"):
            if pwd == "ENSV-2024": # <--- VOTRE MOT DE PASSE
                st.session_state.auth = True
                st.rerun()
            else: st.error("Code incorrect")
    st.stop()

if not st.session_state.auth: login()

# --- 2. STYLE CSS PREMIUM (CADRE BLEU & CAMÉRA) ---
st.markdown("""
    <style>
    div[data-testid="stCameraInput"] { border: 4px solid #1f77b4 !important; border-radius: 20px !important; overflow: hidden !important; }
    div[data-testid="stCameraInput"] video { width: 100% !important; height: 500px !important; object-fit: cover !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stButton>button { height: 55px; border-radius: 12px; font-weight: bold; }
    [data-testid="stMetric"] { background-color: white; border-radius: 15px; border-left: 5px solid #1f77b4; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 3. DATA & GÉO ---
DB_FILE = "data_ovinstat_V18.csv"
COLONNES = [
    "Date", "ID", "Race", "Age", "Poids", 
    "HG", "HS", "LB", "LQ", "LT_tronc", "LC_cou", "LH", "LI", "LP", "PP", "TP",
    "Lcornes", "LTete", "LtTete", "LO", "Lo", "TC", "LY", "TS", "PS", "LG", "LL", 
    "Wilaya", "Commune", "Ration"
]

@st.cache_resource
def load_yolo(): return YOLO('yolov8n.pt')

@st.cache_data
def get_geo():
    wilayas = ["01-Adrar", "02-Chlef", "03-Laghouat", "16-Alger", "17-Djelfa", "51-Ouled Djellal"]
    if os.path.exists("algeria_geo.csv"):
        df_geo = pd.read_csv("algeria_geo.csv", sep=";", encoding='utf-8').apply(lambda x: x.str.strip())
    else: df_geo = pd.DataFrame({"wilaya_name":["Djelfa"], "commune_name":["Djelfa"]})
    return wilayas, df_geo

model = load_yolo()
list_wilayas, df_communes = get_geo()

if not os.path.exists(DB_FILE): pd.DataFrame(columns=COLONNES).to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')

# --- 4. INTERFACE ---
tabs = st.tabs(["📥 Saisie", "🔍 Historique & Modif", "📊 Analyse", "ℹ️ À Propos"])

with tabs[0]:
    # Localisation dynamique
    st.subheader("📍 Localisation de l'étude")
    cw, cc = st.columns(2)
    wilaya_sel = cw.selectbox("Wilaya", list_wilayas, key="w_top")
    w_clean = wilaya_sel.split("-")[-1].strip()
    communes_list = sorted(df_communes[df_communes['wilaya_name'].str.lower() == w_clean.lower()]['commune_name'].tolist())
    commune_sel = cc.selectbox(f"Communes de {w_clean}", communes_list if communes_list else ["Saisir..."], key="c_top")

    st.divider()
    
    # Zone de Scan
    st.write(f"### 📸 Étape {st.session_state.step}/3 : {['Profil', 'Dessus', 'Tête'][st.session_state.step-1]}")
    if st.button(f"✅ VALIDER LA PHOTO {st.session_state.step}", type="primary", use_container_width=True):
        if st.session_state.last_photo:
            if st.session_state.step < 3:
                st.session_state.step += 1
                st.session_state.last_photo = None
                st.rerun()
            else: st.success("🎯 Scan complet ! Suggestions IA appliquées.")
        else: st.error("⚠️ Prenez la photo avant de valider.")

    photo = st.camera_input("Capturer l'animal", key=f"cam_v3_{st.session_state.step}")
    
    ia_vals = {"HG":0.0, "TP":0.0, "LB":0.0}
    if photo:
        st.session_state.last_photo = photo
        results = model(Image.open(photo))
        nb = sum(1 for r in results for b in r.boxes if int(b.cls) == 18)
        if nb == 1: 
            st.success("✅ Animal unique détecté.")
            ia_vals = {"HG":68.5, "TP":84.0, "LB":78.0} # Simulation
        elif nb > 1: st.warning(f"⚠️ {nb} moutons vus. Isolez l'animal.")

    st.divider()

    with st.form("form_global"):
        st.subheader("🆔 Identification & Mensurations")
        c1, c2, c3 = st.columns(3)
        id_in = c1.text_input("ID Animal")
        age_in = c2.number_input("Âge (mois)", 0, 120, 12)
        race_in = c3.selectbox("Race", ["Ouled Djellal", "Rembi", "Hamra"])

        with st.expander("1️⃣ Dimensions Corporelles", expanded=True):
            g1, g2, g3, g4 = st.columns(4)
            poids = g1.number_input("Poids (kg)", 0.0, 150.0, 0.0)
            hg = g2.number_input("HG (Garrot)", 0.0, 120.0, ia_vals["HG"])
            hs = g3.number_input("HS (Sacrum)", 0.0, 120.0, 0.0)
            lb = g4.number_input("LB (Corps)", 0.0, 150.0, ia_vals["LB"])
            lq = g1.number_input("LQ (Queue)", 0.0, 50.0, 0.0)
            lt_t = g2.number_input("LT tronc", 0.0, 120.0, 0.0)
            lc_c = g3.number_input("LC cou", 0.0, 60.0, 0.0)
            lh = g4.number_input("LH bassin", 0.0, 60.0, 0.0)

        with st.expander("2️⃣ Poitrine & Largeurs"):
            g5, g6, g7, g8 = st.columns(4)
            li = g5.number_input("LI (Ischions)", 0.0, 50.0, 0.0)
            lp = g6.number_input("LP (Poitrine)", 0.0, 60.0, 0.0)
            pp = g7.number_input("PP (Profondeur)", 0.0, 60.0, 0.0)
            tp = g8.number_input("TP (Tour)", 0.0, 150.0, ia_vals["TP"])

        with st.expander("3️⃣ Tête & Oreilles"):
            g9, g10, g11 = st.columns(3)
            lcor = g9.number_input("L. Cornes", 0.0)
            ltet = g10.number_input("L. Tête", 0.0)
            ltet_l = g11.number_input("Larg. Tête", 0.0)
            lo = g9.number_input("L. Oreille", 0.0)
            lo_l = g10.number_input("Larg. Oreille", 0.0)
            tc = g11.number_input("T. Canon", 0.0)

        with st.expander("4️⃣ Reproduction & Laine"):
            g12, g13, g14 = st.columns(3)
            ly = g12.number_input("L. Trayons", 0.0); ts = g13.number_input("T. Scrotal", 0.0); ps = g14.number_input("P. Scrotale", 0.0)
            lg = g12.number_input("L. Gigot", 0.0); ll = g13.number_input("L. Laine", 0.0)

        if st.form_submit_button("💾 ENREGISTRER LA FICHE COMPLÈTE"):
            id_f = id_in if id_in else f"TEMP-{datetime.now().strftime('%H%M%S')}"
            row = [date.today(), id_f, race_in, age_in, poids, hg, hs, lb, lq, lt_t, lc_c, lh, li, lp, pp, tp, lcor, ltet, ltet_l, lo, lo_l, tc, ly, ts, ps, lg, ll, wilaya_sel, commune_sel, round(poids*0.035,2)]
            pd.DataFrame([row], columns=COLONNES).to_csv(DB_FILE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
            st.session_state.step = 1
            st.success(f"✅ Animal {id_f} enregistré !")
            st.rerun()

with tabs[1]:
    data = pd.read_csv(DB_FILE, sep=';')
    if not data.empty:
        st.dataframe(data, use_container_width=True)
        st.divider()
        id_mod = st.selectbox("Sélectionner un ID pour modif totale", data["ID"].unique())
        idx = data[data["ID"] == id_mod].index[-1]
        with st.expander(f"📝 Modifier les 30 paramètres de {id_mod}"):
            with st.form("edit_all"):
                new_data = {}
                c_e1, c_edit2 = st.columns(2)
                for i, col in enumerate(COLONNES[2:]):
                    tgt = c_e1 if i % 2 == 0 else c_edit2
                    val = data.at[idx, col]
                    if col in ["Race", "Wilaya", "Commune"]: new_data[col] = tgt.text_input(col, value=str(val))
                    else: new_data[col] = tgt.number_input(col, value=float(val))
                if st.form_submit_button("💾 Sauver"):
                    for k, v in new_data.items(): data.at[idx, k] = v
                    data.to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')
                    st.success("Mise à jour !"); st.rerun()
        if st.button(f"❌ Supprimer {id_mod}"):
            data[data["ID"] != id_mod].to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')
            st.rerun()
