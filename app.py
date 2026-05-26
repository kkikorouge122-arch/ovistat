import streamlit as st
import pandas as pd
import os
from datetime import datetime, date
from ultralytics import YOLO
from PIL import Image
from fpdf import FPDF

# --- 1. CONFIGURATION INITIALE ---
st.set_page_config(page_title="OviStat Vision Pro v2.4", page_icon="🐑", layout="wide")

# --- STYLE CSS DU VISEUR (COLLÉ TOUT EN HAUT À L'INDENTATION ZÉRO) ---
st.markdown("""
<style>
/* 1. Grand écran pour la visée à distance */
div[data-testid="stCameraInput"] {
    border: 5px solid #1f77b4 !important;
    border-radius: 25px !important;
    background-color: #000;
    max-width: 800px;
    margin: auto;
    position: relative;
}

/* 2. Optimisation de la vidéo (Zoom Logiciel 1.2x) */
div[data-testid="stCameraInput"] video {
    width: 100% !important;
    height: 600px !important;
    object-fit: cover !important;
    transform: scale(1.1);
    filter: contrast(1.1) brightness(1.1);
}

/* 3. Viseur de précision Sniper : PETIT ET TRANSPARENT */
div[data-testid="stCameraInput"]::after {
    content: "";
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 80px; height: 80px;
    border: 2px dashed rgba(255, 255, 255, 0.35);
    border-radius: 50%;
    box-shadow: 0 0 0 1000px rgba(0, 0, 0, 0.25);
    pointer-events: none;
}

/* 4. Bouton de capture géant pour éviter les flous de bougé au clic */
div[data-testid="stCameraInput"] button {
    height: 75px !important;
    width: 75px !important;
    border-radius: 50% !important;
    border: 4px solid white !important;
    background-color: #1f77b4 !important;
    position: absolute !important;
    bottom: 20px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    z-index: 20;
    opacity: 0.85;
    transition: all 0.2s ease;
}
div[data-testid="stCameraInput"] button:active {
    transform: translateX(-50%) scale(0.92) !important;
}
</style>
""", unsafe_allow_html=True)

# --- 2. LOGIQUE DE SÉCURITÉ ---
if 'auth' not in st.session_state: st.session_state.auth = False
if 'step' not in st.session_state: st.session_state.step = 1
if 'last_photo' not in st.session_state: st.session_state.last_photo = None

def login():
    c1, c2, c3 = st.columns()
    with c2:
        st.image("https://flaticon.com", width=100)
        st.title("🛡️ Accès Sécurisé ENSV")
        pwd = st.text_input("🔑 Code d'accès chercheur :", type="password")
        if st.button("DÉVERROUILLER"):
            if pwd == "ENSV-2024":
                st.session_state.auth = True
                st.rerun()
            else: st.error("Code incorrect")
    st.stop()

if not st.session_state.auth: login()

# --- 3. DATA & GÉO ---
DB_FILE = "data_ovinstat_V19.csv"
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
        df = pd.read_csv("algeria_geo.csv", sep=";", encoding='utf-8')
    else: df = pd.DataFrame({"wilaya_name":["Djelfa"], "commune_name":["Djelfa"]})
    return wilayas, df

model = load_yolo()
list_wilayas, df_communes = get_geo()
if not os.path.exists(DB_FILE): pd.DataFrame(columns=COLONNES).to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')

# Initialisation de la mémoire IA
if 'mesures_ia' not in st.session_state:
    st.session_state.mesures_ia = {col: 0.0 for col in COLONNES}

# --- 4. INTERFACE ---
tabs = st.tabs(["📥 Saisie", "🔍 Historique", "ℹ️ À Propos"])

# --- ONGLET 1 : SAISIE ---
with tabs[0]:
    st.subheader("📍 Localisation de l'étude")
    col_w, col_c = st.columns(2)
    wilaya_sel = col_w.selectbox("Wilaya", list_wilayas, key="w_dyn")
    
    w_clean = wilaya_sel.split("-")[-1].strip()
    mask = df_communes['wilaya_name'].str.strip().str.lower() == w_clean.lower()
    communes_possibles = df_communes[mask]['commune_name'].unique().tolist()
    
    if communes_possibles:
        commune_sel = col_c.selectbox(f"Communes de {w_clean}", sorted(communes_possibles), key="c_dyn")
    else:
        commune_sel = col_c.text_input("Commune (Saisie manuelle)", key="c_man")
        st.warning(f"⚠️ Aucune commune trouvée pour {w_clean}")

    st.divider()
    
    st.write(f"### 📸 Étape {st.session_state.step}/3 : {['Profil', 'Dessus', 'Tête'][st.session_state.step-1]}")
    
    if st.button(f"✅ VALIDER LA PHOTO {st.session_state.step}", type="primary", use_container_width=True):
        if st.session_state.last_photo:
            if st.session_state.step < 3:
                st.session_state.step += 1
                st.session_state.last_photo = None
                st.rerun()
            else: st.success("🎯 Étude morphométrique complète !")
        else: st.error("⚠️ Capturez l'animal d'abord avec le bouton cercle.")

    photo = st.camera_input("Scanner l'animal", key=f"precision_cam_v4_{st.session_state.step}")
    
    if photo:
        st.session_state.last_photo = photo
        img = Image.open(photo)
        w, h = img.size
        results = model(img)
        
        target_sheep = None
        min_dist = float('inf')
        center_x, center_y = w / 2, h / 2
        threshold = w * 0.20
        
        for r in results:
            for b in r.boxes:
                if int(b.cls) == 18:
                    x1, y1, x2, y2 = b.xyxy.tolist()
                    m_x = (x1 + x2) / 2
                    m_y = (y1 + y2) / 2
                    dist = ((center_x - m_x)**2 + (center_y - m_y)**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        target_sheep = b

        if target_sheep and min_dist < threshold:
            st.success("🔒 CIBLE CENTRALE VERROUILLÉE : Calcul de l'anatomie réelle...")
            x1, y1, x2, y2 = target_sheep.xyxy.tolist()
            pixel_width = x2 - x1
            pixel_height = y2 - y1
            ratio = 0.14
            
            if st.session_state.step == 1:
                cal_hg = round(pixel_height * ratio, 1)
                cal_lb = round(pixel_width * ratio, 1)
                cal_hs = round(cal_hg * 0.98, 1)
                est_poids = round((cal_hg * cal_lb) / 110, 1)
                
                st.session_state.mesures_ia.update({
                    "HG": cal_hg, "HS": cal_hs, "LB": cal_lb, "Poids": est_poids,
                    "LT_tronc": round(cal_lb * 0.58, 1), "LC_cou": round(cal_lb * 0.28, 1),
                    "LH": round(cal_lb * 0.23, 1), "LQ": 22.0
                })
            elif st.session_state.step == 2:
                cal_larg = round(pixel_width * ratio, 1)
                cal_tp = round((pixel_height * ratio * 2) + (cal_larg * 2), 1)
                st.session_state.mesures_ia.update({
                    "TP": cal_tp, "LI": round(cal_larg * 0.35, 1),
                    "LP": round(cal_larg * 0.45, 1), "PP": round(pixel_height * ratio * 0.7, 1)
                })
            elif st.session_state.step == 3:
                cal_tete = round(pixel_height * ratio * 0.4, 1)
                st.session_state.mesures_ia.update({
                    "Lc_cornes": 15.0, "LTete": cal_tete, "LtTete": round(cal_tete * 0.5, 1),
                    "LO": round(cal_tete * 1.1, 1), "Lo": round(cal_tete * 0.3, 1),
                    "TC": round(cal_tete * 0.38, 1), "LY": 2.5, "LL": 5.0
                })
        else:
            if target_sheep: st.error("❌ CIBLE TROP EXCENTRÉE : Placez l'animal bien sous le viseur.")
            else: st.error("❌ AUCUN OVIN DÉTECTÉ.")

    st.divider()

    with st.form("form_final"):
        st.subheader("📋 Fiche Identité & Morphométrie")
        m = st.session_state.mesures_ia
        
        c1, c2, c3 = st.columns(3)
        id_in = c1.text_input("ID / Boucle")
        age_in = c2.number_input("Âge (mois)", 0, 120, 12)
        race_in = c3.selectbox("Race", ["Ouled Djellal", "Rembi", "Hamra"])

        with st.expander("1️⃣ Dimensions Corporelles", expanded=True):
            g1, g2, g3, g4 = st.columns(4)
            poids = g1.number_input("Poids (kg)", 0.0, 150.0, value=float(m.get("Poids", 0.0)))
            hg = g2.number_input("HG (cm)", value=float(m.get("HG", 0.0)))
            hs = g3.number_input("HS (cm)", value=float(m.get("HS", 0.0)))
            lb = g4.number_input("LB (cm)", value=float(m.get("LB", 0.0)))
            lq = g1.number_input("LQ (cm)", value=float(m.get("LQ", 0.0)))
            lt_t = m.get("LT_tronc", 0.0)
            lc_c = m.get("LC_cou", 0.0)
            lh = m.get("LH", 0.0)

        with st.expander("2️⃣ Poitrine & Largeurs"):
            g5, g6, g7, g8 = st.columns(4)
            li = g5.number_input("LI (cm)", value=float(m.get("LI", 0.0)))
            lp = g6.number_input("LP (cm)", value=float(m.get("LP", 0.0)))
            pp = g7.number_input("PP (cm)", value=float(m.get("PP", 0.0)))
            tp = g8.number_input("TP (cm)", value=float(m.get("TP", 0.0)))

        with st.expander("3️⃣ Tête & Oreilles"):
            t1, t2, t3 = st.columns(3)
            lc_cornes = t1.number_input("L. Cornes", value=float(m.get("Lc_cornes", 0.0)))
            lt_tete = t2.number_input("L. Tête", value=float(m.get("LTete", 0.0)))
            lt_la = t3.number_input("Larg. Tête", value=float(m.get("LtTete", 0.0)))
            lo_lo = t1.number_input("L. Oreille (LO)", value=float(m.get("LO", 0.0)))
            lo_la = t2.number_input("Larg. Oreille (Lo)", value=float(m.get("Lo", 0.0)))
            tc = t3.number_input("T. Canon (TC)", value=float(m.get("TC", 0.0)))

        with st.expander("4️⃣ Reproduction & Laine"):
            r1, r2, r3 = st.columns(3)
            ly = r1.number_input("LY", value=float(m.get("LY", 0.0)))
            ts = r2.number_input("TS", value=float(m.get("TS", 0.0)))
            ps = r3.number_input("PS", value=float(m.get("PS", 0.0)))
            r4, r5, r6 = st.columns(3)
            lg = r4.number_input("LG", value=float(m.get("LG", 0.0)))
            lg = r4.number_input("LG", value=float(m.get("LG", 0.0)))
            ration_estim = round(poids * 0.035, 2)
            r6.metric("Ration Sug. (kg)", f"{ration_estim} kg")
            if st.form_submit_button("💾 ENREGISTRER LA FICHE COMPLÈTE"):
            id_f = id_in if id_in else f"T-{datetime.now().strftime('%H%M%S')}"
            row = [date.today(), id_f, race_in, age_in, poids, hg, hs, lb, lq, lt_t, lc_c, lh, li, lp, pp, tp,
            lc_cornes, lt_tete, lt_la, lo_lo, lo_la, tc, ly, ts, ps, lg, ll, wilaya_sel, commune_sel,
            ation_estim]
            pd.DataFrame([row], columns=COLONNES).to_csv(DB_FILE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
            st.session_state.step = 1
            st.session_state.mesures_ia = {k: 0.0 for k in st.session_state.mesures_ia}
            st.success(f"✅ Animal {id_f} enregistré !")
            st.rerun()
