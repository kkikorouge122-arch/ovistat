import streamlit as st
import pandas as pd
import os
from datetime import datetime, date
from ultralytics import YOLO
from PIL import Image
from fpdf import FPDF
import numpy as np

# --- CONFIGURATION INITIALE ---
st.set_page_config(page_title="OviStat Vision Pro v2.4", page_icon="🐑", layout="wide")

# --- LE STYLE CSS FUSIONNÉ (ZÉRO ESPACE À GAUCHE) ---
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

/* 4. BOUTON DE CAPTURE RELOCALISÉ SOUS LE CADRAN SANS DEBORDEMENT */
div[data-testid="stCameraInput"] button {
    height: 75px !important;
    width: 75px !important;
    border-radius: 50% !important;
    border: 4px solid white !important;
    background-color: #1f77b4 !important;
    position: relative !important; 
    margin: 15px auto !important; 
    display: block !important;
    z-index: 20;
    opacity: 0.95;
    transition: all 0.2s ease;
}
div[data-testid="stCameraInput"] button:active {
    transform: scale(0.92) !important;
}
</style>
""", unsafe_allow_html=True)

# --- SESSIONS STATE INITIALISATION ---
if 'auth' not in st.session_state: st.session_state.auth = False
if 'step' not in st.session_state: st.session_state.step = 1
if 'last_photo' not in st.session_state: st.session_state.last_photo = None
if 'dark_mode' not in st.session_state: st.session_state.dark_mode = False
if 'mesures_ia' not in st.session_state:
    st.session_state.mesures_ia = {
        "Poids": 0.0, "HG": 0.0, "HS": 0.0, "LB": 0.0, "LT_tronc": 0.0, "LC_cou": 0.0, "LH": 0.0, "LQ": 0.0,
        "TP": 0.0, "LI": 0.0, "LP": 0.0, "PP": 0.0,
        "Lc_cornes": 0.0, "LTete": 0.0, "LtTete": 0.0, "LO": 0.0, "Lo": 0.0, "TC": 0.0,
        "LY": 0.0, "TS": 0.0, "PS": 0.0, "LG": 0.0, "LL": 0.0
    }

# --- SYSTÈME D'AUTHENTIFICATION ---
def login():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image("https://flaticon.com", width=100)
        st.title("🛡️ Accès Sécurisé ENSV")
        pwd = st.text_input("🔑 Code d'accès chercheur :", type="password")
        if st.button("DÉVERROUILLER"):
            if pwd == "ENSVAlger2026":
                st.session_state.auth = True
                st.rerun()
            else: st.error("Code incorrect")
    st.stop()

if not st.session_state.auth: login()

# --- CONFIGURATION COULEURS DYNAMIQUES (MODE NUIT) ---
if st.session_state.dark_mode:
    bg_c, card_c, text_c, border_c = "#0e1117", "#1d2129", "#e0e0e0", "#3d4450"
    metric_bg = "#12141d"
else:
    bg_c, card_c, text_c, border_c = "#f8f9fa", "#ffffff", "#1f77b4", "#dee2e6"
    metric_bg = "#ffffff"

# --- FICHIERS ET STRUCTURE BASE DE DONNÉES ---
DB_FILE = "data_ovinstat_V16.csv"
DB_SANTE = "data_sante_ovins.csv"

COLONNES = [
    "Date", "ID", "Race", "Age", "Poids", 
    "HG", "HS", "LB", "LQ", "LT_tronc", "LC_cou", "LH", "LI", "LP", "PP", "TP",
    "Lcornes", "LTete", "LtTete", "LO", "Lo", "TC", "LY", "TS", "PS", "LG", "LL", 
    "Wilaya", "Commune", "Ration"
]
COL_SANTE = ["Date", "ID", "Type", "Produit", "Veterinaire", "Prochain_RDV"]

# --- FONCTIONS FILTRAGE & CHARGEMENT ---
@st.cache_resource
def load_yolo_model(): return YOLO('yolov8n.pt')

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
        df_geo = pd.read_csv("algeria_geo.csv", sep=";", encoding='utf-8', on_bad_lines='skip')
        df_geo['wilaya_name'] = df_geo['wilaya_name'].astype(str).str.strip()
        df_geo['commune_name'] = df_geo['commune_name'].astype(str).str.strip()
    else:
        df_geo = pd.DataFrame(columns=["wilaya_name", "commune_name"])
    return wilayas, df_geo

def load_data(file, cols):
    if os.path.exists(file):
        try: return pd.read_csv(file, sep=';', encoding='utf-8-sig', on_bad_lines='skip')
        except: return pd.DataFrame(columns=cols)
    return pd.DataFrame(columns=cols)

# Traitement préventif des bases de données
for f, c in zip([DB_FILE, DB_SANTE], [COLONNES, COL_SANTE]):
    if not os.path.exists(f) or os.path.getsize(f) == 0:
        pd.DataFrame(columns=c).to_csv(f, index=False, sep=';', encoding='utf-8-sig')

model = load_yolo_model()
list_wilayas, df_communes = get_algeria_geo()
data = load_data(DB_FILE, COLONNES)

# --- CONFIGURATION INTERFACE & MENUS ---
with st.sidebar:
    st.image("https://flaticon.com", width=80)
    st.title("OviStat Menu")
    if st.button("🌙 Basculer Mode Nuit / Jour"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    st.divider()
    st.subheader("🔐 Sécurité")
    if st.button("🚪 Se déconnecter / Verrouiller"):
        st.session_state.auth = False
        st.success("Session fermée avec succès.")
        st.rerun()
    st.divider()
    with st.expander("⚙️ MAINTENANCE"):
        if st.button("🗑️ Vider la base"):
            if os.path.exists(DB_FILE): os.remove(DB_FILE)
            st.rerun()

st.title("🐑 OviStat IA v2.4.6")
tabs = st.tabs(["📥 Saisie", "🔍 Historique & Modif", "📊 Analyse", "🩺 Santé", "ℹ️ À Propos"])

# --- ONTLET 1 : SAISIE PROGRESSIVE ---
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
    st.caption("💡 Centrez précisément la silhouette de l'ovin sous le collimateur circulaire discret.")
    
    if st.button(f"✅ VALIDER LA PHOTO {st.session_state.step}", type="primary", use_container_width=True):
        if st.session_state.last_photo:
            if st.session_state.step < 3:
                st.session_state.step += 1
                st.session_state.last_photo = None
                st.rerun()
            else: st.success("🎯 Étude morphométrique et anatomique complète !")
        else: st.error("⚠️ Capturez d'abord l'animal avec le déclencheur circulaire.")

    photo = st.camera_input("Scanner l'animal", key=f"precision_cam_v4_{st.session_state.step}")
    
    if photo:
        st.session_state.last_photo = photo
        img = Image.open(photo)
        w, h = img.size
        results = model(img)
        
               # --- APPORT CALCULS MORPHOMÉTRIQUES RÉELS (FINI LES COPIES) ---
        if target_sheep and min_dist < threshold:
            st.success("🔒 ANIMAL MAÎTRE VERROUILLÉE AU CENTRE : Extraction cm...")
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
                }) # <-- Correctement fermé ici
                st.info(f"📈 Profil calculé : HG={cal_hg}cm | LB={cal_lb}cm | Poids Estimé={est_poids}kg")
                
            elif st.session_state.step == 2:
                cal_larg = round(pixel_width * ratio, 1)
                cal_tp = round((pixel_height * ratio * 2) + (cal_larg * 2), 1)
                
                st.session_state.mesures_ia.update({
                    "TP": cal_tp, "LI": round(cal_larg * 0.35, 1),
                    "LP": round(cal_larg * 0.45, 1), "PP": round(pixel_height * ratio * 0.7, 1)
                }) # <-- Correctement fermé ici
                st.info(f"📈 Poitrine calculée : TP={cal_tp}cm")
                
            elif st.session_state.step == 3:
                cal_tete = round(pixel_height * ratio * 0.4, 1)
                
                st.session_state.mesures_ia.update({
                    "Lc_cornes": 15.0, "LTete": cal_tete, "LtTete": round(cal_tete * 0.5, 1),
                    "LO": round(cal_tete * 1.1, 1), "Lo": round(cal_tete * 0.3, 1),
                    "TC": round(cal_tete * 0.38, 1), "LY": 2.5, "LL": 5.0
                }) # <-- LA FERMETURE QUI MANQUAIT ET CRÉAIT L'ERREUR SYNTAXE
                st.info("📈 Extrémités faciales et céphaliques calculées.")
        else:
            if target_sheep: st.error("❌ SUJET TROP EXCENTRÉ : Ajustez le viseur 🎯 sur l'ovin cible.")
            else: st.error("❌ AUCUN ANIMAL DÉTECTÉ AU CENTRE : Visez à 2 mètres.")
S": cal_hs, "LB": cal_lb, "Poids": est_poids,
