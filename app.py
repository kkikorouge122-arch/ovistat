import streamlit as st
import pandas as pd
import os
from datetime import datetime, date
from ultralytics import YOLO
from PIL import Image

# --- 1. CONFIGURATION INITIALE & DESIGN CSS ---
st.set_page_config(page_title="OviStat Vision v2.5", page_icon="🐑", layout="wide")

st.markdown("""
<style>
/* Grand écran centré pour la visée à distance (2 mètres) */
div[data-testid="stCameraInput"] {
    border: 5px solid #1f77b4 !important;
    border-radius: 25px !important;
    background-color: #000;
    max-width: 800px;
    margin: auto;
    position: relative;
    overflow: hidden !important;
}
/* Optimisation de la vidéo (Zoom Logiciel 1.2x & Contraste des bords) */
div[data-testid="stCameraInput"] video {
    width: 100% !important;
    height: 600px !important;
    object-fit: cover !important;
    transform: scale(1.1);
    filter: contrast(1.1) brightness(1.1);
}
/* Viseur de précision Sniper : Petit, pointillé et semi-transparent */
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
/* Bouton de capture géant relocalisé proprement en dessous du cadran */
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
</style>
""", unsafe_allow_html=True)

# --- 2. SESSIONS STATE & AUTHENTIFICATION ---
if 'auth' not in st.session_state: st.session_state.auth = False
if 'step' not in st.session_state: st.session_state.step = 1
if 'last_photo' not in st.session_state: st.session_state.last_photo = None
if 'mesures_ia' not in st.session_state:
    st.session_state.mesures_ia = {col: 0.0 for col in ["Poids", "HG", "HS", "LB", "LT_tronc", "LC_cou", "LH", "LQ", "TP", "LI", "LP", "PP", "Lc_cornes", "LTete", "LtTete", "LO", "Lo", "TC", "LY", "TS", "PS", "LG", "LL"]}

def login():
    c1, c2, c3 = st.columns(3)
    with c2:
        st.title("🛡️ OviStat Sécurisé")
        pwd = st.text_input("🔑 Code chercheur :", type="password")
        if st.button("DÉVERROUILLER"):
            if pwd == "ENSVAlger2026":
                st.session_state.auth = True
                st.rerun()
            else: st.error("Code incorrect")
    st.stop()

if not st.session_state.auth: login()
# --- 3. CONFIGURATION DES BASES DE DONNÉES & GÉO ---
# --- 3. CONFIGURATION DES BASES DE DONNÉES & GÉO ---
DB_FILE = "data_ovinstat_V16.csv"
DB_SANTE = "data_sante_ovins.csv"

COLONNES = [
    "Date", "ID", "Race", "Age", "Poids", 
    "HG", "HS", "LB", "LQ", "LT_tronc", "LC_cou", "LH", "LI", "LP", "PP", "TP",
    "Lcornes", "LTete", "LtTete", "LO", "Lo", "TC", "LY", "TS", "PS", "LG", "LL", 
    "Wilaya", "Commune", "Ration"
]
COL_SANTE = ["Date", "ID", "Type", "Produit", "Veterinaire", "Prochain_RDV"]

@st.cache_resource
def load_yolo(): 
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
        df_geo = pd.read_csv("algeria_geo.csv", sep=";", encoding='utf-8', on_bad_lines='skip')
        df_geo['wilaya_name'] = df_geo['wilaya_name'].str.strip()
        df_geo['commune_name'] = df_geo['commune_name'].str.strip()
    else:
        df_geo = pd.DataFrame(columns=["wilaya_name", "commune_name"])
    return wilayas, df_geo

def load_data(file, cols):
    if os.path.exists(file):
        try: return pd.read_csv(file, sep=';', encoding='utf-8-sig', on_bad_lines='skip')
        except: return pd.DataFrame(columns=cols)
    return pd.DataFrame(columns=cols)

# Initialisation des fichiers
for f, c in zip([DB_FILE, DB_SANTE], [COLONNES, COL_SANTE]):
    if not os.path.exists(f) or os.path.getsize(f) == 0:
        pd.DataFrame(columns=c).to_csv(f, index=False, sep=';', encoding='utf-8-sig')

# Appels synchronisés des fonctions
model = load_yolo()
list_wilayas, df_communes = get_algeria_geo()

if not os.path.exists(DB_FILE) or os.path.getsize(DB_FILE) == 0:
    pd.DataFrame(columns=COLONNES).to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')

# Variable globale contenant les données chargée en priorité
data = pd.read_csv(DB_FILE, sep=';')

# Interface structurelle des Onglets
st.title("🐑 OviStat IA v2.5.1")
tabs = st.tabs(["📥 Saisie", "🔍 Historique", "📊 Analyse", "ℹ️ À Propos"])

# --- 4. ONGLET 1 : SCANNAGE IA PROGRESSIF ---
with tabs[0]:
    st.subheader("📍 Localisation de l'étude")
    
    # Éléments dynamiques (Parfaitement alignés à 4 espaces)
    col_w, col_c = st.columns(2)
    wilaya_sel = col_w.selectbox("Wilaya", list_wilayas, key="w_dyn")
    
    # Filtrage
    w_clean = wilaya_sel.split("-")[-1].strip()
    mask = df_communes['wilaya_name'].str.strip().str.lower() == w_clean.lower()
    communes_possibles = df_communes[mask]['commune_name'].unique().tolist()
    
    if communes_possibles:
        commune_sel = col_c.selectbox(f"Communes de {w_clean}", sorted(communes_possibles), key="c_dyn")
    else:
        commune_sel = col_c.text_input("Commune (Saisie manuelle)", key="c_man")
        st.warning(f"⚠️ Aucune commune trouvée pour {w_clean}")

    st.write(f"### 📸 Étape {st.session_state.step}/3 : {['Profil', 'Dessus', 'Tête'][st.session_state.step-1]}")
    
    if st.button(f"✅ ÉTAPE SUIVANTE : VALIDER LA PHOTO {st.session_state.step}", type="primary", use_container_width=True):
        if st.session_state.last_photo:
            if st.session_state.step < 3:
                st.session_state.step += 1
                st.session_state.last_photo = None
                st.rerun()
            else: st.success("🎯 Étude morphométrique et anatomique complète !")
        else: st.error("⚠️ Capturez d'abord l'animal avec le bouton cercle bleu ci-dessous.")

    photo = st.camera_input("Scanner l'animal", key=f"precision_cam_v4_{st.session_state.step}")
    
    if photo:
        st.session_state.last_photo = photo
        img = Image.open(photo)
        w, h = img.size
        results = model(img)
        
        target_sheep = None
        min_dist = float('inf')
        for r in results:
            for b in r.boxes:
                if int(b.cls) == 18:
                    x1, y1, x2, y2 = b.xyxy.tolist()
                    dist = (((w/2) - ((x1+x2)/2))**2 + ((h/2) - ((y1+y2)/2))**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        target_sheep = b

        if target_sheep and min_dist < (w * 0.20):
            st.success("🔒 SUJET CENTRAL VERROUILLÉ !")
            x1, y1, x2, y2 = target_sheep.xyxy.tolist()
            pixel_width = x2 - x1
            pixel_height = y2 - y1
            ratio = 0.14
            
            if st.session_state.step == 1:
                cal_hg = round(pixel_height * ratio, 1)
                cal_lb = round(pixel_width * ratio, 1)
                est_poids = round((cal_hg * cal_lb) / 110, 1)
                st.session_state.mesures_ia.update({"HG": cal_hg, "HS": round(cal_hg*0.98, 1), "LB": cal_lb, "Poids": est_poids, "LT_tronc": round(cal_lb*0.58, 1), "LC_cou": round(cal_lb*0.28, 1), "LH": round(cal_lb*0.23, 1), "LQ": 22.0})
            elif st.session_state.step == 2:
                cal_larg = round(pixel_width * ratio, 1)
                st.session_state.mesures_ia.update({"TP": round((pixel_height*ratio*2)+(cal_larg*2), 1), "LI": round(cal_larg*0.35, 1), "LP": round(cal_larg*0.45, 1), "PP": round(pixel_height*ratio*0.7, 1)})
            elif st.session_state.step == 3:
                cal_tete = round(pixel_height * ratio * 0.4, 1)
                st.session_state.mesures_ia.update({"Lc_cornes": 15.0, "LTete": cal_tete, "LtTete": round(cal_tete*0.5, 1), "LO": round(cal_tete*1.1, 1), "Lo": round(cal_tete*0.3, 1), "TC": round(cal_tete*0.38, 1), "LY": 2.5, "LL": 5.0})
        else: st.error("❌ RECADREZ LA CIBLE SOUS LE VISEUR 🎯")
    st.divider()
    with st.form("form_final"):
        st.subheader("📋 Fiche d'Analyse d'Identité")
        m = st.session_state.mesures_ia
        
        c1, c2, c3 = st.columns(3)
        id_in = c1.text_input("ID / Boucle")
        age_in = c2.number_input("Âge (mois)", 0, 120, 12)
        race_in = c3.selectbox("Race", ["Ouled Djellal", "Rembi", "Hamra"])

        with st.expander("1️⃣ Dimensions Corporelles", expanded=True):
            g1, g2, g3, g4 = st.columns(4)
            poids = g1.number_input("Poids (kg)", value=float(m["Poids"]))
            hg = g2.number_input("HG (cm)", value=float(m["HG"]))
            hs = g3.number_input("HS (cm)", value=float(m["HS"]))
            lb = g4.number_input("LB (cm)", value=float(m["LB"]))
            lq = g1.number_input("LQ (cm)", value=float(m["LQ"]))

        with st.expander("2️⃣ Poitrine & Largeurs"):
            g5, g6, g7, g8 = st.columns(4)
            li = g5.number_input("LI (cm)", value=float(m["LI"]))
            lp = g6.number_input("LP (cm)", value=float(m["LP"]))
            pp = g7.number_input("PP (cm)", value=float(m["PP"]))
            tp = g8.number_input("TP (cm)", value=float(m["TP"]))

        with st.expander("3️⃣ Tête & Oreilles"):
            t1, t2, t3 = st.columns(3)
            lc_cornes = t1.number_input("L. Cornes", value=float(m["Lc_cornes"]))
            lt_tete = t2.number_input("L. Tête", value=float(m["LTete"]))
            lt_la = t3.number_input("Larg. Tête", value=float(m["LtTete"]))
            lo_lo = t1.number_input("L. Oreille", value=float(m["LO"]))
            lo_la = t2.number_input("Larg. Oreille", value=float(m["Lo"]))
            tc = t3.number_input("T. Canon", value=float(m["TC"]))

        with st.expander("4️⃣ Reproduction & Laine"):
            r1, r2, r3 = st.columns(3)
            ly = r1.number_input("LY", value=float(m["LY"]))
            ts = r2.number_input("TS", value=float(m["TS"]))
            ps = r3.number_input("PS", value=float(m["PS"]))
            lg = r1.number_input("LG", value=float(m["LG"]))
            ll = r2.number_input("LL", value=float(m["LL"]))
            ration_estim = round(poids * 0.035, 2)
            r3.metric("Ration Sug.", f"{ration_estim} kg")

        if st.form_submit_button("💾 ENREGISTRER LA FICHE COMPLÈTE"):
            id_f = id_in if id_in else f"T-{datetime.now().strftime('%H%M%S')}"
            is_dup = False
            if not data.empty:
                try:
                    p_hist = pd.to_numeric(data['Poids'], errors='coerce')
                    h_hist = pd.to_numeric(data['HG'], errors='coerce')
                    if not data[p_hist.between(poids-1, poids+1) & h_hist.between(hg-1, hg+1)].empty: is_dup = True
                except: is_dup = False

            if is_dup: st.error("⚠️ Animal déjà présent dans le registre.")
            else:
                row = [date.today(), id_f, race_in, age_in, poids, hg, hs, lb, lq, m["LT_tronc"], m["LC_cou"], m["LH"], li, lp, pp, tp, lc_cornes, lt_tete, lt_la, lo_lo, lo_la, tc, ly, ts, ps, lg, ll, wilaya_sel, commune_sel, ration_estim]
                pd.DataFrame([row], columns=COLONNES).to_csv(DB_FILE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
                st.session_state.step = 1
                st.session_state.mesures_ia = {k: 0.0 for k in st.session_state.mesures_ia}
                st.success("✅ Fiche sauvegardée !")
                st.rerun()
# --- ONGLET 2 : HISTORIQUE ---
with tabs[1]:
    st.subheader("📋 Registre d'étude des troupeaux")
    if not data.empty:
        st.dataframe(data, use_container_width=True)
        st.divider()
        id_m = st.selectbox("ID de l'ovin pour Action", data["ID"].unique(), key="sel_hist_action")
        if st.button(f"❌ Supprimer la fiche de {id_m}", type="primary"):
            data[data["ID"] != id_m].to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')
            st.rerun()
        st.download_button("📥 Télécharger la base (CSV)", data.to_csv(sep=';', index=False).encode('utf-8-sig'), "OviStat_Export.csv")
    else: st.info("Le registre est vide.")

# --- ONGLET 3 : ANALYSE ---
with tabs[2]:
    st.subheader("📊 Audit Morphométrique")
    if not data.empty:
        target = st.selectbox("Sélectionner le sujet", data["ID"].unique(), key="sel_audit")
        anim = data[data["ID"] == target].iloc[-1]
        try:
            ic = float(anim['Poids']) / float(anim['LB']) if float(anim['LB']) > 0 else 0
            ir = (float(anim['TP'])**2) / float(anim['HG']) if float(anim['HG']) > 0 else 0
            c_an1, c_an2 = st.columns(2)
            c_an1.metric("🥩 Indice de Compacité", f"{ic:.2f}")
            c_an2.metric("🏗️ Indice de Robustesse", f"{ir:.2f}")
        except Exception as e: st.error(f"Erreur indices : {e}")
    else: st.info("Aucune donnée disponible.")

# --- ONGLET 4 : À PROPOS ---
with tabs[3]:
    st.header("ℹ️ Informations Système")
    st.write("OviStat Vision Pro v2.5.1 | ENSV Alger")
    st.write("Auteur : MERABIA KAWTHER")
    if st.button("🔄 Reset Scan"):
        st.session_state.step = 1
        st.session_state.mesures_ia = {k: 0.0 for k in st.session_state.mesures_ia}
        st.rerun()
