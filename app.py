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
        
        # --- ALGORITHME TARGET LOCK SÉLECTIF ---
        target_sheep = None
        min_dist = float('inf')
        center_x, center_y = w / 2, h / 2
        threshold = w * 0.20 
        
        for r in results:
            for b in r.boxes:
                if int(b.cls) == 18:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    m_x = (x1 + x2) / 2
                    m_y = (y1 + y2) / 2
                    dist = ((center_x - m_x)**2 + (center_y - m_y)**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        target_sheep = b

                # --- APPORT CALCULS MORPHOMÉTRIQUES RÉELS (FINI LES COPIES) ---
        if target_sheep and min_dist < threshold:
            st.success("🔒 ANIMAL MAÎTRE VERROUILLÉ AU CENTRE : Extraction cm...")
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
                st.info(f"📈 Profil calculé : HG={cal_hg}cm | LB={cal_lb}cm | Poids Estimé={est_poids}kg")
                
            elif st.session_state.step == 2:
                cal_larg = round(pixel_width * ratio, 1)
                cal_tp = round((pixel_height * ratio * 2) + (cal_larg * 2), 1)
                st.session_state.mesures_ia.update({
                    "TP": cal_tp, "LI": round(cal_larg * 0.35, 1),
                    "LP": round(cal_larg * 0.45, 1), "PP": round(pixel_height * ratio * 0.7, 1)
                })
                st.info(f"📈 Poitrine calculée : TP={cal_tp}cm")
                
            elif st.session_state.step == 3:
                cal_tete = round(pixel_height * ratio * 0.4, 1)
                st.session_state.mesures_ia.update({
                    "Lc_cornes": 15.0, "LTete": cal_tete, "LtTete": round(cal_tete * 0.5, 1),
                    "LO": round(cal_tete * 1.1, 1), "Lo": round(cal_tete * 0.3, 1),
                    "TC": round(cal_tete * 0.38, 1), "LY": 2.5, "LL": 5.0
                })
                st.info("📈 Extrémités faciales et céphaliques calculées.")
        else:
            if target_sheep: 
                st.error("❌ SUJET TROP EXCENTRÉ : Ajustez le viseur 🎯 sur l'ovin cible.")
            else: 
                st.error("❌ AUCUN ANIMAL DÉTECTÉ AU CENTRE : Visez à 2 mètres.")
                
        st.divider()
    # --- C. LE FORMULAIRE GLOBAL DE SYNTHÈSE ---
    with st.form("form_final"):
        st.subheader("📋 Fiche d'Analyse d'Identité")
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
            lq = g1.number_input("LQ (Longueur de Queue en cm)", value=float(m.get("LQ", 0.0)))
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
            lc_cornes = t1.number_input("L. Cornes (Lc cm)", value=float(m.get("Lc_cornes", 0.0)))
            lt_tete = t2.number_input("L. Tête (LT cm)", value=float(m.get("LTete", 0.0)))
            lt_la = t3.number_input("Larg. Tête (Lt cm)", value=float(m.get("LtTete", 0.0)))
            lo_lo = t1.number_input("L. Oreille (LO cm)", value=float(m.get("LO", 0.0)))
            lo_la = t2.number_input("Larg. Oreille (Lo cm)", value=float(m.get("Lo", 0.0)))
            tc = t3.number_input("T. Canon (TC cm)", value=float(m.get("TC", 0.0)))
            
        with st.expander("4️⃣ Reproduction & Laine"):
            r1, r2, r3 = st.columns(3)
            ly = r1.number_input("LY (cm)", value=float(m.get("LY", 0.0)))
            ts = r2.number_input("TS (cm)", value=float(m.get("TS", 0.0)))
            ps = r3.number_input("PS (cm)", value=float(m.get("PS", 0.0)))
            
            r4, r5, r6 = st.columns(3)
            lg = r4.number_input("LG (cm)", value=float(m.get("LG", 0.0)))
            ll = r5.number_input("LL (cm)", value=float(m.get("LL", 0.0)))
            
            ration_estim = round(poids * 0.035, 2)
            r6.metric("Ration Sug. (kg)", f"{ration_estim} kg MS/j")
            
        # Validation alignée à 8 espaces exacts du bord
        if st.form_submit_button("💾 ENREGISTRER LA FICHE COMPLÈTE"):
            id_f = id_in if id_in else f"T-{datetime.now().strftime('%H%M%S')}"
            
            # 🔍 PROTECTION CONSOLIDÉE CONTRE LES VALUEERROR DU FICHIER CSV
            doublon_potentiel = pd.DataFrame()
            if os.path.exists(DB_FILE):
                try:
                    data_check = pd.read_csv(DB_FILE, sep=';')
                    if not data_check.empty:
                        poids_hist = pd.to_numeric(data_check['Poids'], errors='coerce')
                        hg_hist = pd.to_numeric(data_check['HG'], errors='coerce')
                        doublon_potentiel = data_check[
                            (poids_hist.between(poids - 1.0, poids + 1.0)) & 
                            (hg_hist.between(hg - 1.0, hg + 1.0))
                        ]
                except:
                    doublon_potentiel = pd.DataFrame()
                    
            if not doublon_potentiel.empty:
                st.error("⚠️ ALERTE DE SÉCURITÉ : Un ovin avec des mensurations presque identiques est déjà présent dans la base. Veuillez vérifier le marquage physique de l'animal.")
            else:
                row = [
                    date.today(), id_f, race_in, age_in, poids, hg, hs, lb, lq, lt_t, lc_c, lh, li, lp, pp, tp,
                    lc_cornes, lt_tete, lt_la, lo_lo, lo_la, tc, ly, ts, ps, lg, ll, wilaya_sel, commune_sel, 
                    ration_estim
                ]
                pd.DataFrame([row], columns=COLONNES).to_csv(DB_FILE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
                
                # Réinitialisation propre
                st.session_state.step = 1
                st.session_state.mesures_ia = {k: 0.0 for k in st.session_state.mesures_ia}
                st.success(f"✅ Fiche de l'animal {id_f} validée et ajoutée au registre !")
                st.rerun()
# --- ONGLET 2 : HISTORIQUE DE TERRAIN ---
with tabs[1]:
    # Note : Assurez-vous d'avoir défini la fonction load_data ou utilisez pd.read_csv
    try:
        data = pd.read_csv(DB_FILE, sep=';')
    except:
        data = pd.DataFrame(columns=COLONNES)
        
    st.subheader("📋 Registre d'étude des troupeaux")
    
    if not data.empty:
        st.dataframe(data, use_container_width=True)
        st.divider()
        
        id_m = st.selectbox("ID de l'ovin pour Action", data["ID"].unique(), key="sel_hist_action")
        c_del, c_edt = st.columns(2)
        
        if c_del.button(f"❌ Supprimer la fiche de {id_m}", type="primary"):
            data[data["ID"] != id_m].to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')
            st.success("Fiche retirée.")
            st.rerun()
            
        st.download_button(
            label="📥 Télécharger la base complète (Excel/CSV)", 
            data=data.to_csv(sep=';', index=False).encode('utf-8-sig'), 
            file_name=f"OviStat_Export_{date.today()}.csv"
        )
    else:
        st.info("Le registre est actuellement vide.")

                        
