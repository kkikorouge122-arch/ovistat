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
TV_LARGEUR_REELLE_CM = 95.5  # Largeur de votre TV sur le mur
TV_HAUTEUR_REELLE_CM = 55.2   # Hauteur de votre TV sur le mur
if 'mode_calibration_tv' not in st.session_state:
    st.session_state.mode_calibration_tv = False

st.sidebar.subheader("🎯 Étalonnage & Métrologie")
st.session_state.mode_calibration_tv = st.sidebar.toggle(
    "🔬 Mode Étalon TV Mural", 
    value=st.session_state.mode_calibration_tv
)

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
        
        # 📺 Choix automatique de la cible : 62 pour la TV, 18 pour le mouton
        classe_cible = 62 if st.session_state.mode_calibration_tv else 18
        
        target_sheep = None
        min_dist = float('inf')
        center_x, center_y = w / 2, h / 2
        threshold = w * 0.20
        
        for r in results:
            for b in r.boxes:
                if int(b.cls) == classe_cible:




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
            st.success("🔒 CIBLE CENTRALE VERROUILLÉE : Calcul de l'anatomie réelle...")
            x1, y1, x2, y2 = target_sheep.xyxy[0].tolist()  # 2ème correction ici
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
                    data_check = pd.read_csv(DB_FILE, sep=';', encoding='utf-8-sig')
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
                
                # SÉCURITÉ EXCEL : Si le fichier a été supprimé par mégarde, on réécrit l'en-tête propre
                if not os.path.exists(DB_FILE):
                    pd.DataFrame(columns=COLONNES).to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')
                
                # Enregistrement de la nouvelle ligne (séparateur POINT-VIRGULE STRICT pour Excel)
                pd.DataFrame([row]).to_csv(DB_FILE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
                
                # 🛠️ CONVERTISSEUR DE SAUVEGARDE AUTOMATIQUE EN COLOUNES EXCEL (Intégré sans erreur)
                try:
                    df_verif = pd.read_csv(DB_FILE, sep=None, engine='python', encoding='utf-8-sig')
                    if df_verif.shape[1] == 1:  # Si tout est coincé sur 1 seule colonne
                        df_nettoye = pd.read_csv(DB_FILE, sep=r'[,,\t]', engine='python', encoding='utf-8-sig')
                        df_nettoye.to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')
                except:
                    pass

                # Réinitialisation propre
                st.session_state.step = 1
                st.session_state.mesures_ia = {k: 0.0 for k in st.session_state.mesures_ia}
                st.success(f"✅ Fiche de l'animal {id_f} validée et ajoutée au registre !")
                st.rerun()

# --- ONGLET 2 : HISTORIQUE DE TERRAIN ---
with tabs[1]:  #  Ajout de [1] pour cibler le deuxième onglet
    st.subheader("📋 Gestion de la base de données")

    data = load_data(DB_FILE, COLONNES)
    
    if not data.empty:
        st.dataframe(data, use_container_width=True)
        st.divider()
        id_m = st.selectbox("Sélectionner l'ID pour Modification/Suppression", data["ID"].unique())
        idx = data[data["ID"] == id_m].index[-1]
        
        c_act1, c_act2 = st.columns(2)
        if c_act1.button(f"❌ Supprimer définitivement {id_m}"):
            data[data["ID"] != id_m].to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')
            st.success("Fiche supprimée avec succès.")
            st.rerun()

        if c_act2.checkbox(f"📝 Modifier TOUTES les données de {id_m}"):
            with st.form("edit_all"):
                new_vals = {}
                ce1, ce2 = st.columns(2)
                
                # Parcours sécurisé de toutes les colonnes à modifier
                for i, col in enumerate(COLONNES[2:]):
                    tgt = ce1 if i % 2 == 0 else ce2
                    valeur_actuelle = data.at[idx, col]
                    
                    # 1️⃣ MENUS DÉROULANTS FERMÉS POUR LES TEXTES (Saisie libre interdite)
                    if col == "Race":
                        new_vals[col] = tgt.selectbox(col, ["Ouled Djellal", "Rembi", "Hamra"], index=["Ouled Djellal", "Rembi", "Hamra"].index(str(valeur_actuelle)) if str(valeur_actuelle) in ["Ouled Djellal", "Rembi", "Hamra"] else 0)
                    elif col == "Wilaya":
                        new_vals[col] = tgt.selectbox(col, list_wilayas, index=list_wilayas.index(str(valeur_actuelle)) if str(valeur_actuelle) in list_wilayas else 0)
                    elif col == "Commune":
                        # On réutilise les communes possibles de la géographie locale pour la sécurité
                        w_actuelle = str(data.at[idx, "Wilaya"]).split("-")[-1].strip()
                        mask_c = df_communes['wilaya_name'].str.strip().str.lower() == w_actuelle.lower()
                        c_list = sorted(df_communes[mask_c]['commune_name'].unique().tolist())
                        if c_list:
                            new_vals[col] = tgt.selectbox(col, c_list, index=c_list.index(str(valeur_actuelle)) if str(valeur_actuelle) in c_list else 0)
                        else:
                            new_vals[col] = tgt.text_input(col, value=str(valeur_actuelle))
                    elif col == "Age":
                        new_vals[col] = tgt.number_input(col, min_value=0, max_value=120, value=int(valeur_actuelle))
                    else:
                        # 2️⃣ SAISIE DES MESURES NUMÉRIQUES
                        new_vals[col] = tgt.number_input(col, value=float(valeur_actuelle))
                
                # Traitement à la soumission du formulaire de modification
                if st.form_submit_button("💾 Sauvegarder les modifications"):
                    # 3️⃣ BARRIÈRE BIOLOGIQUE STRICTE LORS DE LA MODIFICATION
                    poids_m = float(new_vals.get("Poids", 0))
                    hg_m = float(new_vals.get("HG", 0))
                    lb_m = float(new_vals.get("LB", 0))
                    tp_m = float(new_vals.get("TP", 0))
                    
                    erreurs_mod = []
                    if not (1.0 <= poids_m <= 180.0): 
                        erreurs_mod.append(f"Poids incohérent ({poids_m} kg). Plage [1 - 180].")
                    if not (30.0 <= hg_m <= 120.0): 
                        erreurs_mod.append(f"Hauteur garrot (HG) suspecte ({hg_m} cm). Plage [30 - 120].")
                    if not (30.0 <= lb_m <= 140.0): 
                        erreurs_mod.append(f"Longueur bassin (LB) suspecte ({lb_m} cm). Plage [30 - 140].")
                    if not (40.0 <= tp_m <= 160.0): 
                        erreurs_mod.append(f"Tour poitrine (TP) suspect ({tp_m} cm). Plage [40 - 160].")
                    
                    if erreurs_mod:
                        st.error("🛑 MODIFICATION REFUSÉE : Des données aberrantes empêchent la mise à jour.")
                        for err in erreurs_mod:
                            st.warning(f"🔹 {err}")
                    else:
                        # Application et forçage strict du type de chaque champ avant écriture
                        for k, v in new_vals.items():
                            if k in ["Race", "Wilaya", "Commune"]:
                                data.at[idx, k] = str(v).strip()
                            elif k == "Age":
                                data.at[idx, k] = int(v)
                            else:
                                data.at[idx, k] = float(v)
                                
                        # Recalcul automatique de la ration suggérée liée au nouveau poids modifié
                        data.at[idx, "Ration"] = round(float(new_vals["Poids"]) * 0.035, 2)
                        
                        # Sauvegarde définitive au format Excel point-virgule
                        data.to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')
                        st.success("🎉 Base de données mise à jour avec succès et vérifiée !")
                        st.rerun()
        
        # --- EXPORT EXCEL AUTOMATIQUE ---
        csv_brut = data.to_csv(sep=';', index=False, encoding='utf-8-sig')
        csv_pour_excel = "sep=;\n" + csv_brut
        
        st.download_button(
            label="📥 Export Excel (Colonnes alignées)", 
            data=csv_pour_excel.encode('utf-8-sig'), 
            file_name=f"OviStat_Export_{date.today()}.csv",
            mime="text/csv"
        )
    else:
        st.info("La base est vide.")


# --- ONGLET 3 : ANALYSE ---
with tabs[2]:
    st.header("📊 Analyse des Performances")
    # Relecture rapide de sécurité pour l'onglet 3 au cas où l'onglet 2 était initialement vide
    data_analyse = load_data(DB_FILE, COLONNES)
    
    if not data_analyse.empty:
        target = st.selectbox("Animal pour audit", data_analyse["ID"].unique(), key="ana_sel")
        anim = data_analyse[data_analyse["ID"] == target].iloc[-1]
        
        # Sécurisation des calculs numériques
        try:
            poids_val = float(anim['Poids'])
            lb_val = float(anim['LB'])
            ic = poids_val / lb_val if lb_val > 0 else 0
        except:
            ic = 0
            
        st.metric("Indice Viande", f"{ic:.2f}")
    else:
        st.info("Aucune donnée disponible pour l'analyse. Enregistrez un animal d'abord.")

 # --- ONGLET 4 : SANTÉ & SUIVI MÉDICAL ---
with tabs[3]:
    st.header("🩺 Carnet de Santé Numérique")
    
    # Rappel des contacts en haut
    with st.expander("📞 Urgences Vétérinaires"):
        st.write("**Dr.Bensemane** : 07 70 90 88 88")
        st.write("**Clinique Vétérinaire** : 021386231/32")

    st.divider()
    
    # Formulaire détaillé
    with st.form("form_sante_pro"):
        c_s1, c_s2 = st.columns(2)
        
        # Sélection de l'animal dans la base actuelle
        target_sante = c_s1.selectbox("Animal concerné", data["ID"].unique()) if not data.empty else "N/A"
        
        # Type d'acte
        acte_type = c_s1.selectbox("Type d'intervention", 
            ["Déparasitage (Interne/Externe)", "Vaccination", "Traitement Curatif", "Soin de plaie", "Suppléments/Vitamines"])
        
        # Maladie / Produit
        pathologie = c_s2.selectbox("Pathologie / Motif", 
            ["Prévention standard", "Enterotoxémie", "Fièvre Aphteuse", "Clavelée", "PPR", "Coccidiose", "Gale/Tiques", "Boiterie", "Autre"])
        
        produit_nom = c_s2.text_input("Nom du médicament / Vaccin")
        
        # Administration et Rappel
        c_s3, c_s4, c_s5 = st.columns(3)
        dose = c_s3.text_input("Dose (ml/mg)")
        date_rappel = c_s4.date_input("Date de rappel prévue")
        delai_viande = c_s5.number_input("Délai d'attente (jours)", value=0)
        
        note_obs = st.text_area("Observations particulières")

        if st.form_submit_button("💉 ENREGISTRER L'INTERVENTION"):
            nouveau_soin = [
                date.today(), target_sante, acte_type, pathologie, 
                produit_nom, dose, date_rappel, delai_viande, note_obs
            ]
            # Sauvegarde dans le fichier santé séparé
            pd.DataFrame([nouveau_soin], columns=["Date", "ID", "Type", "Motif", "Produit", "Dose", "Rappel", "Delai", "Obs"]).to_csv(DB_SANTE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
            st.success(f"Soin enregistré pour {target_sante} !")
            st.rerun()

    # Affichage de l'historique médical
    if os.path.exists(DB_SANTE):
        st.subheader("📋 Historique des soins")
        df_sante = pd.read_csv(DB_SANTE, sep=';')
        if not df_sante.empty:
            st.dataframe(df_sante, use_container_width=True)
            # --- ONGLET 5 : À PROPOS & GUIDE UTILISATEUR ---
with tabs[4]:
    st.header("📘 Guide Utilisateur OviStat Vision Pro")
    
    # --- SECTION 1 : MANUEL PAS À PAS ---
    st.subheader("🚀 Fonctionnement étape par étape")
    
    with st.expander("1️⃣ Localisation & Identification (Début)", expanded=True):
        st.write("""
        *   **Localisation :** Commencez par choisir la **Wilaya** puis la **Commune**. Cette partie est instantanée.
        *   **ID Animal :** Tapez le numéro de boucle. Si l'animal n'est pas marqué, laissez vide : l'IA générera un code **TEMP-**.
        *   **Race & Âge :** Sélectionnez les informations de base de l'ovin.
        """)

    with st.expander("2️⃣ Le Scan IA Triple-Vue"):
        st.write("""
        Le scan se déroule en 3 captures pour une précision maximale :
        1.  **Profil :** Cadrez l'animal de côté (calcul de la Hauteur HG). Cliquez sur 'Capturer' puis sur le bouton de validation.
        2.  **Dessus :** Cadrez l'animal depuis le haut (calcul de la Largeur TP). Validez.
        3.  **Tête :** Prenez la face de l'animal pour confirmer la race.
        *💡 Note : Après la 3ème photo, l'IA affiche ses suggestions de mesures en vert.*
        """)

    with st.expander("3️⃣ Validation des Mensurations"):
        st.write("""
        *   Ouvrez les **4 Expanders** (Dimensions, Poitrine, Tête, Reproduction).
        *   Les cases se remplissent automatiquement grâce à l'IA.
        *   **Contrôle humain :** Vous pouvez modifier n'importe quel chiffre au clavier si vous constatez une erreur de l'IA.
        *   Cliquez sur **💾 ENREGISTRER** pour fixer les données dans le fichier Excel.
        """)

    with st.expander("4️⃣ Gestion, Analyse & Export (Fin)"):
        st.write("""
        *   **Historique :** Consultez le tableau. Les animaux officiels ont une étoile ⭐, les temporaires une horloge 🕒.
        *   **Correction :** En cas d'erreur de saisie, utilisez le module 'Modifier' pour corriger le Poids ou le HG.
        *   **Analyse :** Sélectionnez un animal pour voir son **Indice Viande** et téléchargez son **Certificat PDF** officiel.
        *   **Export :** Téléchargez le fichier global via le bouton 'Export Excel' pour vos rapports.
        """)

    # --- SECTION : MODE HORS-LIGNE (ROUE DE SECOURS) ---
    st.divider()
    with st.expander("📶 Comment utiliser OviStat SANS INTERNET ?"):
        st.write("""
        Si la 5G est absente à la ferme, vous pouvez transformer votre PC en serveur local. 
        Suivez ces étapes sur votre ordinateur portable :
        """)
        
        st.info("**1. Préparation du PC (Une seule fois)**")
        st.code("""
        # Installez Python sur www.python.org
        # Ouvrez un terminal (CMD) et installez les outils :
        pip install streamlit pandas ultralytics fpdf pillow
        """, language="bash")
        
        st.info("**2. Téléchargement du logiciel**")
        st.write("""
        *   Copiez vos fichiers (`app.py`, `algeria_geo.csv`, `yolov8n.pt`) dans un dossier sur votre Bureau.
        """)
        
        st.info("**3. Lancement du serveur local**")
        st.write("Ouvrez le terminal dans votre dossier et tapez :")
        st.code("streamlit run app.py", language="bash")
        
        st.success("**4. Connexion du téléphone (Zéro Data 5G)**")
        st.write("""
        1.  Connectez le PC et le téléphone au même réseau Wi-Fi (même s'il n'y a pas d'internet).
        2.  Regardez l'adresse s'afficher dans le terminal du PC (ex: `192.168.1.XX:8501`).
        3.  Tapez cette adresse dans le navigateur de votre téléphone Android.
        4.  **OviStat fonctionne maintenant à 100% sans internet !**
        """)
 
    # --- SECTION 2 : INFOS LOGICIEL ---
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.image("https://flaticon.com", width=100)
    with col_b:
        st.write("**Nom du Logiciel :** OviStat Vision Pro")
        st.write("**Version :** 1.9.5 (Algérie Edition)")
        st.write("**Auteur :** MERABIA KAWTHER")
        st.write("**Technologie :** IA YOLOv8 & Streamlit Cloud")

    # --- SECTION 3 : LIEN PDF ---
    st.divider()
    if os.path.exists("manuel_ovistat.pdf"):
        with open("manuel_ovistat.pdf", "rb") as f:
            st.download_button("📥 Télécharger le manuel complet en PDF", f.read(), "Manuel_OviStat_Detaille.pdf")
    
    if st.button("🔄 Réinitialiser le cycle de scan (Nouvel animal)"):
        st.session_state.step = 1
        st.rerun()


# --- MAINTENANCE ---
st.divider()
with st.expander("⚙️ MAINTENANCE SYSTÈME"):
    st.warning("⚠️ Action irréversible")
    if st.button("🗑️ VIDER TOUTES LES DONNÉES"):
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        if os.path.exists(DB_SANTE): os.remove(DB_SANTE)
        st.success("Serveur nettoyé !"); st.rerun()
    st.info(f"📍 Chemin serveur : {os.getcwd()}")
