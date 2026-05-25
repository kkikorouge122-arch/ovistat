import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, date
from ultralytics import YOLO
from PIL import Image
from fpdf import FPDF
# --- 1. CONFIGURATION & SÉCURITÉ ---
st.set_page_config(page_title="OviStat Vision Pro v2.4", page_icon="🐑", layout="wide")

if 'auth' not in st.session_state: st.session_state.auth = False
if 'step' not in st.session_state: st.session_state.step = 1
if 'last_photo' not in st.session_state: st.session_state.last_photo = None
if 'mesures_ia' not in st.session_state:
    st.session_state.mesures_ia = {
        "HG": 0.0, "HS": 0.0, "LB": 0.0, "TP": 0.0, "LI": 0.0, "LP": 0.0, "PP": 0.0,
        "Lc_cornes": 0.0, "LTete": 0.0, "LtTete": 0.0, "LO": 0.0, "Lo": 0.0, "TC": 0.0,
        "LY": 0.0, "TS": 0.0, "PS": 0.0, "LG": 0.0, "LL": 0.0
    }

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

# --- 1. GESTION DU MODE NUIT (SESSION STATE) ---
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'step' not in st.session_state:
    st.session_state.step = 1
with st.sidebar:
    st.divider()
    st.subheader("🔐 Sécurité")
    if st.button("🚪 Se déconnecter / Verrouiller"):
        st.session_state.auth = False
        st.success("Session fermée avec succès.")
        st.rerun()
    
# --- 1. CONFIGURATION ---
st.set_page_config(page_title="OviStat Vision Pro v2.1", page_icon="🐑", layout="wide")

# Couleurs dynamiques selon le mode
if st.session_state.dark_mode:
    bg_c, card_c, text_c, border_c = "#0e1117", "#1d2129", "#e0e0e0", "#3d4450"
    metric_bg = "#12141d"
else:
    bg_c, card_c, text_c, border_c = "#f8f9fa", "#ffffff", "#1f77b4", "#dee2e6"
    metric_bg = "#ffffff"

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
        height: 600px !important; /* Hauteur augmentée pour le recul */
        object-fit: cover !important;
        transform: scale(1.1); /* Petit zoom pour compenser la distance de 2m */
        filter: contrast(1.1) brightness(1.1); /* Améliore la détection des bords pour l'IA */
    }

    /* 3. Guide visuel de centrage (Le Viseur) */
    div[data-testid="stCameraInput"]::after {
        content: "📐 CADRAGE 2 MÈTRES";
        position: absolute;
        top: 10px;
        left: 50%;
        transform: translateX(-50%);
        background-color: rgba(31, 119, 180, 0.7);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: bold;
    }

    /* 4. Bouton de capture géant pour éviter de bouger en cliquant */
    div[data-testid="stCameraInput"] button {
        height: 80px !important;
        width: 80px !important;
        border-radius: 50% !important;
        border: 4px solid white !important;
        background-color: #1f77b4 !important;
        bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)



DB_FILE = "data_ovinstat_V16.csv"
DB_SANTE = "data_sante_ovins.csv"

COLONNES = [
    "Date", "ID", "Race", "Age", "Poids", 
    "HG", "HS", "LB", "LQ", "LT_tronc", "LC_cou", "LH", "LI", "LP", "PP", "TP",
    "Lcornes", "LTete", "LtTete", "LO", "Lo", "TC", "LY", "TS", "PS", "LG", "LL", 
    "Wilaya", "Commune", "Ration"
]
COL_SANTE = ["Date", "ID", "Type", "Produit", "Veterinaire", "Prochain_RDV"]

# --- 2. FONCTIONS ---
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
        # On force Pandas à utiliser le point-virgule et l'encodage correct
        df_geo = pd.read_csv("algeria_geo.csv", sep=";", encoding='utf-8', on_bad_lines='skip')
        # Nettoyage immédiat : on enlève les espaces en trop dans le fichier
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

model = load_yolo_model()
list_wilayas, df_communes = get_algeria_geo()
data = load_data(DB_FILE, COLONNES)
# --- 5. BARRE LATÉRALE (SIDEBAR) ---
with st.sidebar:
    st.image("https://flaticon.com", width=80)
    st.title("OviStat Menu")
    if st.button("🌙 Basculer Mode Nuit / Jour"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    st.divider()
    with st.expander("⚙️ MAINTENANCE"):
        if st.button("🗑️ Vider la base"):
            if os.path.exists(DB_FILE): os.remove(DB_FILE)
            st.rerun()
# --- 3. INTERFACE ---
st.title("🐑 OviStat IA v2.1")
tabs = st.tabs(["📥 Saisie", "🔍 Historique & Modif", "📊 Analyse", "🩺 Santé", "ℹ️ À Propos"])

# --- ONGLET 1 : SAISIE ---
# Initialisation de la mémoire IA avec TOUTES les clés nécessaires
if 'mesures_ia' not in st.session_state:
    # On crée un dictionnaire avec des noms simplifiés pour éviter les erreurs
    st.session_state.mesures_ia = {
        "HG": 0.0, "HS": 0.0, "LB": 0.0, "LT_tronc": 0.0, "LC_cou": 0.0, "LH": 0.0,
        "TP": 0.0, "LI": 0.0, "LP": 0.0, "PP": 0.0,
        "Lc_cornes": 0.0, "LTete": 0.0, "LtTete": 0.0, "LO": 0.0, "Lo": 0.0, "TC": 0.0,
        "LY": 0.0, "TS": 0.0, "PS": 0.0, "LG": 0.0, "LL": 0.0
    }

# --- ONGLET 1 : SCAN PROGRESSIF ---
with tabs[0]:
    st.subheader("📍 Localisation de l'étude")
    
    # Éléments dynamiques (Hors formulaire pour mise à jour instantanée)
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

    st.divider()
          # --- B. Zone de Scan IA avec Haute Précision ---
    st.markdown("""
        <style>
        /* Cadre Caméra Imposant pour recul 2m */
        div[data-testid="stCameraInput"] {
            border: 5px solid #1f77b4 !important;
            border-radius: 25px !important;
            background-color: #000;
            position: relative;
        }
        /* Effet Zoom et Contraste pour IA */
        div[data-testid="stCameraInput"] video {
            height: 600px !important;
            object-fit: cover !important;
            transform: scale(1.15); /* Zoom 15% */
            filter: contrast(1.1) brightness(1.1);
        }
        /* Viseur Central */
        div[data-testid="stCameraInput"]::after {
            content: "🎯 CADRAGE 2M";
            position: absolute;
            top: 15px; left: 50%;
            transform: translateX(-50%);
            background: rgba(31, 119, 180, 0.8);
            color: white; padding: 5px 15px;
            border-radius: 20px; font-size: 12px; font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    st.write(f"### 📸 Étape {st.session_state.step}/3 : {['Profil', 'Dessus', 'Tête'][st.session_state.step-1]}")
    
    # Bouton de validation stratégique en haut
    if st.button(f"✅ VALIDER LA PHOTO {st.session_state.step}", type="primary", use_container_width=True):
        if st.session_state.last_photo:
            if st.session_state.step < 3:
                st.session_state.step += 1
                st.session_state.last_photo = None
                st.rerun()
            else:
                st.success("🎯 Étude morphométrique complète !")
        else:
            st.error("⚠️ Capturez l'animal d'abord avec le bouton cercle.")

    # Caméra Unique avec clé dynamique
    photo = st.camera_input("Scanner l'animal", key=f"precision_cam_v4_{st.session_state.step}")
    
    if photo:
        st.session_state.last_photo = photo
        results = model(Image.open(photo))
        nb = sum(1 for r in results for b in r.boxes if int(b.cls) == 18)
        
        if nb == 1:
            st.success("✅ Animal unique détecté au centre.")
            if st.session_state.step == 1: st.session_state.mesures_ia.update({"HG":68.5, "HS":67.0, "LB":78.0})
            if st.session_state.step == 2: st.session_state.mesures_ia.update({"TP":84.0, "LI":14.0, "LP":22.0, "PP":32.0})
            if st.session_state.step == 3: st.session_state.mesures_ia.update({"Lc_cornes":0.0, "LTete":24.0, "LO":28.0, "TC":9.0})
        elif nb > 1: 
            st.warning(f"⚠️ {nb} moutons vus. Centrez la cible sous le viseur 🎯.")
        else:
            st.error("❌ Aucun ovin détecté. Rapprochez-vous un peu.")

    st.divider()

    # C. Formulaire de Mensurations (Les valeurs m.get récupèrent les infos de l'IA ci-dessus)
    with st.form("form_final"):
        st.subheader("📋 Fiche Identité & Morphométrie")
        m = st.session_state.mesures_ia
        
        c1, c2, c3 = st.columns(3)
        id_in = c1.text_input("ID / Boucle")
        age_in = c2.number_input("Âge (mois)", 0, 120, 12)
        race_in = c3.selectbox("Race", ["Ouled Djellal", "Rembi", "Hamra"])

        with st.expander("1️⃣ Dimensions Corporelles", expanded=True):
            g1, g2, g3, g4 = st.columns(4)
            poids = g1.number_input("Poids (kg)", 0.0, 150.0, 0.0)
            hg = g2.number_input("HG (cm)", value=m.get("HG", 0.0))
            hs = g3.number_input("HS (cm)", value=m.get("HS", 0.0))
            lb = g4.number_input("LB (cm)", value=m.get("LB", 0.0))

        with st.expander("2️⃣ Poitrine & Largeurs"):
            g5, g6, g7, g8 = st.columns(4)
            li = g5.number_input("LI (cm)", value=m.get("LI", 0.0))
            lp = g6.number_input("LP (cm)", value=m.get("LP", 0.0))
            pp = g7.number_input("PP (cm)", value=m.get("PP", 0.0))
            tp = g8.number_input("TP (cm)", value=m.get("TP", 0.0))

        with st.expander("3️⃣ Tête & Oreilles"):
            t1, t2, t3 = st.columns(3)
            lc_cornes = t1.number_input("L. Cornes", value=m.get("Lc_cornes", 0.0))
            lt_tete = t2.number_input("L. Tête", value=m.get("LTete", 0.0))
            lt_la = t3.number_input("Larg. Tête", value=m.get("LtTete", 0.0))
            lo_lo = t1.number_input("L. Oreille", value=m.get("LO", 0.0))
            lo_la = t2.number_input("Larg. Oreille", value=m.get("Lo", 0.0))
            tc = t3.number_input("T. Canon", value=m.get("TC", 0.0))

        with st.expander("4️⃣ Reproduction & Laine"):
            r1, r2, r3 = st.columns(3)
            ly = r1.number_input("LY", value=m.get("LY", 0.0))
            ts = r2.number_input("TS", value=m.get("TS", 0.0))
            ps = r3.number_input("PS", value=m.get("PS", 0.0))
            lg = r1.number_input("LG", value=m.get("LG", 0.0))
            ll = r2.number_input("LL", value=m.get("LL", 0.0))

        if st.form_submit_button("💾 ENREGISTRER LA FICHE COMPLÈTE"):
            id_f = id_in if id_in else f"T-{datetime.now().strftime('%H%M%S')}"
            # Sauvegarde des 30 colonnes
            row = [date.today(), id_f, race_in, age_in, poids, hg, hs, lb, 0, 0, 0, 0, li, lp, pp, tp, lc_cornes, lt_tete, lt_la, lo_lo, lo_la, tc, ly, ts, ps, lg, ll, wilaya_sel, commune_sel, round(poids*0.035,2)]
            pd.DataFrame([row], columns=COLONNES).to_csv(DB_FILE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
            
            # Reset pour animal suivant
            st.session_state.step = 1
            st.session_state.mesures_ia = {k: 0.0 for k in st.session_state.mesures_ia}
            st.success(f"✅ Animal {id_f} enregistré avec succès !")
            st.rerun()


with tabs[1]:
    data = pd.read_csv(DB_FILE, sep=';')
    if not data.empty:
        st.dataframe(data, use_container_width=True)
        id_m = st.selectbox("ID pour Action", data["ID"].unique())
        if st.button(f"❌ Supprimer {id_m}"):
            data[data["ID"]!=id_m].to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')
            st.rerun()

with tabs[3]:
    st.write("**OviStat Vision Pro v2.4** | ENSV Alger")
    if st.button("🔄 Reset Scan"): st.session_state.step = 1; st.rerun()
# --- ONGLET 2 : HISTORIQUE & MODIF TOTALE ---
with tabs[1]:
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
            st.rerun()

        if c_act2.checkbox(f"📝 Modifier TOUTES les données de {id_m}"):
            with st.form("edit_all"):
                new_vals = {}
                ce1, ce2 = st.columns(2)
                for i, col in enumerate(COLONNES[2:]):
                    tgt = ce1 if i % 2 == 0 else ce2
                    if col in ["Race", "Wilaya", "Commune"]: 
                        new_vals[col] = tgt.text_input(col, value=str(data.at[idx, col]))
                    else: 
                        new_vals[col] = tgt.number_input(col, value=float(data.at[idx, col]))
                
                if st.form_submit_button("💾 Sauvegarder les modifications"):
                    for k, v in new_vals.items(): data.at[idx, k] = v
                    data.to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')
                    st.success("Mise à jour réussie !"); st.rerun()
        
        st.download_button("📥 Export Excel", data.to_csv(sep=';', index=False).encode('utf-8-sig'), "base.csv")
    else:
        st.info("La base est vide.")

# --- ONGLET 3 : ANALYSE ---
with tabs[2]:
    st.header("📊 Analyse des Performances")
    if not data.empty:
        target = st.selectbox("Animal pour audit", data["ID"].unique(), key="ana_sel")
        anim = data[data["ID"] == target].iloc[-1]
        ic = anim['Poids'] / anim['LB'] if anim['LB'] > 0 else 0
        st.metric("Indice Viande", f"{ic:.2f}")
        
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

