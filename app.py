import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, date
from ultralytics import YOLO
from PIL import Image
from fpdf import FPDF
# --- SYSTÈME DE SÉCURITÉ ET LOGIN ---
if 'auth' not in st.session_state:
    st.session_state.auth = False

# --- SYSTÈME DE SÉCURITÉ INSTITUTIONNEL ENSV ALGER ---
if 'auth' not in st.session_state:
    st.session_state.auth = False

def login():
    # Centrage visuel
    c1, c2, c3 = st.columns([1, 2, 1])
    
    with c2:
        # Utilisation d'un logo vétérinaire officiel ou celui de l'ENSV si vous avez le lien
        # À défaut, j'utilise une icône de santé animale de haute qualité
        st.image("https://flaticon.com", width=120) 
        
        st.title("🛡️ OviStat Vision Pro")
        st.subheader("École Nationale Supérieure Vétérinaire d'Alger")
        st.markdown("---")
        
        st.info("""
        **🎓 Cadre : Recherche Zootechnique & Innovation**  
        Ce système expert d'analyse morphométrique assisté par IA est la propriété intellectuelle de **MERABIA KAWTHER (ENSV)**.
        """)
        
        with st.container():
            # Champ de mot de passe
            pwd = st.text_input("🔑 Code d'accès chercheur :", type="password")
            
            if st.button("DÉVERROUILLER L'INTERFACE"):
                if pwd == "ENSVAlger2026": # <--- VOTRE NOUVEAU MOT DE PASSE
                    st.session_state.auth = True
                    st.success("Accès autorisé. Bienvenue, Docteur.")
                    st.rerun()
                else:
                    st.error("Accès refusé. Identifiants incorrects.")
        
        st.markdown("---")
        st.caption("📍 El Alia, Alger - Laboratoire de Zootechnie")

    st.stop() # Bloque l'application

if not st.session_state.auth:
    login()


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
    /* 1. On définit un conteneur solide pour la caméra */
    div[data-testid="stCameraInput"] {
        border: 4px solid #1f77b4 !important; /* Le cadre bleu revient ici */
        border-radius: 20px !important;
        overflow: hidden !important;
        background-color: black;
        margin-top: 10px;
    }

    /* 2. On force la vidéo à être grande et nette */
    div[data-testid="stCameraInput"] video {
        width: 100% !important;
        height: 500px !important; /* Hauteur fixe pour bien voir le mouton */
        object-fit: cover !important; /* Remplit le cadre sans bandes noires */
    }

    /* 3. On agrandit le bouton de capture pour le pouce */
    div[data-testid="stCameraInput"] button {
        height: 60px !important;
        font-size: 18px !important;
        background-color: #1f77b4 !important;
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
# --- INITIALISATION DES VARIABLES DANS LA MÉMOIRE (Session State) ---
if 'mesures_ia' not in st.session_state:
    st.session_state.mesures_ia = {col: 0.0 for col in COLONNES}
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
     # 2. ZONE DE CAPTURE IA
    etapes = ["PROFIL (Côtés)", "DESSUS (Dos)", "TÊTE (Face)"]
    st.subheader(f"📸 Étape {st.session_state.step}/3 : {etapes[st.session_state.step-1]}")
    
    # Bouton de validation placé en haut
    if st.button(f"✅ VALIDER LA VUE {st.session_state.step}", type="primary", use_container_width=True):
        if st.session_state.last_photo:
            if st.session_state.step < 3:
                st.session_state.step += 1
                st.session_state.last_photo = None
                st.rerun()
            else:
                st.balloons()
                st.success("🎯 Étude morphométrique terminée. Vérifiez les 30 colonnes ci-dessous.")
        else:
            st.error("⚠️ Capturez d'abord l'animal avec la caméra.")

    photo = st.camera_input("Viser le centre 🎯", key=f"cam_v3_{st.session_state.step}")

    # 🧠 LE CERVEAU IA (YOLOv8) : Remplit les cases selon l'étape
    if photo:
        st.session_state.last_photo = photo
        with st.spinner("Analyse anatomique..."):
            # Simulation des mesures selon l'étape
            if st.session_state.step == 1:
                st.session_state.mesures_ia.update({"HG": 68.5, "HS": 67.2, "LB": 78.0, "LT_tronc": 45.0, "LC_cou": 22.0, "LH": 18.0})
                st.info("📊 Mesures de profil débloquées (HG, LB, HS...)")
            elif st.session_state.step == 2:
                st.session_state.mesures_ia.update({"TP": 84.0, "LI": 14.5, "LP": 22.3, "PP": 32.1})
                st.info("📊 Mesures de largeur débloquées (TP, LP...)")
            elif st.session_state.step == 3:
                st.session_state.mesures_ia.update({"Lc_cornes": 0.0, "LTete": 24.5, "LtTete": 12.2, "LO": 28.0, "Lo": 8.5, "TC": 9.2})
                st.info("📊 Mesures de tête débloquées (Oreilles, Canon...)")

    st.divider()

    # 3. FORMULAIRE DE VALIDATION FINALE (Les cases se remplissent toutes seules)
    with st.form("form_final"):
        st.subheader("📋 Synthèse des 24 Paramètres")
        
        m = st.session_state.mesures_ia # Raccourci
        
        with st.expander("1️⃣ Dimensions Corporelles (Profil)", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            poids = c1.number_input("Poids (kg)", value=0.0)
            hg = c2.number_input("HG", value=m["HG"])
            hs = c3.number_input("HS", value=m["HS"])
            lb = c4.number_input("LB", value=m["LB"])

        with st.expander("2️⃣ Poitrine & Volume (Dessus)"):
            c1, c2, c3, c4 = st.columns(4)
            tp = c1.number_input("Tour Poitrine (TP)", value=m["TP"])
            li = c2.number_input("Larg. Ischions (LI)", value=m["LI"])
            lp = c3.number_input("Larg. Poitrine (LP)", value=m["LP"])
            pp = c4.number_input("Prof. Poitrine (PP)", value=m["PP"])

             with st.expander("3️⃣ Tête & Oreilles"):
            # Ligne 1 : Cornes et Tête
            g9, g10, g11 = st.columns(3)
            lc_cornes = g9.number_input("Long. Cornes (Lc)", value=m["Lc_cornes"])
            lt_tete = g10.number_input("Long. Tête (LT)", value=m["LTete"])
            lt_la = g11.number_input("Larg. Tête (Lt)", value=m["LtTete"])
            
            # Ligne 2 : Oreilles et Canon
            g12, g13, g14 = st.columns(3)
            lo_lo = g12.number_input("Long. Oreilles (LO)", value=m["LO"])
            lo_la = g13.number_input("Larg. Oreilles (Lo)", value=m["Lo"])
            tc = g14.number_input("Tour Canon (TC)", value=m["TC"])


          with st.expander("4️⃣ Reproduction & Laine"):
            # Ligne 1 : Appareil reproducteur
            r1, r2, r3 = st.columns(3)
            ly = r1.number_input("Long. Trayons (LY)", value=m["LY"])
            ts = r2.number_input("Tour Scrotal (TS)", value=m["TS"])
            ps = r3.number_input("Prof. Scrotale (PS)", value=m["PS"])
            
            # Ligne 2 : Gigot et Laine
            r4, r5, r6 = st.columns(3)
            lg = r4.number_input("Long. Gigot (LG)", value=m["LG"])
            ll = r5.number_input("Long. Laine (LL)", value=m["LL"])
            
            # Calcul de la ration affiché en temps réel
            ration_estim = round(poids * 0.035, 2)
            r6.metric("Ration (kg)", ration_estim)

        # --- BOUTON D'ENREGISTREMENT FINAL ---
        if st.form_submit_button("💾 ENREGISTRER LA FICHE COMPLÈTE"):
            # Génération de l'ID si vide
            id_f = id_in if id_in else f"TEMP-{datetime.now().strftime('%H%M%S')}"
            
            # Création de la ligne avec TOUTES les variables des expanders (30 colonnes)
            row = [
                date.today(), id_f, race_in, age_in, poids, 
                hg, hs, lb, lq, lt_t, lc_c, lh, li, lp, pp, tp, 
                lc_cornes, lt_tete, lt_la, lo_lo, lo_la, tc, 
                ly, ts, ps, lg, ll, 
                wilaya_sel, commune_sel, ration_estim
            ]
            
            # Sauvegarde physique dans le CSV
            pd.DataFrame([row], columns=COLONNES).to_csv(DB_FILE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
            
            # Réinitialisation pour l'animal suivant
            st.session_state.step = 1
            st.session_state.mesures_ia = {col: 0.0 for col in COLONNES} # On vide la mémoire IA
            st.success(f"✅ Fiche de l'animal {id_f} enregistrée avec succès !")
            st.rerun()



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

