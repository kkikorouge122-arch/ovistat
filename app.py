import streamlit as st
import pandas as pd
import os
from datetime import datetime, date
from ultralytics import YOLO
from PIL import Image

# --- 1. CONFIGURATION & DESIGN MOBILE ---
st.set_page_config(page_title="OviStat Vision v2.5", page_icon="🐑", layout="wide")

st.markdown("""
<style>
div[data-testid="stCameraInput"] { border: 5px solid #1f77b4 !important; border-radius: 25px !important; background-color: #000; max-width: 800px; margin: auto; position: relative; overflow: hidden !important; }
div[data-testid="stCameraInput"] video { width: 100% !important; height: 600px !important; object-fit: cover !important; transform: scale(1.1); filter: contrast(1.1) brightness(1.1); }
div[data-testid="stCameraInput"]::after { content: ""; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80px; height: 80px; border: 2px dashed rgba(255, 255, 255, 0.35); border-radius: 50%; box-shadow: 0 0 0 1000px rgba(0, 0, 0, 0.25); pointer-events: none; }
div[data-testid="stCameraInput"] button { height: 75px !important; width: 75px !important; border-radius: 50% !important; border: 4px solid white !important; background-color: #1f77b4 !important; position: relative !important; margin: 15px auto !important; display: block !important; z-index: 20; opacity: 0.95; }
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
        st.title("🛡️ OviStat Securisé")
        pwd = st.text_input("🔑 Code chercheur :", type="password")
        if st.button("DÉVERROUILLER"):
            if pwd == "ENSVAlger2026":
                st.session_state.auth = True
                st.rerun()
            else: st.error("Code incorrect")
    st.stop()

if not st.session_state.auth: login()

# --- 3. CONFIGURATION DES BASES DE DONNÉES & GÉO ---
DB_FILE = "data_ovinstat_V16.csv"
COLONNES = ["Date", "ID", "Race", "Age", "Poids", "HG", "HS", "LB", "LQ", "LT_tronc", "LC_cou", "LH", "LI", "LP", "PP", "TP", "Lcornes", "LTete", "LtTete", "LO", "Lo", "TC", "LY", "TS", "PS", "LG", "LL", "Wilaya", "Commune", "Ration"]

@st.cache_resource
def load_yolo(): return YOLO('yolov8n.pt')

@st.cache_data
def get_algeria_geo():
    wilayas = ["16-Alger", "17-Djelfa", "51-Ouled Djellal"]
    if os.path.exists("algeria_geo.csv"):
        df = pd.read_csv("algeria_geo.csv", sep=";", encoding='utf-8', on_bad_lines='skip')
    else: df = pd.DataFrame(columns=["wilaya_name", "commune_name"])
    return wilayas, df

model = load_yolo()
list_wilayas, df_communes = get_algeria_geo()

if not os.path.exists(DB_FILE) or os.path.getsize(DB_FILE) == 0:
    pd.DataFrame(columns=COLONNES).to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')

# --- 4. INTERFACE PRINCIPALE ---
tabs = st.tabs(["📥 Saisie", "🔍 Historique", "ℹ️ À Propos"])

# --- ONGLET 1 : SAISIE PROGRESSIVE ---
with tabs[0]:
    st.subheader("📍 Localisation de l'étude")
    c_w, c_c = st.columns(2)
    wilaya_sel = c_w.selectbox("Wilaya", list_wilayas, key="w_v5")
    commune_sel = c_c.text_input("Commune", value="Djelfa", key="c_v5")

    st.divider()
    st.write(f"### 📸 Étape {st.session_state.step}/3 : {['Profil', 'Dessus', 'Tête'][st.session_state.step-1]}")
    
    if st.button(f"✅ VALIDER LA PHOTO {st.session_state.step}", type="primary", use_container_width=True):
        if st.session_state.last_photo:
            if st.session_state.step < 3:
                st.session_state.step += 1
                st.session_state.last_photo = None
                st.rerun()
            else: st.success("🎯 Analyse IA Complète !")
        else: st.error("⚠️ Prenez d'abord la photo ci-dessous.")

    photo = st.camera_input("Scanner", key=f"cam_v5_{st.session_state.step}")
    
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
            st.success("🔒 CIBLE VERROUILLÉE AU CENTRE !")
            x1, y1, x2, y2 = target_sheep.xyxy.tolist()
            p_w, p_h = x2 - x1, y2 - y1
            
            if st.session_state.step == 1:
                st.session_state.mesures_ia.update({"HG": round(p_h*0.14, 1), "LB": round(p_w*0.14, 1), "HS": round(p_h*0.13, 1), "Poids": round((p_h*p_w*0.14*0.14)/110, 1), "LT_tronc": round(p_w*0.08, 1), "LC_cou": round(p_w*0.04, 1), "LH": round(p_w*0.03, 1), "LQ": 22.0})
            elif st.session_state.step == 2:
                st.session_state.mesures_ia.update({"TP": round((p_h*0.28)+(p_w*0.28), 1), "LI": round(p_w*0.05, 1), "LP": round(p_w*0.06, 1), "PP": round(p_h*0.1, 1)})
            elif st.session_state.step == 3:
                st.session_state.mesures_ia.update({"Lc_cornes": 15.0, "LTete": round(p_h*0.05, 1), "LtTete": round(p_h*0.02, 1), "LO": round(p_h*0.06, 1), "Lo": round(p_h*0.01, 1), "TC": round(p_h*0.02, 1), "LY": 2.5, "LL": 5.0})
        else: st.error("❌ RECADREZ LE MOUTON AU CENTRE DU VISEUR 🎯")

    st.divider()

    with st.form("form_final"):
        st.subheader("📋 Formulaire de Validation Vétérinaire")
        m = st.session_state.mesures_ia
        
        c1, c2, c3 = st.columns(3)
        id_in = c1.text_input("ID / Boucle")
        age_in = c2.number_input("Âge (mois)", 0, 120, 12)
        race_in = c3.selectbox("Race", ["Ouled Djellal", "Rembi", "Hamra"])

        with st.expander("1️⃣ Dimensions Corporelles", expanded=True):
            f1, f2, f3, f4 = st.columns(4)
            poids = f1.number_input("Poids (kg)", value=float(m["Poids"]))
            hg = f2.number_input("HG (cm)", value=float(m["HG"]))
            hs = f3.number_input("HS (cm)", value=float(m["HS"]))
            lb = f4.number_input("LB (cm)", value=float(m["LB"]))
            lq = f1.number_input("LQ (cm)", value=float(m["LQ"]))

        with st.expander("2️⃣ Poitrine & Largeurs"):
            f5, f6, f7, f8 = st.columns(4)
            li = f5.number_input("LI (cm)", value=float(m["LI"]))
            lp = f6.number_input("LP (cm)", value=float(m["LP"]))
            pp = f7.number_input("PP (cm)", value=float(m["PP"]))
            tp = f8.number_input("TP (cm)", value=float(m["TP"]))

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
            ration = round(poids * 0.035, 2)
            r3.metric("Ration Sug.", f"{ration} kg")

        if st.form_submit_button("💾 ENREGISTRER LA FICHE COMPLÈTE"):
            id_f = id_in if id_in else f"T-{datetime.now().strftime('%H%M%S')}"
            
            # Algorithme anti-doublon compact
            df_curr = pd.read_csv(DB_FILE, sep=';')
            is_dup = False
            if not df_curr.empty:
                p_h = pd.to_numeric(df_curr['Poids'], errors='coerce')
                h_h = pd.to_numeric(df_curr['HG'], errors='coerce')
                if not df_curr[p_h.between(poids-1, poids+1) & h_h.between(hg-1, hg+1)].empty: is_dup = True

            if is_dup: st.error("⚠️ Animal déjà enregistré aujourd'hui (Mensurations identiques).")
            else:
                row = [date.today(), id_f, race_in, age_in, poids, hg, hs, lb, lq, m["LT_tronc"], m["LC_cou"], m["LH"], li, lp, pp, tp, lc_cornes, lt_tete, lt_la, lo_lo, lo_la, tc, ly, ts, ps, lg, ll, wilaya_sel, commune_sel, ration]
                pd.DataFrame([row], columns=COLONNES).to_csv(DB_FILE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
                st.session_state.step = 1
                st.session_state.mesures_ia = {k: 0.0 for k in st.session_state.mesures_ia}
                st.success("✅ Enregistré !")
                st.rerun()

# --- ONGLET 2 : HISTORIQUE ---
with tabs[1]:
    df_read = pd.read_csv(DB_FILE, sep=';')
    st.subheader("📋 Registre d'étude des troupeaux")
    if not df_read.empty:
        st.dataframe(df_read, use_container_width=True)
        id_m = st.selectbox("ID de l'ovin", df_read["ID"].unique())
        if st.button(f"❌ Supprimer {id_m}"):
            df_read[df_read["ID"] != id_m].to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')
            st.rerun()
    else: st.info("Registre vide.")


      # --- ONGLET 3 : ANALYSE ---
with tabs[2]:
    st.subheader("📊 Audit Morphométrique")
    if not data.empty:
        target = st.selectbox("Sélectionner le sujet d'étude", data["ID"].unique(), key="sel_audit")
        anim = data[data["ID"] == target].iloc[-1]
        
        try:
            # Toutes ces lignes ont été décalées de 4 espaces vers la droite
            ic = float(anim['Poids']) / float(anim['LB']) if float(anim['LB']) > 0 else 0
            ir = (float(anim['TP'])**2) / float(anim['HG']) if float(anim['HG']) > 0 else 0
            
            c_an1, c_an2 = st.columns(2)
            c_an1.metric("🥩 Indice de Compacité", f"{ic:.2f}")
            c_an2.metric("🏗️ Indice de Robustesse", f"{ir:.2f}")
            
        except Exception as e: 
            st.error(f"Erreur indices : {e}")

        
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
