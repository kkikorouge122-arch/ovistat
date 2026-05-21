import streamlit as st
import pandas as pd
import os
from datetime import datetime, date
from ultralytics import YOLO
from PIL import Image
from fpdf import FPDF

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="OviStat Vision Pro v2.1", page_icon="🐑", layout="wide")

st.markdown("""
    <style>
    .block-container { padding-top: 1rem; }
    div[data-testid="stCameraInput"] video { width: 100% !important; object-fit: cover !important; border-radius: 15px; border: 3px solid #1f77b4; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; border-radius: 10px; }
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

# --- 3. INTERFACE ---
st.title("🐑 OviStat IA v2.1")
tabs = st.tabs(["📥 Saisie", "🔍 Historique & Modif", "📊 Analyse", "🩺 Santé", "ℹ️ À Propos"])

# --- ONGLET 1 : SAISIE ---
with tabs[0]:
    if 'step' not in st.session_state: st.session_state.step = 1
    with st.form("form_global"):
                st.subheader("📍 Localisation de l'étude")
        col_w, col_c = st.columns(2)
        
        # 1. Sélection de la Wilaya avec une clé unique pour forcer le rafraîchissement
        wilaya_sel = col_w.selectbox("Sélectionnez la Wilaya", list_wilayas, key="wilaya_choice")
        
        # 2. On extrait le nom (ex: "Djelfa")
        w_clean = wilaya_sel.split("-")[-1].strip()
        
        # 3. Filtrage STRICT des communes
        # On s'assure que le DataFrame n'est pas vide et on filtre
        if not df_communes.empty:
            # On compare le nom de la wilaya en minuscules pour éviter les erreurs
            mask = df_communes['wilaya_name'].str.strip().str.lower() == w_clean.lower()
            communes_possibles = df_communes[mask]['commune_name'].unique().tolist()
        else:
            communes_possibles = []

        # 4. Affichage de la liste correspondante
        if communes_possibles:
            # On ajoute un 'key' dynamique basé sur la wilaya pour forcer la mise à jour de la liste
            commune_sel = col_c.selectbox(
                f"Communes de {w_clean}", 
                sorted(communes_possibles), 
                key=f"commune_list_{w_clean}" 
            )
        else:
            # Si le CSV est vide ou mal lu, on propose la saisie manuelle
            commune_sel = col_c.text_input("Commune (Saisie manuelle)", key="commune_manual")
            st.warning(f"⚠️ Aucune commune trouvée dans le fichier pour {w_clean}")

        c1, c2, c3 = st.columns(3)
        id_in = c1.text_input("ID Animal")
        age_in = c2.number_input("Âge (mois)", value=12)
        race_in = c3.selectbox("Race", ["Ouled Djellal", "Rembi", "Hamra", "Taadmit"])

        st.write(f"📸 **Scan Étape {st.session_state.step}/3**")
        photo = st.camera_input("Capturer")
        if photo and st.session_state.step < 3:
            if st.form_submit_button(f"➡️ Continuer vers étape {st.session_state.step+1}"):
                st.session_state.step += 1; st.rerun()

        st.divider()
        st.subheader("📏 Mensurations (24 Paramètres)")
        
        with st.expander("1️⃣ Dimensions Corporelles (HG, HS, LB, LQ, LT, LC, LH)", expanded=True):
            g1, g2, g3, g4 = st.columns(4)
            poids = g1.number_input("Poids (kg)", value=0.0)
            hg = g2.number_input("H. Garrot (HG)", value=0.0); hs = g3.number_input("H. Sacrum (HS)", value=0.0); lb = g4.number_input("Long. Totale (LB)", value=0.0)
            lq = g1.number_input("Long. Queue (LQ)", value=0.0); lt_t = g2.number_input("Long. Tronc (LT)", value=0.0); lc_c = g3.number_input("Long. Cou (LC)", value=0.0); lh = g4.number_input("Long. Bassin (LH)", value=0.0)

        with st.expander("2️⃣ Poitrine & Largeurs (LI, LP, PP, TP)"):
            g5, g6, g7, g8 = st.columns(4)
            li = g5.number_input("Larg. Ischions (LI)", value=0.0); lp = g6.number_input("Larg. Poitrine (LP)", value=0.0)
            pp = g7.number_input("Prof. Poitrine (PP)", value=0.0); tp = g8.number_input("Tour Poitrine (TP)", value=0.0)

        with st.expander("3️⃣ Tête & Oreilles (Lc, LT, Lt, LO, Lo, TC)"):
            g9, g10, g11 = st.columns(3)
            lc_cornes = g9.number_input("Cornes (Lc)", value=0.0); lt_te = g10.number_input("Long. Tête (LTête)", value=0.0); lt_la = g11.number_input("Larg. Tête (LtTete)", value=0.0)
            lo_lo = g9.number_input("Long. Oreille (LO)", value=0.0); lo_la = g10.number_input("Larg. Oreille (Lo)", value=0.0); tc = g11.number_input("Tour Canon (TC)", value=0.0)

        with st.expander("4️⃣ Reproduction & Laine (LY, TS, PS, LG, LL)"):
            g12, g13, g14 = st.columns(3)
            ly = g12.number_input("Trayons (LY)", value=0.0); ts = g13.number_input("T. Scrotal (TS)", value=0.0); ps = g14.number_input("P. Scrotale (PS)", value=0.0)
            lg = g12.number_input("Gigot (LG)", value=0.0); ll = g13.number_input("Laine (LL)", value=0.0)

        if st.form_submit_button("💾 ENREGISTRER TOUT"):
            id_f = id_in if id_in else f"TEMP-{datetime.now().strftime('%H%M%S')}"
            row = [date.today(), id_f, race_in, age_in, poids, hg, hs, lb, lq, lt_t, lc_c, lh, li, lp, pp, tp, lc_cornes, lt_te, lt_la, lo_lo, lo_la, tc, ly, ts, ps, lg, ll, wilaya_sel, commune_sel, round(poids*0.035,2)]
            pd.DataFrame([row], columns=COLONNES).to_csv(DB_FILE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
            st.session_state.step = 1; st.success("✅ Fiche sauvegardée !"); st.rerun()

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

# --- ONGLET 4 : SANTÉ ---
with tabs[3]:
    st.header("🩺 Suivi Sanitaire")
    with st.form("f_sante"):
        ids = st.selectbox("Animal", data["ID"].unique()) if not data.empty else "N/A"
        acte = st.text_input("Soin effectué")
        if st.form_submit_button("💉 Enregistrer"):
            st.success("Soin noté !")

# --- ONGLET 5 : À PROPOS ---
with tabs[4]:
    st.header("ℹ️ À Propos")
    st.write("**Auteur :** MERABIA KAWTHER | **Version :** 1.9.0")
    if os.path.exists("manuel_ovistat.pdf"):
        with open("manuel_ovistat.pdf", "rb") as f:
            st.download_button("📖 Ouvrir le Manuel PDF", f.read(), "Manuel_OviStat.pdf")
    if st.button("🔄 Reset Scan"): st.session_state.step = 1; st.rerun()

# --- MAINTENANCE ---
st.divider()
with st.expander("⚙️ MAINTENANCE SYSTÈME"):
    st.warning("⚠️ Action irréversible")
    if st.button("🗑️ VIDER TOUTES LES DONNÉES"):
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        if os.path.exists(DB_SANTE): os.remove(DB_SANTE)
        st.success("Serveur nettoyé !"); st.rerun()
    st.info(f"📍 Chemin serveur : {os.getcwd()}")
