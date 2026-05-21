import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, date
from ultralytics import YOLO
from PIL import Image
from fpdf import FPDF

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="OviStat Vision Pro v1.8", page_icon="🐑", layout="wide")

st.markdown("""
    <style>
    .block-container { padding-top: 1rem; }
    div[data-testid="stCameraInput"] video { width: 100% !important; object-fit: cover !important; border-radius: 15px; border: 3px solid #1f77b4; }
    div[data-testid="stCameraInput"] button { height: 60px !important; background-color: #1f77b4 !important; color: white !important; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

DB_FILE = "data_ovinstat_V13.csv"
DB_SANTE = "data_sante_ovins.csv"

COLONNES = [
    "Date", "ID", "Race", "Age", "Poids", 
    "HG", "HS", "LB", "LQ", "LT_tronc", "LC_cou", "LH", "LI", "LP", "PP", "TP",
    "Lc_cornes", "LT_tete", "Lt_tete", "LO", "Lo_oreille", "TC", "LY", "TS", "PS", "LG", "LL", 
    "Wilaya", "Commune", "Ration"
]
COL_SANTE = ["Date", "ID", "Type", "Produit", "Veterinaire", "Prochain_RDV"]

# --- 2. FONCTIONS ---
@st.cache_resource
def load_yolo_model(): return YOLO('yolov8n.pt')

@st.cache_data
def get_algeria_geo():
    wilayas = ["01-Adrar", "02-Chlef", "03-Laghouat", "04-Oum El Bouaghi", "05-Batna", "06-Béjaïa", "07-Biskra", "08-Béchar", "09-Blida", "10-Bouira", "11-Tamanrasset", "12-Tébessa", "13-Tlemcen", "14-Tiaret", "15-Tizi Ouzou", "16-Alger", "17-Djelfa", "18-Jijel", "19-Sétif", "20-Saïda", "21-Skikda", "22-Sidi Bel Abbès", "23-Annaba", "24-Guelma", "25-Constantine", "26-Médéa", "27-Mostaganem", "28-M'Sila", "29-Mascara", "30-Ouargla", "31-Oran", "32-El Bayadh", "33-Illizi", "34-Bordj Bou Arreridj", "35-Boumerès", "36-El Tarf", "37-Tindouf", "38-Tissemsilt", "39-El Oued", "40-Khenchela", "41-Souk Ahras", "42-Tipaza", "43-Mila", "44-Aïn Defla", "45-Naâma", "46-Aïn Témouchent", "47-Ghardaïa", "48-Relizane", "49-El M'Ghair", "50-El Meniaa", "51-Ouled Djellal", "52-Bordj Baji Mokhtar", "53-Béni Abbès", "54-Timimoun", "55-Touggourt", "56-Djanet", "57-In Salah", "58-In Guezzam"]
    if os.path.exists("algeria_geo.csv"):
        try: df_geo = pd.read_csv("algeria_geo.csv", sep=";", encoding='utf-8')
        except: df_geo = pd.DataFrame(columns=["wilaya_name", "commune_name"])
    else: df_geo = pd.DataFrame({"wilaya_name": ["Djelfa"], "commune_name": ["Djelfa"]})
    return wilayas, df_geo

def load_data(file, cols):
    if os.path.exists(file):
        try:
            df = pd.read_csv(file, sep=';', encoding='utf-8-sig', on_bad_lines='skip')
            return df if not df.empty else pd.DataFrame(columns=cols)
        except: return pd.DataFrame(columns=cols)
    return pd.DataFrame(columns=cols)

for f, c in zip([DB_FILE, DB_SANTE], [COLONNES, COL_SANTE]):
    if not os.path.exists(f) or os.path.getsize(f) == 0:
        pd.DataFrame(columns=c).to_csv(f, index=False, sep=';', encoding='utf-8-sig')

model = load_yolo_model()
data = load_data(DB_FILE, COLONNES)
list_wilayas, df_communes = get_algeria_geo()

# --- 3. INTERFACE ---
st.title("🐑 OviStat IA : Expert Algérie")
tabs = st.tabs(["📥 Saisie", "🔍 Historique", "📊 Analyse", "🩺 Santé", "ℹ️ À Propos"])

# --- ONGLET 1 : SAISIE ---
with tabs[0]:
    if 'step' not in st.session_state: st.session_state.step = 1
    with st.form("form_global", clear_on_submit=False):
        st.subheader("📍 Localisation")
        cw, cc = st.columns(2)
        wilaya_sel = cw.selectbox("Wilaya", list_wilayas)
        w_clean = wilaya_sel.split("-")[-1].strip()
        mask = df_communes['wilaya_name'].str.strip().str.lower() == w_clean.lower()
        communes_possibles = df_communes[mask]['commune_name'].unique().tolist()
        commune_sel = cc.selectbox("Commune", sorted(communes_possibles) if communes_possibles else ["Saisir..."])

        st.divider()
        st.subheader("🆔 Identité")
        c1, c2, c3 = st.columns(3)
        id_in = c1.text_input("ID / Boucle (Vide = Auto-ID)")
        age_in = c2.number_input("Âge (mois)", value=12)
        race_in = c3.selectbox("Race", ["Ouled Djellal", "Rembi", "Hamra", "Taadmit"])

        st.write(f"📸 **Étape {st.session_state.step}/3 :** " + ["Profil", "Dessus", "Tête"][st.session_state.step-1])
        photo = st.camera_input("Capturer")
        
        ia_hg, ia_tp = 0.0, 0.0
        if photo:
            if st.session_state.step < 3:
                if st.form_submit_button(f"➡️ Valider Étape {st.session_state.step}"):
                    st.session_state.step += 1; st.rerun()
            else:
                st.success("✅ Photos prêtes !"); ia_hg, ia_tp = 68.5, 84.0

        st.divider()
        with st.expander("📏 Mensurations (24 Paramètres)", expanded=True):
            e1, e2, e3, e4 = st.columns(4)
            poids = e1.number_input("Poids (kg)", value=45.0)
            hg = e2.number_input("HG (Garrot)", value=ia_hg)
            hs = e3.number_input("HS (Sacrum)", value=0.0)
            lb = e4.number_input("LB (Corps)", value=0.0)
            lq = e1.number_input("LQ (Queue)", value=0.0)
            lt_t = e2.number_input("LT (Tronc)", value=0.0)
            lc_c = e3.number_input("LC (Cou)", value=0.0)
            lh = e4.number_input("LH (Bassin)", value=0.0)
            li = e1.number_input("LI (Ischions)", value=0.0)
            lp = e2.number_input("LP (Poitrine)", value=0.0)
            pp = e3.number_input("PP (Prof. Poitrine)", value=0.0)
            tp = e4.number_input("TP (Tour Poitrine)", value=ia_tp)
            lc_cornes = e1.number_input("Lc (Cornes)", value=0.0)
            lt_tete = e2.number_input("LTête", value=0.0)
            lt_la = e3.number_input("LtTete", value=0.0)
            lo_lo = e4.number_input("LO (Oreille)", value=0.0)
            lo_la = e1.number_input("Lo (Oreille)", value=0.0)
            tc = e2.number_input("TC (Canon)", value=0.0)
            ly = e3.number_input("LY (Trayon)", value=0.0)
            ts = e4.number_input("TS (Scrotal)", value=0.0)
            ps = e1.number_input("PS (Scrotale)", value=0.0)
            lg = e2.number_input("LG (Gigot)", value=0.0)
            ll = e3.number_input("LL (Laine)", value=0.0)

        if st.form_submit_button("💾 ENREGISTRER"):
            id_final = id_in if id_in else f"TEMP-{datetime.now().strftime('%d%H%M%S')}"
            row = [date.today(), id_final, race_in, age_in, poids, hg, hs, lb, lq, lt_t, lc_c, lh, li, lp, pp, tp, lc_cornes, lt_tete, lt_la, lo_lo, lo_la, tc, ly, ts, ps, lg, ll, wilaya_sel, commune_sel, round(poids*0.035,2)]
            pd.DataFrame([row], columns=COLONNES).to_csv(DB_FILE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
            st.session_state.step = 1; st.success(f"Enregistré : {id_final}"); st.rerun()

# --- ONGLET 2 : HISTORIQUE & MODIFICATIONS ---
with tabs[1]:
    st.subheader("📋 Gestion de la base de données")
    if not data.empty:
        df_v = data.copy()
        df_v.insert(0, 'Statut', df_v['ID'].apply(lambda x: "⭐ Officiel" if "TEMP-" not in str(x) else "🕒 Temporaire"))
        st.dataframe(df_v, use_container_width=True)
        
        st.divider()
        st.subheader("🛠️ Modifier ou Supprimer")
        id_target = st.selectbox("Choisir l'ID de l'animal", data["ID"].unique())
        
        c_m1, c_m2 = st.columns(2)
        if c_m1.button(f"❌ Supprimer {id_target}", type="primary"):
            data[data["ID"] != id_target].to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')
            st.rerun()
            
        if c_m2.checkbox(f"📝 Modifier {id_target}"):
            old = data[data["ID"] == id_target].iloc[-1]
            with st.form("edit_form"):
                new_p = st.number_input("Nouveau Poids", value=float(old["Poids"]))
                new_hg = st.number_input("Nouveau HG", value=float(old["HG"]))
                if st.form_submit_button("💾 Sauvegarder"):
                    df_clean = data[data["ID"] != id_target]
                    new_line = old.copy()
                    new_line["Poids"], new_line["HG"] = new_p, new_hg
                    pd.concat([df_clean, pd.DataFrame([new_line])]).to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')
                    st.rerun()
        
        st.download_button("📥 Export Excel", data.to_csv(sep=';', index=False).encode('utf-8-sig'), "base.csv")

# --- ONGLET 3 : ANALYSE ---
with tabs[2]:
    if not data.empty:
        target = st.selectbox("Animal", data["ID"].unique(), key="ana_id")
        anim = data[data["ID"] == target].iloc[-1]
        try:
            ic = anim['Poids'] / anim['LB'] if anim['LB'] > 0 else 0
            st.metric("Indice Viande", f"{ic:.2f}")
            if st.button("📄 PDF"):
                pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", 'B', 16)
                pdf.cell(200, 10, f"ID: {target}", ln=True)
                st.download_button("📥 Télécharger", pdf.output(dest="S").encode("latin-1"), f"{target}.pdf")
        except: st.error("Données insuffisantes")

# --- MAINTENANCE ---
st.divider()
with st.expander("⚙️ Maintenance"):
    if st.button("🗑️ Vider tout"):
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()
