import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, date
from ultralytics import YOLO
from PIL import Image

# --- CONFIGURATION ---
st.set_page_config(page_title="OviStat Vision Pro", layout="wide")
DB_FILE = "data_ovins_final.csv"

# MISE À JOUR DES COLONNES (20 colonnes au total avec Date, ID, Race)
COLONNES = [
    "Date", "ID", "Race", "HG", "HS", "LB", "LQ", "LT", "LC", "LH", 
    "LI", "LP", "TP", "Lt_min", "LO", "Lo_min", "TC", "LY", "TS", "LG"
]

@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n.pt')

model = load_yolo_model()

if not os.path.exists(DB_FILE):
    df_init = pd.DataFrame(columns=COLONNES)
    df_init.to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')

def load_data():
    if os.path.exists(DB_FILE):
        return pd.read_csv(DB_FILE, sep=';', encoding='utf-8-sig')
    return pd.DataFrame(columns=COLONNES)

data = load_data()

def get_next_id(df):
    if df.empty: return "OVIN-1"
    return f"OVIN-{len(df['ID'].unique()) + 1}"

st.title("🐑 OviStat Vision Pro : Intelligence Augmentée")

st.markdown("""
    <style>
    div[data-testid="stCameraInput"] video {
        width: 100% !important;
        height: auto !important;
        border-radius: 15px;
        border: 2px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📥 Saisie & Morphométrie", "🔍 Historique", "📊 Analyse"])

with tab1:
    st.header("📥 Enregistrement des Mensurations")
    
    with st.form("form_expert", clear_on_submit=True):
        c1, c2 = st.columns(2)
        id_in = c1.text_input("🆔 ID Animal", value=get_next_id(data))
        race_in = c2.selectbox("🧬 Race", ["Ouled Djellal", "Hamra", "Taadmit"])
        
        st.divider()
        st.write("📸 **Capture Optique (Analyse Morphométrique)**")
        cv1, c_photo, cv2 = st.columns([0.1, 0.8, 0.1])
        with c_photo:
            photo = st.camera_input("Scanner l'animal")
        
        if photo:
            st.success("✅ Image capturée. Veuillez compléter les mesures ci-dessous.")

        st.divider()
        st.write("📏 **Mensurations Détaillées (cm)**")
        
        # Organisation en grille de 3 pour la lisibilité
        g1, g2, g3 = st.columns(3)
        hg = g1.number_input("Hauteur_G (HG)", min_value=0.0)
        hs = g2.number_input("Hauteur_S (HS)", min_value=0.0)
        lb = g3.number_input("Longueur_B (LB)", min_value=0.0)
        
        g4, g5, g6 = st.columns(3)
        lq = g4.number_input("Longueur_Q (LQ)", min_value=0.0)
        lt = g5.number_input("Longueur_T (LT)", min_value=0.0)
        lc = g6.number_input("Longueur_C (LC)", min_value=0.0)
        
        g7, g8, g9 = st.columns(3)
        lh = g7.number_input("Longueur_H (LH)", min_value=0.0)
        li = g8.number_input("Longueur_I (LI)", min_value=0.0)
        lp = g9.number_input("Longueur_P (LP)", min_value=0.0)
        
        g10, g11, g12 = st.columns(3)
        tp = g10.number_input("Taille_P (TP)", min_value=0.0)
        lt_m = g11.number_input("Longueur_t (Lt)", min_value=0.0)
        lo = g12.number_input("Longueur_O (LO)", min_value=0.0)
        
        g13, g14, g15 = st.columns(3)
        lo_m = g13.number_input("Longueur_o (Lo)", min_value=0.0)
        tc = g14.number_input("Taille_C (TC)", min_value=0.0)
        ly = g15.number_input("Longueur_Y (LY)", min_value=0.0)
        
        g16, g17 = st.columns(2)
        ts = g16.number_input("Taille_S (TS)", min_value=0.0)
        lg = g17.number_input("Longueur_G (LG)", min_value=0.0)
        
        st.divider()
        if st.form_submit_button("💾 ENREGISTRER LA FICHE"):
            if id_in:
                date_now = datetime.now().strftime("%Y-%m-%d")
                nouvelle_ligne = pd.DataFrame([[
                    date_now, id_in, race_in, hg, hs, lb, lq, lt, lc, lh, 
                    li, lp, tp, lt_m, lo, lo_m, tc, ly, ts, lg
                ]], columns=COLONNES)
                
                nouvelle_ligne.to_csv(DB_FILE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
                st.success(f"Enregistrement réussi pour {id_in}")
                st.balloons()
                st.rerun()

with tab2:
    st.subheader("🔍 Base de données")
    if not data.empty:
        st.dataframe(data, use_container_width=True)
        csv_data = data.to_csv(sep=';', index=False).encode('utf-8-sig')
        st.download_button("📥 Télécharger CSV", csv_data, "donnees_ovins.csv", "text/csv")

with tab3:
    if not data.empty and len(data["ID"].unique()) >= 2:
        ani1 = st.selectbox("Animal A", data["ID"].unique(), index=0)
        ani2 = st.selectbox("Animal B", data["ID"].unique(), index=1)
        
        df1 = data[data["ID"] == ani1]
        df2 = data[data["ID"] == ani2]
        
        # On vérifie si la colonne HG existe bien avant de dessiner
        if "HG" in data.columns:
            fig, ax = plt.subplots()
            # On prend la dernière mesure pour chaque animal
            val1 = df1["HG"].iloc[-1] if not df1.empty else 0
            val2 = df2["HG"].iloc[-1] if not df2.empty else 0
            
            ax.bar([ani1, ani2], [val1, val2], color=['#1f77b4', '#ff7f0e'])
            ax.set_ylabel("Hauteur Garrot (HG) en cm")
            ax.set_title("Comparaison des Hauteurs")
            st.pyplot(fig)
        else:
            st.error("⚠️ Les nouvelles colonnes (HG, HS...) ne sont pas encore présentes dans votre fichier CSV. Allez dans 'Maintenance' et videz la base.")
    else:
        st.warning("Veuillez enregistrer au moins 2 animaux avec les nouvelles mesures.")


with st.expander("⚙️ Maintenance"):
    if st.button("🗑️ Vider la base"):
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

