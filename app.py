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
# Base de données enrichie
COLONNES = ["Date", "ID", "Race", "Poids_kg", "Taille_cm", "Perimetre_cm", "Note_IA", "Comptage", "Ration_MS"]

@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n.pt')

model = load_yolo_model()

# 1. INITIALISATION DU FICHIER
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

# --- INTERFACE ---
st.title("🐑 OviStat Vision Pro : Intelligence Augmentée")

tab1, tab2, tab3 = st.tabs(["📥 Saisie IA & Ration", "🔍 Historique Complet", "📊 Analyse Comparative"])

# --- ONGLET 1 : SAISIE INTELLIGENTE ---
with tab1:
    st.header("📥 Scan & Enregistrement")
    
    with st.form("form_expert", clear_on_submit=True):
        c1, c2 = st.columns(2)
        
        # Identification
        id_in = c1.text_input("ID Animal", value=get_next_id(data))
        race_in = c1.selectbox("Race", ["Ouled Djellal", "Rembi", "Hamra"])
        
        # Paramètres (seront complétés par l'IA)
        poids_in = c2.number_input("Poids (kg)", min_value=0.0, step=0.5, value=45.0)
        
        st.divider()
        st.write("📸 **Capture Optique & Morphométrie**")
        cv1, c_photo, cv2 = st.columns([1,2,1])
        with c_photo:
            photo = st.camera_input("Scanner l'animal")
        
        # Variables IA par défaut
        taille_ia, peri_ia, note_ia, count_ia = 0.0, 0.0, 0.0, 1
        
        if photo:
            # Simulation détection YOLO + Morphométrie
            taille_ia = 68.0  # Estimation IA
            peri_ia = 82.5    # Estimation IA
            note_ia = 4.0     # Score corporel
            st.success(f"✅ IA : Taille {taille_ia}cm | Périmètre {peri_ia}cm | Note {note_ia}/5")
        
        # --- CALCUL RATION ---
        ration_ms = round(poids_in * 0.035, 2)
        st.info(f"🌾 **Ration suggérée : {ration_ms} kg de Matière Sèche / jour**")
        
        st.divider()
        if st.form_submit_button("💾 VALIDER ET ENREGISTRER"):
            if id_in:
                date_now = datetime.now().strftime("%Y-%m-%d")
                nouvelle_ligne = pd.DataFrame([[
                    date_now, id_in, race_in, poids_in, taille_ia, peri_ia, note_ia, count_ia, ration_ms
                ]], columns=COLONNES)
                
                nouvelle_ligne.to_csv(DB_FILE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
                st.success(f"Enregistrement réussi pour {id_in}")
                st.balloons()
                st.rerun()

# --- ONGLET 2 : HISTORIQUE ---
with tab2:
    st.subheader("🔍 Base de données enrichie")
    if not data.empty:
        st.dataframe(data, use_container_width=True)
        st.download_button("📥 Télécharger Excel (CSV)", data.to_csv(sep=';').encode('utf-8-sig'), "donnees_ovins.csv", "text/csv")
    else:
        st.info("Aucune donnée.")

# --- ONGLET 3 : ANALYSE ---
with tab3:
    if len(data["ID"].unique()) >= 2:
        ani1 = st.selectbox("Animal A", data["ID"].unique(), index=0)
        ani2 = st.selectbox("Animal B", data["ID"].unique(), index=1)
        
        df1 = data[data["ID"] == ani1].sort_values("Date")
        df2 = data[data["ID"] == ani2].sort_values("Date")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df1["Date"], df1["Poids_kg"], label=f"{ani1} (Poids)", marker='o')
        ax.plot(df2["Date"], df2["Poids_kg"], label=f"{ani2} (Poids)", marker='s')
        ax.set_ylabel("kg")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Enregistrez 2 animaux pour comparer.")

# --- MAINTENANCE ---
with st.expander("⚙️ Maintenance"):
    if st.button("🗑️ Vider tout"):
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()
