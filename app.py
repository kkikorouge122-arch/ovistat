import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, date
from ultralytics import YOLO
from PIL import Image
from fpdf import FPDF

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="OviStat Vision Pro v1.6", page_icon="🐑", layout="wide")

st.markdown("""
    <style>
    .block-container { padding-top: 1rem; }
    div[data-testid="stCameraInput"] video { width: 100% !important; object-fit: cover !important; border-radius: 15px; border: 4px solid #1f77b4; }
    div[data-testid="stCameraInput"] button { height: 60px !important; background-color: #1f77b4 !important; color: white !important; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

DB_FILE = "data_ovinstat_V11.csv"
DB_SANTE = "data_sante_ovins.csv"

# LISTE COMPLÈTE DES 28 COLONNES
COLONNES = [
    "Date", "ID", "Race", "Age", "Poids", 
    "HG", "HS", "LB", "LQ", "LT_tronc", "LC_cou", "LH", "LI", "LP", "PP", "TP",
    "Lc_cornes", "LT_tete", "Lt_tete", "LO", "Lo_oreille", "TC", "LY", "TS", "PS", "LG", "LL", "Ration"
]
COL_SANTE = ["Date", "ID", "Type", "Produit", "Veterinaire", "Prochain_RDV"]

# --- 2. FONCTIONS ---
@st.cache_resource
def load_yolo_model(): return YOLO('yolov8n.pt')

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

# --- 3. INTERFACE ---
st.title("🐑 OviStat IA : Expert Morphométrie")

tabs = st.tabs(["📥 Saisie", "🔍 Historique", "📊 Analyse", "🩺 Santé", "ℹ️ À Propos"])

# --- ONGLET 1 : SAISIE SÉQUENTIELLE ---
with tabs[0]:
    if 'step' not in st.session_state: st.session_state.step = 1
    
    with st.form("form_global", clear_on_submit=False):
        st.subheader("🆔 Identité")
        c1, c2, c3 = st.columns(3)
        id_in = c1.text_input("ID Animal", value=f"OVIN-{len(data)+1}")
        age_in = c2.number_input("Âge (mois)", value=12)
        race_in = c3.selectbox("Race", ["Ouled Djellal", "Rembi", "Hamra", "Taadmit"])

        st.divider()
        st.write(f"📸 **Étape {st.session_state.step}/3 :** " + ["Profil", "Dessus", "Tête"][st.session_state.step-1])
        photo = st.camera_input("Capturer l'angle actuel")
        
        ia_hg, ia_tp = 0.0, 0.0
        if photo:
            if st.session_state.step < 3:
                if st.form_submit_button(f"➡️ Valider et passer à l'étape {st.session_state.step + 1}"):
                    st.session_state.step += 1
                    st.rerun()
            else:
                st.success("✅ Toutes les photos sont prêtes !")
                ia_hg, ia_tp = 68.5, 84.0 # Simulation IA

        st.divider()
        st.subheader("📏 Mensurations (24 Paramètres)")
        
        with st.expander("🏗️ Dimensions du Corps", expanded=True):
            g1, g2, g3, g4 = st.columns(4)
            poids = g1.number_input("Poids (kg)", value=45.0)
            hg = g2.number_input("Hauteur Garrot (HG)", value=ia_hg)
            hs = g3.number_input("Hauteur Sacrum (HS)", value=0.0)
            lb = g4.number_input("Long. Corps (LB)", value=0.0)
            
            g5, g6, g7, g8 = st.columns(4)
            lq = g5.number_input("Long. Queue (LQ)", value=0.0)
            lt_t = g6.number_input("Long. Tronc (LT)", value=0.0)
            lc_c = g7.number_input("Long. Cou (LC)", value=0.0)
            lh = g8.number_input("Long. Bassin (LH)", value=0.0)

        with st.expander("📐 Largeurs & Poitrine"):
            g9, g10, g11, g12 = st.columns(4)
            li = g9.number_input("Larg. Ischions (LI)", value=0.0)
            lp = g10.number_input("Larg. Poitrine (LP)", value=0.0)
            pp = g11.number_input("Prof. Poitrine (PP)", value=0.0)
            tp = g12.number_input("Tour Poitrine (TP)", value=ia_tp)

        with st.expander("👤 Tête & Oreilles"):
            g13, g14, g15, g16, g17, g18 = st.columns(6)
            lc_c = g13.number_input("Long. Cornes (Lc)", value=0.0)
            lt_te = g14.number_input("Long. Tête (LT)", value=0.0)
            lt_la = g15.number_input("Larg. Tête (Lt)", value=0.0)
            lo_lo = g16.number_input("Long. Oreilles (LO)", value=0.0)
            lo_la = g17.number_input("Larg. Oreilles (Lo)", value=0.0)
            tc = g18.number_input("Tour Canon (TC)", value=0.0)

        with st.expander("🧬 Reproduction & Laine"):
            g19, g20, g21, g22, g23 = st.columns(5)
            ly = g19.number_input("Long. Trayons (LY)", value=0.0)
            ts = g20.number_input("Tour Scrotal (TS)", value=0.0)
            ps = g21.number_input("Prof. Scrotale (PS)", value=0.0)
            lg = g22.number_input("Long. Gigot (LG)", value=0.0)
            ll = g23.number_input("Long. Laine (LL)", value=0.0)

        ration = round(poids * 0.035, 2)
        
        if st.form_submit_button("💾 ENREGISTRER LA FICHE COMPLÈTE"):
            row = [date.today(), id_in, race_in, age_in, poids, hg, hs, lb, lq, lt_t, lc_c, lh, li, lp, pp, tp, lc_c, lt_te, lt_la, lo_lo, lo_la, tc, ly, ts, ps, lg, ll, ration]
            pd.DataFrame([row], columns=COLONNES).to_csv(DB_FILE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
            st.session_state.step = 1
            st.success("✅ Enregistré !"); st.rerun()

# --- ONGLET 2 : HISTORIQUE & MODIFICATION ---
with tabs[1]:
    st.subheader("📋 Gestion de la base de données")
    
    if not data.empty:
        # --- A. AFFICHAGE ---
        st.dataframe(data, use_container_width=True)
        
        st.divider()
        
        # --- B. MODIFICATION / SUPPRESSION ---
        st.subheader("🛠️ Modifier ou Supprimer une fiche")
        
        # Sélection de l'animal à traiter
        id_a_modifier = st.selectbox("Choisir l'ID de l'animal à gérer", data["ID"].unique(), key="select_modif")
        
        col_btn1, col_btn2 = st.columns(2)
        
        # 1. BOUTON SUPPRIMER
        if col_btn1.button(f"❌ Supprimer définitivement {id_a_modifier}", type="secondary"):
            df_new = data[data["ID"] != id_a_modifier]
            df_new.to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')
            st.success(f"Animal {id_a_modifier} supprimé.")
            st.rerun()

        # 2. ZONE DE MODIFICATION
        if col_btn2.checkbox(f"📝 Modifier les données de {id_a_modifier}"):
            # On récupère les données actuelles
            ligne_actuelle = data[data["ID"] == id_a_modifier].iloc[-1]
            
            with st.form("form_edit"):
                st.write(f"Modification de : {id_a_modifier}")
                new_poids = st.number_input("Nouveau Poids (kg)", value=float(ligne_actuelle["Poids"]))
                new_hg = st.number_input("Nouvelle Hauteur (HG)", value=float(ligne_actuelle["HG"]))
                
                if st.form_submit_button("💾 Enregistrer les modifications"):
                    # On supprime l'ancienne ligne et on ajoute la nouvelle
                    df_clean = data[data["ID"] != id_a_modifier]
                    
                    # On crée la nouvelle ligne (copie de l'ancienne avec les changements)
                    nouvelle_ligne = ligne_actuelle.copy()
                    nouvelle_ligne["Poids"] = new_poids
                    nouvelle_ligne["HG"] = new_hg
                    nouvelle_ligne["Date"] = date.today() # On met à jour la date de modif
                    
                    # Sauvegarde
                    df_final = pd.concat([df_clean, pd.DataFrame([nouvelle_ligne])], ignore_index=True)
                    df_final.to_csv(DB_FILE, index=False, sep=';', encoding='utf-8-sig')
                    
                    st.success("Modification enregistrée !")
                    st.rerun()

        st.divider()
        st.download_button("📥 Télécharger l'archive Excel", data.to_csv(sep=';', index=False).encode('utf-8-sig'), "OviStat_Data.csv")
    else:
        st.info("La base de données est vide.")


# --- ONGLET 3 : ANALYSE & CERTIFICAT ---
with tabs[2]:
    st.header("🧠 Expertise & Benchmark Zootechnique")
    
    if not data.empty:
        # 1. Sélection de l'animal
        target = st.selectbox("Sélectionner l'animal pour l'audit complet", data["ID"].unique())
        # On récupère la ligne correspondante
        anim = data[data["ID"] == target].iloc[-1] 
        
        # 2. Calculs des Indices (Sécurisés)
        try:
            # On récupère les valeurs pour les calculs
            poids_val = anim['Poids'] if 'Poids' in anim else 0
            lb_val = anim['LB'] if 'LB' in anim else 0
            tp_val = anim['TP'] if 'TP' in anim else 0
            hg_val = anim['HG'] if 'HG' in anim else 0
            
            # Formules mathématiques
            ic = poids_val / lb_val if lb_val > 0 else 0
            ir = (tp_val**2) / hg_val if hg_val > 0 else 0
            ip = hg_val / lb_val if lb_val > 0 else 0
            
            # 3. Affichage des Scores
            st.subheader("🥇 Scores Individuels")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Rendement Viande", f"{ic:.2f}")
                st.caption("Indice de Compacité")
            with c2:
                st.metric("Robustesse", f"{ir:.2f}")
                st.caption("Indice d'Anamorphose")
            with c3:
                format_anim = "Longiligne" if ip < 0.95 else "Médioligne"
                st.metric("Format", format_anim)
                st.caption(f"Ratio HG/LB: {ip:.2f}")

            # 4. Conseils de l'IA
            st.divider()
            if ic > 0.5:
                st.success("✅ **Conseil IA :** Excellente conformation bouchère. Potentiel élevé.")
            else:
                st.warning("⚠️ **Conseil IA :** Animal un peu frêle. Surveiller l'alimentation.")

            # 5. Génération du Certificat PDF
            st.subheader("📄 Certificat Officiel")
            if st.button(f"Générer le certificat pour {target}"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(200, 10, "CERTIFICAT DE QUALITÉ OVISTAT IA", ln=True, align='C')
                pdf.ln(10)
                pdf.set_font("Arial", '', 12)
                pdf.cell(0, 10, f"ID Animal : {target}", ln=True)
                pdf.cell(0, 10, f"Race : {anim['Race']}", ln=True)
                pdf.cell(0, 10, f"Indice Compacité : {ic:.2f}", ln=True)
                pdf.cell(0, 10, f"Indice Robustesse : {ir:.2f}", ln=True)
                
                pdf_bytes = pdf.output(dest="S").encode("latin-1")
                st.download_button(f"📥 Télécharger PDF {target}", data=pdf_bytes, file_name=f"Certificat_{target}.pdf")

        except Exception as e:
            st.error(f"Erreur lors du calcul des indices : {e}")
            
    else:
        st.info("📊 Les analyses apparaîtront après le premier enregistrement.")


# --- ONGLET 4 : SANTÉ ---
with tabs[3]:
    st.subheader("🩺 Suivi Sanitaire")
    with st.form("form_sante"):
        id_s = st.selectbox("Animal", data["ID"].unique()) if not data.empty else "N/A"
        acte = st.text_input("Vaccin / Soin")
        if st.form_submit_button("💉 Noter le soin"):
            row_s = [date.today(), id_s, "Soin", acte, "Dr. Ahmed", date.today()]
            pd.DataFrame([row_s], columns=COL_SANTE).to_csv(DB_SANTE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
            st.success("Soin enregistré !")


# --- ONGLET 5 : À PROPOS ---
with tabs[4]:
    st.header("ℹ️ À Propos")
    st.write("**Auteur :** MERABIA KAWTHER | **Version :** 1.6.0")
    if os.path.exists("manuel_ovistat.pdf"):
        with open("manuel_ovistat.pdf", "rb") as f:
            st.download_button("📖 Ouvrir le Manuel PDF", f.read(), "Manuel_OviStat.pdf")
    if st.button("🔄 Réinitialiser le cycle de scan"):
        st.session_state.step = 1
        st.rerun()

# --- MAINTENANCE ---
st.divider()
with st.expander("⚙️ Maintenance"):
    if st.button("🗑️ Vider la base"):
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

