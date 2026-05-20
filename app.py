import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, date
from ultralytics import YOLO
from PIL import Image
from fpdf import FPDF

# --- 1. CONFIGURATION & DESIGN ---
st.set_page_config(page_title="OviStat Vision Pro", layout="wide")

# CSS pour une interface mobile élégante
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; border-radius: 10px; padding: 10px; }
    div[data-testid="stCameraInput"] video { border: 3px solid #1f77b4; border-radius: 15px; }
    .stMetric { background-color: #ffffff; border-left: 5px solid #1f77b4; padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

DB_FILE = "data_ovinstat_V4.csv" # On passe en V4 pour repartir de zéro

# Les 24 paramètres + métadonnées
COLONNES = [
    "Date", "ID", "Race", "Age", "Poids", 
    "HG", "HS", "LB", "LQ", "LT", "LC", "LH", "LI", "LP", "PP", "TP",
    "Lc_cornes", "LT_tete", "Lt_tete", "LO", "Lo_oreille", "TC", "LY", "TS", "PS", "LG", "LL"
]

# --- 2. FONCTIONS TECHNIQUES ---
@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n.pt')

def load_data():
    if os.path.exists(DB_FILE):
        return pd.read_csv(DB_FILE, sep=';', encoding='utf-8-sig')
    return pd.DataFrame(columns=COLONNES)

model = load_yolo_model()
data = load_data()

# --- 3. INTERFACE ---
st.title("🐑 OviStat IA : Morphométrie Avancée")

tab1, tab2, tab3 = st.tabs(["📥 Saisie Terrain", "🔍 Historique", "📊 Analyse"])

with tab1:
    with st.form("form_expert", clear_on_submit=True):
        # SECTION 1 : IDENTITÉ
        st.subheader("🆔 Identification")
        c1, c2, c3 = st.columns(3)
        id_in = c1.text_input("ID Animal", value=f"OVIN-{len(data)+1}")
        age_in = c2.number_input("Âge (mois)", min_value=0, value=12)
        race_in = c3.selectbox("Race", ["Ouled Djellal", "Rembi", "Hamra", "Taadmit"])

        st.divider()

        # SECTION 2 : VISION IA
        st.subheader("📸 Diagnostic Optique")
        col_cam, col_info = st.columns([2, 1])
        with col_cam:
            photo = st.camera_input("Scanner l'animal")
        with col_info:
            if photo:
                st.success("✅ Animal détecté")
                st.info("💡 L'IA suggère de vérifier les mesures de hauteur et de longueur du tronc.")
            else:
                st.warning("Prenez une photo pour activer l'assistance morphométrique.")

        st.divider()

        # SECTION 3 : MENSURATIONS (Organisées par zones)
        st.subheader("📏 Paramètres Morphométriques")

        with st.expander("🏗️ Hauteurs & Longueurs (Corps)", expanded=True):
            g1, g2, g3, g4 = st.columns(4)
            poids = g1.number_input("Poids (kg)", value=0.0)
            hg = g2.number_input("H. Garrot (HG)", value=0.0)
            hs = g3.number_input("H. Sacrum (HS)", value=0.0)
            lb = g4.number_input("Long. Corps (LB)", value=0.0)
            
            g5, g6, g7, g8 = st.columns(4)
            lq = g5.number_input("Long. Queue (LQ)", value=0.0)
            lt = g6.number_input("Long. Tronc (LT)", value=0.0)
            lc = g7.number_input("Long. Cou (LC)", value=0.0)
            lh = g8.number_input("Long. Bassin (LH)", value=0.0)

        with st.expander("📐 Largeurs & Poitrine"):
            g9, g10, g11, g12 = st.columns(4)
            li = g9.number_input("Larg. Ischions (LI)", value=0.0)
            lp = g10.number_input("Larg. Poitrine (LP)", value=0.0)
            pp = g11.number_input("Prof. Poitrine (PP)", value=0.0)
            tp = g12.number_input("Tour Poitrine (TP)", value=0.0)

        with st.expander("👤 Tête & Oreilles"):
            g13, g14, g15, g16, g17 = st.columns(5)
            l_cornes = g13.number_input("Long. Cornes (Lc)", value=0.0)
            lt_tete = g14.number_input("Long. Tête (LT)", value=0.0)
            lt_tete_larg = g15.number_input("Larg. Tête (Lt)", value=0.0)
            lo = g16.number_input("Long. Oreille (LO)", value=0.0)
            lo_larg = g17.number_input("Larg. Oreille (Lo)", value=0.0)

        with st.expander("🧬 Reproduction & Laine"):
            g18, g19, g20, g21, g22, g23 = st.columns(6)
            tc = g18.number_input("Tour Canon (TC)", value=0.0)
            ly = g19.number_input("Long. Trayons (LY)", value=0.0)
            ts = g20.number_input("Tour Scrotal (TS)", value=0.0)
            ps = g21.number_input("Prof. Scrotale (PS)", value=0.0)
            lg = g22.number_input("Long. Gigot (LG)", value=0.0)
            ll = g23.number_input("Long. Laine (LL)", value=0.0)

        st.divider()

        # BOUTON FINAL
        if st.form_submit_button("💾 ENREGISTRER LA FICHE COMPLÈTE"):
            if id_in:
                date_now = datetime.now().strftime("%Y-%m-%d")
                new_row = [
                    date_now, id_in, race_in, age_in, poids,
                    hg, hs, lb, lq, lt, lc, lh, li, lp, pp, tp,
                    l_cornes, lt_tete, lt_tete_larg, lo, lo_larg, tc, ly, ts, ps, lg, ll
                ]
                pd.DataFrame([new_row], columns=COLONNES).to_csv(DB_FILE, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
                st.success(f"Fiche de {id_in} archivée avec succès !")
                st.balloons()
                st.rerun()

with tab2:
    st.subheader("📋 Historique du troupeau")
    
    # On affiche les données si elles existent
    if not data.empty:
        st.dataframe(data, use_container_width=True)
    else:
        st.info("ℹ️ La base de données est actuellement vide. Enregistrez un animal pour voir les données.")

    st.divider()
    st.subheader("📥 Exportation")
    
    # Préparation du téléchargement (même si vide, cela téléchargera les en-têtes)
    csv_ready = data.to_csv(sep=';', index=False).encode('utf-8-sig')
    
    st.download_button(
        label="📥 Télécharger la base Excel (CSV)",
        data=csv_ready,
        file_name=f"OviStat_Export_{date.today()}.csv",
        mime="text/csv",
        help="Cliquez ici pour récupérer vos données et les ouvrir dans Excel"
    )


with tab3:
    st.header("🧠 Expertise & Benchmark Zootechnique")
    
    if not data.empty:
        # 1. SÉLECTION DE L'ANIMAL
        target = st.selectbox("Sélectionner l'animal pour l'audit complet", data["ID"].unique())
        anim = data[data["ID"] == target].iloc[-1] 
        
        # --- PARTIE A : SCORES INDIVIDUELS (VOTRE CODE) ---
        st.subheader("🥇 Scores Individuels")
        try:
            # Note : Assurez-vous que les noms de colonnes 'Poids_kg', 'LB', 'TP', 'HG' correspondent à votre CSV
            poids_val = anim['Poids_kg'] if 'Poids_kg' in anim else 0
            indice_compacite = poids_val / anim['LB'] if anim['LB'] > 0 else 0
            indice_anamorphose = (anim['TP']**2) / anim['HG'] if anim['HG'] > 0 else 0
            indice_proportion = anim['HG'] / anim['LB'] if anim['LB'] > 0 else 0
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Rendement Viande", f"{indice_compacite:.2f}")
                st.caption("Indice de Compacité")
            with c2:
                st.metric("Robustesse", f"{indice_anamorphose:.2f}")
                st.caption("Indice d'Anamorphose")
            with c3:
                format_animal = "Longiligne" if indice_proportion < 0.95 else "Médioligne"
                st.metric("Format", format_animal)
                st.caption(f"Ratio HG/LB: {indice_proportion:.2f}")

            # Conseil IA basé sur l'individu
            if indice_compacite < 0.5:
                st.warning("💡 **Conseil Individuel :** Animal un peu frêle. Surveiller l'alimentation.")
            else:
                st.success("💡 **Conseil Individuel :** Excellente conformation bouchère.")

        except Exception as e:
            st.error(f"Données insuffisantes pour les indices : {e}")

        st.divider()

        # --- PARTIE B : BENCHMARK (COMPARAISON AU GROUPE) ---
        if len(data) >= 2:
            st.subheader(f"📊 Position de {target} par rapport au troupeau")
            
            moyenne_poids = data['Poids_kg'].mean()
            diff_poids = ((poids_val - moyenne_poids) / moyenne_poids) * 100 if moyenne_poids > 0 else 0
            
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.metric("Performance Poids", f"{poids_val} kg", f"{diff_poids:.1f}% vs Moyenne")
            with col_b2:
                # Classement
                rang = data['Poids_kg'].rank(ascending=False).iloc[-1]
                st.metric("Rang", f"{int(rang)} / {len(data)}")

            # Graphique de Distribution
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.hist(data['Poids_kg'], bins=10, color='#d1dceb', edgecolor='#1f77b4', alpha=0.7)
            ax.axvline(poids_val, color='red', linestyle='--', label=f"{target}")
            ax.axvline(moyenne_poids, color='green', linestyle='-', label="Moyenne")
            ax.set_xlabel("Poids (kg)")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("💡 Les statistiques de groupe apparaîtront quand vous aurez au moins 2 animaux.")
            
    else:
        st.info("📊 Les analyses apparaîtront après le premier enregistrement.")

        st.divider()
        st.subheader("📄 Certificat de Vente Officiel")
        
    if st.button(f"Générer le certificat pour {target}"):
            # Création du PDF
            pdf = FPDF()
            pdf.add_page()
            
            # --- ENTÊTE ---
            pdf.set_font("Arial", 'B', 20)
            pdf.set_text_color(31, 119, 180) # Bleu OviStat
            pdf.cell(200, 15, "CERTIFICAT DE QUALITÉ OVISTAT IA", ln=True, align='C')
            
            pdf.set_font("Arial", 'I', 10)
            pdf.set_text_color(100)
            pdf.cell(200, 10, f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}", ln=True, align='C')
            pdf.ln(10)

            # --- INFOS IDENTITÉ ---
            pdf.set_fill_color(240, 242, 246)
            pdf.set_font("Arial", 'B', 12)
            pdf.set_text_color(0)
            pdf.cell(0, 10, f" IDENTIFICATION DE L'ANIMAL : {target}", ln=True, fill=True)
            
            pdf.set_font("Arial", '', 11)
            pdf.cell(100, 10, f"Race : {anim['Race']}")
            pdf.cell(100, 10, f"Age : {anim['Age']} mois", ln=True)
            pdf.cell(100, 10, f"Poids actuel : {poids_val} kg")
            pdf.cell(100, 10, f"Rang Troupeau : {int(rang)} / {len(data)}", ln=True)
            pdf.ln(5)

            # --- SCORES ZOOTECHNIQUES ---
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, " EXPERTISE MORPHOMÉTRIQUE", ln=True, fill=True)
            
            pdf.set_font("Arial", '', 11)
            pdf.cell(100, 10, f"Indice de Compacité (Viande) : {indice_compacite:.2f}")
            pdf.cell(100, 10, f"Indice de Robustesse : {indice_anamorphose:.2f}", ln=True)
            pdf.cell(100, 10, f"Format : {format_animal} (Ratio: {indice_proportion:.2f})", ln=True)
            pdf.ln(5)

            # --- MENSURATIONS CLÉS ---
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, " MENSURATIONS PRINCIPALES (CM)", ln=True, fill=True)
            
            pdf.set_font("Arial", '', 10)
            # On liste quelques mesures clés parmi les 24
            mesures_pdf = f"HG: {anim['HG']} | HS: {anim['HS']} | LB: {anim['LB']} | TP: {anim['TP']} | TS: {anim['TS']}"
            pdf.multi_cell(0, 10, mesures_pdf)
            
            pdf.ln(15)
            pdf.set_font("Arial", 'I', 9)
            pdf.multi_cell(0, 5, "Ce document est généré par l'IA OviStat Vision Pro sur la base des mesures biométriques relevées. Il certifie la conformité de l'animal aux standards de performance du troupeau.")

            # Sauvegarde et bouton de téléchargement
            pdf_output = pdf.output(dest="S").encode("latin-1")
            st.download_button(
                label=f"📥 Télécharger le Certificat de {target} (PDF)",
                data=pdf_output,
                file_name=f"Certificat_{target}.pdf",
                mime="application/pdf"
            )
