"""
MedGemma Sentinel - Longitudinal Analysis Launcher
Main Streamlit app for longitudinal patient analysis and report generation
"""

import streamlit as st
import json
import os
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "longitudinal_src"))

from longitudinal_analysis import (
    extract_medical_data,
    analyze_longitudinal_evolution,
    create_longitudinal_visualizations,
    generate_enhanced_report_with_longitudinal_analysis,
    load_medgemma_model,
    SEVERITY_COLORS
)


def setup_page():
    """Configure la page Streamlit"""
    st.set_page_config(
        page_title="MedGemma Longitudinal Analysis",
        page_icon="🏥",
        layout="wide"
    )
    
    st.markdown("""
    <style>
    .header-container {
        display: flex;
        align-items: center;
        padding: 10px 0;
        margin-bottom: 20px;
        border-bottom: 1px solid #eaeaea;
    }
    .logo {
        width: 80px;
        height: 80px;
        margin-right: 20px;
        border-radius: 50%;
        background: linear-gradient(135deg, #0d47a1 0%, #1565c0 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 24px;
    }
    .logo-text {
        display: flex;
        flex-direction: column;
    }
    .logo-title {
        font-size: 24px;
        font-weight: 700;
        color: #0d47a1;
        margin-bottom: 5px;
    }
    .logo-subtitle {
        font-size: 14px;
        color: #607d8b;
    }
    </style>
    <div class="header-container">
        <div class="logo">M</div>
        <div class="logo-text">
            <div class="logo-title">MEDGEMMA SENTINEL</div>
            <div class="logo-subtitle">ANALYSE LONGITUDINALE AVANCÉE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Analyse longitudinale complète utilisant MedGemma")


def main():
    """Fonction principale"""
    setup_page()
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model configuration
        st.subheader("Configuration du Modèle")
        use_local_model = st.checkbox("Utiliser un modèle MedGemma local", value=False)
        
        model_path = None
        if use_local_model:
            model_path = st.text_input(
                "Chemin du modèle GGUF",
                value="models/medgemma-1.5-medical-Q4_K_M.gguf"
            )
            if not Path(model_path).exists():
                st.warning(f"Modèle non trouvé: {model_path}")
                use_local_model = False
        
        analysis_type = st.selectbox(
            "Type d'analyse",
            ["Comparaison (2 rapports)", "Série temporelle (multiple)"],
            index=0
        )
        analysis_type = "comparison" if analysis_type.startswith("Comparaison") else "time_series"
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Téléchargement des Rapports PDF")
        uploaded_files = st.file_uploader(
            "Sélectionnez les rapports PDF (2 minimum)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Téléchargez 2 rapports pour une comparaison, ou plusieurs pour une analyse longitudinale"
        )
    
    with col2:
        st.subheader("📊 Paramètres d'Analyse")
        
        # Options
        generate_visualizations = st.checkbox("Générer les visualisations", value=True)
        generate_report = st.checkbox("Générer un rapport PDF", value=True)
        use_medgemma = st.checkbox("Utiliser MedGemma pour l'analyse (recommandé)", value=use_local_model)
    
    # Analyze button
    if st.button("🚀 Lancer l'Analyse", type="primary", use_container_width=True):
        if not uploaded_files or len(uploaded_files) < 2:
            st.error("❌ Veuillez télécharger au moins 2 rapports PDF")
            return
        
        with st.spinner("📥 Extraction des données médicales..."):
            # Extract data from PDFs
            patient_reports = []
            for pdf_file in uploaded_files:
                # Save temporary file
                temp_path = f"/tmp/{pdf_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(pdf_file.getbuffer())
                
                # Extract data
                report_data = extract_medical_data(temp_path)
                patient_reports.append(report_data)
                os.remove(temp_path)
        
        st.success(f"✅ {len(patient_reports)} rapports extraits avec succès")
        
        # Load model if needed
        llm_model = None
        if use_medgemma and use_local_model and model_path:
            with st.spinner("🔄 Chargement du modèle MedGemma..."):
                llm_model = load_medgemma_model(model_path)
                if llm_model:
                    st.success("✅ Modèle MedGemma chargé")
                else:
                    st.warning("⚠️ Impossible de charger le modèle, utilisation de l'analyse par défaut")
        
        # Perform analysis
        with st.spinner("🧠 Analyse longitudinale en cours..."):
            longitudinal_analysis = analyze_longitudinal_evolution(
                patient_reports,
                llm_model,
                analysis_type
            )
        
        st.success("✅ Analyse complétée")
        
        # Display results
        st.subheader("📋 Résultats de l'Analyse")
        
        # Summary
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.metric(
                "Évaluation Globale",
                longitudinal_analysis.get('overall_assessment', 'N/A')
            )
        
        with summary_col2:
            st.metric(
                "Niveau de Risque",
                longitudinal_analysis.get('risk_level', 'N/A')
            )
        
        with summary_col3:
            st.metric(
                "Nombre de Rapports",
                len(patient_reports)
            )
        
        # Tabs for detailed results
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Résumé",
            "🔄 Changements",
            "📈 Visualisations",
            "💾 Téléchargement"
        ])
        
        with tab1:
            st.write("**Résumé de l'Évolution**")
            st.write(longitudinal_analysis.get('summary', 'N/A'))
            
            if 'severity_evolution' in longitudinal_analysis:
                st.write("**Évolution de la Gravité**")
                st.write(longitudinal_analysis['severity_evolution'])
            
            if 'critical_alerts' in longitudinal_analysis:
                alerts = longitudinal_analysis['critical_alerts']
                if alerts:
                    st.warning("**⚠️ Alertes Critiques**")
                    for alert in alerts:
                        st.write(f"- {alert}")
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'symptom_changes' in longitudinal_analysis:
                    st.write("**Changements Symptomatiques**")
                    changes = longitudinal_analysis['symptom_changes']
                    if changes.get('new_symptoms'):
                        st.success(f"**Nouveaux:** {', '.join(changes['new_symptoms'])}")
                    if changes.get('resolved_symptoms'):
                        st.info(f"**Résólus:** {', '.join(changes['resolved_symptoms'])}")
                    if changes.get('worsened_symptoms'):
                        st.error(f"**Aggravés:** {', '.join(changes['worsened_symptoms'])}")
            
            with col2:
                if 'diagnosis_evolution' in longitudinal_analysis:
                    st.write("**Évolution des Diagnostics**")
                    diag = longitudinal_analysis['diagnosis_evolution']
                    if isinstance(diag, dict):
                        if diag.get('new_diagnoses'):
                            st.success(f"**Nouveaux:** {', '.join(diag['new_diagnoses'])}")
                        if diag.get('excluded_diagnoses'):
                            st.info(f"**Exclus:** {', '.join(diag['excluded_diagnoses'])}")
        
        with tab3:
            if generate_visualizations:
                with st.spinner("📊 Génération des visualisations..."):
                    fig = create_longitudinal_visualizations(patient_reports, longitudinal_analysis)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Télécharger JSON", use_container_width=True):
                    json_str = json.dumps(longitudinal_analysis, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="📄 Analyse JSON",
                        data=json_str,
                        file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col2:
                if generate_report and st.button("📥 Télécharger PDF", use_container_width=True):
                    with st.spinner("PDF en génération..."):
                        output_file = f"rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        generate_enhanced_report_with_longitudinal_analysis(
                            patient_reports[0],
                            longitudinal_analysis,
                            patient_reports[1:],
                            output_file
                        )
                        
                        with open(output_file, "rb") as f:
                            st.download_button(
                                label="📄 Rapport PDF",
                                data=f.read(),
                                file_name=output_file,
                                mime="application/pdf"
                            )
                        os.remove(output_file)


if __name__ == "__main__":
    main()
