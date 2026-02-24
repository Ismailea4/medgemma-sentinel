"""
MedGemma Longitudinal Analysis Module
Handles longitudinal patient data analysis and report generation
"""

import pdfplumber
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import datetime
import json
import os
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak
from reportlab.lib import colors
from reportlab.lib.units import inch
from collections import defaultdict
import textwrap
from pathlib import Path

# Mapping de gravité
SEVERITY_MAP = {"Elevee": 3, "High": 3, "Moyenne": 2, "Medium": 2, "Basse": 1, "Low": 1, "None": 0, "": 0}
SEVERITY_COLORS = {
    "Elevee": "red", "High": "red", 
    "Moyenne": "orange", "Medium": "orange", 
    "Basse": "green", "Low": "green", 
    "None": "gray", "": "gray"
}


def load_medgemma_model(model_path: str):
    """Charge le modèle MedGemma local"""
    try:
        from llama_cpp import Llama
        model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=0,
            n_batch=512,
            verbose=False
        )
        return model
    except Exception as e:
        print(f"Error loading MedGemma model: {str(e)}")
        return None


def extract_medical_data(pdf_path):
    """Extrait les données médicales structurées du PDF"""
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    
    data = {
        "patient_id": "",
        "patient_name": "",
        "date": "",
        "time": "",
        "room": "",
        "main_complaint": "",
        "symptoms": [],
        "vital_signs": {},
        "physical_exam": {},
        "diagnoses": [],
        "severity": "",
        "management_plan": [],
        "critical_alerts": False,
        "raw_text": text,
        "full_date": None,
        "filename": os.path.basename(pdf_path)
    }
    
    # Patient Name
    patient_name_match = re.search(r"Patient:\s*([^\n]+)", text)
    if patient_name_match:
        data["patient_name"] = patient_name_match.group(1).strip()
    
    # Patient ID
    patient_id_match = re.search(r"ID:\s*([A-Z0-9_]+)", text)
    if patient_id_match:
        data["patient_id"] = patient_id_match.group(1)
    
    # Date
    date_match = re.search(r"Date:\s*(\d{2}/\d{2}/\d{4})", text)
    if date_match:
        data["date"] = date_match.group(1)
    
    # Time from filename
    time_match = re.search(r"_(\d{4})\.pdf", os.path.basename(pdf_path))
    if time_match:
        time_str = time_match.group(1)
        data["time"] = f"{time_str[:2]}:{time_str[2:]}"
    
    # Room
    room_match = re.search(r"Chambre:\s*([^\n]+)", text)
    if room_match:
        data["room"] = room_match.group(1).strip()
    
    # Main complaint
    complaint_match = re.search(r"Plainte principale:\s*([^\n]+)", text)
    if complaint_match:
        data["main_complaint"] = complaint_match.group(1).strip()
    
    # Symptoms
    symptoms_match = re.search(r"Symptômes associés:\s*([\s\S]*?)(?:■|Examen|Analyse|Diagnostics|■■)", text)
    if symptoms_match:
        symptoms_text = symptoms_match.group(1)
        symptoms = [s.strip().lstrip('•-').strip() for s in symptoms_text.split('\n') if s.strip()]
        data["symptoms"] = symptoms
    
    # Vital signs
    vital_signs = {}
    vs_match = re.search(r"Constantes:\s*([\s\S]*?)(?:Examen|Analyse|Diagnostics|■■)", text)
    if vs_match:
        vs_text = vs_match.group(1)
        vs_items = [s.strip() for s in vs_text.split('\n') if s.strip()]
        for item in vs_items:
            if ':' in item:
                key, value = item.split(':', 1)
                vital_signs[key.strip()] = value.strip()
        data["vital_signs"] = vital_signs
    
    # Physical exam
    physical_exam = {}
    pe_match = re.search(r"Examen physique:\s*([\s\S]*?)(?:Analyse|Diagnostics|Plan|■■)", text)
    if pe_match:
        pe_text = pe_match.group(1)
        pe_items = [s.strip() for s in pe_text.split('\n') if s.strip()]
        for item in pe_items:
            if ':' in item:
                key, value = item.split(':', 1)
                physical_exam[key.strip()] = value.strip()
        data["physical_exam"] = physical_exam
    
    # Diagnoses
    diagnoses = []
    diag_match = re.search(r"Diagnostics différentiels:\s*([\s\S]*?)(?:Gravité|Plan|■■)", text)
    if diag_match:
        diag_text = diag_match.group(1)
        diag_items = re.split(r'\d+\.', diag_text)
        for d in diag_items:
            clean_d = d.strip()
            if clean_d:
                diagnoses.append(clean_d)
        data["diagnoses"] = diagnoses
    
    # Severity
    severity_match = re.search(r"Gravité évaluée:\s*([^\n]+)", text)
    if severity_match:
        data["severity"] = severity_match.group(1).strip()
    
    # Management plan
    plan_match = re.search(r"Plan de Prise en Charge\s*([\s\S]*?)(?:Rapport|Version|■■)", text)
    if plan_match:
        plan_text = plan_match.group(1)
        plan_items = [s.strip().lstrip('✓*•-').strip() for s in plan_text.split('\n') if s.strip()]
        data["management_plan"] = plan_items
    
    # Critical alerts
    data["critical_alerts"] = "alertes critiques" in text.lower() or "■■ ATTENTION" in text
    
    # Full date for sorting
    if data["date"] and data["time"]:
        try:
            data["full_date"] = datetime.datetime.strptime(
                f"{data['date']} {data['time']}", "%d/%m/%Y %H:%M"
            )
        except:
            pass
    
    return data


def analyze_longitudinal_evolution(patient_reports, llm_model, analysis_type="comparison"):
    """
    Analyse longitudinale utilisant MedGemma
    analysis_type: "comparison" (2 rapports) ou "time_series" (plusieurs rapports)
    """
    
    # Préparer les données pour MedGemma
    reports_text = ""
    for i, report in enumerate(patient_reports, 1):
        reports_text += f"Consultation {i} ({report['date']} {report['time']}):\n"
        reports_text += f"- Plainte principale: {report['main_complaint'] if report['main_complaint'] else 'Aucune'}\n"
        reports_text += f"- Symptômes: {', '.join(report['symptoms']) if report['symptoms'] else 'Aucun'}\n"
        reports_text += f"- Diagnostics: {', '.join(report['diagnoses']) if report['diagnoses'] else 'Aucun'}\n"
        reports_text += f"- Gravité: {report['severity'] if report['severity'] else 'Non évaluée'}\n"
        reports_text += f"- Plan: {', '.join(report['management_plan']) if report['management_plan'] else 'Non défini'}\n\n"
    
    # Prompt pour l'analyse
    if analysis_type == "comparison":
        prompt = f"""Tu es un expert médical analysant l'évolution clinique entre deux consultations.
{reports_text}

Fournis une analyse JSON structurée avec:
- summary
- severity_evolution
- symptom_changes (new, resolved, persistent, worsened)
- diagnosis_evolution
- management_changes
- critical_alerts
- clinical_recommendations
- prognosis
- monitoring_parameters
- overall_assessment
- risk_level"""
    else:
        prompt = f"""Tu es un expert médical analysant l'évolution longitudinale sur plusieurs consultations.
{reports_text}

Fournis une analyse JSON structurée complète."""
    
    try:
        if llm_model:
            response = llm_model(
                prompt,
                max_tokens=1536,
                temperature=0.3,
                stop=["</s>"],
                echo=False
            )
            generated_text = response['choices'][0]['text'].strip()
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
    except Exception as e:
        print(f"Error during longitudinal analysis: {str(e)}")
    
    return generate_fallback_longitudinal_analysis(patient_reports, analysis_type)


def generate_fallback_longitudinal_analysis(patient_reports, analysis_type="comparison"):
    """Analyse de secours si MedGemma échoue"""
    
    severities = [SEVERITY_MAP.get(report["severity"], 0) for report in patient_reports]
    
    if len(severities) > 1:
        if severities[-1] > severities[0]:
            trend = "détérioration"
            risk = "Élevé"
        elif severities[-1] < severities[0]:
            trend = "amélioration"
            risk = "Faible"
        else:
            trend = "stable"
            risk = "Moyen"
    else:
        trend = "stable"
        risk = "Moyen"
    
    symptom_changes = {
        "new_symptoms": [], 
        "resolved_symptoms": [], 
        "persistent_symptoms": [],
        "worsened_symptoms": []
    }
    
    if len(patient_reports) >= 2:
        prev_symptoms = set([s.lower() for s in patient_reports[0]["symptoms"]])
        curr_symptoms = set([s.lower() for s in patient_reports[-1]["symptoms"]])
        
        symptom_changes["new_symptoms"] = list(curr_symptoms - prev_symptoms)
        symptom_changes["resolved_symptoms"] = list(prev_symptoms - curr_symptoms)
        symptom_changes["persistent_symptoms"] = list(prev_symptoms & curr_symptoms)
    
    diagnosis_evolution = {
        "new_diagnoses": [],
        "excluded_diagnoses": [],
        "prioritized_diagnoses": [],
        "analysis": "Évolution des diagnostics analysée"
    }
    
    if len(patient_reports) >= 2:
        prev_diagnoses = set([d.lower() for d in patient_reports[0]["diagnoses"]])
        curr_diagnoses = set([d.lower() for d in patient_reports[-1]["diagnoses"]])
        
        diagnosis_evolution["new_diagnoses"] = list(curr_diagnoses - prev_diagnoses)
        diagnosis_evolution["excluded_diagnoses"] = list(prev_diagnoses - curr_diagnoses)
        diagnosis_evolution["prioritized_diagnoses"] = patient_reports[-1]["diagnoses"][:3] if patient_reports[-1]["diagnoses"] else []
    
    critical_alerts = []
    for report in patient_reports:
        if report["critical_alerts"]:
            critical_alerts.append(f"Alerte critique le {report['date']}")
        if "Elevee" in report["severity"]:
            critical_alerts.append(f"Gravité élevée le {report['date']}")
    
    if analysis_type == "comparison":
        return {
            "summary": f"Analyse longitudinale de l'évolution clinique",
            "severity_evolution": f"Tendance: {trend}",
            "symptom_changes": symptom_changes,
            "diagnosis_evolution": diagnosis_evolution,
            "management_changes": {
                "new_interventions": patient_reports[-1]["management_plan"] if patient_reports else [],
                "removed_interventions": [],
                "modified_interventions": [],
                "justification": "Plan adapté selon l'évolution"
            },
            "critical_alerts": critical_alerts if critical_alerts else ["Aucune alerte"],
            "clinical_recommendations": [
                "Surveillance clinique étroite",
                "Adapter le traitement selon l'évolution",
                "Réévaluer les diagnostics"
            ],
            "prognosis": "À évaluer selon la réponse au traitement",
            "monitoring_parameters": ["Signes vitaux", "Symptômes", "Paramètres biologiques"],
            "overall_assessment": "Amélioration" if "amélioration" in trend else "Détérioration" if "détérioration" in trend else "Stable",
            "risk_level": risk
        }
    else:
        return {
            "summary": f"Analyse longitudinale complète",
            "overall_trend": "Amélioration" if "amélioration" in trend else "Détérioration" if "détérioration" in trend else "Stable",
            "key_milestones": [
                {"date": patient_reports[0]['date'], "event": "Première consultation", "impact": "Baseline"}
            ],
            "severity_analysis": {
                "trend": trend,
                "critical_periods": [report['date'] for report in patient_reports if "Elevee" in report['severity']],
                "analysis": "Analyse de la gravité"
            },
            "symptom_evolution": {
                "timeline": [{"date": patient_reports[-1]['date'] if patient_reports else "", "new": symptom_changes["new_symptoms"]}],
                "pattern": "Pattern analysé"
            },
            "diagnosis_evolution": diagnosis_evolution,
            "management_evolution": {
                "adaptations": "Adaptations appliquées",
                "effectiveness": "À évaluer",
                "recommendations": ["Surveillance", "Ajustements"]
            },
            "critical_alerts": critical_alerts if critical_alerts else ["Aucune alerte majeure"],
            "clinical_recommendations": [
                "Surveillance régulière",
                "Suivi des paramètres",
                "Adaptation du traitement"
            ],
            "prognosis": {
                "short_term": "Variable selon la réponse",
                "medium_term": "À réévaluer après stabilisation",
                "long_term": "Dépend de l'évolution"
            },
            "monitoring_plan": {
                "priority_parameters": ["FC", "TA", "SpO2", "Symptômes"],
                "frequency": "Surveillance continue",
                "warning_signs": ["Aggravation", "Nouveaux symptômes"]
            },
            "risk_assessment": {
                "current_risk": risk,
                "risk_factors": ["Évolution de la gravité"],
                "mitigation_strategies": ["Surveillance étroite"]
            }
        }


def create_longitudinal_visualizations(patient_reports, analysis):
    """Crée des visualisations longitudinales complètes"""
    
    dates = [report['date'] for report in patient_reports]
    severities = [SEVERITY_MAP.get(report['severity'], 0) for report in patient_reports]
    symptom_counts = [len(report['symptoms']) for report in patient_reports]
    diagnosis_counts = [len(report['diagnoses']) for report in patient_reports]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Évolution de la Gravité',
            'Évolution des Symptômes',
            'Évolution des Diagnostics',
            'Tendance Globale'
        ),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "indicator"}]]
    )
    
    # Gravité
    fig.add_trace(
        go.Scatter(
            x=dates, y=severities,
            mode='lines+markers+text',
            name='Gravité',
            line=dict(color='red', width=3),
            marker=dict(size=15, color='red', symbol='circle'),
            text=severities,
            textposition='top center'
        ),
        row=1, col=1
    )
    
    # Symptômes
    fig.add_trace(
        go.Scatter(
            x=dates, y=symptom_counts,
            mode='lines+markers+text',
            name='Symptômes',
            line=dict(color='blue', width=3),
            marker=dict(size=15, color='blue', symbol='circle'),
            text=symptom_counts,
            textposition='top center'
        ),
        row=1, col=2
    )
    
    # Diagnostics
    fig.add_trace(
        go.Scatter(
            x=dates, y=diagnosis_counts,
            mode='lines+markers+text',
            name='Diagnostics',
            line=dict(color='purple', width=3),
            marker=dict(size=15, color='purple', symbol='circle'),
            text=diagnosis_counts,
            textposition='top center'
        ),
        row=2, col=1
    )
    
    # Gauge
    overall = analysis.get('overall_assessment', 'Stable')
    trend_value = 50
    if 'Amélioration' in overall:
        trend_value = 80
        trend_color = 'green'
    elif 'Détérioration' in overall:
        trend_value = 20
        trend_color = 'red'
    else:
        trend_value = 50
        trend_color = 'orange'
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=trend_value,
            title={'text': "Tendance"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': trend_color},
                'steps': [
                    {'range': [0, 33], 'color': 'red'},
                    {'range': [33, 66], 'color': 'orange'},
                    {'range': [66, 100], 'color': 'green'}
                ]
            }
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=700,
        showlegend=False,
        title_text="Évolution Clinique"
    )
    
    return fig


def generate_enhanced_report_with_longitudinal_analysis(original_report, longitudinal_analysis, previous_reports, output_path):
    """Génère un rapport PDF amélioré avec l'analyse longitudinale"""
    
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = styles['Title']
    title_style.fontSize = 14
    title = Paragraph("Rapport de Consultation Médicale", title_style)
    story.append(title)
    story.append(Spacer(1, 0.1*inch))
    
    header_style = styles['Normal']
    header_style.fontSize = 9
    header_style.textColor = colors.grey
    story.append(Paragraph("MedGemma Sentinel - Longitudinal Analysis", header_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Patient info
    patient_info = [
        ["Patient:", original_report["patient_name"] if original_report["patient_name"] else "N/A"],
        ["ID:", original_report["patient_id"]],
        ["Date:", original_report["date"]],
        ["Heure:", original_report["time"]],
    ]
    patient_table = Table(patient_info, colWidths=[1.2*inch, 4.5*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 0.15*inch))
    
    # Longitudinal analysis section
    story.append(PageBreak())
    story.append(Paragraph("<b>ANALYSE LONGITUDINALE</b>", styles['Title']))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph(f"<b>Résumé:</b> {longitudinal_analysis.get('summary', 'N/A')}", styles['Normal']))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph(f"<b>Évaluation:</b> {longitudinal_analysis.get('overall_assessment', 'N/A')}", styles['Normal']))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph(f"<b>Risque:</b> {longitudinal_analysis.get('risk_level', 'N/A')}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Build PDF
    doc.build(story)
    return output_path
