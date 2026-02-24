"""
Report Templates - HTML/Markdown templates for clinical reports
Used for generating formatted Rap1 (night) and Rap2 (day) reports
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from string import Template


class ReportTemplate(ABC):
    """Base class for report templates"""
    
    def __init__(self):
        self.generated_at = datetime.now()
    
    @abstractmethod
    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render the report as Markdown"""
        pass
    
    @abstractmethod
    def render_html(self, data: Dict[str, Any]) -> str:
        """Render the report as HTML"""
        pass
    
    def _format_date(self, dt: Optional[datetime] = None) -> str:
        """Format datetime for display"""
        dt = dt or self.generated_at
        return dt.strftime("%d/%m/%Y %H:%M")
    
    def _format_date_only(self, dt: Optional[datetime] = None) -> str:
        """Format date only"""
        dt = dt or self.generated_at
        return dt.strftime("%d/%m/%Y")


class NightReportTemplate(ReportTemplate):
    """Template for Rap1 - Night Surveillance Report"""
    
    MARKDOWN_TEMPLATE = """# 🌙 Rapport de Surveillance Nocturne

**MedGemma Sentinel - The Scribe**

---

## 📋 Informations Générales

| Champ | Valeur |
|-------|--------|
| **Patient** | $patient_name |
| **ID** | $patient_id |
| **Chambre** | $room |
| **Date** | $date |
| **Période** | $period |
| **Généré le** | $generated_at |

---

## 🎯 Résumé Exécutif

$executive_summary

---

## 🚨 Alertes et Événements

### Statistiques
- **Total événements**: $total_events
- **Alertes critiques**: $critical_alerts 🔴
- **Alertes élevées**: $high_alerts 🟠
- **Alertes modérées**: $medium_alerts 🟡

### Détail des Événements Critiques

$critical_events_detail

### Chronologie des Événements

$events_timeline

---

## 💓 Constantes Vitales

### Résumé

| Paramètre | Min | Max | Moyenne | Anomalies |
|-----------|-----|-----|---------|-----------|
$vitals_table

### Observations

$vitals_observations

---

## 😴 Qualité du Sommeil

| Indicateur | Valeur |
|------------|--------|
| **Score global** | $sleep_score/100 |
| **Qualité** | $sleep_quality |
| **Interruptions** | $sleep_interruptions |
| **Temps sommeil estimé** | $sleep_time |

$sleep_observations

---

## 🔊 Analyse Audio

$audio_analysis

---

## 👁️ Analyse Vision (IR)

$vision_analysis

---

## ✅ Interventions Effectuées

$interventions

---

## 📌 Recommandations pour l'Équipe de Jour

$recommendations

---

## 📊 Graphiques

*[Les graphiques de tendances sont disponibles dans la version PDF complète]*

---

## ⚠️ Points de Vigilance

$vigilance_points

---

**Rapport généré automatiquement par MedGemma Sentinel**

*Ce rapport est un outil d'aide à la décision. Il ne remplace pas l'évaluation clinique par un professionnel de santé qualifié.*

---
*Version: 1.0 | Système: MedGemma Sentinel - The Scribe*
"""
    
    HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport de Surveillance Nocturne - $patient_name</title>
    <style>
        :root {
            --primary-color: #1a365d;
            --secondary-color: #2c5282;
            --accent-color: #3182ce;
            --success-color: #38a169;
            --warning-color: #d69e2e;
            --danger-color: #e53e3e;
            --bg-color: #f7fafc;
            --card-bg: #ffffff;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: #2d3748;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: var(--card-bg);
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .header {
            text-align: center;
            border-bottom: 3px solid var(--primary-color);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        .header h1 {
            color: var(--primary-color);
            margin: 0;
            font-size: 2em;
        }
        
        .header .subtitle {
            color: var(--secondary-color);
            font-size: 1.1em;
            margin-top: 5px;
        }
        
        .header .logo {
            font-size: 3em;
            margin-bottom: 10px;
        }
        
        .section {
            margin-bottom: 30px;
            padding: 20px;
            background: #f8fafc;
            border-radius: 6px;
            border-left: 4px solid var(--accent-color);
        }
        
        .section h2 {
            color: var(--primary-color);
            margin-top: 0;
            font-size: 1.3em;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .info-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .info-table td {
            padding: 8px 12px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .info-table td:first-child {
            font-weight: 600;
            width: 150px;
            color: var(--secondary-color);
        }
        
        .alert-box {
            padding: 15px;
            border-radius: 6px;
            margin: 10px 0;
        }
        
        .alert-critical {
            background-color: #fed7d7;
            border-left: 4px solid var(--danger-color);
        }
        
        .alert-high {
            background-color: #feebc8;
            border-left: 4px solid var(--warning-color);
        }
        
        .alert-medium {
            background-color: #fefcbf;
            border-left: 4px solid #d69e2e;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin: 20px 0;
        }
        
        .stat-card {
            text-align: center;
            padding: 15px;
            background: white;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .stat-card .number {
            font-size: 2em;
            font-weight: bold;
        }
        
        .stat-card .label {
            font-size: 0.9em;
            color: #718096;
        }
        
        .vitals-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        
        .vitals-table th, .vitals-table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .vitals-table th {
            background: var(--primary-color);
            color: white;
        }
        
        .recommendation {
            background: #e6fffa;
            border-left: 4px solid var(--success-color);
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }
        
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #e2e8f0;
            color: #718096;
            font-size: 0.9em;
        }
        
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: 600;
        }
        
        .badge-critical { background: var(--danger-color); color: white; }
        .badge-high { background: var(--warning-color); color: white; }
        .badge-medium { background: #ecc94b; color: #744210; }
        .badge-low { background: #9ae6b4; color: #22543d; }
        
        @media print {
            body { padding: 0; }
            .container { box-shadow: none; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">🌙</div>
            <h1>Rapport de Surveillance Nocturne</h1>
            <div class="subtitle">MedGemma Sentinel - The Scribe</div>
        </div>
        
        <div class="section">
            <h2>📋 Informations Générales</h2>
            <table class="info-table">
                <tr><td>Patient</td><td><strong>$patient_name</strong></td></tr>
                <tr><td>ID</td><td>$patient_id</td></tr>
                <tr><td>Chambre</td><td>$room</td></tr>
                <tr><td>Date</td><td>$date</td></tr>
                <tr><td>Période</td><td>$period</td></tr>
                <tr><td>Généré le</td><td>$generated_at</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>🎯 Résumé Exécutif</h2>
            <p>$executive_summary</p>
        </div>
        
        <div class="section">
            <h2>🚨 Alertes et Événements</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="number">$total_events</div>
                    <div class="label">Total</div>
                </div>
                <div class="stat-card">
                    <div class="number" style="color: var(--danger-color);">$critical_alerts</div>
                    <div class="label">Critiques</div>
                </div>
                <div class="stat-card">
                    <div class="number" style="color: var(--warning-color);">$high_alerts</div>
                    <div class="label">Élevées</div>
                </div>
                <div class="stat-card">
                    <div class="number" style="color: #d69e2e;">$medium_alerts</div>
                    <div class="label">Modérées</div>
                </div>
            </div>
            $critical_events_html
        </div>
        
        <div class="section">
            <h2>💓 Constantes Vitales</h2>
            $vitals_html
        </div>
        
        <div class="section">
            <h2>😴 Qualité du Sommeil</h2>
            <table class="info-table">
                <tr><td>Score global</td><td><strong>$sleep_score/100</strong></td></tr>
                <tr><td>Qualité</td><td>$sleep_quality</td></tr>
                <tr><td>Interruptions</td><td>$sleep_interruptions</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>📌 Recommandations</h2>
            $recommendations_html
        </div>
        
        <div class="footer">
            <p><strong>Rapport généré automatiquement par MedGemma Sentinel</strong></p>
            <p>Ce rapport est un outil d'aide à la décision. Il ne remplace pas l'évaluation clinique par un professionnel de santé qualifié.</p>
            <p>Version 1.0 | Système: MedGemma Sentinel - The Scribe</p>
        </div>
    </div>
</body>
</html>"""
    
    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render night report as Markdown"""
        template = Template(self.MARKDOWN_TEMPLATE)
        
        # Prepare data
        prepared = self._prepare_data(data)
        
        return template.safe_substitute(prepared)
    
    def render_html(self, data: Dict[str, Any]) -> str:
        """Render night report as HTML"""
        template = Template(self.HTML_TEMPLATE)
        
        # Prepare data with HTML formatting
        prepared = self._prepare_data(data)
        prepared.update(self._prepare_html_sections(data))
        
        return template.safe_substitute(prepared)
    
    def _prepare_data(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Prepare template data"""
        events = data.get("events", [])
        night_data = data.get("night_data", {})
        
        critical = [e for e in events if e.get("level") == "critical"]
        high = [e for e in events if e.get("level") == "high"]
        medium = [e for e in events if e.get("level") == "medium"]
        
        sleep_score = night_data.get("sleep_quality_score", 0)
        sleep_quality = "Excellente" if sleep_score > 80 else \
                       "Bonne" if sleep_score > 60 else \
                       "Modérée" if sleep_score > 40 else "Mauvaise"
        
        return {
            "patient_name": data.get("patient_name", "N/A"),
            "patient_id": data.get("patient_id", "N/A"),
            "room": data.get("room", "N/A"),
            "date": self._format_date_only(),
            "period": data.get("period", "21:00 - 07:00"),
            "generated_at": self._format_date(),
            "executive_summary": data.get("summary", "Aucun résumé disponible."),
            "total_events": str(len(events)),
            "critical_alerts": str(len(critical)),
            "high_alerts": str(len(high)),
            "medium_alerts": str(len(medium)),
            "critical_events_detail": self._format_events_markdown(critical),
            "events_timeline": self._format_timeline_markdown(events),
            "vitals_table": self._format_vitals_table(data.get("vitals_summary", {})),
            "vitals_observations": data.get("vitals_observations", "Pas d'observations particulières."),
            "sleep_score": str(int(sleep_score)) if sleep_score else "N/A",
            "sleep_quality": sleep_quality,
            "sleep_interruptions": str(night_data.get("alerts_triggered", 0)),
            "sleep_time": data.get("sleep_time", "N/A"),
            "sleep_observations": data.get("sleep_observations", ""),
            "audio_analysis": data.get("audio_analysis", "Aucune anomalie audio significative détectée."),
            "vision_analysis": data.get("vision_analysis", "Aucune anomalie visuelle significative détectée."),
            "interventions": data.get("interventions", "Aucune intervention requise."),
            "recommendations": self._format_recommendations_markdown(data.get("recommendations", [])),
            "vigilance_points": data.get("vigilance_points", "Surveillance standard recommandée.")
        }
    
    def _prepare_html_sections(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Prepare HTML-specific sections"""
        events = data.get("events", [])
        critical = [e for e in events if e.get("level") == "critical"]
        
        critical_html = ""
        for event in critical:
            critical_html += f"""
            <div class="alert-box alert-critical">
                <strong>{event.get('type', 'Événement')}</strong> - {event.get('timestamp', 'N/A')}<br>
                {event.get('description', '')}
            </div>
            """
        
        recommendations = data.get("recommendations", [])
        rec_html = ""
        for rec in recommendations:
            rec_html += f'<div class="recommendation">• {rec}</div>'
        
        vitals = data.get("vitals_summary", {})
        vitals_html = """<table class="vitals-table">
            <tr><th>Paramètre</th><th>Min</th><th>Max</th><th>Moyenne</th></tr>
        """
        for param, values in vitals.items():
            vitals_html += f"""
            <tr>
                <td>{param}</td>
                <td>{values.get('min', 'N/A')}</td>
                <td>{values.get('max', 'N/A')}</td>
                <td>{values.get('avg', 'N/A')}</td>
            </tr>
            """
        vitals_html += "</table>"
        
        return {
            "critical_events_html": critical_html or "<p>Aucun événement critique.</p>",
            "recommendations_html": rec_html or "<p>Poursuivre surveillance standard.</p>",
            "vitals_html": vitals_html
        }
    
    def _format_events_markdown(self, events: List[Dict]) -> str:
        """Format events list as Markdown"""
        if not events:
            return "*Aucun événement critique détecté.*"
        
        lines = []
        for event in events:
            lines.append(f"- **{event.get('timestamp', 'N/A')}** - {event.get('type', 'Événement')}: {event.get('description', '')}")
        
        return "\n".join(lines)
    
    def _format_timeline_markdown(self, events: List[Dict]) -> str:
        """Format events as timeline"""
        if not events:
            return "*Aucun événement enregistré.*"
        
        lines = ["| Heure | Type | Niveau | Description |", "|-------|------|--------|-------------|"]
        for event in sorted(events, key=lambda e: e.get("timestamp", "")):
            time = event.get("timestamp", "N/A")
            if "T" in str(time):
                time = str(time).split("T")[1][:5]
            level_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(event.get("level", "low"), "⚪")
            lines.append(f"| {time} | {event.get('type', 'N/A')} | {level_emoji} | {event.get('description', '')[:50]} |")
        
        return "\n".join(lines)
    
    def _format_vitals_table(self, vitals: Dict[str, Dict]) -> str:
        """Format vitals as Markdown table rows"""
        if not vitals:
            return "| SpO2 | - | - | - | - |\n| FC | - | - | - | - |"
        
        rows = []
        for param, values in vitals.items():
            anomalies = "✓" if values.get("anomalies", 0) == 0 else f"⚠️ {values.get('anomalies', 0)}"
            rows.append(f"| {param} | {values.get('min', '-')} | {values.get('max', '-')} | {values.get('avg', '-')} | {anomalies} |")
        
        return "\n".join(rows)
    
    def _format_recommendations_markdown(self, recommendations: List[str]) -> str:
        """Format recommendations list"""
        if not recommendations:
            return "- Poursuivre surveillance standard\n- Pas d'action particulière requise"
        
        return "\n".join([f"- {rec}" for rec in recommendations])


class DayReportTemplate(ReportTemplate):
    """Template for Rap2 - Day Consultation Report"""
    
    MARKDOWN_TEMPLATE = """# ☀️ Rapport de Consultation Médicale

**MedGemma Sentinel - The Scribe**

---

## 📋 Identification

| Champ | Valeur |
|-------|--------|
| **Patient** | $patient_name |
| **ID** | $patient_id |
| **Date** | $date |
| **Consultant** | $provider |
| **Mode** | $consultation_mode |
| **Généré le** | $generated_at |

---

## 🌙 Contexte Nocturne

$night_context

---

## 📝 Motif de Consultation

**Plainte principale:** $chief_complaint

### Histoire de la Maladie

$illness_history

### Symptômes Associés

$symptoms_list

---

## 🩺 Examen Clinique

### Constantes Vitales

| Paramètre | Valeur | Statut |
|-----------|--------|--------|
$vitals_table

### Examen Physique

$physical_exam

---

## 🔬 Analyses Complémentaires

$additional_tests

---

## 🧠 Analyse IA (MedGemma)

### Diagnostics Différentiels

$differential_diagnosis

### Évaluation de la Gravité

**Niveau:** $severity_level

$severity_details

---

## 📋 Conclusion

### Diagnostic Retenu/Suspecté

$diagnosis

### Argumentation

$diagnosis_reasoning

---

## 💊 Plan de Prise en Charge

### Traitement Proposé

$treatment_plan

### Examens à Réaliser

$tests_to_order

### Suivi Recommandé

$follow_up

---

## ⚠️ Points de Vigilance

$vigilance_points

---

## 📨 Orientation

$orientation

---

**Rapport généré automatiquement par MedGemma Sentinel**

*Ce rapport est un outil d'aide à la décision. Il ne remplace pas l'évaluation clinique par un professionnel de santé qualifié. Le diagnostic final et les décisions thérapeutiques relèvent de la responsabilité du médecin.*

---
*Version: 1.0 | Système: MedGemma Sentinel - The Scribe*
"""
    
    HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport de Consultation - $patient_name</title>
    <style>
        :root {
            --primary-color: #744210;
            --secondary-color: #975a16;
            --accent-color: #d69e2e;
            --success-color: #38a169;
            --warning-color: #dd6b20;
            --danger-color: #e53e3e;
            --bg-color: #fffaf0;
            --card-bg: #ffffff;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: #2d3748;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: var(--card-bg);
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .header {
            text-align: center;
            border-bottom: 3px solid var(--primary-color);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        .header h1 {
            color: var(--primary-color);
            margin: 0;
            font-size: 2em;
        }
        
        .section {
            margin-bottom: 25px;
            padding: 20px;
            background: #fefcf3;
            border-radius: 6px;
            border-left: 4px solid var(--accent-color);
        }
        
        .section h2 {
            color: var(--primary-color);
            margin-top: 0;
            font-size: 1.3em;
        }
        
        .diagnosis-box {
            background: #fef3c7;
            border: 2px solid var(--accent-color);
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
        }
        
        .severity-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            text-transform: uppercase;
        }
        
        .severity-low { background: #c6f6d5; color: #22543d; }
        .severity-moderate { background: #fef3c7; color: #744210; }
        .severity-high { background: #fed7aa; color: #7c2d12; }
        .severity-critical { background: #fed7d7; color: #742a2a; }
        
        .treatment-box {
            background: #e6fffa;
            border-left: 4px solid var(--success-color);
            padding: 15px;
            margin: 10px 0;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }
        
        th {
            background: var(--primary-color);
            color: white;
        }
        
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #e2e8f0;
            color: #718096;
            font-size: 0.9em;
        }
        
        @media print {
            body { padding: 0; background: white; }
            .container { box-shadow: none; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div style="font-size: 3em;">☀️</div>
            <h1>Rapport de Consultation Médicale</h1>
            <div>MedGemma Sentinel - The Scribe</div>
        </div>
        
        <div class="section">
            <h2>📋 Identification</h2>
            <table>
                <tr><td><strong>Patient</strong></td><td>$patient_name</td></tr>
                <tr><td><strong>ID</strong></td><td>$patient_id</td></tr>
                <tr><td><strong>Date</strong></td><td>$date</td></tr>
                <tr><td><strong>Mode</strong></td><td>$consultation_mode</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>📝 Motif de Consultation</h2>
            <p><strong>$chief_complaint</strong></p>
            <p>$illness_history</p>
        </div>
        
        <div class="section">
            <h2>🩺 Examen Clinique</h2>
            $vitals_html
            <h3>Examen Physique</h3>
            <p>$physical_exam</p>
        </div>
        
        <div class="section">
            <h2>🧠 Analyse IA</h2>
            <h3>Diagnostics Différentiels</h3>
            <ol>$differential_html</ol>
            
            <h3>Gravité</h3>
            <span class="severity-badge severity-$severity_class">$severity_level</span>
        </div>
        
        <div class="section">
            <h2>📋 Conclusion</h2>
            <div class="diagnosis-box">
                <strong>Diagnostic:</strong> $diagnosis
            </div>
        </div>
        
        <div class="section">
            <h2>💊 Plan de Prise en Charge</h2>
            <div class="treatment-box">
                $treatment_html
            </div>
        </div>
        
        <div class="footer">
            <p><strong>Rapport généré automatiquement par MedGemma Sentinel</strong></p>
            <p>Ce rapport ne remplace pas l'évaluation clinique par un professionnel qualifié.</p>
        </div>
    </div>
</body>
</html>"""
    
    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render day consultation report as Markdown"""
        template = Template(self.MARKDOWN_TEMPLATE)
        prepared = self._prepare_data(data)
        return template.safe_substitute(prepared)
    
    def render_html(self, data: Dict[str, Any]) -> str:
        """Render day consultation report as HTML"""
        template = Template(self.HTML_TEMPLATE)
        prepared = self._prepare_data(data)
        prepared.update(self._prepare_html_sections(data))
        return template.safe_substitute(prepared)
    
    def _prepare_data(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Prepare template data"""
        day_data = data.get("day_data", {})
        
        # Build vitals table
        vitals = day_data.get("vitals", {})
        vitals_rows = []
        for param, value in vitals.items():
            status = "✓"  # Simplified
            vitals_rows.append(f"| {param} | {value} | {status} |")
        
        # Build symptoms list
        symptoms = day_data.get("symptoms", [])
        symptoms_list = "\n".join([f"- {s}" for s in symptoms]) if symptoms else "*Aucun symptôme associé rapporté*"
        
        # Build differential diagnosis
        differentials = day_data.get("differential_diagnosis", [])
        diff_text = "\n".join([f"{i}. {d}" for i, d in enumerate(differentials, 1)]) if differentials else "*En cours d'évaluation*"
        
        # Build treatment plan
        actions = day_data.get("recommended_actions", [])
        treatment = "\n".join([f"- {a}" for a in actions]) if actions else "*À définir après examens*"
        
        return {
            "patient_name": data.get("patient_name", "N/A"),
            "patient_id": data.get("patient_id", "N/A"),
            "date": self._format_date_only(),
            "provider": data.get("provider", "MedGemma Sentinel"),
            "consultation_mode": day_data.get("consultation_mode", "Général").capitalize(),
            "generated_at": self._format_date(),
            "night_context": data.get("night_context", "*Pas de données nocturnes disponibles*"),
            "chief_complaint": day_data.get("presenting_complaint", "Non spécifié"),
            "illness_history": data.get("illness_history", "*À compléter*"),
            "symptoms_list": symptoms_list,
            "vitals_table": "\n".join(vitals_rows) if vitals_rows else "| - | - | - |",
            "physical_exam": self._format_exam(day_data.get("physical_exam", {})),
            "additional_tests": data.get("additional_tests", "*Aucun examen complémentaire réalisé*"),
            "differential_diagnosis": diff_text,
            "severity_level": day_data.get("severity_assessment", "Modérée"),
            "severity_details": data.get("severity_details", ""),
            "diagnosis": day_data.get("final_diagnosis", "*Diagnostic en attente de confirmation*"),
            "diagnosis_reasoning": data.get("diagnosis_reasoning", ""),
            "treatment_plan": treatment,
            "tests_to_order": data.get("tests_to_order", "*Selon évolution clinique*"),
            "follow_up": data.get("follow_up", "Réévaluation selon évolution"),
            "vigilance_points": data.get("vigilance_points", "Surveillance des signes d'aggravation"),
            "orientation": data.get("orientation", "Suivi ambulatoire / Hospitalisation selon gravité")
        }
    
    def _prepare_html_sections(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Prepare HTML-specific sections"""
        day_data = data.get("day_data", {})
        
        # Vitals HTML
        vitals = day_data.get("vitals", {})
        vitals_html = "<table><tr><th>Paramètre</th><th>Valeur</th></tr>"
        for param, value in vitals.items():
            vitals_html += f"<tr><td>{param}</td><td>{value}</td></tr>"
        vitals_html += "</table>"
        
        # Differential HTML
        differentials = day_data.get("differential_diagnosis", [])
        diff_html = "".join([f"<li>{d}</li>" for d in differentials]) if differentials else "<li>En évaluation</li>"
        
        # Treatment HTML
        actions = day_data.get("recommended_actions", [])
        treatment_html = "<ul>" + "".join([f"<li>{a}</li>" for a in actions]) + "</ul>" if actions else "<p>À définir</p>"
        
        # Severity class
        severity = day_data.get("severity_assessment", "Modérée").lower()
        severity_map = {"faible": "low", "modérée": "moderate", "élevée": "high", "critique": "critical"}
        severity_class = severity_map.get(severity, "moderate")
        
        return {
            "vitals_html": vitals_html,
            "differential_html": diff_html,
            "treatment_html": treatment_html,
            "severity_class": severity_class
        }
    
    def _format_exam(self, exam: Dict[str, str]) -> str:
        """Format physical exam findings"""
        if not exam:
            return "*Examen physique non documenté*"
        
        lines = []
        for system, finding in exam.items():
            lines.append(f"- **{system}:** {finding}")
        
        return "\n".join(lines)
