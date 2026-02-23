"""
PDF Report Generator - Generate clinical PDF reports
Uses ReportLab for PDF generation with professional medical formatting
Integrates MedGemma for dynamic content generation
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from enum import Enum
import io
import re

# PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, HRFlowable
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: ReportLab not installed. PDF generation will be limited.")

# MedGemma integration
try:
    from src.reporting.medgemma_engine import MedGemmaEngine
    MEDGEMMA_AVAILABLE = True
except ImportError:
    MEDGEMMA_AVAILABLE = False

# Clinical plots integration
try:
    from src.reporting.clinical_plots import (
        generate_night_report_plots,
        plot_vitals_dashboard,
        plot_events_timeline,
        plot_severity_distribution
    )
    PLOTS_AVAILABLE = True
except ImportError:
    PLOTS_AVAILABLE = False


class ReportStyle(str, Enum):
    """Visual styles for PDF reports"""
    CLINICAL = "clinical"      # Professional medical style
    SUMMARY = "summary"        # Condensed summary style
    DETAILED = "detailed"      # Full detailed report


class PDFReportGenerator:
    """
    PDF Report Generator for MedGemma Sentinel.
    
    Generates professional medical PDF reports for:
    - Night surveillance (Rap1)
    - Day consultation (Rap2)
    - Triage assessments
    - Longitudinal summaries
    """
    
    # Color scheme
    COLORS = {
        "primary": colors.HexColor("#1a365d"),
        "secondary": colors.HexColor("#2c5282"),
        "accent": colors.HexColor("#3182ce"),
        "success": colors.HexColor("#38a169"),
        "warning": colors.HexColor("#d69e2e"),
        "danger": colors.HexColor("#e53e3e"),
        "light_bg": colors.HexColor("#f7fafc"),
        "text": colors.HexColor("#2d3748"),
        "muted": colors.HexColor("#718096")
    }
    
    def __init__(self, output_dir: str = "./data/reports", initialize_medgemma: bool = False):
        """
        Initialize the PDF generator.
        
        Args:
            output_dir: Directory for saving generated PDFs
            initialize_medgemma: If True, initialize MedGemmaEngine for dynamic text generation.
                                 Default False to avoid reloading heavy guard/model stack during PDF export.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Lazy by default: PDF export does not require MedGemma inference.
        self.medgemma_engine = None
        if initialize_medgemma and MEDGEMMA_AVAILABLE:
            try:
                self.medgemma_engine = MedGemmaEngine()
            except Exception as e:
                print(f"Warning: MedGemma engine not initialized: {e}")
        
        if REPORTLAB_AVAILABLE:
            self._init_styles()
    
    def _init_styles(self) -> None:
        """Initialize PDF styles"""
        self.styles = getSampleStyleSheet()
        
        # Custom styles
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=self.COLORS["primary"],
            spaceAfter=20,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionTitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=self.COLORS["secondary"],
            spaceBefore=15,
            spaceAfter=10,
            borderPadding=5
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubSection',
            parent=self.styles['Heading3'],
            fontSize=12,
            textColor=self.COLORS["accent"],
            spaceBefore=10,
            spaceAfter=5
        ))
        
        # Modify existing BodyText style instead of adding new one
        self.styles['BodyText'].fontSize = 10
        self.styles['BodyText'].textColor = self.COLORS["text"]
        self.styles['BodyText'].spaceBefore = 3
        self.styles['BodyText'].spaceAfter = 3
        self.styles['BodyText'].alignment = TA_JUSTIFY
        
        self.styles.add(ParagraphStyle(
            name='AlertCritical',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=colors.white,
            backColor=self.COLORS["danger"],
            borderPadding=8,
            spaceBefore=2,
            spaceAfter=2,
            alignment=TA_LEFT
        ))
        
        self.styles.add(ParagraphStyle(
            name='AlertWarning',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=13,
            textColor=colors.black,
            backColor=self.COLORS["warning"],
            borderPadding=8,
            spaceBefore=2,
            spaceAfter=2,
            alignment=TA_LEFT
        ))
        
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=self.COLORS["muted"],
            alignment=TA_CENTER
        ))
    
    def generate_night_report(
        self,
        data: Dict[str, Any],
        filename: Optional[str] = None,
        style: ReportStyle = ReportStyle.CLINICAL
    ) -> str:
        """
        Generate a night surveillance report PDF (Rap1).
        
        Args:
            data: Report data from the workflow
            filename: Optional custom filename
            style: Visual style for the report
            
        Returns:
            Path to the generated PDF file
        """
        if not REPORTLAB_AVAILABLE:
            return self._generate_fallback_report(data, "night", filename)
        
        # Generate filename
        patient_id = data.get("patient_id", "unknown")
        date_str = datetime.now().strftime("%Y%m%d_%H%M")
        filename = filename or f"rap1_night_{patient_id}_{date_str}.pdf"
        filepath = self.output_dir / filename
        
        # Create document
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        # Build content
        story = []
        
        # Header
        story.extend(self._build_header("Rapport de Surveillance Nocturne", data))
        
        # Executive Summary
        story.extend(self._build_executive_summary(data))

        # Longitudinal insights from the two most recent sessions
        story.extend(self._build_history_evolution_section(data))
        
        # Alerts section
        story.extend(self._build_alerts_section(data))
        
        # Vitals section
        story.extend(self._build_vitals_section(data))
        
        # Clinical plots (vitals trends, events timeline, severity distribution)
        story.extend(self._build_plots_section(data))
        
        # Sleep quality
        story.extend(self._build_sleep_section(data))
        
        # Recommendations
        story.extend(self._build_recommendations_section(data))
        
        # Footer
        story.extend(self._build_footer())
        
        # Build PDF
        doc.build(story)
        
        return str(filepath)
    
    def generate_day_report(
        self,
        data: Dict[str, Any],
        filename: Optional[str] = None,
        style: ReportStyle = ReportStyle.CLINICAL
    ) -> str:
        """
        Generate a day consultation report PDF (Rap2).
        
        Args:
            data: Report data from the workflow
            filename: Optional custom filename
            style: Visual style for the report
            
        Returns:
            Path to the generated PDF file
        """
        if not REPORTLAB_AVAILABLE:
            return self._generate_fallback_report(data, "day", filename)
        
        # Generate filename
        patient_id = data.get("patient_id", "unknown")
        date_str = datetime.now().strftime("%Y%m%d_%H%M")
        filename = filename or f"rap2_day_{patient_id}_{date_str}.pdf"
        filepath = self.output_dir / filename
        
        # Create document
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        # Build content
        story = []
        
        # Header
        story.extend(self._build_header("Rapport de Consultation Medicale", data))
        
        # Night context if available
        if data.get("night_context"):
            story.extend(self._build_night_context_section(data))

        # Longitudinal insights from the two most recent sessions
        story.extend(self._build_history_evolution_section(data))
        
        # Chief complaint
        story.extend(self._build_complaint_section(data))
        
        # Clinical exam
        story.extend(self._build_exam_section(data))
        
        # AI Analysis
        story.extend(self._build_ai_analysis_section(data))
        
        # Treatment plan
        story.extend(self._build_treatment_section(data))
        
        # Footer
        story.extend(self._build_footer())
        
        # Build PDF
        doc.build(story)
        
        return str(filepath)
    
    def _build_header(self, title: str, data: Dict[str, Any]) -> List:
        """Build report header"""
        elements = []
        
        # Title
        elements.append(Paragraph(title, self.styles['ReportTitle']))
        elements.append(Paragraph(
            "MedGemma Sentinel - The Scribe",
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 10))
        
        # Horizontal line
        elements.append(HRFlowable(
            width="100%",
            thickness=2,
            color=self.COLORS["primary"],
            spaceAfter=15
        ))
        
        # Patient info table
        patient_data = [
            ["Patient:", data.get("patient_name", "N/A"), "ID:", data.get("patient_id", "N/A")],
            ["Chambre:", data.get("room", "N/A"), "Date:", datetime.now().strftime("%d/%m/%Y")],
        ]
        
        info_table = Table(patient_data, colWidths=[3*cm, 5*cm, 2*cm, 5*cm])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (-1, -1), self.COLORS["text"]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(info_table)
        elements.append(Spacer(1, 15))
        
        return elements
    
    def _parse_markdown_to_elements(self, md_text: str) -> List:
        """Convert Markdown text (from MedGemma) into ReportLab flowable elements.
        
        Handles: **bold**, *italic*, ## headings, bullet/numbered lists,
        horizontal rules (---), and plain paragraphs.
        """
        elements = []
        if not md_text:
            return elements

        def _md_inline(text: str) -> str:
            """Convert inline markdown to ReportLab XML markup."""
            # Bold: **text** or __text__
            text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
            text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)
            # Italic: *text* or _text_ (but not inside bold tags)
            text = re.sub(r'(?<!\w)\*(?!\*)(.+?)\*(?!\*)', r'<i>\1</i>', text)
            text = re.sub(r'(?<!\w)_(?!_)(.+?)_(?!_)', r'<i>\1</i>', text)
            # Backtick code: `text`
            text = re.sub(r'`(.+?)`', r'<font face="Courier">\1</font>', text)
            # Escape any remaining angle-bracket issues for ReportLab
            return text

        lines = md_text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                elements.append(Spacer(1, 4))
                i += 1
                continue

            # Horizontal rule
            if re.match(r'^-{3,}$|^\*{3,}$|^_{3,}$', stripped):
                elements.append(HRFlowable(
                    width="100%", thickness=0.5, color=self.COLORS["muted"], spaceAfter=6
                ))
                i += 1
                continue

            # Headings: ## or ###
            heading_match = re.match(r'^(#{1,4})\s+(.+)$', stripped)
            if heading_match:
                level = len(heading_match.group(1))
                heading_text = _md_inline(heading_match.group(2))
                if level <= 2:
                    elements.append(Paragraph(heading_text, self.styles['SectionTitle']))
                else:
                    elements.append(Paragraph(heading_text, self.styles['SubSection']))
                i += 1
                continue

            # Bullet list: - item or * item (not bold **)
            bullet_match = re.match(r'^\s*[-*+]\s+(.+)$', stripped)
            if bullet_match and not stripped.startswith('**'):
                bullet_text = _md_inline(bullet_match.group(1))
                elements.append(Paragraph(
                    f"\u2022  {bullet_text}", self.styles['BodyText']
                ))
                i += 1
                continue

            # Numbered list: 1. item
            num_match = re.match(r'^\s*(\d+)\.\s+(.+)$', stripped)
            if num_match:
                num = num_match.group(1)
                item_text = _md_inline(num_match.group(2))
                elements.append(Paragraph(
                    f"{num}.  {item_text}", self.styles['BodyText']
                ))
                i += 1
                continue

            # Regular paragraph — collect consecutive non-blank non-special lines
            para_lines = []
            while i < len(lines):
                l = lines[i].strip()
                if not l or re.match(r'^#{1,4}\s', l) or re.match(r'^-{3,}$', l):
                    break
                # Stop if next line is a list item (unless it's mid-paragraph)
                if re.match(r'^\s*[-*+]\s+', l) and not l.startswith('**'):
                    break
                if re.match(r'^\s*\d+\.\s+', l):
                    break
                para_lines.append(l)
                i += 1

            if para_lines:
                full_text = _md_inline(' '.join(para_lines))
                elements.append(Paragraph(full_text, self.styles['BodyText']))
            continue

        return elements

    def _build_executive_summary(self, data: Dict[str, Any]) -> List:
        """Build executive summary section"""
        elements = []
        
        elements.append(Paragraph("Resume Executif", self.styles['SectionTitle']))
        
        summary = data.get("summary", "Aucun résumé disponible.")
        # Parse MedGemma markdown output into proper PDF elements
        elements.extend(self._parse_markdown_to_elements(summary))
        elements.append(Spacer(1, 10))
        
        return elements
    
    def _build_alerts_section(self, data: Dict[str, Any]) -> List:
        """Build alerts and events section"""
        elements = []
        
        elements.append(Paragraph("Alertes et Evenements", self.styles['SectionTitle']))
        
        events = data.get("events", [])
        night_data = data.get("night_data", {})
        
        # Statistics
        critical = len([e for e in events if e.get("level") == "critical"])
        high = len([e for e in events if e.get("level") == "high"])
        medium = len([e for e in events if e.get("level") == "medium"])
        
        stats_data = [
            ["Total", "Critiques", "Elevees", "Moderees"],
            [str(len(events)), str(critical), str(high), str(medium)]
        ]
        
        stats_table = Table(stats_data, colWidths=[4*cm, 4*cm, 4*cm, 4*cm])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.COLORS["primary"]),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('BACKGROUND', (0, 1), (-1, 1), self.COLORS["light_bg"]),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (-1, 1), 14),
        ]))
        
        elements.append(stats_table)
        elements.append(Spacer(1, 10))
        
        # Critical events detail
        critical_events = [e for e in events if e.get("level") == "critical"]
        if critical_events:
            elements.append(Paragraph("Événements critiques:", self.styles['SubSection']))
            for event in critical_events:
                alert_text = f"ALERTE - {event.get('type', 'Evenement')}: {event.get('description', '')}"
                alert_box = Table([[Paragraph(alert_text, self.styles['AlertCritical'])]], colWidths=[16*cm])
                alert_box.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), self.COLORS["danger"]),
                    ('LEFTPADDING', (0, 0), (-1, -1), 8),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ]))
                elements.append(alert_box)
                elements.append(Spacer(1, 6))
        
        elements.append(Spacer(1, 10))
        return elements
    
    def _build_vitals_section(self, data: Dict[str, Any]) -> List:
        """Build vital signs section"""
        elements = []
        
        elements.append(Paragraph("Constantes Vitales", self.styles['SectionTitle']))
        
        vitals_summary = data.get("vitals_summary", {})
        
        if vitals_summary:
            vitals_data = [["Paramètre", "Min", "Max", "Moyenne", "Anomalies"]]
            
            for param, values in vitals_summary.items():
                anomalies = "OK" if values.get("anomalies", 0) == 0 else f"! {values.get('anomalies', 0)}"
                vitals_data.append([
                    param,
                    str(values.get("min", "-")),
                    str(values.get("max", "-")),
                    str(values.get("avg", "-")),
                    anomalies
                ])
            
            vitals_table = Table(vitals_data, colWidths=[4*cm, 2.5*cm, 2.5*cm, 2.5*cm, 3*cm])
            vitals_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.COLORS["secondary"]),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.COLORS["light_bg"]]),
            ]))
            
            elements.append(vitals_table)
        else:
            elements.append(Paragraph("Aucune donnée de constantes disponible.", self.styles['BodyText']))
        
        elements.append(Spacer(1, 10))
        return elements
    
    def _build_sleep_section(self, data: Dict[str, Any]) -> List:
        """Build sleep quality section"""
        elements = []
        
        elements.append(Paragraph("Qualite du Sommeil", self.styles['SectionTitle']))
        
        night_data = data.get("night_data", {})
        score = night_data.get("sleep_quality_score", 0) or 0
        
        quality = "Excellente" if score > 80 else \
                  "Bonne" if score > 60 else \
                  "Modérée" if score > 40 else "Mauvaise"
        
        sleep_data = [
            ["Score Global", f"{int(score)}/100"],
            ["Qualité", quality],
            ["Interruptions", str(night_data.get("alerts_triggered", 0))],
        ]
        
        sleep_table = Table(sleep_data, colWidths=[6*cm, 6*cm])
        sleep_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (-1, -1), self.COLORS["light_bg"]),
        ]))
        
        elements.append(sleep_table)
        elements.append(Spacer(1, 10))
        
        return elements
    
    def _build_recommendations_section(self, data: Dict[str, Any]) -> List:
        """Build recommendations section"""
        elements = []
        
        elements.append(Paragraph("Recommandations", self.styles['SectionTitle']))
        
        recommendations = data.get("recommendations", [
            "Poursuivre surveillance standard",
            "Réévaluer selon évolution clinique"
        ])
        
        for rec in recommendations:
            elements.append(Paragraph(f"* {rec}", self.styles['BodyText']))
        
        elements.append(Spacer(1, 10))
        return elements

    def _build_history_evolution_section(self, data: Dict[str, Any]) -> List:
        """Build optional cross-session evolution insights section."""
        elements = []
        insights = data.get("history_evolution_insights", "")
        if not insights:
            return elements

        elements.append(Paragraph("Evolution Inter-Cycles (2 sessions)", self.styles['SectionTitle']))
        elements.extend(self._parse_markdown_to_elements(insights))

        sources = data.get("history_sources", [])
        if sources:
            source_text = ", ".join(str(source) for source in sources)
            elements.append(Paragraph(f"<b>Sources sessions:</b> {source_text}", self.styles['BodyText']))

        elements.append(Spacer(1, 10))
        return elements
    
    def _build_plots_section(self, data: Dict[str, Any]) -> List:
        """Build clinical plots section for night report (vitals trends, events)."""
        elements = []
        
        if not PLOTS_AVAILABLE or not REPORTLAB_AVAILABLE:
            return elements
        
        vitals_timeline = data.get("vitals_timeline", [])
        events = data.get("events", [])
        patient_id = data.get("patient_id", "unknown")
        
        if not vitals_timeline and not events:
            return elements
        
        elements.append(Paragraph("Graphiques de Tendances", self.styles['SectionTitle']))
        elements.append(Spacer(1, 5))
        
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate all plots at once
        plot_paths = generate_night_report_plots(
            vitals_timeline=vitals_timeline,
            events=events,
            output_dir=str(plots_dir),
            patient_id=patient_id
        )
        
        # Vitals dashboard (combined 3-panel chart)
        if "vitals_dashboard" in plot_paths:
            try:
                elements.append(Paragraph("Tableau de Bord des Constantes", self.styles['Heading2']))
                img = Image(plot_paths["vitals_dashboard"], width=16*cm, height=13*cm)
                img.hAlign = 'CENTER'
                elements.append(img)
                elements.append(Spacer(1, 8))
            except Exception:
                pass
        
        # Individual trend plots
        individual_plots = [
            ("spo2_trend", "Tendance SpO2", 14, 6),
            ("heart_rate_trend", "Tendance Frequence Cardiaque", 14, 6),
            ("temperature_trend", "Tendance Temperature", 14, 6),
        ]
        for plot_key, title, w, h in individual_plots:
            if plot_key in plot_paths:
                try:
                    elements.append(Paragraph(title, self.styles['Heading2']))
                    img = Image(plot_paths[plot_key], width=w*cm, height=h*cm)
                    img.hAlign = 'CENTER'
                    elements.append(img)
                    elements.append(Spacer(1, 8))
                except Exception:
                    pass
        
        # Events timeline
        if "events_timeline" in plot_paths:
            try:
                elements.append(Paragraph("Chronologie des Evenements", self.styles['Heading2']))
                img = Image(plot_paths["events_timeline"], width=14*cm, height=5*cm)
                img.hAlign = 'CENTER'
                elements.append(img)
                elements.append(Spacer(1, 8))
            except Exception:
                pass
        
        # Severity distribution
        if "severity_distribution" in plot_paths:
            try:
                elements.append(Paragraph("Distribution des Alertes", self.styles['Heading2']))
                img = Image(plot_paths["severity_distribution"], width=8*cm, height=7*cm)
                img.hAlign = 'CENTER'
                elements.append(img)
                elements.append(Spacer(1, 8))
            except Exception:
                pass
        
        # Store paths for reference
        data["_plot_paths"] = plot_paths
        
        return elements
    
    def _build_night_context_section(self, data: Dict[str, Any]) -> List:
        """Build night context section for day report"""
        elements = []
        
        elements.append(Paragraph("Contexte Nocturne", self.styles['SectionTitle']))
        
        night_context = data.get("night_context", "Pas de données nocturnes disponibles.")
        # Parse MedGemma markdown output into proper PDF elements
        elements.extend(self._parse_markdown_to_elements(night_context))
        elements.append(Spacer(1, 10))
        
        return elements
    
    def _build_complaint_section(self, data: Dict[str, Any]) -> List:
        """Build chief complaint section"""
        elements = []
        
        elements.append(Paragraph("Motif de Consultation", self.styles['SectionTitle']))
        
        day_data = data.get("day_data", {})
        complaint = day_data.get("presenting_complaint", "Non spécifié")
        
        elements.append(Paragraph(f"<b>Plainte principale:</b> {complaint}", self.styles['BodyText']))
        
        symptoms = day_data.get("symptoms", [])
        if symptoms:
            elements.append(Paragraph("<b>Symptômes associés:</b>", self.styles['BodyText']))
            for symptom in symptoms:
                elements.append(Paragraph(f"  • {symptom}", self.styles['BodyText']))
        
        elements.append(Spacer(1, 10))
        return elements
    
    def _build_exam_section(self, data: Dict[str, Any]) -> List:
        """Build clinical examination section"""
        elements = []
        
        elements.append(Paragraph("Examen Clinique", self.styles['SectionTitle']))
        
        day_data = data.get("day_data", {})
        
        # Vitals
        vitals = day_data.get("vitals", {})
        if vitals:
            elements.append(Paragraph("<b>Constantes:</b>", self.styles['BodyText']))
            for param, value in vitals.items():
                elements.append(Paragraph(f"  • {param}: {value}", self.styles['BodyText']))
        
        # Physical exam
        exam = day_data.get("physical_exam", {})
        if exam:
            elements.append(Paragraph("<b>Examen physique:</b>", self.styles['BodyText']))
            for system, finding in exam.items():
                elements.append(Paragraph(f"  • {system}: {finding}", self.styles['BodyText']))
        
        elements.append(Spacer(1, 10))
        return elements
    
    def _build_ai_analysis_section(self, data: Dict[str, Any]) -> List:
        """Build AI analysis section"""
        elements = []
        
        elements.append(Paragraph("Analyse IA (MedGemma)", self.styles['SectionTitle']))
        
        day_data = data.get("day_data", {})
        
        # Differential diagnosis
        differentials = day_data.get("differential_diagnosis", [])
        if differentials:
            elements.append(Paragraph("<b>Diagnostics différentiels:</b>", self.styles['BodyText']))
            for i, dx in enumerate(differentials, 1):
                elements.append(Paragraph(f"  {i}. {dx}", self.styles['BodyText']))
        
        # Severity
        severity = day_data.get("severity_assessment", "Modérée")
        severity_style = self.styles['AlertWarning'] if severity in ["Élevée", "Critique"] else self.styles['BodyText']
        elements.append(Paragraph(f"<b>Gravité évaluée:</b> {severity}", severity_style))
        
        elements.append(Spacer(1, 10))
        return elements
    
    def _build_treatment_section(self, data: Dict[str, Any]) -> List:
        """Build treatment plan section"""
        elements = []
        
        elements.append(Paragraph("Plan de Prise en Charge", self.styles['SectionTitle']))
        
        day_data = data.get("day_data", {})
        actions = day_data.get("recommended_actions", [])
        
        if actions:
            for action in actions:
                elements.append(Paragraph(f"  ✓ {action}", self.styles['BodyText']))
        else:
            elements.append(Paragraph("À définir selon évolution clinique.", self.styles['BodyText']))
        
        elements.append(Spacer(1, 10))
        return elements
    
    def _build_footer(self) -> List:
        """Build report footer"""
        elements = []
        
        elements.append(Spacer(1, 20))
        elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["muted"]))
        elements.append(Spacer(1, 10))
        
        footer_text = """
        <b>Rapport généré automatiquement par MedGemma Sentinel</b><br/>
        Ce rapport est un outil d'aide à la décision. Il ne remplace pas l'évaluation clinique 
        par un professionnel de santé qualifié.<br/>
        Version 1.0 | Système: MedGemma Sentinel - The Scribe
        """
        
        elements.append(Paragraph(footer_text, self.styles['Footer']))
        
        return elements
    
    def _generate_fallback_report(
        self,
        data: Dict[str, Any],
        report_type: str,
        filename: Optional[str]
    ) -> str:
        """Generate a text fallback when ReportLab is not available"""
        patient_id = data.get("patient_id", "unknown")
        date_str = datetime.now().strftime("%Y%m%d_%H%M")
        filename = filename or f"rap_{report_type}_{patient_id}_{date_str}.txt"
        filepath = self.output_dir / filename
        
        # Simple text report
        content = f"""
=================================================================
           MEDGEMMA SENTINEL - RAPPORT {'DE NUIT' if report_type == 'night' else 'DE CONSULTATION'}
=================================================================

Patient: {data.get('patient_name', 'N/A')}
ID: {data.get('patient_id', 'N/A')}
Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}

-----------------------------------------------------------------
RÉSUMÉ
-----------------------------------------------------------------
{data.get('summary', 'Aucun résumé disponible.')}

-----------------------------------------------------------------
RECOMMANDATIONS
-----------------------------------------------------------------
{chr(10).join(['- ' + r for r in data.get('recommendations', ['Surveillance standard'])])}

=================================================================
Rapport généré par MedGemma Sentinel - The Scribe
Ce rapport ne remplace pas l'évaluation clinique professionnelle.
=================================================================
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(filepath)
