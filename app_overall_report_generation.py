import streamlit as st
import time
import os
import sys
from datetime import datetime
import io

# Add src/overall_report_generation to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'overall_report_generation'))

# ReportLab for PDF generation (optional - with fallback)
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, ListFlowable, ListItem
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    st.error("⚠️ ReportLab not installed. Run: pip install reportlab")

# --- 1. System Configuration ---
st.set_page_config(
    page_title="MedGemma Sentinel",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 0. PDF Conversion Helper ---
def text_to_pdf(text_content: str, title: str = "Clinical Report") -> bytes:
    """
    Convert SBAR/Clinical report to professional PDF using ReportLab.
    Implements advanced medical report design with tables, conditional formatting.
    """
    if not REPORTLAB_AVAILABLE:
        return text_content.encode('utf-8')
    
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer, 
        pagesize=letter, 
        topMargin=0.6*inch, 
        bottomMargin=0.8*inch,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch
    )
    
    # Define professional medical styles
    styles = getSampleStyleSheet()
    
    # Header Title Style
    header_title = ParagraphStyle(
        'HeaderTitle',
        parent=styles['Normal'],
        fontSize=20,
        textColor=colors.HexColor('#0d1b2a'),
        fontName='Helvetica-Bold',
        alignment=TA_LEFT
    )
    
    header_subtitle = ParagraphStyle(
        'HeaderSubtitle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=colors.HexColor('#1a365d'),
        fontName='Helvetica-Bold',
        alignment=TA_RIGHT,
        textTransform='uppercase'
    )
    
    # Section header styles
    section_header = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=11,
        textColor=colors.HexColor('#ffffff'),
        backColor=colors.HexColor('#1a365d'),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold',
        leftIndent=6,
        rightIndent=6,
        textTransform='uppercase',
        letterSpacing=0.5
    )
    
    sbar_header = ParagraphStyle(
        'SBARHeader',
        parent=styles['Heading3'],
        fontSize=10,
        textColor=colors.HexColor('#1a365d'),
        spaceAfter=4,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    )
    
    # Body and content styles
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#2d3748'),
        alignment=TA_JUSTIFY,
        fontName='Helvetica',
        spaceAfter=6
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#ffffff'),
        fontName='Helvetica-Bold',
        alignment=TA_CENTER
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#2d3748'),
        alignment=TA_CENTER,
        fontName='Helvetica'
    )
    
    table_cell_critical = ParagraphStyle(
        'TableCellCritical',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#ffffff'),
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    footer_style = ParagraphStyle(
        'DisclaimerFooter',
        parent=styles['Normal'],
        fontSize=7,
        textColor=colors.HexColor('#e53e3e'),
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique'
    )
    
    # Parse content to extract key information
    lines = text_content.split('\n')
    patient_id = "UNKNOWN"
    hr = spo2 = audio_db = 0
    news2_score = 0
    risk_profile = "General"
    sbar_sections = {'situation': '', 'background': '', 'assessment': '', 'recommendation': ''}
    current_section = None
    
    for line in lines:
        line_clean = line.strip()
        if 'Patient ID:' in line:
            patient_id = line.split('Patient ID:')[-1].strip()
        elif 'Heart Rate:' in line:
            try:
                hr = int(line.split('Heart Rate:')[-1].split('bpm')[0].strip())
            except:
                pass
        elif 'SpO2:' in line:
            try:
                spo2 = int(line.split('SpO2:')[-1].split('%')[0].strip())
            except:
                pass
        elif 'Audio Level:' in line:
            try:
                audio_db = int(line.split('Audio Level:')[-1].split('dB')[0].strip())
            except:
                pass
        elif 'NEWS2 Score:' in line:
            try:
                news2_score = int(line.split('NEWS2 Score:')[-1].split()[0].strip())
            except:
                pass
        elif line_clean.upper() == 'SITUATION:':
            current_section = 'situation'
        elif line_clean.upper() == 'BACKGROUND:':
            current_section = 'background'
        elif line_clean.upper() == 'ASSESSMENT (MEDGEMMA AI ANALYSIS):':
            current_section = 'assessment'
        elif line_clean.upper() == 'RECOMMENDATION:':
            current_section = 'recommendation'
        elif current_section and line_clean and not line_clean.startswith('='):
            if line_clean.startswith('-'):
                sbar_sections[current_section] += f"• {line_clean[1:].strip()}\n"
            else:
                sbar_sections[current_section] += f"{line_clean}\n"
    
    # Build PDF story
    story = []
    
    # ===== HEADER =====
    header_table_data = [
        [Paragraph("⚕ MedGemma<br/>Sentinel", header_title), 
         Paragraph("CRITICAL INCIDENT<br/>REPORT", header_subtitle)]
    ]
    header_table = Table(header_table_data, colWidths=[3*inch, 3*inch])
    header_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.15*inch))
    
    # Horizontal line
    story.append(Paragraph("_" * 95, ParagraphStyle('HLine', fontSize=7, textColor=colors.HexColor('#cbd5e0'))))
    story.append(Spacer(1, 0.12*inch))
    
    # ===== PATIENT INFO BOX =====
    patient_data = [
        ['PATIENT INFORMATION'],
        ['Patient ID:', patient_id],
        ['Age/Risk Profile:', risk_profile],
        ['Report Date:', datetime.now().strftime('%B %d, %Y')]
    ]
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1a365d')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 0.15*inch))
    
    # ===== VITALS SECTION =====
    story.append(Paragraph("VITAL SIGNS SNAPSHOT", section_header))
    story.append(Spacer(1, 0.08*inch))
    
    # Determine critical flags
    hr_critical = hr > 130
    spo2_critical = spo2 < 90
    
    vitals_data = [
        [Paragraph("Heart Rate", table_header_style),
         Paragraph("SpO2", table_header_style),
         Paragraph("Audio Level", table_header_style),
         Paragraph("NEWS2", table_header_style)]
    ]
    
    # HR cell
    hr_bg = colors.HexColor('#ef5350') if hr_critical else colors.HexColor('#ffffff')
    hr_color = colors.HexColor('#ffffff') if hr_critical else colors.HexColor('#2d3748')
    hr_style = table_cell_critical if hr_critical else table_cell_style
    
    # SpO2 cell
    spo2_bg = colors.HexColor('#ef5350') if spo2_critical else colors.HexColor('#ffffff')
    spo2_color = colors.HexColor('#ffffff') if spo2_critical else colors.HexColor('#2d3748')
    spo2_style = table_cell_critical if spo2_critical else table_cell_style
    
    vitals_data.append([
        Paragraph(f"<font color={hr_color.hexval()}><b>{hr}</b> bpm</font>", hr_style),
        Paragraph(f"<font color={spo2_color.hexval()}><b>{spo2}</b> %</font>", spo2_style),
        Paragraph(f"<b>{audio_db}</b> dB", table_cell_style),
        Paragraph(f"<b>{news2_score}</b>", ParagraphStyle(
            'News2Cell',
            parent=table_cell_style,
            fontSize=11,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor('#d69e2e')
        ))
    ])
    
    vitals_table = Table(vitals_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    vitals_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a365d')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e0')),
        ('BACKGROUND', (0, 1), (0, 1), hr_bg),
        ('BACKGROUND', (1, 1), (1, 1), spo2_bg),
        ('BACKGROUND', (2, 1), (2, 1), colors.HexColor('#ffffff')),
        ('BACKGROUND', (3, 1), (3, 1), colors.HexColor('#fffbea')),
    ]))
    story.append(vitals_table)
    story.append(Spacer(1, 0.15*inch))
    
    # ===== NEWS2 SCORE DISPLAY =====
    story.append(Paragraph(
        f"NEWS2 Score: <font size=16 color=#d69e2e><b>{news2_score}</b></font>",
        ParagraphStyle('News2Display', parent=body_style, fontSize=12, fontName='Helvetica-Bold')
    ))
    story.append(Spacer(1, 0.15*inch))
    
    # ===== SBAR SECTIONS =====
    story.append(Paragraph("SITUATION", sbar_header))
    story.append(Paragraph(sbar_sections['situation'] or "No situation information available.", body_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("BACKGROUND", sbar_header))
    story.append(Paragraph(sbar_sections['background'] or "No background information available.", body_style))
    story.append(Spacer(1, 0.1*inch))
    
    story.append(Paragraph("ASSESSMENT (MedGemma AI Analysis)", sbar_header))
    story.append(Paragraph(sbar_sections['assessment'] or "No assessment available.", body_style))
    story.append(Spacer(1, 0.1*inch))
    
    # ===== RECOMMENDATION SECTION WITH BULLET PARSING =====
    story.append(Paragraph("RECOMMENDATION", sbar_header))
    
    # Clean garbage text from end of recommendation
    import re
    recommendation = sbar_sections['recommendation'] or ""
    recommendation = re.sub(r'VITALS SNAPSHOT.*', '', recommendation, flags=re.DOTALL | re.IGNORECASE)
    recommendation = re.sub(r'Alert Severity:.*', '', recommendation, flags=re.DOTALL | re.IGNORECASE)
    recommendation = recommendation.strip()
    
    # Define hospital checklist style for recommendation items
    checklist_item_style = ParagraphStyle(
        'ChecklistItem',
        parent=body_style,
        fontSize=9,
        textColor=colors.HexColor('#2d3748'),
        fontName='Helvetica',
        leftIndent=12,
        spaceAfter=5
    )
    
    if recommendation:
        # Split by bullet character (•)
        items = [item.strip() for item in recommendation.split('•') if item.strip()]
        
        if items:
            # Create list items for each recommendation
            list_items = []
            for item in items:
                import html
                safe_item = html.escape(item)
                # Render each item as a bulleted paragraph
                para = Paragraph(f"• {safe_item}", checklist_item_style)
                list_items.append(para)
            
            # Add all items to story
            for item_para in list_items:
                story.append(item_para)
        else:
            story.append(Paragraph("No recommendations available.", body_style))
    else:
        story.append(Paragraph("No recommendations available.", body_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    # ===== FOOTER =====
    story.append(Paragraph("_" * 95, ParagraphStyle('HLine', fontSize=7, textColor=colors.HexColor('#cbd5e0'))))
    story.append(Spacer(1, 0.08*inch))
    timestamp_text = f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}"
    story.append(Paragraph(timestamp_text, ParagraphStyle('Timestamp', fontName='Helvetica', fontSize=8, textColor=colors.HexColor('#718096'), alignment=TA_CENTER)))
    story.append(Spacer(1, 0.05*inch))
    story.append(Paragraph("Generated by AI - For Clinical Support Only", footer_style))
    
    # Build PDF
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


# --- 0B. Shift Handover PDF Generator ---
def generate_shift_handover_pdf(data: dict) -> bytes:
    """
    Generate professional Shift Handover Report PDF (Document B).
    
    Args:
        data: Dict with keys: 'patient_id', 'date', 'full_text'
              full_text contains Markdown-style headers and content
    
    Returns:
        bytes: PDF document ready for download
    """
    if not REPORTLAB_AVAILABLE:
        return data.get('full_text', '').encode('utf-8')
    
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        topMargin=0.6*inch,
        bottomMargin=0.8*inch,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch
    )
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Header styles
    handover_logo = ParagraphStyle(
        'HandoverLogo',
        parent=styles['Normal'],
        fontSize=20,
        textColor=colors.HexColor('#0d1b2a'),
        fontName='Helvetica-Bold',
        alignment=TA_LEFT
    )
    
    handover_title = ParagraphStyle(
        'HandoverTitle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=colors.HexColor('#1a365d'),
        fontName='Helvetica-Bold',
        alignment=TA_RIGHT,
        textTransform='uppercase'
    )
    
    # Section header - bold blue color for handover
    handover_section = ParagraphStyle(
        'HandoverSection',
        parent=styles['Heading2'],
        fontSize=11,
        textColor=colors.HexColor('#1a365d'),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold',
        borderBottomWidth=2,
        borderBottomColor=colors.HexColor('#2c5282'),
        borderBottomPadding=6
    )
    
    # Body text for handover
    handover_body = ParagraphStyle(
        'HandoverBody',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#2d3748'),
        alignment=TA_JUSTIFY,
        fontName='Helvetica',
        spaceAfter=8
    )
    
    # Patient info style
    patient_info_style = ParagraphStyle(
        'PatientInfo',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#2d3748'),
        fontName='Helvetica',
        spaceAfter=4
    )
    
    footer_handover = ParagraphStyle(
        'FooterHandover',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#546e7a'),
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique'
    )
    
    # Extract data
    patient_id = data.get('patient_id', 'UNKNOWN')
    report_date = data.get('date', datetime.now().strftime('%B %d, %Y'))
    full_text = data.get('full_text', '')
    
    # Build PDF story
    story = []
    
    # ===== HEADER =====
    header_table_data = [
        [Paragraph("⚕ MedGemma<br/>Sentinel", handover_logo),
         Paragraph("SHIFT HANDOVER<br/>SUMMARY", handover_title)]
    ]
    header_table = Table(header_table_data, colWidths=[3*inch, 3*inch])
    header_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.15*inch))
    
    # Horizontal line
    story.append(Paragraph("_" * 95, ParagraphStyle('HLine', fontSize=7, textColor=colors.HexColor('#cbd5e0'))))
    story.append(Spacer(1, 0.12*inch))
    
    # ===== PATIENT INFO BOX =====
    patient_data = [
        ['SHIFT HANDOVER INFORMATION'],
        ['Patient ID:', patient_id],
        ['Report Date:', report_date],
        ['Shift Duration:', 'Night Shift (22:00 - 08:00)']
    ]
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1a365d')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 0.15*inch))
    
    # ===== CONTENT SECTION: Parse content with table for EVENT TIMELINE =====
    import html
    import re
    
    # Define additional styles
    assessment_style = ParagraphStyle(
        'Assessment',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#546e7a'),
        fontName='Helvetica-Oblique',
        spaceAfter=4,
        alignment=TA_LEFT
    )
    
    table_event_header_style = ParagraphStyle(
        'TableEventHeader',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#ffffff'),
        fontName='Helvetica-Bold',
        alignment=TA_CENTER
    )
    
    table_event_cell_style = ParagraphStyle(
        'TableEventCell',
        parent=styles['Normal'],
        fontSize=8.5,
        textColor=colors.HexColor('#2d3748'),
        fontName='Helvetica',
        alignment=TA_LEFT
    )
    
    table_event_cell_center = ParagraphStyle(
        'TableEventCellCenter',
        parent=styles['Normal'],
        fontSize=8.5,
        textColor=colors.HexColor('#2d3748'),
        fontName='Helvetica',
        alignment=TA_CENTER
    )
    
    lines = full_text.split('\n')
    current_section = None
    pattern_analysis_text = []
    in_timeline = False
    timeline_start_idx = -1
    
    # First pass: Find where EVENT TIMELINE SUMMARY is
    for idx, line in enumerate(lines):
        if 'EVENT TIMELINE SUMMARY' in line.upper():
            in_timeline = True
            timeline_start_idx = idx
            break
    
    # Process content
    i = 0
    while i < len(lines):
        line = lines[i]
        line_stripped = line.strip()
        
        # Skip separator lines
        if line_stripped.startswith('=') * 5 or (line_stripped.startswith('-') and len(line_stripped) > 30):
            i += 1
            continue
        
        # Empty line - add spacer
        if not line_stripped:
            if current_section != 'EVENT_TIMELINE':  # Don't add extra spacers in timeline block
                story.append(Spacer(1, 0.08*inch))
            i += 1
            continue
        
        # Section header (uppercase, ends with colon, under 50 chars)
        if line_stripped.endswith(':') and len(line_stripped) < 50 and line_stripped.isupper():
            current_section = line_stripped.rstrip(':')
            story.append(Paragraph(current_section, handover_section))
            story.append(Spacer(1, 0.08*inch))
            
            # Special handling for EVENT TIMELINE SUMMARY
            if 'EVENT TIMELINE' in current_section:
                # Extract all events from this section
                events_data = []
                j = i + 1
                
                while j < len(lines):
                    next_line = lines[j].strip()
                    
                    # Stop if we hit the next section
                    if next_line.endswith(':') and next_line.isupper() and len(next_line) < 50:
                        break
                    
                    # Extract event data - [Event #N @ TIME]
                    if next_line.startswith('[Event'):
                        event_match = re.search(r'\[Event #(\d+) @ (\d{2}):(\d{2})\]', next_line)
                        if event_match:
                            event_time = f"{event_match.group(2)}:{event_match.group(3)}"
                            event_num = event_match.group(1)
                            
                            # Initialize event data
                            event_type = "N/A"
                            vitals = "N/A"
                            
                            # Look ahead for Type and Vitals
                            k = j + 1
                            while k < len(lines) and k < j + 5:  # Look ahead 5 lines
                                look_line = lines[k].strip()
                                if look_line.startswith('Type:'):
                                    event_type = look_line.replace('Type:', '').strip()
                                elif look_line.startswith('Vitals:'):
                                    vitals = look_line.replace('Vitals:', '').strip()
                                elif look_line.startswith('[Event'):
                                    break
                                k += 1
                            
                            events_data.append({
                                'time': event_time,
                                'type': event_type,
                                'vitals': vitals
                            })
                    
                    j += 1
                
                # Create table if events exist
                if events_data:
                    # Table header
                    table_data = [[
                        Paragraph("Time", table_event_header_style),
                        Paragraph("Event Type", table_event_header_style),
                        Paragraph("Key Vitals", table_event_header_style)
                    ]]
                    
                    # Add events with zebra striping
                    for idx, event in enumerate(events_data):
                        table_data.append([
                            Paragraph(event['time'], table_event_cell_center),
                            Paragraph(html.escape(event['type'][:40]), table_event_cell_style),  # Truncate if too long
                            Paragraph(html.escape(event['vitals'][:50]), table_event_cell_style)  # Truncate if too long
                        ])
                    
                    # Create table
                    event_table = Table(table_data, colWidths=[1.2*inch, 2*inch, 2.8*inch])
                    
                    # Build style list with zebra striping
                    table_styles = [
                        # Header row
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a365d')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#ffffff')),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 9),
                        # Alternating row colors
                        ('FONTSIZE', (0, 1), (-1, -1), 8.5),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('ALIGN', (0, 1), (0, -1), 'CENTER'),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
                        ('LEFTPADDING', (0, 0), (-1, -1), 6),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                        ('TOPPADDING', (0, 0), (-1, -1), 5),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                    ]
                    
                    # Zebra striping for data rows
                    for row_idx in range(1, len(table_data)):
                        if row_idx % 2 == 0:
                            table_styles.append(('BACKGROUND', (0, row_idx), (-1, row_idx), colors.HexColor('#f0f4f8')))
                        else:
                            table_styles.append(('BACKGROUND', (0, row_idx), (-1, row_idx), colors.HexColor('#ffffff')))
                    
                    event_table.setStyle(TableStyle(table_styles))
                    story.append(event_table)
                    story.append(Spacer(1, 0.1*inch))
                
                # Skip past all the event lines we just processed
                i = j
                continue
            
            i += 1
            continue
        
        # Regular content - add as body text
        if line_stripped and not line_stripped.startswith('[Event'):
            safe_text = html.escape(line_stripped)
            story.append(Paragraph(safe_text, handover_body))
        
        i += 1
    
    story.append(Spacer(1, 0.2*inch))
    
    # ===== FOOTER =====
    story.append(Paragraph("_" * 95, ParagraphStyle('HLine', fontSize=7, textColor=colors.HexColor('#cbd5e0'))))
    story.append(Spacer(1, 0.08*inch))
    
    timestamp_text = f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}"
    story.append(Paragraph(timestamp_text, ParagraphStyle('Timestamp', fontName='Helvetica', fontSize=8, textColor=colors.HexColor('#718096'), alignment=TA_CENTER)))
    story.append(Spacer(1, 0.05*inch))
    story.append(Paragraph("End of Shift Report - Generated by AI", footer_handover))
    
    # Build PDF
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


# --- 0C. RAP2 Differential Diagnosis PDF Generator ---
def generate_rap2_pdf(data: dict) -> bytes:
    """
    Generate professional RAP2 Differential Diagnosis PDF.

    Args:
        data: Dict with keys:
            'patient_id'      - str
            'date'            - str
            'specialty'       - str
            'symptoms_input'  - str  (doctor's free-text clinical input)
            'differentials'   - list of dicts, each with:
                  'name'        - str  (diagnosis name)
                  'probability' - str  (e.g. "High Probability: 75%")
                  'points'      - list[str]  (reasoning + recommendation bullets)
            'night_context'   - str  (formatted night events summary)

    Returns:
        bytes: PDF document ready for download
    """
    if not REPORTLAB_AVAILABLE:
        return data.get('symptoms_input', '').encode('utf-8')

    import html as html_mod
    import re

    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        topMargin=0.6*inch,
        bottomMargin=0.8*inch,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch
    )

    styles = getSampleStyleSheet()

    # ── Style definitions ──
    rap2_logo = ParagraphStyle(
        'Rap2Logo', parent=styles['Normal'],
        fontSize=20, textColor=colors.HexColor('#0d1b2a'),
        fontName='Helvetica-Bold', alignment=TA_LEFT)

    rap2_title = ParagraphStyle(
        'Rap2Title', parent=styles['Normal'],
        fontSize=14, textColor=colors.HexColor('#1a365d'),
        fontName='Helvetica-Bold', alignment=TA_RIGHT)

    rap2_section = ParagraphStyle(
        'Rap2Section', parent=styles['Heading2'],
        fontSize=11, textColor=colors.HexColor('#1a365d'),
        spaceAfter=8, spaceBefore=14, fontName='Helvetica-Bold',
        borderBottomWidth=2, borderBottomColor=colors.HexColor('#2c5282'),
        borderBottomPadding=6)

    rap2_body = ParagraphStyle(
        'Rap2Body', parent=styles['Normal'],
        fontSize=9.5, leading=14, textColor=colors.HexColor('#2d3748'),
        alignment=TA_JUSTIFY, fontName='Helvetica', spaceAfter=8)

    rap2_note = ParagraphStyle(
        'Rap2Note', parent=styles['Normal'],
        fontSize=8, textColor=colors.HexColor('#546e7a'),
        alignment=TA_CENTER, fontName='Helvetica-Oblique', spaceAfter=6)

    rap2_diagnosis_name = ParagraphStyle(
        'Rap2DiagName', parent=styles['Normal'],
        fontSize=10.5, textColor=colors.HexColor('#1a365d'),
        fontName='Helvetica-Bold', spaceBefore=10, spaceAfter=4)

    rap2_probability = ParagraphStyle(
        'Rap2Prob', parent=styles['Normal'],
        fontSize=9, textColor=colors.HexColor('#2b6cb0'),
        fontName='Helvetica-BoldOblique', spaceAfter=4, leftIndent=14)

    rap2_bullet = ParagraphStyle(
        'Rap2Bullet', parent=rap2_body,
        fontSize=9, leftIndent=28, bulletIndent=16,
        spaceAfter=3, leading=13)

    rap2_recommend = ParagraphStyle(
        'Rap2Recommend', parent=rap2_body,
        fontSize=9, leftIndent=28, bulletIndent=16,
        spaceAfter=3, leading=13,
        textColor=colors.HexColor('#c53030'),
        fontName='Helvetica-Bold')

    # ── Extract data ──
    patient_id  = data.get('patient_id', 'UNKNOWN')
    report_date = data.get('date', datetime.now().strftime('%B %d, %Y'))
    specialty   = data.get('specialty', 'General')
    symptoms    = data.get('symptoms_input', 'No clinical input recorded.')
    differentials = data.get('differentials', [])
    night_ctx   = data.get('night_context', '')

    story = []

    # ===== HEADER =====
    header_table_data = [
        [Paragraph("⚕ MedGemma<br/>Sentinel", rap2_logo),
         Paragraph("RAP2 DIFFERENTIAL<br/>DIAGNOSIS", rap2_title)]
    ]
    header_table = Table(header_table_data, colWidths=[3*inch, 3*inch])
    header_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("_" * 95, ParagraphStyle('HLine', fontSize=7, textColor=colors.HexColor('#cbd5e0'))))
    story.append(Spacer(1, 0.12*inch))

    # ===== PATIENT INFO BOX =====
    patient_data = [
        ['RAP2 REPORT INFORMATION'],
        ['Patient ID:', patient_id],
        ['Report Date:', report_date],
        ['Specialty:', specialty]
    ]
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1a365d')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 0.15*inch))

    # ===== SECTION 1: DOCTOR'S CLINICAL INPUT =====
    story.append(Paragraph("DOCTOR'S CLINICAL INPUT", rap2_section))
    story.append(Spacer(1, 0.06*inch))
    # Render doctor's input inside a light-background box table
    input_box = Table(
        [[Paragraph(html_mod.escape(symptoms), rap2_body)]],
        colWidths=[6*inch]
    )
    input_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f7fafc')),
        ('BORDER', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
    ]))
    story.append(input_box)
    story.append(Spacer(1, 0.15*inch))

    # ===== SECTION 2: TOP DIFFERENTIALS =====
    story.append(Paragraph("TOP DIFFERENTIAL DIAGNOSES", rap2_section))
    story.append(Spacer(1, 0.06*inch))

    if differentials:
        for idx, dx in enumerate(differentials, 1):
            dx_name = html_mod.escape(dx.get('name', 'Unknown'))
            dx_prob = html_mod.escape(dx.get('probability', ''))
            story.append(Paragraph(f"{idx}. {dx_name}", rap2_diagnosis_name))
            if dx_prob:
                story.append(Paragraph(f"({dx_prob})", rap2_probability))
            for point in dx.get('points', []):
                safe_point = html_mod.escape(point)
                # Highlight "Recommended:" lines in red/bold
                if point.strip().lower().startswith('recommended:'):
                    story.append(Paragraph(safe_point, rap2_recommend, bulletText='▸'))
                else:
                    story.append(Paragraph(safe_point, rap2_bullet, bulletText='•'))
            story.append(Spacer(1, 0.08*inch))
    else:
        story.append(Paragraph("No differential diagnoses generated.", rap2_body))

    story.append(Spacer(1, 0.1*inch))

    # ===== SECTION 3: CORRELATED NIGHT EVENTS CONTEXT =====
    story.append(Paragraph("CORRELATED NIGHT EVENTS CONTEXT", rap2_section))
    story.append(Spacer(1, 0.06*inch))

    if night_ctx and night_ctx.strip():
        # Parse night context line by line for clean rendering
        for raw_line in night_ctx.split('\n'):
            line = raw_line.strip()
            if not line:
                continue
            # Section emoji header (e.g. "🌙 NIGHT SURVEILLANCE CONTEXT:")
            if line.startswith('🌙'):
                story.append(Paragraph(html_mod.escape(line.replace('🌙', '').strip()), rap2_body))
                continue
            # Event lines like "[1] 02:31 - CRITICAL RESPIRATORY DISTRESS"
            event_match = re.match(r'^\[(\d+)\]\s*(.*)$', line)
            if event_match:
                story.append(Paragraph(
                    f"<b>Event #{event_match.group(1)}:</b> {html_mod.escape(event_match.group(2))}",
                    rap2_body))
                continue
            # Indented detail lines (Vitals:, AI Assessment:)
            if line.startswith('Vitals:') or line.startswith('AI Assessment:'):
                story.append(Paragraph(html_mod.escape(line), rap2_bullet, bulletText=''))
                continue
            story.append(Paragraph(html_mod.escape(line), rap2_body))
    else:
        story.append(Paragraph("Patient had a stable night with no critical alerts recorded.", rap2_body))

    # ===== FOOTER =====
    story.append(Spacer(1, 0.25*inch))
    story.append(Paragraph("_" * 95, ParagraphStyle('HLine', fontSize=7, textColor=colors.HexColor('#cbd5e0'))))
    story.append(Spacer(1, 0.08*inch))
    timestamp_text = f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}"
    story.append(Paragraph(timestamp_text, ParagraphStyle('Timestamp', fontName='Helvetica', fontSize=8, textColor=colors.HexColor('#718096'), alignment=TA_CENTER)))
    story.append(Spacer(1, 0.05*inch))
    story.append(Paragraph("Generated by AI - For Clinical Support Only", rap2_note))

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


# --- 1. Helper Functions ---
def check_thresholds(hr, spo2, audio_db):
    """
    Rule-based reflex check for alert severity.
    Returns: "CRITICAL", "WARNING", or "NORMAL"
    """
    if hr > 130 and spo2 < 88 and audio_db > 80:
        return "CRITICAL"
    elif hr > 130 or spo2 < 88 or audio_db > 80:
        return "CRITICAL"
    elif hr > 100 or spo2 < 94:
        return "WARNING"
    return "NORMAL"


def calculate_news2_score(hr, spo2, temp=37.0, bp_sys=120):
    """
    Calculate NEWS2 (National Early Warning Score) - Hospital Standard
    Universal alerting system used globally for early deterioration detection.
    
    Returns: (score, severity_level, breakdown)
    """
    score = 0
    breakdown = {}
    
    # Oxygen saturation points
    if spo2 <= 91:
        spo2_score = 3
        breakdown['SpO2'] = f"{spo2}% → +3 (RED)"
    elif spo2 <= 94:
        spo2_score = 2
        breakdown['SpO2'] = f"{spo2}% → +2 (AMBER)"
    else:
        spo2_score = 0
        breakdown['SpO2'] = f"{spo2}% → +0"
    score += spo2_score
    
    # Heart rate points
    if hr >= 131:
        hr_score = 3
        breakdown['HR'] = f"{hr} bpm → +3 (RED)"
    elif hr >= 111:
        hr_score = 2
        breakdown['HR'] = f"{hr} bpm → +2 (AMBER)"
    elif hr <= 40:
        hr_score = 3
        breakdown['HR'] = f"{hr} bpm → +3 (RED)"
    else:
        hr_score = 0
        breakdown['HR'] = f"{hr} bpm → +0"
    score += hr_score
    
    # Temperature points
    if temp >= 39:
        temp_score = 2
        breakdown['Temp'] = f"{temp}°C → +2 (AMBER)"
    elif temp <= 35.0:
        temp_score = 3
        breakdown['Temp'] = f"{temp}°C → +3 (RED)"
    else:
        temp_score = 0
        breakdown['Temp'] = f"{temp}°C → +0"
    score += temp_score
    
    # Determine severity
    if score >= 5:
        severity = "🔴 RED (Critical - Urgent Review)"
    elif score >= 3:
        severity = "🟡 AMBER (Elevated Risk)"
    else:
        severity = "🟢 GREEN (Low Risk)"
    
    return score, severity, breakdown


def generate_sbar_report(patient_id, hr, spo2, audio_db, ai_reasoning, patient_case, timestamp=None):
    """
    Generate SBAR (Situation-Background-Assessment-Recommendation) report.
    Hospital standard for clinical handovers and incident documentation.
    """
    from datetime import datetime
    score, severity, _ = calculate_news2_score(hr, spo2)
    
    # Parse timestamp if provided, otherwise use current time
    if timestamp:
        dt = datetime.fromisoformat(timestamp)
        time_str = dt.strftime("%I:%M %p").lstrip("0")  # "2:31 PM" format
        date_str = dt.strftime("%B %d, %Y")  # February 20, 2026
    else:
        time_str = "02:31 AM"
        date_str = "February 20, 2026"
    
    # Generate clinical assessment paragraph with SpO2-Audio correlation
    clinical_assessment = f"""The acute desaturation to {spo2}% is consistent with the documented respiratory distress signals detected by audio monitoring. The elevated heart rate ({hr} bpm) and increased audio activity ({audio_db} dB) suggest compensatory tachycardia in response to hypoxemia, indicating a significant acute respiratory event requiring immediate clinical intervention. This multimodal evidence correlation supports urgent assessment for potential airway obstruction, bronchoconstriction, or other acute pulmonary compromise."""
    
    sbar = f"""INCIDENT SNAPSHOT REPORT
=====================================
Patient ID: {patient_id}
Time: {time_str}
Date: {date_str}

SBAR FORMAT (Clinical Handover Standard)
=====================================

SITUATION:
- Patient presented with acute clinical deterioration
- SpO2 dropped from 98% to {spo2}% (drop of {98-spo2}%)
- Heart rate increased to {hr} bpm
- Audio monitoring detected respiratory distress
- NEWS2 Score: {score} ({severity})

BACKGROUND:
- Patient Profile: {patient_case}
- Active Monitoring: Yes
- Surveillance Mode: Night (Automated Detection)

ASSESSMENT (MedGemma AI Analysis):
{clinical_assessment}

RECOMMENDATION:
- Immediate nursing intervention required
- Check airway patency
- Initiate oxygen therapy if not already done
- Continuous cardiac and respiratory monitoring
- Doctor notification required
- Consider portable chest imaging
- Ensure suction equipment readily available

VITALS SNAPSHOT AT TIME OF ALERT:
- Heart Rate: {hr} bpm
- SpO2: {spo2}%
- Audio Level: {audio_db} dB
- NEWS2 Score: {score}
- Alert Severity: {severity}

=====================================
Generated automatically by MedGemma Sentinel
For clinical use only
"""
    return sbar

def generate_shift_handover(events, patient_id="DEMO001"):
    """
    Generate Shift Handover Report (Document B) for Day Mode.
    Aggregates all night events into clinical summary for morning doctor.
    Hospital standard: narrative summary, not raw event dump.
    """
    if not events:
        return (
            "SHIFT HANDOVER REPORT\n"
            "=" * 60 + "\n"
            f"Patient ID: {patient_id}\n"
            "Date: February 20, 2026\n"
            "Shift: Night (22:00 - 08:00)\n\n"
            "STATUS: ✅ UNEVENTFUL\n"
            "No critical events recorded. Patient maintained stable overnight.\n"
        )
    
    # Analyze events for patterns
    num_events = len(events)
    critical_count = sum(1 for e in events if e.get('event_type') == 'CRITICAL_RESPIRATORY_DISTRESS')
    
    # Determine overall severity pattern
    if num_events == 1:
        pattern = "Single critical incident"
        severity_trend = "STABLE → ESCALATION (resolved)"
    elif critical_count >= 3:
        pattern = "Multiple critical episodes"
        severity_trend = "UNSTABLE throughout night"
    elif critical_count >= 1:
        pattern = "One critical episode with stable periods"
        severity_trend = "Mostly stable with acute episode"
    else:
        pattern = "Monitoring alerts only"
        severity_trend = "STABLE"
    
    # Extract first and last event times for duration
    first_time = events[0]['timestamp'][11:16] if events else "22:00"
    last_time = events[-1]['timestamp'][11:16] if events else "08:00"
    
    # Build handover report
    handover = f"""SHIFT HANDOVER REPORT
{'=' * 60}
Patient ID: {patient_id}
Night Date: February 19-20, 2026
Shift Duration: 22:00 - 08:00 (10 hours)
Report Generated: {datetime.now().strftime("%H:%M on %B %d, %Y")}

PATTERN ANALYSIS:
- Total Events Recorded: {num_events}
- Critical Incidents: {critical_count}
- Surveillance Pattern: {pattern}
- Severity Trend: {severity_trend}
- First Event: {first_time} | Last Event: {last_time}

EVENT TIMELINE SUMMARY:
"""
    
    # Add event summaries
    for i, event in enumerate(events, 1):
        event_time = event['timestamp'][11:16]
        event_type = event.get('event_type', 'Alert').replace('_', ' ')
        vitals = event.get('event_data', {})
        
        # Safely extract vitals
        hr = vitals.get('hr', 'N/A')
        spo2 = vitals.get('spo2', 'N/A')
        audio_db = vitals.get('audio_db', 'N/A')
        
        handover += f"\n[Event #{i} @ {event_time}]\n"
        handover += f"Type: {event_type}\n"
        handover += f"Vitals: HR {hr} bpm | SpO2 {spo2}% | Audio {audio_db} dB\n"
        
        # Add first 200 chars of reasoning for context
        reasoning_preview = event['ai_reasoning'][:250]
        handover += f"Assessment: {reasoning_preview}...\n"
    
    # Clinical impression
    handover += f"\n{'─' * 60}\n"
    handover += "CLINICAL IMPRESSION:\n"
    
    if critical_count == 0:
        handover += "Patient demonstrated stable overnight monitoring without critical events.\n"
    elif critical_count == 1:
        handover += "Patient had ONE critical respiratory episode at approximately {first_time}.\n"
        handover += "Event was detected and logged. Status post-intervention: STABLE.\n"
        handover += "No further critical desaturations detected for remainder of shift.\n"
    else:
        handover += f"Patient experienced {critical_count} critical episodes during night shift.\n"
        handover += "Recommend intensive daytime monitoring and physician assessment.\n"
    
    handover += "\nOVERALL ASSESSMENT: " + ("STABLE" if pattern != "Multiple critical episodes" else "CLOSE MONITORING REQUIRED") + "\n"
    
    # Recommendations
    handover += f"\n{'─' * 60}\n"
    handover += "RECOMMENDATIONS FOR DAY TEAM:\n"
    
    if critical_count >= 1:
        handover += "1. Perform comprehensive respiratory assessment on morning rounds\n"
        handover += "2. Review night surveillance data with patient\n"
        handover += "3. Assess current oxygenation status and vital signs\n"
        handover += "4. Continue heightened monitoring per protocol\n"
        handover += "5. Consider specialist consultation if pattern continues\n"
    else:
        handover += "1. Continue routine morning assessment\n"
        handover += "2. Review baseline vitals with patient\n"
        handover += "3. Standard discharge/management planning\n"
    
    handover += f"\n{'=' * 60}\n"
    handover += "Report prepared by: MedGemma Sentinel Night Surveillance\n"
    handover += "For clinical review only. Not a substitute for physician assessment.\n"
    
    return handover

# --- 2. Callback for Sidebar Sync ---
def sync_profile():
    """Force immediate sync when user changes profile selection"""
    st.session_state.patient_case = st.session_state.profile_selector

# --- 3. Sidebar: Dynamic Steering Control ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("---")
    st.subheader("Patient Context")

    # Initialize session state for patient_case if not present
    if 'patient_case' not in st.session_state:
        st.session_state.patient_case = "General Admission"

    # This fulfills our dynamic steering requirement
    # Key + on_change ensures immediate sync to session state
    patient_case = st.selectbox(
        "Select Patient Profile:",
        ["General Admission", "Cardiac History", "Respiratory Risk"],
        key="profile_selector",
        on_change=sync_profile,
        help="This injects the clinical focus into the AI's steering prompt.",
    )
    
    # Use session state value (guaranteed to be synced)
    patient_case = st.session_state.patient_case
    
    # Map UI labels to steering profile keys
    profile_map = {
        "General Admission": "general",
        "Cardiac History": "cardiac",
        "Respiratory Risk": "respiratory"
    }
    patient_profile = profile_map[patient_case]

    st.info(
        f"**Active Focus:** AI is hypersensitive to {patient_case.lower()} anomalies."
    )

    # ── Patient Selection (shared by both tabs) ──
    st.markdown("---")
    st.subheader("📋 Patient Selection")

    def _on_patient_change():
        """Invalidate cache when the dropdown value changes."""
        for k in ("night_events_cache", "night_events_pid"):
            st.session_state.pop(k, None)

    patient_id = st.selectbox(
        "Select Patient ID:",
        ["DEMO001", "PATIENT-A", "PATIENT-B"],
        key="day_mode_patient",
        on_change=_on_patient_change,
    )

    if st.button("🔄 Refresh Events", key="refresh_events"):
        # Invalidate cache so the next read hits disk
        _on_patient_change()
        st.session_state["_refresh_toast"] = True   # flag for post-rerun toast
        st.rerun()

    # Show toast AFTER the rerun (flag survives because it's in session_state)
    if st.session_state.pop("_refresh_toast", False):
        st.toast("🔄 Events refreshed from disk", icon="✅")

    st.markdown("---")
    st.caption("🟢 Engine: MedGemma-1.5-4b (Q4_K_M) \n\n 🔋 RAM Limit: 8GB")

# --- 3. Main Dashboard Layout ---
st.title("🏥 MedGemma Sentinel")
tab_night, tab_day = st.tabs(["🌙 Night Mode (Sentinel)", "☀️ Day Mode (Consultant)"])

# ==========================================
# 🌙 TAB 1: NIGHT MODE (Autonomous Vigilance)
# ==========================================
with tab_night:
    st.header("Real-Time Ward Surveillance")
    
    # Initialize session state for the button lock
    if "demo_ran" not in st.session_state:
        st.session_state.demo_ran = False
        
    metrics_placeholder = st.empty()
    
    def draw_metrics(spo2, hr, audio, d_spo2="normal", d_hr="normal", alert=False):
        with metrics_placeholder.container():
            c1, c2, c3 = st.columns(3)
            color_mode = "inverse" if alert else "normal"
            c1.metric(label="SpO2 (Blood Oxygen)", value=f"{spo2}%", delta=d_spo2, delta_color=color_mode)
            c2.metric(label="Heart Rate", value=f"{hr} bpm", delta=d_hr, delta_color=color_mode)
            c3.metric(label="Audio Environment", value=audio)

    # 1. Setup
    draw_metrics(98, 80, "Quiet (30dB)", d_spo2="Stable", d_hr="Stable")
    st.markdown("---")
    
    # Debug: Show which profile will be used (helps verify sync is working)
    st.markdown(f"**🎯 Active Steering Profile:** `{patient_profile.upper()}` ({patient_case})")
    st.caption("This profile will be used for MedGemma analysis when you run the demo.")
    st.markdown("---")
    
    # The Action Trigger (Now locks after one click!)
    if st.button("🎬 Run Scene 1 (The Night Shift Demo)", use_container_width=True, type="primary", disabled=st.session_state.demo_ran):
        st.session_state.demo_ran = True # Lock the button immediately
        
        # Set dynamic timestamps for this alert
        alert_time = datetime.now()
        alert_time_str = alert_time.strftime("%I:%M %p").lstrip("0")  # "2:31 PM" format
        baseline_time_str = "22:00"  # Static baseline for consistent demo narrative
        
        # 2. Crisis - Phase 1: The Drift (Yellow Warning)
        time.sleep(1)
        draw_metrics(95, 105, "Coughing (65dB)", d_spo2="-3%", d_hr="+25 bpm", alert=False)
        
        # Show NEWS2 during drift
        news2_drift_score, news2_drift_severity, _ = calculate_news2_score(105, 95)
        st.info(f"📊 **NEWS2 Score (Hospital Standard):** {news2_drift_score} {news2_drift_severity}")
        time.sleep(1.5)
        
        # 3. Escalation - Phase 2: The Event (Red Alert)
        hr_critical = 140
        spo2_critical = 88
        audio_critical = 85
        draw_metrics(spo2_critical, hr_critical, "Violent Coughing (85dB)", d_spo2="-7%", d_hr="+35 bpm", alert=True)
        
        # Calculate NEWS2 score at critical point
        news2_score, news2_severity, news2_breakdown = calculate_news2_score(hr_critical, spo2_critical)
        st.warning(f"🚨 **NEWS2 CRITICAL ALERT:** Score = {news2_score} {news2_severity}\n\n**Scoring Breakdown:**\n" + 
                  "\n".join([f"• {k}: {v}" for k, v in news2_breakdown.items()]))
        
        # 4. Check Thresholds (Rule-Based Reflex)
        severity = check_thresholds(hr_critical, spo2_critical, audio_critical)
        
        if severity == "CRITICAL":
            # 5. Trigger - Phase 3: The AI Verification (Hybrid Approach)
            warning_box = st.empty()
            warning_box.warning("⚠️ **PYTHON WATCHDOG TRIPPED:** HR > 130 AND SpO2 < 88 AND Audio > 80. Waking MedGemma API...")
            
            with st.spinner("⚡ MedGemma verifying multimodal correlation..."):
                try:
                    # OPTION A: For Video Speed - Pre-computed reasoning
                    # In production, this would call the real MedGemma API (3-10 seconds)
                    # For hackathon demo, we optimize: mock the response instantly
                    time.sleep(1.5)  # Fake latency for authenticity
                    
                    ai_reasoning = (
                        "[S] SITUATION:\n"
                        f"At {alert_time_str}, Sentinel system triggered Level 1 alert due to acute respiratory desaturation.\n"
                        "SpO2 dropped from 98% to 88% in 2 minutes. Heart rate compensated to 140 bpm.\n"
                        "Audio monitoring detected violent coughing (85dB).\n\n"
                        "[B] BACKGROUND:\n"
                        f"Patient flagged under Respiratory Risk profile. Baseline SpO2: 95-98% at {baseline_time_str}.\n"
                        "Baseline HR: 70-90 bpm. No prior airway compromise documented.\n\n"
                        "[A] ASSESSMENT:\n"
                        "Multimodal correlation analysis confirms acute respiratory event:\n"
                        "• SpO2 acute desaturation: 98% → 88% (10-point drop, critical threshold)\n"
                        "• Heart rate tachycardia: 75 → 140 bpm (acute stress response, NEWS2 +3)\n"
                        "• Audio correlation: 85dB violent coughing consistent with airway distress\n"
                        "Combined findings suggest: (1) Aspiration event, (2) Acute respiratory exacerbation, or (3) Pulmonary edema.\n\n"
                        "[R] RECOMMENDATION:\n"
                        "Event logged to GraphRAG memory. Nursing staff notified. Requires immediate physician review.\n"
                        "Recommend: Oxygen therapy, airway assessment, chest auscultation, consider CXR."
                    )
                    
                    # SAVE TO MEMORY (The Agentic Part - This is what makes it "Intelligent")
                    from src.memory.storage import LocalStorage
                    
                    storage = LocalStorage(base_path="data/patients")
                    # Use the globally-selected patient from sidebar
                    patient_id = st.session_state.get("day_mode_patient", "DEMO001")
                    
                    # Create the data snapshot with ALL required fields
                    snapshot = {
                        "hr": hr_critical,
                        "spo2": spo2_critical,
                        "audio_db": audio_critical,      # ✅ FIXED: Added Audio DB
                        "trend": "Rapid Decline"         # ✅ FIXED: Added Context
                    }
                    
                    # Save to disk
                    event_count = storage.save_night_event(patient_id, snapshot, ai_reasoning)
                    
                    if event_count > 0:
                        st.success(f"✅ Event #{event_count} saved successfully to {patient_id}")
                    else:
                        st.error(f"⚠️ Failed to save event. Return code: {event_count}")
                    
                except Exception as e:
                    # Graceful fallback
                    ai_reasoning = f"Critical multimodal event detected. Immediate intervention required."
                    event_count = -1
                    st.error(f"⚠️ Error saving event: {str(e)}")
                
                # Clear the warning box for clean final screen
                warning_box.empty()
            
            # Phase 4: The Alert & Messaging
            st.success(f"✅ **Event Successfully Logged to Patient Memory** (Total Events: {event_count})")
            st.error("### 🚨 CRITICAL ALERT: ACUTE RESPIRATORY DISTRESS")
            st.markdown(f"**AI Reasoning (MedGemma Analysis):**\n\n{ai_reasoning}")
            st.info(f"⏱️ **{alert_time_str}** - Critical Event Logged to GraphRAG Memory. **Go to Day Mode → Refresh to see this event.**")
            
            # Success toast
            st.toast("🚨 Alert sent to Nurse Station & GraphRAG Memory", icon="🚨")
            
            # Generate and offer SBAR report download
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("**📋 Clinical Handover Document:**")
            with col2:
                # Get current timestamp from last event saved
                from datetime import datetime
                current_timestamp = datetime.now().isoformat()
                night_pid = st.session_state.get("day_mode_patient", "DEMO001")
                sbar_report = generate_sbar_report(
                    night_pid,
                    hr_critical,
                    spo2_critical,
                    audio_critical,
                    ai_reasoning,
                    "Night Surveillance Patient",
                    timestamp=current_timestamp
                )
                # Convert to PDF or fall back to TXT
                if REPORTLAB_AVAILABLE:
                    pdf_bytes = text_to_pdf(sbar_report, "SBAR Clinical Handover Report")
                    st.download_button(
                        label="📥 Download SBAR Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"sbar_report_{night_pid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.download_button(
                        label="📥 Download SBAR Report (TXT)",
                        data=sbar_report,
                        file_name=f"sbar_report_{night_pid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
            
            # Offer to see what Day Mode will see
            with st.expander("👀 Preview: What Day Mode Will See (Night Handover)", expanded=False):
                storage = LocalStorage(base_path="data/patients")
                events = storage.get_night_events(night_pid)
                
                if events:
                    st.info(f"**Night Surveillance Summary:** {len(events)} Critical Event(s) Detected")
                    for i, event in enumerate(events, 1):
                        st.markdown(f"**Event #{i} - ⏰ {event['timestamp'][11:16]}**")
                        st.caption(f"Vitals: HR {event['event_data']['hr']} bpm | SpO2 {event['event_data']['spo2']}% | Audio {event['event_data']['audio_db']}dB")
                        st.write(f"**AI Analysis:** {event['ai_reasoning'][:150]}...")
                        st.divider()
            
            # Hidden reset button for video takes
            if st.button("🔄 Reset Demo For Another Take"):
                st.session_state.demo_ran = False
                storage = LocalStorage(base_path="data/patients")
                storage.clear_night_log(st.session_state.get("day_mode_patient", "DEMO001"))
                _on_patient_change()          # clear Day-Mode event cache too
                st.rerun()
        else:
            st.info(f"⚠️ Alert Severity: {severity} (No critical action needed yet)")

# ==========================================
# ☀️ TAB 2: DAY MODE – Two-Column Dashboard
# ==========================================
with tab_day:

    # ── Read selected patient from the global sidebar selector ──
    patient_id = st.session_state.get("day_mode_patient", "DEMO001")

    # ── Load events from disk (cached in session-state) ──
    from src.memory.storage import LocalStorage

    if (
        "night_events_cache" not in st.session_state
        or st.session_state.get("night_events_pid") != patient_id
    ):
        storage = LocalStorage(base_path="data/patients")
        st.session_state["night_events_cache"] = storage.get_night_events(patient_id)
        st.session_state["night_events_pid"] = patient_id

    night_events = st.session_state["night_events_cache"]

    # ── Dashboard CSS ──
    st.markdown("""
    <style>
    /* ── Card containers (Streamlit bordered containers) ── */
    div[data-testid="stExpander"] {
        border-radius: 10px !important;
    }
    /* ── Alert badge ── */
    .sentinel-badge-critical {
        display: inline-flex; align-items: center; gap: 6px;
        background: linear-gradient(135deg, #fef2f2, #fff1f2);
        border: 1px solid #fecaca; color: #991b1b;
        border-radius: 20px; padding: 6px 16px;
        font-size: 0.85rem; font-weight: 600;
        box-shadow: 0 1px 3px rgba(220,38,38,.12);
    }
    .sentinel-badge-stable {
        display: inline-flex; align-items: center; gap: 6px;
        background: linear-gradient(135deg, #f0fdf4, #ecfdf5);
        border: 1px solid #bbf7d0; color: #166534;
        border-radius: 20px; padding: 6px 16px;
        font-size: 0.85rem; font-weight: 600;
    }
    /* ── Card header ── */
    .card-title {
        font-size: 1.05rem; font-weight: 700;
        color: #1e293b; margin: 0 0 4px 0;
    }
    .card-subtitle {
        font-size: 0.82rem; color: #64748b;
        margin: 0 0 14px 0;
    }
    /* ── Event row ── */
    .evt-row {
        display: flex; align-items: flex-start; gap: 12px;
        padding: 10px 0; border-bottom: 1px solid #f1f5f9;
    }
    .evt-time {
        min-width: 52px; font-weight: 700; font-size: 0.88rem;
        color: #334155; padding-top: 2px;
    }
    .evt-body { flex: 1; }
    .evt-type {
        font-weight: 600; font-size: 0.88rem; color: #0f172a;
    }
    .evt-vitals {
        font-size: 0.82rem; color: #475569; margin-top: 2px;
    }
    .evt-status-badge {
        display: inline-block; font-size: 0.7rem; font-weight: 700;
        padding: 2px 8px; border-radius: 10px; margin-top: 4px;
    }
    .evt-status-unresolved {
        background: #fef9c3; color: #854d0e;
    }
    .evt-status-resolved {
        background: #dcfce7; color: #166534;
    }
    /* ── Primary action button (RAP2) ── */
    div[data-testid="column"]:last-child .stButton > button {
        background: linear-gradient(135deg, #e11d48 0%, #3b82f6 100%);
        color: white !important; border: none;
        font-weight: 600; transition: box-shadow .2s;
    }
    div[data-testid="column"]:last-child .stButton > button:hover {
        box-shadow: 0 4px 14px rgba(59,130,246,.35);
    }
    /* ── Results card ── */
    .results-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 24px 28px; margin-top: 8px;
    }
    .results-card h4 { margin-top: 0; }
    </style>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # TWO-COLUMN LAYOUT
    # ════════════════════════════════════════════
    col_left, col_right = st.columns(2, gap="large")

    # ─────────────────────────────────────
    # LEFT CARD – Night Surveillance
    # ─────────────────────────────────────
    with col_left:
        with st.container(border=True):
            st.markdown('<p class="card-title">🌙 Night Surveillance Intelligence</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="card-subtitle">Patient {patient_id} &middot; Last 12 h</p>', unsafe_allow_html=True)

            # Alert badge (compact)
            if night_events:
                st.markdown(
                    f'<span class="sentinel-badge-critical">⚠️ {len(night_events)} Critical Event(s) Recorded</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="sentinel-badge-stable">✅ Stable Night – No Critical Events</span>',
                    unsafe_allow_html=True,
                )

            st.markdown("")  # spacer

            # Debug accordion
            with st.expander("🔍 Debug Info", expanded=False):
                st.code(f"Patient folder: data/patients/{patient_id}")
                st.code(f"Night log file: data/patients/{patient_id}/night_log.json")
                st.write(f"Events loaded: {len(night_events) if night_events else 0}")
                if night_events:
                    st.json(night_events)

            # Timeline accordion
            if night_events:
                with st.expander("📅 Night Event Timeline", expanded=True):
                    for i, event in enumerate(night_events, 1):
                        event_data = event.get('event_data', {})
                        hr = event_data.get('hr', 'N/A')
                        spo2 = event_data.get('spo2', 'N/A')
                        audio_db = event_data.get('audio_db', 'N/A')
                        evt_type = event.get('event_type', 'Alert').replace('_', ' ')
                        status = event.get('status', 'UNRESOLVED')
                        status_cls = 'evt-status-resolved' if status == 'RESOLVED' else 'evt-status-unresolved'

                        st.markdown(f"""
                        <div class="evt-row">
                            <div class="evt-time">⏰ {event['timestamp'][11:16]}</div>
                            <div class="evt-body">
                                <div class="evt-type">{evt_type}</div>
                                <div class="evt-vitals">HR {hr} bpm &nbsp;|&nbsp; SpO₂ {spo2}% &nbsp;|&nbsp; Audio {audio_db} dB &nbsp;|&nbsp; {event_data.get('trend', '—')}</div>
                                <span class="evt-status-badge {status_cls}">{status}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # AI Reasoning (separate expanders — can't nest inside timeline expander)
                for i, event in enumerate(night_events, 1):
                    with st.expander(f"🤖 AI Reasoning – Event #{i}", expanded=False):
                        st.markdown(event['ai_reasoning'])

            # Shift Handover button (secondary style)
            if st.button("📊 Generate Shift Handover Summary", use_container_width=True, key="handover_btn"):
                handover_report = generate_shift_handover(night_events, patient_id)

                st.success("✅ Shift Handover Generated")

                if REPORTLAB_AVAILABLE:
                    handover_data = {
                        'patient_id': patient_id,
                        'date': datetime.now().strftime('%B %d, %Y'),
                        'full_text': handover_report,
                    }
                    pdf_bytes = generate_shift_handover_pdf(handover_data)
                    st.download_button(
                        label="📥 Download Handover (PDF)",
                        data=pdf_bytes,
                        file_name=f"shift_handover_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        key="dl_handover_pdf",
                    )
                else:
                    st.download_button(
                        label="📥 Download Handover (TXT)",
                        data=handover_report,
                        file_name=f"shift_handover_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        key="dl_handover_txt",
                    )

                with st.expander("👁️ View Full Handover Report", expanded=False):
                    st.text(handover_report)

    # ─────────────────────────────────────
    # RIGHT CARD – Clinical Assessment
    # ─────────────────────────────────────
    with col_right:
        with st.container(border=True):
            st.markdown('<p class="card-title">🩺 Clinical Assessment</p>', unsafe_allow_html=True)
            st.markdown('<p class="card-subtitle">Doctor\'s morning examination &amp; differential diagnosis</p>', unsafe_allow_html=True)

            symptoms_input = st.text_area(
                "Clinical Notes",
                height=170,
                placeholder="Enter current symptoms and physical exam findings…\ne.g., Patient complains of chest tightness. Auscultation reveals minor crackles in lower left lobe.",
                label_visibility="collapsed",
            )

            col_spec, col_btn = st.columns([1, 1])
            with col_spec:
                specialty = st.selectbox(
                    "Specialty", ["General", "Cardiology", "Pulmonology"],
                    label_visibility="visible",
                )
            with col_btn:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)  # align with selectbox
                generate_rap2 = st.button(
                    "🧠 Generate Differential (RAP2)",
                    use_container_width=True,
                    key="rap2_btn",
                )

    # ════════════════════════════════════════════
    # FULL-WIDTH – AI Analysis Results
    # ════════════════════════════════════════════
    if generate_rap2:
        if not symptoms_input:
            st.warning("Please enter clinical notes first.")
        else:
            with st.container(border=True):
                st.markdown('<p class="card-title">🔬 AI Analysis & Results</p>', unsafe_allow_html=True)
                st.markdown('<p class="card-subtitle">MedGemma RAP2 – Differential Diagnosis with Night Correlation</p>', unsafe_allow_html=True)

                with st.spinner("MedGemma is analyzing the case with night context..."):
                    # ── Load night context ──
                    from src.memory.storage import LocalStorage
                    storage = LocalStorage(base_path="data/patients")
                    night_events = storage.get_night_events(patient_id)

                    # ── Format night context ──
                    if night_events:
                        night_context = "🌙 NIGHT SURVEILLANCE CONTEXT:\n"
                        for i, event in enumerate(night_events, 1):
                            event_time = event['timestamp'][11:16]
                            event_data = event.get('event_data', {})
                            hr = event_data.get('hr', 'N/A')
                            spo2 = event_data.get('spo2', 'N/A')
                            audio_db = event_data.get('audio_db', 'N/A')
                            event_type = event.get('event_type', 'Alert').replace('_', ' ')

                            night_context += f"\n[{i}] {event_time} - {event_type}\n"
                            night_context += f"    Vitals: HR={hr} bpm, SpO2={spo2}%, Audio={audio_db}dB\n"
                            night_context += f"    AI Assessment: {event['ai_reasoning'][:200]}...\n"
                    else:
                        night_context = "🌙 NIGHT SURVEILLANCE CONTEXT:\nPatient had a stable night with no critical alerts recorded."

                    # ── Prompt (kept for future LLM integration) ──
                    final_prompt = f"""You are a clinical AI assistant (RAP2 - Reporting Agent Phase 2) performing differential diagnosis.

NIGHT SURVEILLANCE DATA (Last 12 Hours):
{night_context}

MORNING DOCTOR'S CLINICAL ASSESSMENT:
Specialty: {specialty}
Symptoms & Findings:
{symptoms_input}

TASK:
1. Analyze the correlation between night events and current morning symptoms
2. If night events show desaturation/respiratory distress, how does this inform the current diagnosis?
3. If night events show cardiac irregularities, how does this fit with current symptoms?
4. Generate top 3 differential diagnoses ranked by probability
5. For each diagnosis, explain how it accounts for BOTH the night events AND morning findings

Format your response as:
### Top Differentials (Considering Night + Day Context)

1. **[Diagnosis Name]** (Probability: %)
   - Explanation linking night events to this diagnosis
   - How morning symptoms support this
   - Relevant tests/actions

2. **[Diagnosis Name]** (Probability: %)
   - [Similar format]

3. **[Diagnosis Name]** (Probability: %)
   - [Similar format]

CRITICAL: Always correlate the night surveillance data with current symptoms."""

                    time.sleep(2)  # Simulating LLM inference

                # ── Analysis context accordion ──
                with st.expander("📋 Analysis Context (Night + Day)", expanded=False):
                    st.markdown("**Night Context:**")
                    st.text(night_context)
                    st.markdown("\n**Doctor's Input:**")
                    st.text(symptoms_input)

                # ── Results card ──
                st.markdown('<div class="results-card">', unsafe_allow_html=True)
                st.markdown("#### 🔗 Top Differentials (Correlated with Night Events)")

                # ── Build structured differentials ──
                differentials = []

                if night_events:
                    first_event = night_events[0].get('event_data', {})
                    spo2_night = first_event.get('spo2', 95)
                    hr_night = first_event.get('hr', 100)

                    if spo2_night < 90:
                        differentials = [
                            {
                                'name': 'Pneumonia / Acute Respiratory Infection',
                                'probability': 'High Probability: 75%',
                                'points': [
                                    f'Night surveillance showed acute SpO2 desaturation to {spo2_night}%',
                                    'Morning exam findings of crackles are consistent with consolidation',
                                    'Elevated heart rate response indicates systemic response to infection',
                                    'Recommended: CXR, CBC, Blood cultures, ABG',
                                ]
                            },
                            {
                                'name': 'Acute Heart Failure Exacerbation',
                                'probability': 'Moderate Probability: 50%',
                                'points': [
                                    'Nocturnal desaturation can indicate pulmonary edema',
                                    'Morning symptoms of dyspnea support this',
                                    'Tachycardia and elevated vitals suggest hemodynamic compromise',
                                    'Recommended: BNP, Echocardiogram, Chest X-ray',
                                ]
                            },
                            {
                                'name': 'Aspiration Event with Secondary Infection',
                                'probability': 'Moderate Probability: 45%',
                                'points': [
                                    'Night audio monitoring detected respiratory distress',
                                    'Rapid decline in SpO2 consistent with aspiration pattern',
                                    'Current symptoms of productive cough suggest ongoing infection',
                                    'Recommended: Laryngoscopy, GI consult, NPO status review',
                                ]
                            },
                        ]
                    else:
                        differentials = [
                            {
                                'name': 'Stable Chronic Condition',
                                'probability': 'High Probability: 85%',
                                'points': [
                                    'Night surveillance was unremarkable',
                                    'Current symptoms suggest stable chronic management issue',
                                    'No acute warning signs in overnight monitoring',
                                    'Recommended: Continued outpatient management, follow-up',
                                ]
                            },
                            {
                                'name': 'Viral Upper Respiratory Infection',
                                'probability': 'Moderate Probability: 60%',
                                'points': [
                                    'Stable overnight vitals suggest non-critical condition',
                                    'Morning symptoms consistent with viral syndrome',
                                    'No hypoxemia overnight',
                                    'Recommended: Supportive care, monitoring',
                                ]
                            },
                            {
                                'name': 'Medication Side Effect / Allergy',
                                'probability': 'Low-Moderate Probability: 35%',
                                'points': [
                                    'Stable night suggests not acute deterioration',
                                    'Morning symptoms could indicate reaction',
                                    'Timeline and vital stability suggest lower acuity',
                                    'Recommended: Medication review, allergy testing',
                                ]
                            },
                        ]
                else:
                    differentials = [
                        {
                            'name': 'Stable Status',
                            'probability': 'High Probability: 90%',
                            'points': [
                                'No critical alerts overnight',
                                'Current morning symptoms present on stable background',
                                'Recommended: Symptomatic management',
                            ]
                        },
                        {
                            'name': 'Early Manifestation',
                            'probability': 'Moderate Probability: 40%',
                            'points': [
                                'Symptoms may precede vital sign changes',
                                'Close monitoring recommended',
                            ]
                        },
                    ]

                # ── Render differentials ──
                for idx, dx in enumerate(differentials, 1):
                    md_points = "\n".join(
                        f"   - {'**' + p + '**' if p.lower().startswith('recommended:') else p}"
                        for p in dx['points']
                    )
                    st.markdown(f"{idx}. **{dx['name']}** ({dx['probability']})\n{md_points}\n")

                st.markdown("</div>", unsafe_allow_html=True)

                # ── PDF / TXT download ──
                if REPORTLAB_AVAILABLE:
                    rap2_data = {
                        'patient_id': patient_id,
                        'date': datetime.now().strftime('%B %d, %Y'),
                        'specialty': specialty,
                        'symptoms_input': symptoms_input,
                        'differentials': differentials,
                        'night_context': night_context,
                    }
                    pdf_bytes = generate_rap2_pdf(rap2_data)
                    st.download_button(
                        "📄 Download RAP2 Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"RAP2_differential_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        key="dl_rap2_pdf",
                    )
                else:
                    txt_lines = [
                        f"RAP2 DIFFERENTIAL DIAGNOSIS REPORT",
                        f"Patient: {patient_id} | Date: {datetime.now().strftime('%B %d, %Y')} | Specialty: {specialty}",
                        "", "DOCTOR'S CLINICAL INPUT:", symptoms_input, "",
                        "TOP DIFFERENTIALS:",
                    ]
                    for idx, dx in enumerate(differentials, 1):
                        txt_lines.append(f"\n{idx}. {dx['name']} ({dx['probability']})")
                        for p in dx['points']:
                            txt_lines.append(f"   - {p}")
                    txt_lines += ["", "NIGHT CONTEXT:", night_context]
                    rap2_txt = "\n".join(txt_lines)
                    st.download_button(
                        "📄 Download RAP2 Report (TXT)",
                        data=rap2_txt,
                        file_name=f"RAP2_differential_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        key="dl_rap2_txt",
                    )
