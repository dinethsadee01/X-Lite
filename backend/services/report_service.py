from pathlib import Path
import sys
from datetime import datetime
from typing import List, Dict
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import Config


class ReportService:
    """Service for generating professional PDF reports"""

    def __init__(self):
        self.reports_dir = Config.UPLOAD_FOLDER / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_pdf_report(
        self,
        patient_id: str,
        predictions: List[Dict],
        image_filename: str,
        notes: str = ""
    ) -> Path:
        """
        Generate detailed professional PDF report from prediction results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"report_{patient_id}_{timestamp}.pdf"
        report_path = self.reports_dir / report_filename

        doc = SimpleDocTemplate(
            str(report_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        styles = getSampleStyleSheet()
        elements = []

        # Styles
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1E3A8A'),
            alignment=1, # Center
            spaceAfter=20
        )
        sub_style = ParagraphStyle(
            'SubTitle',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.gray,
            alignment=1,
            spaceAfter=20
        )
        h2_style = ParagraphStyle(
            'H2',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1E3A8A'),
            spaceAfter=12
        )
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.red,
            alignment=1,
            spaceBefore=30
        )

        # Header
        elements.append(Paragraph("X-Lite Radiological AI Analysis", title_style))
        elements.append(Paragraph("Automated Chest X-Ray Clinical Report", sub_style))

        # Basic Info Table
        info_data = [
            ["Patient ID:", patient_id, "Report Date:", datetime.now().strftime('%B %d, %Y - %H:%M:%S')],
            ["Image Source:", image_filename, "Analysis Mode:", "15-Class AI Inference"]
        ]
        info_table = Table(info_data, colWidths=[1.2*inch, 2.3*inch, 1.2*inch, 2*inch])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('TEXTCOLOR', (0,0), (0,-1), colors.HexColor('#1E3A8A')),
            ('TEXTCOLOR', (2,0), (2,-1), colors.HexColor('#1E3A8A')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LINEBELOW', (0, -1), (-1, -1), 1, colors.HexColor('#E5E7EB'))
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 20))

        # Findings Summary
        elements.append(Paragraph("AI Diagnostic Findings", h2_style))
        
        # Sort predictions manually inside: High > Medium > Low
        risk_order = {"High": 1, "Medium": 2, "Low": 3}
        preds_sorted = sorted(predictions, key=lambda x: risk_order.get(str(x.get('risk_level')).title(), 4))

        table_data = [["Disease Feature", "Confidence Probability", "Condition Risk Level"]]
        
        for pred in preds_sorted:
            disease = pred.get('disease', 'Unknown').replace('_', ' ')
            prob = pred.get('probability', 0.0)
            risk = str(pred.get('risk_level', 'unknown')).title()
            
            prob_str = f"{prob * 100:.2f}%"
            table_data.append([disease, prob_str, risk])

        # Table formatting
        t = Table(table_data, colWidths=[3*inch, 2*inch, 1.7*inch])
        
        # Build dynamic row colors based on risk
        table_style = [
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 12),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8FAFC')),
            ('GRID', (0,0), (-1,-1), 1, colors.white)
        ]
        
        for i, row in enumerate(table_data[1:], start=1):
            risk_val = row[2]
            if risk_val == 'High':
                table_style.append(('TEXTCOLOR', (2,i), (2,i), colors.HexColor('#DC2626'))) # Red
                table_style.append(('FONTNAME', (2,i), (2,i), 'Helvetica-Bold'))
            elif risk_val == 'Medium':
                table_style.append(('TEXTCOLOR', (2,i), (2,i), colors.HexColor('#D97706'))) # Orange
                table_style.append(('FONTNAME', (2,i), (2,i), 'Helvetica-Bold'))
            else:
                table_style.append(('TEXTCOLOR', (2,i), (2,i), colors.HexColor('#16A34A'))) # Green
                
        t.setStyle(TableStyle(table_style))
        elements.append(t)

        if notes:
            elements.append(Spacer(1, 20))
            elements.append(Paragraph("Clinical Notes", h2_style))
            elements.append(Paragraph(notes, styles['Normal']))

        # Disclaimer Table/Block
        elements.append(Spacer(1, 30))
        disclaimer_text = (
            "DISCLAIMER: This report is generated by an Artificial Intelligence system (X-Lite) and is intended FOR INVESTIGATIONAL AND TRIAGING PURPOSES ONLY. "
            "It does not constitute a definitive medical diagnosis. A qualified radiologist or medical professional must review the original X-ray images and confirm all findings before making any clinical decisions."
        )
        elements.append(Paragraph(disclaimer_text, disclaimer_style))

        # Build PDF
        doc.build(elements)

        return report_path
