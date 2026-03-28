from pathlib import Path
import sys
from datetime import datetime
from typing import List, Dict
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage

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
        notes: str = "",
        heatmap_path: str = None
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

        # Sort predictions manually inside: High > Medium > Low
        preds_sorted = sorted(predictions, key=lambda x: x.get('probability', 0.0), reverse=True)

        # Key Findings
        elements.append(Paragraph("Key Findings", h2_style))
        significant_findings = [p for p in preds_sorted if p.get('probability', 0) >= p.get('threshold', 0.5)]

        bullet_style = ParagraphStyle(
            'Bullet',
            parent=styles['Normal'],
            fontSize=11,
            leftIndent=20,
            bulletIndent=10,
            spaceAfter=6
        )

        if not significant_findings:
            elements.append(Paragraph("No significant findings detected.", styles['Normal']))
            elements.append(Spacer(1, 10))
        else:
            for pred in significant_findings:
                disease = pred.get('disease', 'Unknown').replace('_', ' ')
                prob = pred.get('probability', 0.0)
                elements.append(Paragraph(f"• <b>{disease}</b> (Confidence: {prob * 100:.1f}%)", bullet_style))
            elements.append(Spacer(1, 15))

        # Complete Feature Rankings Table
        elements.append(Paragraph("Complete Feature Rankings", h2_style))

        table_data = [["Disease Feature", "Confidence Probability"]]
        
        for pred in preds_sorted:
            disease = pred.get('disease', 'Unknown').replace('_', ' ')
            prob = pred.get('probability', 0.0)
            
            prob_str = f"{prob * 100:.2f}%"
            table_data.append([disease, prob_str])

        # Table formatting
        t = Table(table_data, colWidths=[4*inch, 2.5*inch])
        
        # Build dynamic row colors based on risk
        table_style = [
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 12),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8FAFC')),
            ('GRID', (0,0), (-1,-1), 1, colors.white),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,1), (-1,-1), 10)
        ]
        
        t.setStyle(TableStyle(table_style))
        elements.append(t)

        # Images section
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Imaging Visualization", h2_style))

        img_data = []
        img_headers = []
        
        # Check original image
        raw_image_path = Config.UPLOAD_FOLDER / image_filename
        if raw_image_path.exists():
            img_headers.append(Paragraph("Original Radiograph", ParagraphStyle('ImgHeader', parent=styles['Normal'], alignment=1)))
            img_item = RLImage(str(raw_image_path), width=3.2*inch, height=3.2*inch)
            img_item.hAlign = 'CENTER'
            img_data.append(img_item)
            
        if heatmap_path:
            heatmap_file = Path(heatmap_path).name
            h_path = Config.UPLOAD_FOLDER / "heatmaps" / heatmap_file
            if h_path.exists():
                img_headers.append(Paragraph("Grad-CAM Overlay", ParagraphStyle('ImgHeader', parent=styles['Normal'], alignment=1)))
                himg_item = RLImage(str(h_path), width=3.2*inch, height=3.2*inch)
                himg_item.hAlign = 'CENTER'
                img_data.append(himg_item)
                
        if img_data:
            img_table = Table([img_headers, img_data], colWidths=[3.5*inch]*len(img_data))
            img_table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
                ('BOTTOMPADDING', (0,0), (-1,0), 10)
            ]))
            elements.append(img_table)

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
