"""
Report Service
Generates PDF reports from prediction results
"""

from pathlib import Path
import sys
from datetime import datetime
from typing import List, Dict
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import Config


class ReportService:
    """Service for generating prediction reports"""
    
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
        Generate PDF report from prediction results
        
        Args:
            patient_id: Patient identifier
            predictions: List of prediction results
            image_filename: Original X-ray filename
            notes: Additional notes
        
        Returns:
            Path: Path to generated PDF report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"report_{patient_id}_{timestamp}.pdf"
        report_path = self.reports_dir / report_filename
        
        # Generate report content
        content = self._generate_report_content(
            patient_id, predictions, image_filename, notes
        )

        # Save report as PDF
        self._write_pdf_report(report_path, content)
        
        return report_path

    def _write_pdf_report(self, report_path: Path, content: str):
        """Write text content to a simple multi-page PDF."""
        pdf = canvas.Canvas(str(report_path), pagesize=letter)
        width, height = letter
        left_margin = 40
        top_margin = 40
        line_height = 14
        y = height - top_margin

        pdf.setFont("Helvetica", 10)

        for raw_line in content.splitlines():
            line = raw_line

            # Basic hard wrap for long lines
            while len(line) > 110:
                chunk = line[:110]
                pdf.drawString(left_margin, y, chunk)
                y -= line_height
                line = line[110:]

                if y < top_margin:
                    pdf.showPage()
                    pdf.setFont("Helvetica", 10)
                    y = height - top_margin

            pdf.drawString(left_margin, y, line)
            y -= line_height

            if y < top_margin:
                pdf.showPage()
                pdf.setFont("Helvetica", 10)
                y = height - top_margin

        pdf.save()
    
    def _generate_report_content(
        self,
        patient_id: str,
        predictions: List[Dict],
        image_filename: str,
        notes: str
    ) -> str:
        """Generate report text content"""
        
        lines = [
            "=" * 60,
            "X-LITE CHEST X-RAY ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Patient ID: {patient_id}",
            f"Image: {image_filename}",
            f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "-" * 60,
            "FINDINGS:",
            "-" * 60,
            ""
        ]
        
        # Add predictions
        for pred in predictions:
            disease = pred.get('disease', 'Unknown')
            prob = pred.get('probability', 0.0)
            risk = pred.get('risk_level', 'unknown')
            
            lines.append(f"{disease:25} | Probability: {prob:5.1%} | Risk: {risk.upper()}")
        
        lines.extend([
            "",
            "-" * 60,
            "POSITIVE FINDINGS (>50% probability):",
            "-" * 60,
            ""
        ])
        
        positive = [p for p in predictions if p.get('probability', 0) > 0.5]
        if positive:
            for pred in positive:
                lines.append(f"- {pred['disease']}: {pred['probability']:.1%}")
        else:
            lines.append("No significant findings detected.")
        
        if notes:
            lines.extend([
                "",
                "-" * 60,
                "ADDITIONAL NOTES:",
                "-" * 60,
                "",
                notes
            ])
        
        lines.extend([
            "",
            "=" * 60,
            "Note: This is an AI-assisted analysis. Please consult with",
            "a qualified radiologist for final diagnosis.",
            "=" * 60
        ])
        
        return "\n".join(lines)
