"""
PDF Generator — Creates downloadable accessibility reports.

Generates professional, printable PDF reports from audit results.
"""
import io
import logging
from datetime import datetime

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
except ImportError:
    logging.getLogger(__name__).warning("reportlab not installed. PDF generation disabled.")
    
logger = logging.getLogger(__name__)

class PDFGenerator:
    """Generates PDF accessibility audit reports."""
    
    def generate(self, report_dict: dict) -> io.BytesIO:
        """Generate PDF report and return as BytesIO buffer."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=letter,
            rightMargin=72, leftMargin=72,
            topMargin=72, bottomMargin=18
        )
        
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Center', alignment=1))
        styles.add(ParagraphStyle(name='Warning', textColor=colors.orange))
        styles.add(ParagraphStyle(name='Critical', textColor=colors.red))
        styles.add(ParagraphStyle(name='Success', textColor=colors.green))
        
        elements = []
        
        # 1. Header
        elements.append(Paragraph("AccessLens", styles['Title']))
        elements.append(Paragraph("AI-Powered Accessibility Audit Report", styles['Heading2']))
        elements.append(Spacer(1, 0.25 * inch))
        
        # 2. Executive Summary
        url = report_dict.get('url', 'Unknown URL')
        score = report_dict.get('overall_score', 0)
        grade = report_dict.get('grade', 'F')
        timestamp = report_dict.get('timestamp', datetime.now().isoformat())
        
        summary_data = [
            ["Target URL:", url],
            ["Date Scanned:", timestamp.split('T')[0]],
            ["Overall Score:", f"{score}/100"],
            ["Grade:", grade],
            ["Total Issues:", str(report_dict.get('total_issues', 0))],
            ["Critical Issues:", str(report_dict.get('critical_count', 0))]
        ]
        
        t = Table(summary_data, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (1, 0), (1, -1), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.silver),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.5 * inch))
        
        # 3. Category Breakdown
        elements.append(Paragraph("Category Breakdown", styles['Heading2']))
        categories = report_dict.get('categories', [])
        
        cat_data = [["Category", "Score", "Issues"]]
        for cat in categories:
            cat_data.append([
                cat.get('name', ''),
                f"{cat.get('score', 0)}/100",
                str(cat.get('issue_count', 0))
            ])
            
        if len(cat_data) > 1:
            t = Table(cat_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#7c3aed')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.silver),
            ]))
            elements.append(t)
        elements.append(Spacer(1, 0.5 * inch))
        
        # 4. Detailed Issues
        elements.append(Paragraph("Issues Found", styles['Heading2']))
        issues = report_dict.get('issues', [])
        
        if not issues:
            elements.append(Paragraph("No issues found! Great job.", styles['Success']))
        else:
            for issue in issues:
                sev = issue.get('severity', '')
                sev_style = 'Critical' if sev == 'critical' else ('Warning' if sev == 'warning' else 'Normal')
                
                elements.append(Paragraph(f"<b>[{sev.upper()}] {issue.get('title', '')}</b>", styles[sev_style]))
                elements.append(Paragraph(f"<i>WCAG {issue.get('wcag_criterion', '')}</i>", styles['Normal']))
                elements.append(Paragraph(f"{issue.get('description', '')}", styles['Normal']))
                elements.append(Paragraph(f"<b>Suggestion:</b> {issue.get('suggestion', '')}", styles['Normal']))
                elements.append(Spacer(1, 0.15 * inch))
                
        # 5. AI Insights
        dl_insights = report_dict.get('dl_insights', report_dict.get('ai_insights', []))
        if dl_insights:
            elements.append(Spacer(1, 0.25 * inch))
            elements.append(Paragraph("AI Model Insights (ViT-B/16)", styles['Heading2']))
            for ins in dl_insights:
                elements.append(Paragraph(f"<b>{ins.get('title', '')}</b> ({ins.get('confidence', 0)*100:.0f}% confidence)", styles['Normal']))
                elements.append(Paragraph(f"{ins.get('description', '')}", styles['Normal']))
                elements.append(Spacer(1, 0.15 * inch))

        # Build PDF
        try:
            doc.build(elements)
        except Exception as e:
            logger.error(f"Failed to build PDF: {e}")
            
        buffer.seek(0)
        return buffer

pdf_generator = PDFGenerator()
