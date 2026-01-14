"""
Report Engine - MoSPI-Compliant
Generates comprehensive PDF reports including:
- Dataset overview
- Descriptive statistics
- Weighted statistics
- Crosstabulations and frequencies
- GenAI narrative insights
- Well-formatted HTML to PDF conversion

All operations are deterministic and reproducible.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from services.file_manager import FileManager
from services.analysis_engine import AnalysisEngine
from services.insight_engine import InsightEngine


class ReportEngine:
    """
    MoSPI-compliant report generation engine for statistical data processing.
    
    Generates comprehensive PDF reports with:
    - Title page with metadata
    - Dataset overview section
    - Descriptive statistics tables
    - Frequency distributions
    - Crosstabulation analysis
    - Weighted statistics (if applicable)
    - AI-generated narrative insights
    """
    
    def __init__(self):
        """Initialize ReportEngine."""
        pass
    
    @staticmethod
    def create_report(filename: str) -> str:
        """
        Generate a comprehensive MoSPI-compliant PDF report.
        
        Process Flow:
        1. Resolve file path using FileManager
        2. Run AnalysisEngine to generate statistics
        3. Run InsightEngine to generate GenAI narrative
        4. Build comprehensive HTML template
        5. Convert HTML to PDF (WeasyPrint preferred, ReportLab fallback)
        6. Save to report directory
        7. Return output path
        
        Args:
            filename: Name of the file to generate report for (e.g., "survey.csv")
            
        Returns:
            Full path to the generated PDF report
            
        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If file is empty or analysis fails
            Exception: For other processing errors
        """
        try:
            # ============================================
            # STEP 1: Resolve File Path
            # ============================================
            file_path = FileManager.get_best_available_file(filename)
            
            # Extract base filename
            base_filename = Path(filename).stem
            if base_filename.endswith('_mapped'):
                base_filename = base_filename[:-7]
            if base_filename.endswith('_cleaned'):
                base_filename = base_filename[:-8]
            if base_filename.endswith('_weighted'):
                base_filename = base_filename[:-9]
            
            # ============================================
            # STEP 2: Run Analysis Engine
            # ============================================
            print("Running statistical analysis...")
            try:
                analysis = AnalysisEngine()
                stats = analysis.generate_statistics(filename)
            except Exception as e:
                print(f"Warning: Analysis failed: {e}")
                stats = {
                    "descriptive_stats": {},
                    "frequencies": {},
                    "crosstabs": {},
                    "weighted_stats": {},
                    "distribution_notes": {}
                }
            
            # ============================================
            # STEP 3: Run Insight Engine (GenAI)
            # ============================================
            print("Generating AI insights...")
            try:
                insight_engine = InsightEngine()
                insights = insight_engine.generate_insights(filename)
                
                narrative_text = insights.get("narrative", "No narrative available")
                summary_dict = insights.get("summary", {})
            except Exception as e:
                print(f"Warning: Insight generation failed: {e}")
                narrative_text = "Insight generation unavailable at this time."
                summary_dict = {}
            
            # ============================================
            # STEP 4: Build HTML Template
            # ============================================
            print("Building HTML report...")
            html_content = ReportEngine._build_html_report(
                filename=base_filename,
                stats=stats,
                narrative=narrative_text,
                summary=summary_dict
            )
            
            # ============================================
            # STEP 5: Generate PDF
            # ============================================
            output_path = FileManager.get_report_path(f"{base_filename}.pdf")
            
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Try WeasyPrint first, fall back to ReportLab
            success = False
            
            # Method 1: WeasyPrint (preferred - better HTML rendering)
            try:
                from weasyprint import HTML
                print("Using WeasyPrint for PDF generation...")
                HTML(string=html_content).write_pdf(output_path)
                success = True
                print("PDF generated successfully with WeasyPrint")
            except ImportError:
                print("WeasyPrint not available, falling back to ReportLab...")
            except Exception as e:
                print(f"WeasyPrint failed: {e}, falling back to ReportLab...")
            
            # Method 2: ReportLab (fallback - simpler but functional)
            if not success:
                try:
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                    from reportlab.lib.styles import getSampleStyleSheet
                    from reportlab.lib.pagesizes import A4
                    from reportlab.lib.units import inch
                    
                    print("Using ReportLab for PDF generation...")
                    ReportEngine._generate_pdf_reportlab(
                        output_path=output_path,
                        filename=base_filename,
                        stats=stats,
                        narrative=narrative_text,
                        summary=summary_dict
                    )
                    success = True
                    print("PDF generated successfully with ReportLab")
                except Exception as e:
                    raise Exception(f"Both PDF generation methods failed: {e}")
            
            if not success:
                raise Exception("PDF generation failed with all available methods")
            
            # ============================================
            # STEP 7: Return Output Path
            # ============================================
            return output_path
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filename}")
        
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                raise
            raise Exception(f"Report generation failed: {str(e)}")
    
    @staticmethod
    def _build_html_report(filename: str, stats: Dict[str, Any], 
                          narrative: str, summary: Dict[str, Any]) -> str:
        """
        Build comprehensive HTML report template.
        
        Args:
            filename: Base filename
            stats: Statistics dictionary from AnalysisEngine
            narrative: GenAI narrative text
            summary: Summary dictionary from InsightEngine
            
        Returns:
            Complete HTML string
        """
        # Get current timestamp
        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        
        # Start building HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>StatFlow AI Report - {filename}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        
        .title-page {{
            text-align: center;
            padding: 100px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 40px;
        }}
        
        .title-page h1 {{
            font-size: 48px;
            margin-bottom: 20px;
            font-weight: bold;
        }}
        
        .title-page h2 {{
            font-size: 24px;
            margin-bottom: 10px;
            font-weight: normal;
        }}
        
        .title-page p {{
            font-size: 18px;
            margin: 10px 0;
        }}
        
        .section {{
            background: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .section h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 28px;
        }}
        
        .section h3 {{
            color: #764ba2;
            margin-top: 25px;
            margin-bottom: 15px;
            font-size: 22px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }}
        
        th {{
            background-color: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }}
        
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        
        tr:hover {{
            background-color: #f0f0f0;
        }}
        
        .stat-card {{
            display: inline-block;
            background: #f8f9fa;
            padding: 15px 20px;
            margin: 10px 10px 10px 0;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        
        .stat-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }}
        
        .narrative {{
            background: #fff9e6;
            border-left: 4px solid #ffc107;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            font-size: 15px;
            line-height: 1.8;
        }}
        
        .warning {{
            background: #fff3cd;
            border-left: 4px solid #ff9800;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 12px;
            margin-top: 40px;
        }}
    </style>
</head>
<body>
"""
        
        # ============================================
        # TITLE PAGE
        # ============================================
        html += f"""
    <div class="title-page">
        <h1>StatFlow AI</h1>
        <h2>Automated Statistical Report</h2>
        <p style="margin-top: 40px; font-size: 20px; font-weight: bold;">Generated for MoSPI</p>
        <p style="font-size: 16px;">Dataset: {filename}</p>
        <p style="font-size: 14px;">Generated: {timestamp}</p>
    </div>
"""
        
        # ============================================
        # SECTION 1: Dataset Overview
        # ============================================
        # Calculate overview stats
        num_rows = summary.get("total_rows", "N/A")
        num_cols = summary.get("total_columns", "N/A")
        num_numeric = len(stats.get("descriptive_stats", {}))
        num_categorical = len(stats.get("frequencies", {}))
        
        html += f"""
    <div class="section">
        <h2>📊 Dataset Overview</h2>
        
        <div class="stat-card">
            <div class="stat-label">Total Rows</div>
            <div class="stat-value">{num_rows}</div>
        </div>
        
        <div class="stat-card">
            <div class="stat-label">Total Columns</div>
            <div class="stat-value">{num_cols}</div>
        </div>
        
        <div class="stat-card">
            <div class="stat-label">Numeric Variables</div>
            <div class="stat-value">{num_numeric}</div>
        </div>
        
        <div class="stat-card">
            <div class="stat-label">Categorical Variables</div>
            <div class="stat-value">{num_categorical}</div>
        </div>
    </div>
"""
        
        # ============================================
        # SECTION 2: Descriptive Statistics
        # ============================================
        descriptive_stats = stats.get("descriptive_stats", {})
        
        if descriptive_stats:
            html += """
    <div class="section">
        <h2>📈 Descriptive Statistics</h2>
        <p>Summary statistics for numeric variables in the dataset.</p>
        <table>
            <thead>
                <tr>
                    <th>Variable</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Skewness</th>
                </tr>
            </thead>
            <tbody>
"""
            
            for var, stat_dict in list(descriptive_stats.items())[:10]:  # Limit to 10 variables
                if "error" in stat_dict:
                    continue
                
                mean_val = stat_dict.get("mean", 0)
                median_val = stat_dict.get("median", 0)
                std_val = stat_dict.get("std", 0)
                min_val = stat_dict.get("min", 0)
                max_val = stat_dict.get("max", 0)
                skew_val = stat_dict.get("skewness", 0)
                
                html += f"""
                <tr>
                    <td><strong>{var}</strong></td>
                    <td>{mean_val:.2f}</td>
                    <td>{median_val:.2f}</td>
                    <td>{std_val:.2f}</td>
                    <td>{min_val:.2f}</td>
                    <td>{max_val:.2f}</td>
                    <td>{skew_val:.2f}</td>
                </tr>
"""
            
            html += """
            </tbody>
        </table>
    </div>
"""
        
        # ============================================
        # SECTION 3: Frequency Distributions
        # ============================================
        frequencies = stats.get("frequencies", {})
        
        if frequencies:
            html += """
    <div class="section">
        <h2>📊 Frequency Distributions</h2>
        <p>Frequency tables for categorical variables.</p>
"""
            
            for var, freq_dict in list(frequencies.items())[:5]:  # Limit to 5 categorical variables
                if "error" in freq_dict:
                    continue
                
                counts = freq_dict.get("counts", {})
                proportions = freq_dict.get("proportions", {})
                
                if counts:
                    html += f"""
        <h3>{var}</h3>
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
"""
                    
                    for category, count in list(counts.items())[:10]:  # Limit to top 10 categories
                        pct = proportions.get(category, 0)
                        html += f"""
                <tr>
                    <td>{category}</td>
                    <td>{count}</td>
                    <td>{pct:.1f}%</td>
                </tr>
"""
                    
                    html += """
            </tbody>
        </table>
"""
            
            html += """
    </div>
"""
        
        # ============================================
        # SECTION 4: Crosstabulations
        # ============================================
        crosstabs = stats.get("crosstabs", {})
        
        if crosstabs:
            html += """
    <div class="section">
        <h2>🔀 Crosstabulation Analysis</h2>
        <p>Two-way frequency tables showing relationships between categorical variables.</p>
"""
            
            for crosstab_name, crosstab_dict in list(crosstabs.items())[:3]:  # Limit to 3 crosstabs
                if "error" in crosstab_dict:
                    continue
                
                # Display crosstab name
                html += f"<h3>{crosstab_name.replace('_', ' ').title()}</h3>"
                
                # Get row percentages
                row_pct = crosstab_dict.get("row_percentages", {})
                
                if row_pct:
                    # Get all unique columns
                    all_cols = set()
                    for row_data in row_pct.values():
                        all_cols.update(row_data.keys())
                    
                    all_cols = sorted(list(all_cols))
                    
                    html += """
        <table>
            <thead>
                <tr>
                    <th>Category</th>
"""
                    
                    for col in all_cols:
                        html += f"<th>{col}</th>"
                    
                    html += """
                </tr>
            </thead>
            <tbody>
"""
                    
                    for row_name, row_data in row_pct.items():
                        html += f"<tr><td><strong>{row_name}</strong></td>"
                        for col in all_cols:
                            val = row_data.get(col, 0)
                            html += f"<td>{val:.1f}%</td>"
                        html += "</tr>"
                    
                    html += """
            </tbody>
        </table>
"""
                
                # Display chi-square test results if available
                if "chi2_statistic" in crosstab_dict:
                    chi2 = crosstab_dict.get("chi2_statistic", 0)
                    p_val = crosstab_dict.get("p_value", 1)
                    significant = crosstab_dict.get("significant", False)
                    
                    sig_text = "statistically significant" if significant else "not significant"
                    
                    html += f"""
        <p style="margin-top: 10px; font-size: 13px; color: #666;">
            Chi-square test: χ² = {chi2:.2f}, p-value = {p_val:.4f} ({sig_text} at α=0.05)
        </p>
"""
            
            html += """
    </div>
"""
        
        # ============================================
        # SECTION 5: Weighted Statistics
        # ============================================
        weighted_stats = stats.get("weighted_stats", {})
        
        if weighted_stats:
            html += """
    <div class="section">
        <h2>⚖️ Weighted Statistics</h2>
        <p>Statistics calculated using survey weights to account for sampling design.</p>
        <table>
            <thead>
                <tr>
                    <th>Variable</th>
                    <th>Weighted Mean</th>
                    <th>Unweighted Mean</th>
                    <th>Design Effect</th>
                </tr>
            </thead>
            <tbody>
"""
            
            for var, wstat_dict in list(weighted_stats.items())[:10]:
                if "error" in wstat_dict or "_frequencies" in var:
                    continue
                
                w_mean = wstat_dict.get("weighted_mean", 0)
                uw_mean = wstat_dict.get("unweighted_mean", 0)
                deff = wstat_dict.get("design_effect", 1)
                
                html += f"""
                <tr>
                    <td><strong>{var}</strong></td>
                    <td>{w_mean:.2f}</td>
                    <td>{uw_mean:.2f}</td>
                    <td>{deff:.2f}</td>
                </tr>
"""
            
            html += """
            </tbody>
        </table>
        <p style="margin-top: 15px; font-size: 13px; color: #666;">
            <strong>Design Effect (DEFF):</strong> Ratio of actual variance to simple random sampling variance. 
            Values > 1 indicate clustering effects in the sample design.
        </p>
    </div>
"""
        
        # ============================================
        # SECTION 6: GenAI Narrative Insights
        # ============================================
        html += f"""
    <div class="section">
        <h2>🤖 AI-Generated Insights</h2>
        <p>Automated narrative analysis powered by Google Gemini AI.</p>
        <div class="narrative">
            {narrative.replace(chr(10), '<br>')}
        </div>
    </div>
"""
        
        # ============================================
        # SECTION 7: Distribution Notes
        # ============================================
        distribution_notes = stats.get("distribution_notes", {})
        
        if distribution_notes:
            html += """
    <div class="section">
        <h2>📉 Distribution Analysis</h2>
        <p>Observations about the shape and characteristics of data distributions.</p>
        <ul style="list-style-type: disc; padding-left: 30px;">
"""
            
            for var, note in distribution_notes.items():
                html += f"<li><strong>{var}:</strong> {note}</li>"
            
            html += """
        </ul>
    </div>
"""
        
        # ============================================
        # FOOTER
        # ============================================
        html += """
    <div class="footer">
        <p>Generated by StatFlow AI - MoSPI-Compliant Statistical Analysis Platform</p>
        <p>This report is automatically generated using deterministic algorithms and AI-powered insights.</p>
    </div>
</body>
</html>
"""
        
        return html
    
    @staticmethod
    def _generate_pdf_reportlab(output_path: str, filename: str, 
                               stats: Dict[str, Any], narrative: str,
                               summary: Dict[str, Any]) -> None:
        """
        Generate PDF using ReportLab (fallback method).
        
        Args:
            output_path: Path to save PDF
            filename: Base filename
            stats: Statistics dictionary
            narrative: GenAI narrative text
            summary: Summary dictionary
        """
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import inch
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib import colors
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Add custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            textColor=colors.HexColor('#1a5490'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=18,
            textColor=colors.HexColor('#2c5aa0'),
            spaceAfter=12,
            spaceBefore=20
        )
        
        # Title Page
        elements.append(Paragraph("StatFlow AI", title_style))
        elements.append(Paragraph("Automated Statistical Report", styles['Heading2']))
        elements.append(Spacer(1, 0.3*inch))
        elements.append(Paragraph(f"Dataset: {filename}", styles['Normal']))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
        elements.append(PageBreak())
        
        # Dataset Overview
        elements.append(Paragraph("Dataset Overview", heading_style))
        overview_text = f"""
        Total Rows: {summary.get('total_rows', 'N/A')}<br/>
        Total Columns: {summary.get('total_columns', 'N/A')}<br/>
        Numeric Variables: {len(stats.get('descriptive_stats', {}))}<br/>
        Categorical Variables: {len(stats.get('frequencies', {}))}
        """
        elements.append(Paragraph(overview_text, styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Descriptive Statistics
        descriptive_stats = stats.get("descriptive_stats", {})
        if descriptive_stats:
            elements.append(Paragraph("Descriptive Statistics", heading_style))
            
            # Create table
            table_data = [['Variable', 'Mean', 'Median', 'Std Dev', 'Min', 'Max']]
            
            for var, stat_dict in list(descriptive_stats.items())[:10]:
                if "error" not in stat_dict:
                    row = [
                        var,
                        f"{stat_dict.get('mean', 0):.2f}",
                        f"{stat_dict.get('median', 0):.2f}",
                        f"{stat_dict.get('std', 0):.2f}",
                        f"{stat_dict.get('min', 0):.2f}",
                        f"{stat_dict.get('max', 0):.2f}"
                    ]
                    table_data.append(row)
            
            if len(table_data) > 1:
                t = Table(table_data)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(t)
                elements.append(Spacer(1, 0.3*inch))
        
        # GenAI Narrative
        elements.append(Paragraph("AI-Generated Insights", heading_style))
        # Clean narrative text for ReportLab
        narrative_clean = narrative.replace('\n', '<br/>')
        elements.append(Paragraph(narrative_clean, styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Build PDF
        doc.build(elements)
