"""
Report Engine - Core business logic for MoSPI-compliant report generation
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib import colors

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class ReportEngine:
    """
    MoSPI-compliant PDF report generation engine for StatFlow AI
    """
    
    def __init__(self, file_id: str):
        """
        Initialize the report engine with a file ID
        
        Args:
            file_id: Unique identifier for the dataset/file
        """
        self.file_id = file_id
        self.base_path = Path("temp_uploads")
        self.reports_dir = self.base_path / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Intermediate results storage
        self.cleaning_results = None
        self.weighting_results = None
        self.analysis_results = None
        self.metadata = {}
        
        # Styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a5490'),
            spaceAfter=30,
            alignment=TA_CENTER,
            bold=True
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.HexColor('#2c5aa0'),
            spaceAfter=12,
            alignment=TA_CENTER
        ))
        
        # Info style
        self.styles.add(ParagraphStyle(
            name='InfoText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            alignment=TA_LEFT
        ))
        
        # Bullet style
        self.styles.add(ParagraphStyle(
            name='BulletText',
            parent=self.styles['Normal'],
            fontSize=10,
            leftIndent=20,
            spaceAfter=6,
            bulletIndent=10
        ))
    
    def attach_chart(self, flowables: list, img_path: str, title_text: str, 
                     width: float = 6, height: float = 4) -> None:
        """
        Embed a chart image into the PDF report with a title.
        
        Args:
            flowables: List of flowable elements to append to
            img_path: Path to the image file (PNG format)
            title_text: Title to display above the chart
            width: Image width in inches (default: 6)
            height: Image height in inches (default: 4)
        """
        from reportlab.platypus import Image
        from reportlab.lib.units import inch
        from pathlib import Path
        
        try:
            # Check if image file exists
            img_file = Path(img_path)
            if not img_file.exists():
                # Silently skip if image doesn't exist
                return
            
            # Add chart title
            flowables.append(Paragraph(title_text, self.styles['ReportSubtitle']))
            flowables.append(Spacer(1, 0.15*inch))
            
            # Embed image
            img = Image(str(img_file), width=width*inch, height=height*inch)
            flowables.append(img)
            flowables.append(Spacer(1, 0.3*inch))
            
        except Exception as e:
            # Handle any errors silently to avoid breaking report generation
            print(f"Warning: Could not attach chart {img_path}: {e}")
    
    def load_intermediate_results(self) -> bool:
        """
        Load intermediate results from cleaning, weighting, and analysis JSON files
        
        Returns:
            True if results loaded successfully, False otherwise
        """
        try:
            # Load cleaning results
            cleaning_path = self.base_path / "cleaned" / "default_user" / f"{self.file_id}_metadata.json"
            if cleaning_path.exists():
                with open(cleaning_path, 'r') as f:
                    self.cleaning_results = json.load(f)
            
            # Load weighting results
            weighting_path = self.base_path / "weighted" / "default_user" / f"{self.file_id}_weighting_metadata.json"
            if weighting_path.exists():
                with open(weighting_path, 'r') as f:
                    self.weighting_results = json.load(f)
            
            # Load analysis results
            analysis_path = self.base_path / "processed" / f"{self.file_id}_analysis.json"
            if analysis_path.exists():
                with open(analysis_path, 'r') as f:
                    self.analysis_results = json.load(f)
            
            # Extract metadata
            if self.cleaning_results:
                self.metadata = self.cleaning_results.get('metadata', {})
            
            return True
            
        except Exception as e:
            print(f"Error loading intermediate results: {e}")
            return False
    
    def save_pdf(self, flowables: List) -> str:
        """
        Save PDF document with given flowables and page numbers
        
        Args:
            flowables: List of ReportLab flowable elements
            
        Returns:
            Path to saved PDF file
        """
        # Generate PDF filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"{self.file_id}_report_{timestamp}.pdf"
        pdf_path = self.reports_dir / pdf_filename
        
        # Page number callback function
        def add_page_number(canvas, doc):
            """Add page number to footer"""
            page_num = canvas.getPageNumber()
            text = f"Page {page_num}"
            canvas.saveState()
            canvas.setFont('Helvetica', 9)
            canvas.setFillColor(colors.grey)
            canvas.drawCentredString(4.25*inch, 0.5*inch, text)
            canvas.restoreState()
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Build PDF with page numbers
        doc.build(flowables, onFirstPage=add_page_number, onLaterPages=add_page_number)
        
        return str(pdf_path)
    
    def build_title_page(self) -> List:
        """
        Build the title page with project and file information
        
        Returns:
            List of flowable elements for the title page
        """
        flowables = []
        
        # Add some spacing from top
        flowables.append(Spacer(1, 0.5 * inch))
        
        # Project Name
        title = Paragraph("StatFlow AI", self.styles['ReportTitle'])
        flowables.append(title)
        flowables.append(Spacer(1, 0.3 * inch))
        
        # MoSPI Problem Title
        subtitle = Paragraph(
            "AI Enhanced Data Preparation & Report Writing",
            self.styles['ReportSubtitle']
        )
        flowables.append(subtitle)
        flowables.append(Spacer(1, 0.5 * inch))
        
        # Horizontal line separator
        flowables.append(Spacer(1, 0.2 * inch))
        
        # File Information Section
        info_title = Paragraph("<b>Dataset Information</b>", self.styles['Heading2'])
        flowables.append(info_title)
        flowables.append(Spacer(1, 0.2 * inch))
        
        # File name
        file_name = self.metadata.get('original_filename', 'Unknown')
        file_info = Paragraph(f"<b>File Name:</b> {file_name}", self.styles['InfoText'])
        flowables.append(file_info)
        
        # Timestamp
        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        time_info = Paragraph(f"<b>Report Generated:</b> {timestamp}", self.styles['InfoText'])
        flowables.append(time_info)
        
        # Dataset dimensions
        num_rows = self.metadata.get('rows', 0)
        num_cols = self.metadata.get('columns', 0)
        dims_info = Paragraph(
            f"<b>Dataset Dimensions:</b> {num_rows:,} rows × {num_cols} columns",
            self.styles['InfoText']
        )
        flowables.append(dims_info)
        
        # File ID
        file_id_info = Paragraph(f"<b>File ID:</b> {self.file_id}", self.styles['InfoText'])
        flowables.append(file_id_info)
        
        flowables.append(Spacer(1, 0.5 * inch))
        
        # Add page break after title page
        flowables.append(PageBreak())
        
        return flowables
    
    def build_schema_section(self) -> List:
        """
        Build the schema section showing column information
        
        Returns:
            List of flowable elements for the schema section
        """
        flowables = []
        
        # Section title
        title = Paragraph("<b>1. Dataset Schema</b>", self.styles['Heading2'])
        flowables.append(title)
        flowables.append(Spacer(1, 0.15 * inch))
        
        # Introduction paragraph
        intro_text = (
            "The following table presents the schema of the dataset, including "
            "column names, inferred data types, missing value percentages, and "
            "validation rules applied during data preparation."
        )
        intro = Paragraph(intro_text, self.styles['Normal'])
        flowables.append(intro)
        flowables.append(Spacer(1, 0.2 * inch))
        
        # Build table data
        table_data = [['Column', 'Type', 'Missing %', 'Rules Applied']]
        
        # Extract schema information from cleaning results
        if self.cleaning_results:
            issue_summary = self.cleaning_results.get('issue_summary', {})
            missing_summary = issue_summary.get('missing_summary', {})
            numeric_summary = issue_summary.get('numeric_summary', {})
            categorical_summary = issue_summary.get('categorical_summary', {})
            
            # Combine all columns
            all_columns = set()
            all_columns.update(missing_summary.keys())
            all_columns.update(numeric_summary.keys())
            all_columns.update(categorical_summary.keys())
            
            # Also get columns from metadata if available (ensure it's iterable)
            if 'columns' in self.metadata and isinstance(self.metadata['columns'], (list, tuple)):
                all_columns.update(self.metadata['columns'])
            
            for col in sorted(all_columns):
                # Determine type
                if col in numeric_summary:
                    col_type = numeric_summary[col].get('dtype', 'numeric')
                elif col in categorical_summary:
                    col_type = categorical_summary[col].get('dtype', 'categorical')
                else:
                    col_type = 'unknown'
                
                # Get missing percentage
                if col in missing_summary:
                    missing_pct = f"{missing_summary[col]['missing_percent']:.2f}%"
                else:
                    missing_pct = "0.00%"
                
                # Determine rules applied
                rules = []
                if col in missing_summary:
                    rules.append('Missing imputation')
                if col in numeric_summary:
                    rules.append('Outlier detection')
                
                rules_text = ', '.join(rules) if rules else 'None'
                
                table_data.append([col, col_type, missing_pct, rules_text])
        
        # If no data, show placeholder
        if len(table_data) == 1:
            table_data.append(['No schema data available', '-', '-', '-'])
        
        # Create table
        table = Table(table_data, colWidths=[2.5*inch, 1.5*inch, 1*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5490')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        flowables.append(table)
        flowables.append(Spacer(1, 0.3 * inch))
        flowables.append(PageBreak())
        
        return flowables
    
    def build_cleaning_summary_section(self) -> List:
        """
        Build the cleaning summary section with issues detected and resolved
        
        Returns:
            List of flowable elements for the cleaning summary section
        """
        flowables = []
        
        # Section title
        title = Paragraph("<b>2. Data Cleaning Summary</b>", self.styles['Heading2'])
        flowables.append(title)
        flowables.append(Spacer(1, 0.15 * inch))
        
        # Introduction
        intro_text = (
            "This section summarizes the data quality issues detected and the "
            "cleaning operations performed to prepare the dataset for analysis."
        )
        intro = Paragraph(intro_text, self.styles['Normal'])
        flowables.append(intro)
        flowables.append(Spacer(1, 0.2 * inch))
        
        if not self.cleaning_results:
            no_data = Paragraph("<i>No cleaning data available.</i>", self.styles['Normal'])
            flowables.append(no_data)
            flowables.append(PageBreak())
            return flowables
        
        # Extract cleaning information
        issue_summary = self.cleaning_results.get('issue_summary', {})
        outlier_summary = self.cleaning_results.get('outlier_summary', {})
        cleaning_logs = self.cleaning_results.get('cleaning_logs', [])
        
        # === SUMMARY BULLET LIST ===
        summary_title = Paragraph("<b>Key Findings:</b>", self.styles['Heading3'])
        flowables.append(summary_title)
        flowables.append(Spacer(1, 0.1 * inch))
        
        # Missing values
        missing_summary = issue_summary.get('missing_summary', {})
        total_missing_cols = len(missing_summary)
        total_missing_values = sum(m['missing_count'] for m in missing_summary.values())
        
        bullet_1 = Paragraph(
            f"• <b>Missing Values:</b> Detected {total_missing_values:,} missing values across "
            f"{total_missing_cols} columns. Applied automatic imputation using median (numeric) and "
            f"mode (categorical) strategies.",
            self.styles['BulletText']
        )
        flowables.append(bullet_1)
        
        # Outliers
        total_outlier_cols = len(outlier_summary)
        total_outliers = sum(o.get('outlier_count', 0) for o in outlier_summary.values())
        
        bullet_2 = Paragraph(
            f"• <b>Outliers:</b> Detected {total_outliers:,} outliers across {total_outlier_cols} "
            f"numeric columns using IQR method. Outliers were capped to boundary values to prevent "
            f"distortion in statistical analyses.",
            self.styles['BulletText']
        )
        flowables.append(bullet_2)
        
        # Data integrity
        rows_dropped = self.cleaning_results.get('rows_dropped', 0)
        original_rows = self.cleaning_results.get('original_row_count', 0)
        
        bullet_3 = Paragraph(
            f"• <b>Data Integrity:</b> {'No rows were dropped' if rows_dropped == 0 else f'{rows_dropped:,} rows dropped'} "
            f"during cleaning. Final dataset contains {self.cleaning_results.get('row_count', 0):,} "
            f"rows and {self.cleaning_results.get('column_count', 0)} columns.",
            self.styles['BulletText']
        )
        flowables.append(bullet_3)
        
        flowables.append(Spacer(1, 0.25 * inch))
        
        # === MISSING VALUES TABLE ===
        if missing_summary:
            missing_title = Paragraph("<b>Missing Values by Column:</b>", self.styles['Heading3'])
            flowables.append(missing_title)
            flowables.append(Spacer(1, 0.1 * inch))
            
            missing_table_data = [['Column', 'Missing Count', 'Missing %', 'Imputation Method']]
            
            for col, info in sorted(missing_summary.items())[:10]:  # Limit to top 10
                imputation_method = 'Median' if col in issue_summary.get('numeric_summary', {}) else 'Mode'
                missing_table_data.append([
                    col,
                    f"{info['missing_count']:,}",
                    f"{info['missing_percent']:.2f}%",
                    imputation_method
                ])
            
            missing_table = Table(missing_table_data, colWidths=[2*inch, 1.5*inch, 1.2*inch, 1.8*inch])
            missing_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 1), (2, -1), 'RIGHT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))
            
            flowables.append(missing_table)
            flowables.append(Spacer(1, 0.25 * inch))
        
        # === OUTLIERS TABLE ===
        if outlier_summary:
            outlier_title = Paragraph("<b>Outliers by Column (IQR Method):</b>", self.styles['Heading3'])
            flowables.append(outlier_title)
            flowables.append(Spacer(1, 0.1 * inch))
            
            outlier_table_data = [['Column', 'Outlier Count', 'Outlier %', 'Lower Bound', 'Upper Bound']]
            
            for col, info in sorted(outlier_summary.items(), 
                                   key=lambda x: x[1].get('outlier_count', 0), 
                                   reverse=True)[:10]:  # Top 10 by outlier count
                if info.get('outlier_count', 0) > 0:
                    outlier_table_data.append([
                        col,
                        f"{info['outlier_count']:,}",
                        f"{info['outlier_percent']:.2f}%",
                        f"{info['lower_bound']:.2f}",
                        f"{info['upper_bound']:.2f}"
                    ])
            
            if len(outlier_table_data) > 1:
                outlier_table = Table(outlier_table_data, colWidths=[2*inch, 1.2*inch, 1*inch, 1.2*inch, 1.2*inch])
                outlier_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                ]))
                
                flowables.append(outlier_table)
                flowables.append(Spacer(1, 0.25 * inch))
        
        # === CLEANING OPERATIONS LOG (Brief) ===
        if cleaning_logs:
            log_title = Paragraph("<b>Operations Performed:</b>", self.styles['Heading3'])
            flowables.append(log_title)
            flowables.append(Spacer(1, 0.1 * inch))
            
            for log_entry in cleaning_logs[:5]:  # Show last 5 operations
                operation = log_entry.get('operation', 'Unknown')
                details = log_entry.get('details', {})
                
                log_text = f"• <b>{operation}:</b> {str(details)[:100]}"
                log_para = Paragraph(log_text, self.styles['BulletText'])
                flowables.append(log_para)
        
        flowables.append(Spacer(1, 0.3 * inch))
        flowables.append(PageBreak())
        
        return flowables
    
    def build_weighting_summary_section(self) -> List:
        """
        Build the weighting methodology and diagnostics section
        
        Returns:
            List of flowable elements for the weighting summary section
        """
        flowables = []
        
        # Section title
        title = Paragraph("<b>3. Weighting Methodology</b>", self.styles['Heading2'])
        flowables.append(title)
        flowables.append(Spacer(1, 0.15 * inch))
        
        # Introduction
        intro_text = (
            "Survey weights were applied to adjust for sampling design and non-response bias, "
            "ensuring the sample represents the target population. This section details the "
            "weighting methodology and diagnostic metrics."
        )
        intro = Paragraph(intro_text, self.styles['Normal'])
        flowables.append(intro)
        flowables.append(Spacer(1, 0.2 * inch))
        
        if not self.weighting_results:
            no_data = Paragraph("<i>No weighting data available.</i>", self.styles['Normal'])
            flowables.append(no_data)
            flowables.append(PageBreak())
            return flowables
        
        # === WEIGHTING METHODOLOGY ===
        method_title = Paragraph("<b>Weighting Procedure:</b>", self.styles['Heading3'])
        flowables.append(method_title)
        flowables.append(Spacer(1, 0.1 * inch))
        
        # Base weight information
        base_weight_text = (
            "• <b>Base Weights:</b> Initial weights derived from sampling design. "
            "For probability samples, base weights are computed as the inverse of inclusion probabilities. "
            "For non-probability samples, uniform base weights (1.0) are applied."
        )
        base_weight_para = Paragraph(base_weight_text, self.styles['BulletText'])
        flowables.append(base_weight_para)
        
        # Post-stratification
        poststrat_logs = [log for log in self.weighting_results.get('operations_log', []) 
                         if log.get('operation') == 'apply_poststrat_weights']
        
        if poststrat_logs:
            poststrat_details = poststrat_logs[-1].get('details', {})
            categories = poststrat_details.get('categories', [])
            categories_str = ', '.join(categories) if categories else 'multiple demographic variables'
            
            poststrat_text = (
                f"• <b>Post-Stratification:</b> Weights adjusted to match known population distributions "
                f"across {categories_str}. This ensures sample proportions align with census benchmarks."
            )
        else:
            poststrat_text = (
                "• <b>Post-Stratification:</b> Weights adjusted to match known population distributions."
            )
        poststrat_para = Paragraph(poststrat_text, self.styles['BulletText'])
        flowables.append(poststrat_para)
        
        # Raking
        raking_logs = [log for log in self.weighting_results.get('operations_log', []) 
                      if log.get('operation') == 'raking']
        
        if raking_logs:
            raking_details = raking_logs[-1].get('details', {})
            converged = raking_details.get('converged', False)
            iterations = raking_details.get('iterations', 0)
            control_margins = raking_details.get('control_margins', [])
            
            convergence_text = "converged successfully" if converged else "reached maximum iterations"
            margins_str = ', '.join(control_margins) if control_margins else 'multiple margins'
            
            raking_text = (
                f"• <b>Raking (Iterative Proportional Fitting):</b> Applied to simultaneously "
                f"adjust weights for {margins_str}. The procedure {convergence_text} "
                f"after {iterations} iterations, achieving balanced marginal distributions."
            )
            raking_para = Paragraph(raking_text, self.styles['BulletText'])
            flowables.append(raking_para)
        
        # Trimming
        trimming_logs = [log for log in self.weighting_results.get('operations_log', []) 
                        if log.get('operation') == 'trim_weights']
        
        if trimming_logs:
            trimming_details = trimming_logs[-1].get('details', {})
            lower_bound = trimming_details.get('lower_bound', 0)
            upper_bound = trimming_details.get('upper_bound', 0)
            trimmed_count = trimming_details.get('weights_trimmed', 0)
            
            trimming_text = (
                f"• <b>Weight Trimming:</b> Applied to reduce variance from extreme weights. "
                f"Weights were bounded between {lower_bound:.3f} and {upper_bound:.3f}, "
                f"affecting {trimmed_count:,} observations. This improves statistical efficiency "
                f"while maintaining population representation."
            )
        else:
            trimming_text = (
                "• <b>Weight Trimming:</b> No trimming applied. All weights retained within acceptable range."
            )
        trimming_para = Paragraph(trimming_text, self.styles['BulletText'])
        flowables.append(trimming_para)
        
        flowables.append(Spacer(1, 0.25 * inch))
        
        # === DIAGNOSTIC METRICS TABLE ===
        diagnostics_title = Paragraph("<b>Weight Quality Diagnostics:</b>", self.styles['Heading3'])
        flowables.append(diagnostics_title)
        flowables.append(Spacer(1, 0.1 * inch))
        
        # Extract diagnostics from weighting results
        diagnostics = self.weighting_results.get('diagnostics', {})
        
        # Build diagnostics table
        diag_table_data = [['Metric', 'Value', 'Interpretation']]
        
        # DEFF (Design Effect)
        deff = diagnostics.get('design_effect', 1.0)
        diag_table_data.append([
            'DEFF (Design Effect)',
            f'{deff:.3f}',
            'Good' if deff < 2.0 else 'Moderate' if deff < 3.0 else 'High variance'
        ])
        
        # ESS (Effective Sample Size)
        ess = diagnostics.get('effective_sample_size', 0)
        n_obs = diagnostics.get('n_observations', 0)
        ess_pct = (ess / n_obs * 100) if n_obs > 0 else 0
        diag_table_data.append([
            'ESS (Effective Sample Size)',
            f'{ess:.0f} ({ess_pct:.1f}%)',
            'Good' if ess_pct > 80 else 'Moderate' if ess_pct > 60 else 'Review variance'
        ])
        
        # CV (Coefficient of Variation)
        cv = diagnostics.get('cv', 0)
        diag_table_data.append([
            'CV (Coefficient of Variation)',
            f'{cv:.3f}',
            'Excellent' if cv < 0.25 else 'Good' if cv < 0.50 else 'High variability'
        ])
        
        # Mean Weight
        mean_weight = diagnostics.get('mean_weight', 1.0)
        diag_table_data.append([
            'Mean Weight',
            f'{mean_weight:.3f}',
            'Close to 1.0 indicates balanced weighting'
        ])
        
        # Weight Range
        min_weight = diagnostics.get('min_weight', 0)
        max_weight = diagnostics.get('max_weight', 0)
        diag_table_data.append([
            'Weight Range',
            f'{min_weight:.3f} - {max_weight:.3f}',
            f'Ratio: {(max_weight/min_weight):.2f}:1' if min_weight > 0 else 'N/A'
        ])
        
        # Create diagnostics table
        diag_table = Table(diag_table_data, colWidths=[2.2*inch, 1.8*inch, 2.5*inch])
        diag_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        flowables.append(diag_table)
        flowables.append(Spacer(1, 0.2 * inch))
        
        # Metric explanations
        explanation_title = Paragraph("<b>Metric Definitions:</b>", self.styles['Heading4'])
        flowables.append(explanation_title)
        flowables.append(Spacer(1, 0.05 * inch))
        
        explanations = [
            "<b>DEFF:</b> Measures variance inflation due to weighting. Values close to 1.0 indicate minimal variance increase.",
            "<b>ESS:</b> The equivalent unweighted sample size. Higher values indicate more efficient weighting.",
            "<b>CV:</b> Measures weight variability. Lower values indicate more uniform weights and better precision."
        ]
        
        for exp in explanations:
            exp_para = Paragraph(f"• {exp}", self.styles['BulletText'])
            flowables.append(exp_para)
        
        flowables.append(Spacer(1, 0.3 * inch))
        flowables.append(PageBreak())
        
        return flowables
    
    def build_moe_and_ci_section(self) -> List:
        """
        Build the margin of error and confidence intervals section
        
        Returns:
            List of flowable elements for the MoE and CI section
        """
        flowables = []
        
        # Section title
        title = Paragraph("<b>4. Margin of Error and Confidence Intervals</b>", self.styles['Heading2'])
        flowables.append(title)
        flowables.append(Spacer(1, 0.15 * inch))
        
        # Introduction
        intro_text = (
            "This section presents weighted estimates with their associated standard errors, "
            "margins of error, and 95% confidence intervals. The margin of error (MoE) represents "
            "the maximum expected difference between the sample estimate and the true population value, "
            "calculated as 1.96 times the standard error (±1.96 SE for 95% confidence level)."
        )
        intro = Paragraph(intro_text, self.styles['Normal'])
        flowables.append(intro)
        flowables.append(Spacer(1, 0.2 * inch))
        
        # Check if we have estimates data
        if not self.weighting_results:
            no_data = Paragraph("<i>No weighted estimates available.</i>", self.styles['Normal'])
            flowables.append(no_data)
            flowables.append(PageBreak())
            return flowables
        
        # === BUILD ESTIMATES TABLE ===
        table_title = Paragraph("<b>Weighted Estimates with Confidence Intervals:</b>", self.styles['Heading3'])
        flowables.append(table_title)
        flowables.append(Spacer(1, 0.1 * inch))
        
        # Extract weighted estimates (use diagnostics for now as example)
        # In production, this would come from analysis_engine results
        diagnostics = self.weighting_results.get('diagnostics', {})
        
        # Build MoE table with example statistics
        moe_table_data = [['Variable', 'Estimate', 'Std Error', 'Margin of Error', '95% CI Lower', '95% CI Upper']]
        
        # Example: Mean weight and its precision
        mean_weight = diagnostics.get('mean_weight', 1.0)
        n_obs = diagnostics.get('n_observations', 1)
        std_weight = diagnostics.get('std_weight', 0)
        
        # Calculate SE for mean weight
        se_mean = std_weight / np.sqrt(n_obs) if n_obs > 0 else 0
        moe_mean = 1.96 * se_mean
        ci_lower_mean = mean_weight - moe_mean
        ci_upper_mean = mean_weight + moe_mean
        
        moe_table_data.append([
            'Mean Weight',
            f'{mean_weight:.4f}',
            f'{se_mean:.4f}',
            f'±{moe_mean:.4f}',
            f'{ci_lower_mean:.4f}',
            f'{ci_upper_mean:.4f}'
        ])
        
        # Effective Sample Size (as a count, not with CI)
        ess = diagnostics.get('effective_sample_size', 0)
        moe_table_data.append([
            'Effective Sample Size',
            f'{ess:.0f}',
            'N/A',
            'N/A',
            'N/A',
            'N/A'
        ])
        
        # Design Effect
        deff = diagnostics.get('design_effect', 1.0)
        # DEFF typically doesn't have CI in this context
        moe_table_data.append([
            'Design Effect (DEFF)',
            f'{deff:.3f}',
            'N/A',
            'N/A',
            'N/A',
            'N/A'
        ])
        
        # Add note about example data
        note_text = (
            "<i>Note: This table shows weighting quality metrics. In production use, "
            "this section would include weighted estimates for key survey variables "
            "(proportions, means, totals) with their associated standard errors and confidence intervals.</i>"
        )
        note_para = Paragraph(note_text, self.styles['Normal'])
        flowables.append(note_para)
        flowables.append(Spacer(1, 0.1 * inch))
        
        # Create MoE table
        moe_table = Table(moe_table_data, colWidths=[1.8*inch, 1*inch, 0.9*inch, 1.1*inch, 0.9*inch, 0.9*inch])
        moe_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5490')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        flowables.append(moe_table)
        flowables.append(Spacer(1, 0.25 * inch))
        
        # === INTERPRETATION GUIDE ===
        interp_title = Paragraph("<b>Interpreting the Results:</b>", self.styles['Heading3'])
        flowables.append(interp_title)
        flowables.append(Spacer(1, 0.1 * inch))
        
        interpretations = [
            "<b>Estimate:</b> The weighted point estimate from the sample.",
            "<b>Standard Error (SE):</b> Measures the precision of the estimate. Smaller values indicate more precise estimates.",
            "<b>Margin of Error (MoE):</b> The maximum expected sampling error at 95% confidence (±1.96 × SE).",
            "<b>95% Confidence Interval:</b> We are 95% confident the true population value falls within this range."
        ]
        
        for interp in interpretations:
            interp_para = Paragraph(f"• {interp}", self.styles['BulletText'])
            flowables.append(interp_para)
        
        flowables.append(Spacer(1, 0.2 * inch))
        
        # Formula reference
        formula_text = (
            "<b>Formulas Used:</b><br/>"
            "• Margin of Error (MoE) = 1.96 × Standard Error<br/>"
            "• 95% CI Lower = Estimate - MoE<br/>"
            "• 95% CI Upper = Estimate + MoE"
        )
        formula_para = Paragraph(formula_text, self.styles['Normal'])
        flowables.append(formula_para)
        
        flowables.append(Spacer(1, 0.3 * inch))
        flowables.append(PageBreak())
        
        return flowables
    
    def build_descriptive_stats_section(self) -> List:
        """
        Build the descriptive statistics section with numeric and categorical summaries
        
        Returns:
            List of flowable elements for the descriptive statistics section
        """
        flowables = []
        
        # Section title
        title = Paragraph("<b>5. Descriptive Statistics</b>", self.styles['Heading2'])
        flowables.append(title)
        flowables.append(Spacer(1, 0.15 * inch))
        
        # Introduction
        intro_text = (
            "This section presents descriptive statistics for all variables in the dataset. "
            "Numeric variables are summarized with measures of central tendency, dispersion, "
            "and distribution shape. Categorical variables are presented with frequency distributions."
        )
        intro = Paragraph(intro_text, self.styles['Normal'])
        flowables.append(intro)
        flowables.append(Spacer(1, 0.2 * inch))
        
        if not self.analysis_results:
            no_data = Paragraph("<i>No analysis data available.</i>", self.styles['Normal'])
            flowables.append(no_data)
            flowables.append(PageBreak())
            return flowables
        
        # Extract descriptive stats
        descriptive_stats = self.analysis_results.get('descriptive_stats', {})
        
        if not descriptive_stats:
            no_data = Paragraph("<i>No descriptive statistics available.</i>", self.styles['Normal'])
            flowables.append(no_data)
            flowables.append(PageBreak())
            return flowables
        
        # Separate numeric and categorical variables
        numeric_vars = {}
        categorical_vars = {}
        
        for var_name, var_stats in descriptive_stats.items():
            var_type = var_stats.get('type', var_stats.get('dtype', 'unknown'))
            if var_type in ['numeric', 'int64', 'float64']:
                numeric_vars[var_name] = var_stats
            elif var_type in ['categorical', 'object', 'category']:
                categorical_vars[var_name] = var_stats
        
        # === NUMERIC VARIABLES SUBSECTION ===
        if numeric_vars:
            numeric_title = Paragraph(
                f"<b>A. Numeric Variables Summary ({len(numeric_vars)} variables)</b>",
                self.styles['Heading3']
            )
            flowables.append(numeric_title)
            flowables.append(Spacer(1, 0.1 * inch))
            
            # Build numeric table
            numeric_table_data = [['Variable', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skew', 'Kurtosis']]
            
            for var_name, stats in sorted(numeric_vars.items())[:15]:  # Limit to 15 for readability
                mean = stats.get('mean', 0)
                median = stats.get('median', 0)
                std = stats.get('std', 0)
                min_val = stats.get('min', 0)
                max_val = stats.get('max', 0)
                skew = stats.get('skewness', 0)
                kurt = stats.get('kurtosis', 0)
                
                numeric_table_data.append([
                    var_name[:20],  # Truncate long names
                    f'{mean:.2f}',
                    f'{median:.2f}',
                    f'{std:.2f}',
                    f'{min_val:.2f}',
                    f'{max_val:.2f}',
                    f'{skew:.2f}',
                    f'{kurt:.2f}'
                ])
            
            # Create numeric table
            numeric_table = Table(numeric_table_data, colWidths=[
                1.3*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.6*inch, 0.7*inch
            ])
            numeric_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5490')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 7),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))
            
            flowables.append(numeric_table)
            flowables.append(Spacer(1, 0.2 * inch))
            
            # === NARRATIVE ANALYSIS ===
            findings_title = Paragraph("<b>Key Findings:</b>", self.styles['Heading4'])
            flowables.append(findings_title)
            flowables.append(Spacer(1, 0.05 * inch))
            
            findings = []
            
            # Check for high skewness
            high_skew_vars = []
            for var_name, stats in numeric_vars.items():
                skew = abs(stats.get('skewness', 0))
                if skew > 1.0:
                    high_skew_vars.append((var_name, stats.get('skewness', 0)))
            
            if high_skew_vars:
                skew_text = ', '.join([f"{name} ({skew:.2f})" for name, skew in high_skew_vars[:3]])
                findings.append(
                    f"<b>Skewed Distributions:</b> The following variables show significant skewness "
                    f"(|skew| > 1.0): {skew_text}. This indicates asymmetric distributions that may require "
                    f"transformation for certain analyses."
                )
            
            # Check for high kurtosis
            high_kurt_vars = []
            for var_name, stats in numeric_vars.items():
                kurt = abs(stats.get('kurtosis', 0))
                if kurt > 3.0:
                    high_kurt_vars.append((var_name, stats.get('kurtosis', 0)))
            
            if high_kurt_vars:
                kurt_text = ', '.join([f"{name} ({kurt:.2f})" for name, kurt in high_kurt_vars[:3]])
                findings.append(
                    f"<b>Heavy-Tailed Distributions:</b> Variables with high kurtosis (|kurtosis| > 3.0) "
                    f"include {kurt_text}. These distributions have more extreme values than a normal distribution."
                )
            
            # Check for large value ranges
            large_range_vars = []
            for var_name, stats in numeric_vars.items():
                min_val = stats.get('min', 0)
                max_val = stats.get('max', 0)
                mean_val = stats.get('mean', 1)
                if mean_val != 0:
                    range_ratio = (max_val - min_val) / abs(mean_val)
                    if range_ratio > 10:
                        large_range_vars.append((var_name, min_val, max_val))
            
            if large_range_vars:
                range_text = ', '.join([f"{name} ({min_v:.1f} to {max_v:.1f})" 
                                       for name, min_v, max_v in large_range_vars[:3]])
                findings.append(
                    f"<b>Large Value Ranges:</b> The following variables show wide ranges relative to their means: "
                    f"{range_text}. Consider scale normalization for comparative analyses."
                )
            
            # If no special findings, provide general summary
            if not findings:
                findings.append(
                    "<b>Distribution Quality:</b> All numeric variables show reasonable distributions with "
                    "moderate skewness and kurtosis values, indicating data quality is suitable for standard "
                    "statistical analyses."
                )
            
            for finding in findings:
                finding_para = Paragraph(f"• {finding}", self.styles['BulletText'])
                flowables.append(finding_para)
            
            flowables.append(Spacer(1, 0.25 * inch))
        
        # === CATEGORICAL VARIABLES SUBSECTION ===
        if categorical_vars:
            cat_title = Paragraph(
                f"<b>B. Categorical Variables Summary ({len(categorical_vars)} variables)</b>",
                self.styles['Heading3']
            )
            flowables.append(cat_title)
            flowables.append(Spacer(1, 0.1 * inch))
            
            # Build categorical table (showing top categories for each variable)
            cat_table_data = [['Variable', 'Category', 'Count', 'Percentage']]
            
            for var_name, stats in sorted(categorical_vars.items())[:10]:  # Limit to 10 variables
                frequencies = stats.get('frequencies', {})
                total_count = stats.get('count', sum(frequencies.values()))
                
                # Get top 5 categories
                sorted_freqs = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:5]
                
                for idx, (category, count) in enumerate(sorted_freqs):
                    percentage = (count / total_count * 100) if total_count > 0 else 0
                    
                    # Only show variable name in first row
                    var_display = var_name[:20] if idx == 0 else ''
                    cat_display = str(category)[:25]  # Truncate long category names
                    
                    cat_table_data.append([
                        var_display,
                        cat_display,
                        f'{count:,}',
                        f'{percentage:.1f}%'
                    ])
            
            # Create categorical table
            cat_table = Table(cat_table_data, colWidths=[1.5*inch, 2.5*inch, 1*inch, 1*inch])
            cat_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (2, 1), (3, -1), 'RIGHT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            
            flowables.append(cat_table)
            flowables.append(Spacer(1, 0.15 * inch))
            
            # Note about display limits
            note_text = (
                "<i>Note: For categorical variables with many categories, only the top 5 most "
                "frequent categories are shown. Complete frequency distributions are available "
                "in the detailed analysis output.</i>"
            )
            note_para = Paragraph(note_text, self.styles['Normal'])
            flowables.append(note_para)
        
        flowables.append(Spacer(1, 0.3 * inch))
        flowables.append(PageBreak())
        
        return flowables
    
    def build_crosstab_section(self) -> List:
        """
        Build the cross-tabulation analysis section with chi-square tests
        
        Returns:
            List of flowable elements for the crosstab section
        """
        flowables = []
        
        # Section title
        title = Paragraph("<b>6. Cross-Tabulation Analysis</b>", self.styles['Heading2'])
        flowables.append(title)
        flowables.append(Spacer(1, 0.15 * inch))
        
        # Introduction
        intro_text = (
            "Cross-tabulation analysis examines relationships between categorical variables by "
            "displaying frequency distributions across variable combinations. Chi-square tests "
            "are performed to assess statistical significance of observed associations."
        )
        intro = Paragraph(intro_text, self.styles['Normal'])
        flowables.append(intro)
        flowables.append(Spacer(1, 0.2 * inch))
        
        if not self.analysis_results:
            no_data = Paragraph("<i>No analysis data available.</i>", self.styles['Normal'])
            flowables.append(no_data)
            flowables.append(PageBreak())
            return flowables
        
        # Extract crosstab results
        crosstabs = self.analysis_results.get('crosstabs', [])
        
        if not crosstabs:
            no_data = Paragraph("<i>No crosstabulation data available.</i>", self.styles['Normal'])
            flowables.append(no_data)
            flowables.append(PageBreak())
            return flowables
        
        # Process each crosstab (limit to first 3 for report brevity)
        for idx, crosstab_data in enumerate(crosstabs[:3]):
            if idx > 0:
                flowables.append(Spacer(1, 0.4 * inch))
            
            # Get variable names
            row_var = crosstab_data.get('row_var', 'Row Variable')
            col_var = crosstab_data.get('col_var', 'Column Variable')
            
            # Crosstab title
            crosstab_title = Paragraph(
                f"<b>Crosstab {idx + 1}: {row_var} × {col_var}</b>",
                self.styles['Heading3']
            )
            flowables.append(crosstab_title)
            flowables.append(Spacer(1, 0.1 * inch))
            
            # Get table data
            table_data = crosstab_data.get('table', {})
            
            if not table_data:
                error_msg = Paragraph(
                    "<i>No data available for this crosstab.</i>",
                    self.styles['Normal']
                )
                flowables.append(error_msg)
                continue
            
            # === RAW COUNT TABLE ===
            count_subtitle = Paragraph("<b>A. Frequency Counts</b>", self.styles['Heading4'])
            flowables.append(count_subtitle)
            flowables.append(Spacer(1, 0.05 * inch))
            
            # Build count table
            row_labels = sorted(table_data.keys())
            col_labels = sorted(list(table_data[row_labels[0]].keys())) if row_labels else []
            
            # Create table data with headers
            count_table_data = [[row_var + ' \\ ' + col_var] + [str(c) for c in col_labels]]
            
            for row_label in row_labels:
                row_data = [str(row_label)]
                for col_label in col_labels:
                    value = table_data[row_label].get(str(col_label), 0)
                    row_data.append(f'{value:.0f}' if isinstance(value, float) else str(int(value)))
                count_table_data.append(row_data)
            
            # Calculate column widths
            num_cols = len(col_labels) + 1
            col_width = 6.5 * inch / num_cols
            col_widths = [col_width] * num_cols
            
            # Create count table
            count_table = Table(count_table_data, colWidths=col_widths)
            count_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5490')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))
            
            flowables.append(count_table)
            flowables.append(Spacer(1, 0.2 * inch))
            
            # === PERCENTAGE TABLE ===
            # Calculate percentages from counts
            percent_subtitle = Paragraph("<b>B. Row Percentages</b>", self.styles['Heading4'])
            flowables.append(percent_subtitle)
            flowables.append(Spacer(1, 0.05 * inch))
            
            # Build percentage table
            percent_table_data = [[row_var + ' \\ ' + col_var] + [str(c) for c in col_labels]]
            
            for row_label in row_labels:
                row_data = [str(row_label)]
                row_total = sum(table_data[row_label].values())
                for col_label in col_labels:
                    value = table_data[row_label].get(str(col_label), 0)
                    percentage = (value / row_total * 100) if row_total > 0 else 0
                    row_data.append(f'{percentage:.1f}%')
                percent_table_data.append(row_data)
            
            # Create percentage table
            percent_table = Table(percent_table_data, colWidths=col_widths)
            percent_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))
            
            flowables.append(percent_table)
            flowables.append(Spacer(1, 0.2 * inch))
            
            # === CHI-SQUARE TEST RESULTS ===
            chi_square_test = crosstab_data.get('chi_square_test', {})
            
            if chi_square_test:
                chi_subtitle = Paragraph("<b>C. Chi-Square Test of Independence</b>", self.styles['Heading4'])
                flowables.append(chi_subtitle)
                flowables.append(Spacer(1, 0.05 * inch))
                
                # Extract chi-square statistics
                chi2_stat = chi_square_test.get('chi2_statistic', 0)
                p_value = chi_square_test.get('p_value', 1)
                dof = chi_square_test.get('degrees_of_freedom', 0)
                
                # Build chi-square results table
                chi_table_data = [
                    ['Chi-square', 'p-value', 'Degrees of Freedom'],
                    [f'{chi2_stat:.4f}', f'{p_value:.4f}', str(dof)]
                ]
                
                chi_table = Table(chi_table_data, colWidths=[2*inch, 2*inch, 2.5*inch])
                chi_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5490')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                ]))
                
                flowables.append(chi_table)
                flowables.append(Spacer(1, 0.15 * inch))
                
                # === INTERPRETATION ===
                interp_title = Paragraph("<b>Interpretation:</b>", self.styles['Heading4'])
                flowables.append(interp_title)
                flowables.append(Spacer(1, 0.05 * inch))
                
                # Determine significance
                if p_value < 0.05:
                    if p_value < 0.001:
                        interpretation = (
                            f"<b>Highly Significant Association:</b> The chi-square test reveals a highly "
                            f"statistically significant relationship between {row_var} and {col_var} "
                            f"(χ² = {chi2_stat:.2f}, p < 0.001, df = {dof}). The distribution of {col_var} "
                            f"differs significantly across categories of {row_var}."
                        )
                    else:
                        interpretation = (
                            f"<b>Statistically Significant Association:</b> A statistically significant "
                            f"relationship exists between {row_var} and {col_var} "
                            f"(χ² = {chi2_stat:.2f}, p = {p_value:.4f}, df = {dof}). This indicates that "
                            f"the two variables are not independent."
                        )
                else:
                    interpretation = (
                        f"<b>No Significant Association:</b> The chi-square test did not detect a "
                        f"statistically significant relationship between {row_var} and {col_var} "
                        f"(χ² = {chi2_stat:.2f}, p = {p_value:.4f}, df = {dof}). The variables appear "
                        f"to be independent at the 0.05 significance level."
                    )
                
                interp_para = Paragraph(interpretation, self.styles['Normal'])
                flowables.append(interp_para)
                
                # Add warnings if any
                warnings = chi_square_test.get('warnings', [])
                if warnings:
                    flowables.append(Spacer(1, 0.1 * inch))
                    warning_text = f"<i>Note: {'; '.join(warnings)}</i>"
                    warning_para = Paragraph(warning_text, self.styles['Normal'])
                    flowables.append(warning_para)
        
        # If more than 3 crosstabs exist, add note
        if len(crosstabs) > 3:
            flowables.append(Spacer(1, 0.2 * inch))
            note_text = (
                f"<i>Note: Only the first 3 of {len(crosstabs)} crosstabulations are shown. "
                "Complete results are available in the detailed analysis output.</i>"
            )
            note_para = Paragraph(note_text, self.styles['Normal'])
            flowables.append(note_para)
        
        flowables.append(Spacer(1, 0.3 * inch))
        flowables.append(PageBreak())
        
        return flowables

    def build_regression_section(self) -> List:
        """
        Build the regression analysis section with OLS and Logistic regression results
        """
        flowables = []
        
        # Section header
        flowables.append(Paragraph(
            "<font color='#1a5490'><b>7. REGRESSION ANALYSIS</b></font>",
            self.styles['ReportSubtitle']
        ))
        flowables.append(Spacer(1, 12))
        
        # Section introduction
        intro_text = (
            "This section presents regression modeling results to examine relationships between "
            "variables. Both Ordinary Least Squares (OLS) regression for continuous outcomes and "
            "Logistic regression for binary outcomes are provided where applicable. Statistical "
            "significance is assessed at the α = 0.05 level."
        )
        flowables.append(Paragraph(intro_text, self.styles['InfoText']))
        flowables.append(Spacer(1, 12))
        
        if not self.analysis_results or 'regression' not in self.analysis_results:
            flowables.append(Paragraph(
                "<i>No regression analysis results available.</i>",
                self.styles['InfoText']
            ))
            return flowables
        
        regression_data = self.analysis_results['regression']
        
        # === OLS REGRESSION ===
        if 'ols' in regression_data:
            flowables.append(Paragraph(
                "<b>A. Ordinary Least Squares (OLS) Regression</b>",
                self.styles['Heading3']
            ))
            flowables.append(Spacer(1, 8))
            
            ols_data = regression_data['ols']
            coefficients = ols_data.get('coefficients', {})
            std_errors = ols_data.get('std_errors', {})
            p_values = ols_data.get('p_values', {})
            r_squared = ols_data.get('r_squared', 0)
            adj_r_squared = ols_data.get('adj_r_squared', 0)
            
            if coefficients:
                # Build OLS table
                ols_table_data = [['Variable', 'Coefficient', 'Std Error', 'p-value']]
                
                # Sort variables: Intercept first, then alphabetically
                sorted_vars = sorted(coefficients.keys(), 
                                    key=lambda x: (x != 'Intercept', x))
                
                significant_vars = []
                for var in sorted_vars:
                    coef = coefficients.get(var, 0)
                    se = std_errors.get(var, 0)
                    p_val = p_values.get(var, 1)
                    
                    # Track significant variables with large effects
                    if p_val < 0.05 and var != 'Intercept' and abs(coef) > 0.1:
                        significant_vars.append((var, coef, p_val))
                    
                    ols_table_data.append([
                        var,
                        f"{coef:.4f}",
                        f"{se:.4f}",
                        f"{p_val:.4f}"
                    ])
                
                ols_table = Table(ols_table_data, colWidths=[140, 110, 110, 110])
                ols_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                flowables.append(ols_table)
                flowables.append(Spacer(1, 12))
                
                # Model fit statistics
                fit_text = (
                    f"<b>Model Fit Statistics:</b><br/>"
                    f"• R² = {r_squared:.4f} ({r_squared*100:.2f}% of variance explained)<br/>"
                    f"• Adjusted R² = {adj_r_squared:.4f}"
                )
                flowables.append(Paragraph(fit_text, self.styles['InfoText']))
                flowables.append(Spacer(1, 12))
                
                # Interpretation
                if significant_vars:
                    interp_parts = ["<b>Key Findings:</b> "]
                    
                    if len(significant_vars) == 1:
                        var, coef, p_val = significant_vars[0]
                        direction = "positive" if coef > 0 else "negative"
                        interp_parts.append(
                            f"The variable <i>{var}</i> shows a statistically significant {direction} "
                            f"relationship with the outcome (β = {coef:.4f}, p = {p_val:.4f}). "
                        )
                    else:
                        interp_parts.append(
                            f"The model identified {len(significant_vars)} statistically significant "
                            f"predictors (p < 0.05). "
                        )
                        
                        # Highlight strongest effects
                        strongest = max(significant_vars, key=lambda x: abs(x[1]))
                        var, coef, p_val = strongest
                        direction = "positive" if coef > 0 else "negative"
                        interp_parts.append(
                            f"The strongest effect is observed for <i>{var}</i> "
                            f"(β = {coef:.4f}, p = {p_val:.4f}), indicating a {direction} association. "
                        )
                    
                    # Assess model fit
                    if r_squared >= 0.5:
                        interp_parts.append(
                            f"The model demonstrates strong explanatory power with R² = {r_squared:.4f}, "
                            f"accounting for {r_squared*100:.1f}% of the variance in the outcome."
                        )
                    elif r_squared >= 0.3:
                        interp_parts.append(
                            f"The model shows moderate explanatory power (R² = {r_squared:.4f}), "
                            f"explaining {r_squared*100:.1f}% of outcome variance."
                        )
                    else:
                        interp_parts.append(
                            f"The model has limited explanatory power (R² = {r_squared:.4f}), "
                            f"suggesting other unmeasured factors may influence the outcome."
                        )
                    
                    flowables.append(Paragraph(''.join(interp_parts), self.styles['InfoText']))
                else:
                    flowables.append(Paragraph(
                        "<b>Key Findings:</b> No variables achieved statistical significance at the "
                        "α = 0.05 level. The model may require refinement or additional predictors.",
                        self.styles['InfoText']
                    ))
                
                flowables.append(Spacer(1, 16))
        
        # === LOGISTIC REGRESSION ===
        if 'logistic' in regression_data:
            flowables.append(Paragraph(
                "<b>B. Logistic Regression</b>",
                self.styles['Heading3']
            ))
            flowables.append(Spacer(1, 8))
            
            logistic_data = regression_data['logistic']
            coefficients = logistic_data.get('coefficients', {})
            odds_ratios = logistic_data.get('odds_ratios', {})
            std_errors = logistic_data.get('std_errors', {})
            p_values = logistic_data.get('p_values', {})
            accuracy = logistic_data.get('accuracy', 0)
            precision = logistic_data.get('precision', 0)
            recall = logistic_data.get('recall', 0)
            f1_score = logistic_data.get('f1', 0)
            
            if coefficients:
                # Build logistic table
                logistic_table_data = [['Variable', 'Coefficient', 'Odds Ratio', 'Std Error', 'p-value']]
                
                # Sort variables: Intercept first, then alphabetically
                sorted_vars = sorted(coefficients.keys(), 
                                    key=lambda x: (x != 'Intercept', x))
                
                significant_vars = []
                for var in sorted_vars:
                    coef = coefficients.get(var, 0)
                    odds_ratio = odds_ratios.get(var, 1)
                    se = std_errors.get(var, 0)
                    p_val = p_values.get(var, 1)
                    
                    # Track significant variables
                    if p_val < 0.05 and var != 'Intercept':
                        significant_vars.append((var, coef, odds_ratio, p_val))
                    
                    logistic_table_data.append([
                        var,
                        f"{coef:.4f}",
                        f"{odds_ratio:.4f}",
                        f"{se:.4f}",
                        f"{p_val:.4f}"
                    ])
                
                logistic_table = Table(logistic_table_data, colWidths=[120, 90, 90, 90, 90])
                logistic_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                flowables.append(logistic_table)
                flowables.append(Spacer(1, 12))
                
                # Classification metrics
                metrics_text = (
                    f"<b>Classification Performance Metrics:</b><br/>"
                    f"• Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)<br/>"
                    f"• Precision: {precision:.4f} ({precision*100:.2f}%)<br/>"
                    f"• Recall: {recall:.4f} ({recall*100:.2f}%)<br/>"
                    f"• F1 Score: {f1_score:.4f}"
                )
                flowables.append(Paragraph(metrics_text, self.styles['InfoText']))
                flowables.append(Spacer(1, 12))
                
                # Interpretation
                interp_parts = []
                
                # Assess model performance
                if accuracy >= 0.85:
                    interp_parts.append(
                        f"<b>Model Performance:</b> The logistic regression model demonstrates "
                        f"excellent predictive performance with {accuracy*100:.1f}% accuracy. "
                    )
                elif accuracy >= 0.70:
                    interp_parts.append(
                        f"<b>Model Performance:</b> The model shows good predictive performance "
                        f"with {accuracy*100:.1f}% accuracy. "
                    )
                else:
                    interp_parts.append(
                        f"<b>Model Performance:</b> The model has moderate predictive performance "
                        f"({accuracy*100:.1f}% accuracy) and may benefit from additional features or refinement. "
                    )
                
                # Precision/Recall tradeoff
                if precision >= 0.80 and recall >= 0.80:
                    interp_parts.append(
                        f"Both precision ({precision*100:.1f}%) and recall ({recall*100:.1f}%) are high, "
                        f"indicating balanced performance in identifying positive cases while minimizing false positives. "
                    )
                elif precision > recall + 0.1:
                    interp_parts.append(
                        f"Precision ({precision*100:.1f}%) exceeds recall ({recall*100:.1f}%), "
                        f"suggesting the model is conservative in predicting positive outcomes. "
                    )
                elif recall > precision + 0.1:
                    interp_parts.append(
                        f"Recall ({recall*100:.1f}%) exceeds precision ({precision*100:.1f}%), "
                        f"indicating the model captures most positive cases but with some false positives. "
                    )
                
                # Significant predictors
                if significant_vars:
                    if len(significant_vars) == 1:
                        var, coef, odds_ratio, p_val = significant_vars[0]
                        if odds_ratio > 1:
                            interp_parts.append(
                                f"<br/><b>Key Predictor:</b> <i>{var}</i> significantly increases the odds of the "
                                f"outcome (OR = {odds_ratio:.2f}, p = {p_val:.4f}), meaning each unit increase "
                                f"in {var} is associated with {(odds_ratio-1)*100:.1f}% higher odds."
                            )
                        else:
                            interp_parts.append(
                                f"<br/><b>Key Predictor:</b> <i>{var}</i> significantly decreases the odds of the "
                                f"outcome (OR = {odds_ratio:.2f}, p = {p_val:.4f}), meaning each unit increase "
                                f"in {var} is associated with {(1-odds_ratio)*100:.1f}% lower odds."
                            )
                    else:
                        interp_parts.append(
                            f"<br/><b>Significant Predictors:</b> {len(significant_vars)} variables show "
                            f"statistically significant associations with the outcome (p < 0.05). "
                        )
                        
                        # Highlight strongest effect by odds ratio
                        strongest = max(significant_vars, key=lambda x: abs(x[2] - 1))
                        var, coef, odds_ratio, p_val = strongest
                        if odds_ratio > 1:
                            interp_parts.append(
                                f"The strongest positive predictor is <i>{var}</i> (OR = {odds_ratio:.2f}), "
                                f"increasing odds by {(odds_ratio-1)*100:.1f}%. "
                            )
                        else:
                            interp_parts.append(
                                f"The strongest negative predictor is <i>{var}</i> (OR = {odds_ratio:.2f}), "
                                f"reducing odds by {(1-odds_ratio)*100:.1f}%. "
                            )
                else:
                    interp_parts.append(
                        "<br/>No individual predictors achieved statistical significance, though the overall "
                        "model may still provide useful predictions based on the combined feature set."
                    )
                
                flowables.append(Paragraph(''.join(interp_parts), self.styles['InfoText']))
                flowables.append(Spacer(1, 12))
        
        # Add page break after regression section
        flowables.append(PageBreak())
        
        return flowables

    def build_forecasting_section(self) -> List:
        """
        Build the forecasting section with time series predictions and visualizations
        """
        flowables = []
        
        # Section header
        flowables.append(Paragraph(
            "<font color='#1a5490'><b>8. FORECASTING ANALYSIS</b></font>",
            self.styles['ReportSubtitle']
        ))
        flowables.append(Spacer(1, 12))
        
        # Section introduction
        intro_text = (
            "This section presents time series forecasting results, including predicted values "
            "with confidence intervals, model performance metrics, and visual representations "
            "of forecast trends. The analysis provides insights into future patterns based on "
            "historical data."
        )
        flowables.append(Paragraph(intro_text, self.styles['InfoText']))
        flowables.append(Spacer(1, 12))
        
        if not self.analysis_results or 'forecasting' not in self.analysis_results:
            flowables.append(Paragraph(
                "<i>No forecasting analysis results available.</i>",
                self.styles['InfoText']
            ))
            return flowables
        
        forecast_data = self.analysis_results['forecasting']
        
        # === FORECAST VISUALIZATION ===
        flowables.append(Paragraph(
            "<b>A. Forecast Visualization</b>",
            self.styles['Heading3']
        ))
        flowables.append(Spacer(1, 8))
        
        # Extract forecast data
        forecast_values = forecast_data.get('forecast_values', [])
        lower_bounds = forecast_data.get('lower_bounds', [])
        upper_bounds = forecast_data.get('upper_bounds', [])
        actual_values = forecast_data.get('actual_values', [])
        time_periods = forecast_data.get('time_periods', list(range(len(forecast_values))))
        
        if forecast_values:
            # Generate forecast plot
            try:
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Plot actual values if available
                if actual_values:
                    actual_periods = time_periods[:len(actual_values)]
                    ax.plot(actual_periods, actual_values, 'o-', color='#1a5490', 
                           linewidth=2, markersize=4, label='Actual', alpha=0.8)
                
                # Plot forecast
                forecast_periods = time_periods[len(actual_values):] if actual_values else time_periods
                forecast_start_idx = len(actual_values) if actual_values else 0
                
                ax.plot(forecast_periods, forecast_values[forecast_start_idx:], 
                       's-', color='#ff6b35', linewidth=2, markersize=4, 
                       label='Forecast', alpha=0.8)
                
                # Add confidence interval shading
                if lower_bounds and upper_bounds:
                    ax.fill_between(
                        forecast_periods,
                        lower_bounds[forecast_start_idx:],
                        upper_bounds[forecast_start_idx:],
                        color='#ff6b35', alpha=0.2, label='95% Confidence Interval'
                    )
                
                # Formatting
                ax.set_xlabel('Time Period', fontsize=11, fontweight='bold')
                ax.set_ylabel('Value', fontsize=11, fontweight='bold')
                ax.set_title('Time Series Forecast with Confidence Intervals', 
                           fontsize=12, fontweight='bold', color='#1a5490')
                ax.legend(loc='best', framealpha=0.9)
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # Save plot
                plot_path = self.reports_dir / f"{self.file_id}_forecast_plot.png"
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Embed plot in PDF
                if plot_path.exists():
                    img = Image(str(plot_path), width=6*inch, height=3*inch)
                    flowables.append(img)
                    flowables.append(Spacer(1, 12))
                    
            except Exception as e:
                flowables.append(Paragraph(
                    f"<i>Unable to generate forecast plot: {str(e)}</i>",
                    self.styles['InfoText']
                ))
                flowables.append(Spacer(1, 8))
        
        # === FORECAST SUMMARY TABLE ===
        flowables.append(Paragraph(
            "<b>B. Forecast Summary</b>",
            self.styles['Heading3']
        ))
        flowables.append(Spacer(1, 8))
        
        if forecast_values:
            # Build forecast table (show up to 10 periods)
            max_rows = min(10, len(forecast_values))
            forecast_table_data = [['Time Period', 'Forecast', 'Lower 95%', 'Upper 95%']]
            
            for i in range(max_rows):
                time_label = str(time_periods[i]) if i < len(time_periods) else str(i+1)
                forecast_val = forecast_values[i] if i < len(forecast_values) else 0
                lower_val = lower_bounds[i] if i < len(lower_bounds) else forecast_val
                upper_val = upper_bounds[i] if i < len(upper_bounds) else forecast_val
                
                forecast_table_data.append([
                    time_label,
                    f"{forecast_val:.2f}",
                    f"{lower_val:.2f}",
                    f"{upper_val:.2f}"
                ])
            
            if len(forecast_values) > max_rows:
                forecast_table_data.append(['...', '...', '...', '...'])
            
            forecast_table = Table(forecast_table_data, colWidths=[120, 120, 120, 120])
            forecast_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            flowables.append(forecast_table)
            flowables.append(Spacer(1, 12))
        
        # === ERROR METRICS ===
        flowables.append(Paragraph(
            "<b>C. Forecast Accuracy Metrics</b>",
            self.styles['Heading3']
        ))
        flowables.append(Spacer(1, 8))
        
        metrics = forecast_data.get('metrics', {})
        mae = metrics.get('mae', 0)
        mape = metrics.get('mape', 0)
        rmse = metrics.get('rmse', 0)
        
        # Build metrics table
        metrics_table_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['MAE (Mean Absolute Error)', f"{mae:.4f}", 'Average prediction error'],
            ['MAPE (Mean Abs % Error)', f"{mape:.2f}%", 'Percentage error magnitude'],
            ['RMSE (Root Mean Sq Error)', f"{rmse:.4f}", 'Prediction error variance']
        ]
        
        metrics_table = Table(metrics_table_data, colWidths=[160, 120, 200])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        flowables.append(metrics_table)
        flowables.append(Spacer(1, 12))
        
        # === INTERPRETATION ===
        flowables.append(Paragraph(
            "<b>Forecast Quality Assessment</b>",
            self.styles['Heading4']
        ))
        flowables.append(Spacer(1, 8))
        
        # Interpret MAPE
        interp_parts = []
        
        if mape < 10:
            quality = "excellent"
            reliability = "highly reliable"
            interp_parts.append(
                f"<b>Forecast Quality: Excellent.</b> The model demonstrates {quality} predictive "
                f"performance with a Mean Absolute Percentage Error (MAPE) of {mape:.2f}%. "
                f"Forecasts are {reliability} with typical prediction errors under 10%, indicating "
                f"strong model fit and stable patterns in the data."
            )
        elif mape < 20:
            quality = "good"
            reliability = "reasonably reliable"
            interp_parts.append(
                f"<b>Forecast Quality: Good.</b> The model shows {quality} predictive performance "
                f"with a MAPE of {mape:.2f}%. Forecasts are {reliability}, with prediction errors "
                f"in the 10-20% range. The model captures key patterns but some variability remains."
            )
        else:
            quality = "moderate"
            reliability = "should be used cautiously"
            interp_parts.append(
                f"<b>Forecast Quality: Moderate.</b> The model has {quality} predictive performance "
                f"with a MAPE of {mape:.2f}%. Forecasts {reliability}, as prediction errors exceed 20%. "
                f"Consider model refinement, additional features, or alternative forecasting methods."
            )
        
        # Add context about MAE and RMSE
        interp_parts.append(
            f"<br/><br/><b>Error Magnitude:</b> The Mean Absolute Error (MAE) of {mae:.4f} indicates "
            f"the average deviation between forecasted and actual values. The Root Mean Squared Error "
            f"(RMSE) of {rmse:.4f} penalizes larger errors more heavily; "
        )
        
        if rmse > mae * 1.5:
            interp_parts.append(
                f"the RMSE is notably higher than MAE, suggesting the presence of some large prediction "
                f"errors or outliers that warrant investigation."
            )
        else:
            interp_parts.append(
                f"the RMSE is close to MAE, indicating relatively consistent error magnitudes across "
                f"predictions."
            )
        
        # Confidence interval interpretation
        if lower_bounds and upper_bounds:
            avg_ci_width = np.mean([u - l for u, l in zip(upper_bounds, lower_bounds)])
            interp_parts.append(
                f"<br/><br/><b>Uncertainty Quantification:</b> The 95% confidence intervals average "
                f"{avg_ci_width:.2f} units in width, providing a measure of forecast uncertainty. "
                f"Narrower intervals indicate higher confidence in predictions."
            )
        
        flowables.append(Paragraph(''.join(interp_parts), self.styles['InfoText']))
        flowables.append(Spacer(1, 12))
        
        # Add page break after forecasting section
        flowables.append(PageBreak())
        
        return flowables

    def build_ml_section(self) -> List:
        """
        Build the machine learning models summary section
        """
        flowables = []
        
        # Section header
        flowables.append(Paragraph(
            "<font color='#1a5490'><b>9. MACHINE LEARNING MODELS</b></font>",
            self.styles['ReportSubtitle']
        ))
        flowables.append(Spacer(1, 12))
        
        # Section introduction
        intro_text = (
            "This section presents machine learning model results, including performance metrics, "
            "feature importance analysis, and clustering patterns where applicable. Models are "
            "evaluated using standard metrics appropriate to their task type (classification, "
            "regression, or clustering)."
        )
        flowables.append(Paragraph(intro_text, self.styles['InfoText']))
        flowables.append(Spacer(1, 12))
        
        if not self.analysis_results or 'ml_results' not in self.analysis_results:
            flowables.append(Paragraph(
                "<i>No machine learning model results available.</i>",
                self.styles['InfoText']
            ))
            return flowables
        
        ml_data = self.analysis_results['ml_results']
        model_type = ml_data.get('model_type', 'unknown')
        metrics = ml_data.get('metrics', {})
        feature_importance = ml_data.get('feature_importance', {})
        cluster_assignments = ml_data.get('cluster_assignments', {})
        cluster_centroids = ml_data.get('cluster_centroids', {})
        training_params = ml_data.get('training_params', {})
        
        # === MODEL SUMMARY ===
        flowables.append(Paragraph(
            "<b>A. Model Summary</b>",
            self.styles['Heading3']
        ))
        flowables.append(Spacer(1, 8))
        
        # Model type and parameters
        model_name = model_type.replace('_', ' ').title()
        summary_text = f"<b>Model Type:</b> {model_name}<br/>"
        
        if training_params:
            summary_text += "<b>Training Parameters:</b><br/>"
            for param, value in list(training_params.items())[:5]:  # Show top 5 params
                summary_text += f"  • {param}: {value}<br/>"
        
        flowables.append(Paragraph(summary_text, self.styles['InfoText']))
        flowables.append(Spacer(1, 12))
        
        # === PERFORMANCE METRICS ===
        flowables.append(Paragraph(
            "<b>B. Performance Metrics</b>",
            self.styles['Heading3']
        ))
        flowables.append(Spacer(1, 8))
        
        # Determine model category and display appropriate metrics
        is_classification = any(key in metrics for key in ['accuracy', 'precision', 'recall', 'f1'])
        is_regression = any(key in metrics for key in ['mse', 'rmse', 'r2', 'mae'])
        is_clustering = any(key in metrics for key in ['inertia', 'silhouette_score'])
        
        if is_classification:
            # Classification metrics table
            accuracy = metrics.get('accuracy', 0)
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            f1 = metrics.get('f1', 0)
            
            class_metrics_data = [
                ['Metric', 'Value', 'Description'],
                ['Accuracy', f"{accuracy:.4f} ({accuracy*100:.2f}%)", 'Overall correctness of predictions'],
                ['Precision', f"{precision:.4f} ({precision*100:.2f}%)", 'Proportion of true positives'],
                ['Recall', f"{recall:.4f} ({recall*100:.2f}%)", 'Proportion of positives identified'],
                ['F1 Score', f"{f1:.4f}", 'Harmonic mean of precision and recall']
            ]
            
            class_metrics_table = Table(class_metrics_data, colWidths=[120, 150, 210])
            class_metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (1, -1), 'CENTER'),
                ('ALIGN', (2, 0), (2, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            flowables.append(class_metrics_table)
            flowables.append(Spacer(1, 12))
            
            # Interpretation
            if accuracy >= 0.90:
                perf_assessment = "excellent"
                confidence = "very high confidence"
            elif accuracy >= 0.80:
                perf_assessment = "strong"
                confidence = "high confidence"
            elif accuracy >= 0.70:
                perf_assessment = "good"
                confidence = "reasonable confidence"
            else:
                perf_assessment = "moderate"
                confidence = "limited confidence"
            
            interp_text = (
                f"<b>Performance Assessment:</b> The {model_name} model demonstrates {perf_assessment} "
                f"performance with {accuracy*100:.1f}% accuracy. Predictions can be made with {confidence}. "
            )
            
            # Precision-recall balance
            if abs(precision - recall) > 0.1:
                if precision > recall:
                    interp_text += (
                        f"The model favors precision ({precision*100:.1f}%) over recall ({recall*100:.1f}%), "
                        f"meaning it is conservative in making positive predictions."
                    )
                else:
                    interp_text += (
                        f"The model favors recall ({recall*100:.1f}%) over precision ({precision*100:.1f}%), "
                        f"meaning it captures more positive cases but with some false positives."
                    )
            else:
                interp_text += (
                    f"The model achieves balanced precision ({precision*100:.1f}%) and recall ({recall*100:.1f}%), "
                    f"indicating well-calibrated predictions."
                )
            
            flowables.append(Paragraph(interp_text, self.styles['InfoText']))
            flowables.append(Spacer(1, 16))
        
        elif is_regression:
            # Regression metrics table
            mse = metrics.get('mse', 0)
            rmse = metrics.get('rmse', mse ** 0.5 if mse else 0)
            r2 = metrics.get('r2', 0)
            mae = metrics.get('mae', 0)
            
            reg_metrics_data = [
                ['Metric', 'Value', 'Description'],
                ['R\u00b2 Score', f"{r2:.4f}", 'Proportion of variance explained'],
                ['MSE', f"{mse:.4f}", 'Mean squared prediction error'],
                ['RMSE', f"{rmse:.4f}", 'Root mean squared error'],
                ['MAE', f"{mae:.4f}", 'Mean absolute error']
            ]
            
            reg_metrics_table = Table(reg_metrics_data, colWidths=[120, 150, 210])
            reg_metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (1, -1), 'CENTER'),
                ('ALIGN', (2, 0), (2, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            flowables.append(reg_metrics_table)
            flowables.append(Spacer(1, 12))
            
            # Interpretation
            if r2 >= 0.75:
                fit_quality = "excellent"
                explanation = "strong predictive power"
            elif r2 >= 0.50:
                fit_quality = "good"
                explanation = "moderate predictive power"
            elif r2 >= 0.25:
                fit_quality = "fair"
                explanation = "limited predictive power"
            else:
                fit_quality = "weak"
                explanation = "weak predictive power"
            
            interp_text = (
                f"<b>Performance Assessment:</b> The {model_name} model shows {fit_quality} fit with "
                f"R\u00b2 = {r2:.4f}, indicating {explanation}. The model explains {r2*100:.1f}% of the "
                f"variance in the target variable. "
            )
            
            if rmse > 0:
                interp_text += (
                    f"The RMSE of {rmse:.4f} represents the typical prediction error magnitude, while "
                    f"MAE of {mae:.4f} shows the average absolute deviation."
                )
            
            flowables.append(Paragraph(interp_text, self.styles['InfoText']))
            flowables.append(Spacer(1, 16))
        
        elif is_clustering:
            # Clustering metrics table
            inertia = metrics.get('inertia', 0)
            silhouette = metrics.get('silhouette_score', 0)
            n_clusters = metrics.get('n_clusters', len(cluster_assignments))
            
            cluster_metrics_data = [
                ['Metric', 'Value', 'Description'],
                ['Number of Clusters', str(n_clusters), 'Total clusters identified'],
                ['Inertia', f"{inertia:.4f}", 'Within-cluster sum of squares'],
            ]
            
            if silhouette > 0:
                cluster_metrics_data.append(
                    ['Silhouette Score', f"{silhouette:.4f}", 'Cluster separation quality (-1 to 1)']
                )
            
            cluster_metrics_table = Table(cluster_metrics_data, colWidths=[140, 130, 210])
            cluster_metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (1, -1), 'CENTER'),
                ('ALIGN', (2, 0), (2, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            flowables.append(cluster_metrics_table)
            flowables.append(Spacer(1, 12))
            
            # Interpretation
            if silhouette > 0:
                if silhouette >= 0.7:
                    quality = "excellent separation with well-defined clusters"
                elif silhouette >= 0.5:
                    quality = "good separation with distinct clusters"
                elif silhouette >= 0.25:
                    quality = "moderate separation with some cluster overlap"
                else:
                    quality = "weak separation with significant cluster overlap"
                
                interp_text = (
                    f"<b>Clustering Quality:</b> The {model_name} identified {n_clusters} clusters with "
                    f"{quality} (silhouette score = {silhouette:.4f}). "
                )
            else:
                interp_text = (
                    f"<b>Clustering Summary:</b> The {model_name} identified {n_clusters} clusters with "
                    f"inertia of {inertia:.4f}. "
                )
            
            flowables.append(Paragraph(interp_text, self.styles['InfoText']))
            flowables.append(Spacer(1, 16))
        
        # === FEATURE IMPORTANCE ===
        if feature_importance:
            flowables.append(Paragraph(
                "<b>C. Feature Importance Analysis</b>",
                self.styles['Heading3']
            ))
            flowables.append(Spacer(1, 8))
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:10]  # Show top 10
            
            # Generate feature importance bar chart
            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                
                features = [f[0] for f in top_features]
                importances = [f[1] for f in top_features]
                
                # Create horizontal bar chart
                bars = ax.barh(features, importances, color='#2c5aa0', alpha=0.8)
                
                # Add value labels on bars
                for i, (bar, imp) in enumerate(zip(bars, importances)):
                    ax.text(imp, i, f' {imp:.4f}', va='center', fontsize=9)
                
                ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
                ax.set_title('Top Feature Importance', fontsize=12, fontweight='bold', color='#1a5490')
                ax.invert_yaxis()  # Highest importance at top
                ax.grid(axis='x', alpha=0.3, linestyle='--')
                
                # Save plot
                plot_path = self.reports_dir / f"{self.file_id}_feature_importance.png"
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Embed plot in PDF
                if plot_path.exists():
                    img = Image(str(plot_path), width=5.5*inch, height=3.5*inch)
                    flowables.append(img)
                    flowables.append(Spacer(1, 12))
                    
            except Exception as e:
                flowables.append(Paragraph(
                    f"<i>Unable to generate feature importance chart: {str(e)}</i>",
                    self.styles['InfoText']
                ))
                flowables.append(Spacer(1, 8))
            
            # Feature importance table
            feature_table_data = [['Feature', 'Importance Score']]
            for feature, score in top_features:
                feature_table_data.append([feature, f"{score:.4f}"])
            
            if len(sorted_features) > 10:
                feature_table_data.append(['...', '...'])
            
            feature_table = Table(feature_table_data, colWidths=[280, 200])
            feature_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            flowables.append(feature_table)
            flowables.append(Spacer(1, 12))
            
            # Feature interpretation
            top_feature, top_score = top_features[0]
            interp_parts = [
                f"<b>Key Findings:</b> The most influential feature is <i>{top_feature}</i> "
                f"(importance = {top_score:.4f}), which dominates the model's predictions. "
            ]
            
            if len(top_features) >= 3:
                top3_total = sum(f[1] for f in top_features[:3])
                total_importance = sum(feature_importance.values())
                if total_importance > 0:
                    top3_pct = (top3_total / total_importance) * 100
                    interp_parts.append(
                        f"The top 3 features collectively account for {top3_pct:.1f}% of the total "
                        f"feature importance, suggesting these variables are critical for accurate predictions."
                    )
            
            flowables.append(Paragraph(''.join(interp_parts), self.styles['InfoText']))
            flowables.append(Spacer(1, 16))
        
        # === CLUSTER SUMMARY ===
        if cluster_assignments:
            flowables.append(Paragraph(
                "<b>D. Cluster Distribution</b>",
                self.styles['Heading3']
            ))
            flowables.append(Spacer(1, 8))
            
            # Cluster sizes table
            cluster_table_data = [['Cluster ID', 'Count', 'Percentage']]
            total_points = sum(cluster_assignments.values())
            
            for cluster_id in sorted(cluster_assignments.keys()):
                count = cluster_assignments[cluster_id]
                percentage = (count / total_points * 100) if total_points > 0 else 0
                cluster_table_data.append([
                    f"Cluster {cluster_id}",
                    str(count),
                    f"{percentage:.1f}%"
                ])
            
            cluster_table_data.append(['Total', str(total_points), '100.0%'])
            
            cluster_table = Table(cluster_table_data, colWidths=[160, 160, 160])
            cluster_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LINEBELOW', (0, -2), (-1, -2), 1.5, colors.HexColor('#2c5aa0')),
            ]))
            flowables.append(cluster_table)
            flowables.append(Spacer(1, 12))
            
            # Cluster distribution interpretation
            sorted_clusters = sorted(cluster_assignments.items(), key=lambda x: x[1], reverse=True)
            largest_cluster_id, largest_count = sorted_clusters[0]
            largest_pct = (largest_count / total_points * 100) if total_points > 0 else 0
            
            # Check balance
            counts = list(cluster_assignments.values())
            max_count = max(counts)
            min_count = min(counts)
            balance_ratio = min_count / max_count if max_count > 0 else 0
            
            if balance_ratio >= 0.7:
                balance_desc = "well-balanced"
            elif balance_ratio >= 0.4:
                balance_desc = "moderately balanced"
            else:
                balance_desc = "imbalanced"
            
            cluster_interp = (
                f"<b>Cluster Distribution:</b> The data points are distributed across {len(cluster_assignments)} "
                f"clusters in a {balance_desc} manner. Cluster {largest_cluster_id} is the largest, containing "
                f"{largest_count} observations ({largest_pct:.1f}% of total). "
            )
            
            if balance_ratio < 0.4:
                cluster_interp += (
                    f"The significant size imbalance suggests some groups are more prevalent or that "
                    f"outlier groups exist in the data."
                )
            else:
                cluster_interp += (
                    f"The relatively balanced distribution suggests meaningful segmentation of the population."
                )
            
            flowables.append(Paragraph(cluster_interp, self.styles['InfoText']))
            flowables.append(Spacer(1, 12))
            
            # Cluster centroids (if available)
            if cluster_centroids:
                flowables.append(Paragraph(
                    "<b>Cluster Centroids</b>",
                    self.styles['Heading4']
                ))
                flowables.append(Spacer(1, 8))
                
                # Build centroids table
                if isinstance(cluster_centroids, dict) and cluster_centroids:
                    # Get feature names from first cluster
                    first_cluster = next(iter(cluster_centroids.values()))
                    if isinstance(first_cluster, dict):
                        features = list(first_cluster.keys())[:8]  # Limit to 8 features
                        
                        centroid_table_data = [['Cluster'] + features]
                        for cluster_id in sorted(cluster_centroids.keys()):
                            centroid = cluster_centroids[cluster_id]
                            row = [f"Cluster {cluster_id}"]
                            for feat in features:
                                val = centroid.get(feat, 0)
                                row.append(f"{val:.2f}")
                            centroid_table_data.append(row)
                        
                        col_widths = [80] + [55] * len(features)
                        centroid_table = Table(centroid_table_data, colWidths=col_widths)
                        centroid_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 9),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ]))
                        flowables.append(centroid_table)
                        flowables.append(Spacer(1, 12))
                        
                        flowables.append(Paragraph(
                            "<i>Note: Centroids represent the average feature values for each cluster, "
                            "helping identify how clusters differ from each other.</i>",
                            self.styles['InfoText']
                        ))
                        flowables.append(Spacer(1, 8))
        
        # Add page break after ML section
        flowables.append(PageBreak())
        
        return flowables

    def build_insight_recommendation_section(self) -> List:
        """
        Build the automated insights and recommendations section
        """
        flowables = []
        
        # Section header
        flowables.append(Paragraph(
            "<font color='#1a5490'><b>10. INSIGHTS & RECOMMENDATIONS</b></font>",
            self.styles['ReportSubtitle']
        ))
        flowables.append(Spacer(1, 12))
        
        # Section introduction
        intro_text = (
            "This section provides automated insights derived from comprehensive data analysis, "
            "including correlation patterns, anomaly detection, risk group identification, and "
            "actionable recommendations. These insights are generated to support evidence-based "
            "decision-making and policy formulation."
        )
        flowables.append(Paragraph(intro_text, self.styles['InfoText']))
        flowables.append(Spacer(1, 12))
        
        if not self.analysis_results or 'insight_results' not in self.analysis_results:
            flowables.append(Paragraph(
                "<i>No automated insights available.</i>",
                self.styles['InfoText']
            ))
            return flowables
        
        insight_data = self.analysis_results['insight_results']
        correlations = insight_data.get('correlations', {})
        anomalies = insight_data.get('anomalies', {})
        risk_groups = insight_data.get('risk_groups', {})
        drivers = insight_data.get('drivers', {})
        recommended_actions = insight_data.get('recommended_actions', [])
        missing_patterns = insight_data.get('missing_patterns', '')
        
        # === EXECUTIVE SUMMARY ===
        if correlations or risk_groups or drivers:
            flowables.append(Paragraph(
                "<b>Executive Summary</b>",
                self.styles['Heading3']
            ))
            flowables.append(Spacer(1, 8))
            
            summary_parts = []
            
            # Count key findings
            strong_corr_count = sum(1 for v in correlations.values() if abs(v) > 0.6)
            risk_group_count = len(risk_groups)
            driver_count = len(drivers)
            anomaly_count = len(anomalies)
            
            summary_parts.append(
                f"<b>Key Findings:</b> The analysis identified {strong_corr_count} strong correlations, "
            )
            
            if driver_count > 0:
                summary_parts.append(
                    f"{driver_count} key outcome drivers, "
                )
            
            if risk_group_count > 0:
                summary_parts.append(
                    f"and {risk_group_count} risk groups requiring attention. "
                )
            else:
                summary_parts.append("and notable patterns requiring attention. ")
            
            if anomaly_count > 0:
                summary_parts.append(
                    f"Additionally, {anomaly_count} anomalies were detected that warrant investigation. "
                )
            
            if recommended_actions:
                summary_parts.append(
                    f"Based on these findings, {len(recommended_actions)} actionable recommendations "
                    f"have been generated to guide intervention strategies and policy decisions."
                )
            
            flowables.append(Paragraph(''.join(summary_parts), self.styles['InfoText']))
            flowables.append(Spacer(1, 16))
        
        # === CORRELATION SUMMARY ===
        if correlations:
            flowables.append(Paragraph(
                "<b>A. Correlation Analysis</b>",
                self.styles['Heading3']
            ))
            flowables.append(Spacer(1, 8))
            
            # Sort correlations by absolute value
            sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Build correlation table
            corr_table_data = [['Variable Pair', 'Correlation', 'Strength']]
            strong_positive = []
            strong_negative = []
            
            for var_pair, corr_val in sorted_corr[:15]:  # Show top 15
                # Determine strength
                abs_corr = abs(corr_val)
                if abs_corr > 0.8:
                    strength = "Very Strong"
                elif abs_corr > 0.6:
                    strength = "Strong"
                elif abs_corr > 0.4:
                    strength = "Moderate"
                else:
                    strength = "Weak"
                
                corr_table_data.append([
                    var_pair,
                    f"{corr_val:.4f}",
                    strength
                ])
                
                # Track strong correlations for narrative
                if abs_corr > 0.6:
                    if corr_val > 0:
                        strong_positive.append((var_pair, corr_val))
                    else:
                        strong_negative.append((var_pair, corr_val))
            
            corr_table = Table(corr_table_data, colWidths=[240, 120, 120])
            corr_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            flowables.append(corr_table)
            flowables.append(Spacer(1, 12))
            
            # Narrative interpretation
            interp_parts = []
            
            if strong_positive:
                interp_parts.append(
                    f"<b>Strong Positive Correlations:</b> {len(strong_positive)} variable pairs show "
                    f"strong positive relationships (r > 0.6). "
                )
                if len(strong_positive) >= 1:
                    top_pair, top_corr = strong_positive[0]
                    interp_parts.append(
                        f"The strongest is <i>{top_pair}</i> (r = {top_corr:.3f}), indicating these "
                        f"variables increase together. "
                    )
            
            if strong_negative:
                if strong_positive:
                    interp_parts.append("<br/>")
                interp_parts.append(
                    f"<b>Strong Negative Correlations:</b> {len(strong_negative)} variable pairs show "
                    f"strong inverse relationships (r < -0.6). "
                )
                if len(strong_negative) >= 1:
                    top_pair, top_corr = strong_negative[0]
                    interp_parts.append(
                        f"The strongest is <i>{top_pair}</i> (r = {top_corr:.3f}), indicating these "
                        f"variables move in opposite directions. "
                    )
            
            if not strong_positive and not strong_negative:
                interp_parts.append(
                    "<b>Correlation Patterns:</b> No exceptionally strong correlations (|r| > 0.6) were "
                    "detected, suggesting relatively independent variable relationships in the dataset."
                )
            
            flowables.append(Paragraph(''.join(interp_parts), self.styles['InfoText']))
            flowables.append(Spacer(1, 16))
        
        # === ANOMALY/OUTLIER SUMMARY ===
        if anomalies:
            flowables.append(Paragraph(
                "<b>B. Anomaly Detection</b>",
                self.styles['Heading3']
            ))
            flowables.append(Spacer(1, 8))
            
            flowables.append(Paragraph(
                "The following anomalies and unusual patterns were detected in the dataset:",
                self.styles['InfoText']
            ))
            flowables.append(Spacer(1, 8))
            
            # Create bullet list of anomalies
            for column, description in anomalies.items():
                bullet_text = f"<b>{column}:</b> {description}"
                flowables.append(Paragraph(
                    bullet_text,
                    self.styles['BulletText']
                ))
            
            flowables.append(Spacer(1, 8))
            flowables.append(Paragraph(
                "<i>Note: These anomalies may indicate data quality issues, measurement errors, or "
                "genuine outliers requiring further investigation.</i>",
                self.styles['InfoText']
            ))
            flowables.append(Spacer(1, 16))
        
        # === RISK GROUP SUMMARY ===
        if risk_groups:
            flowables.append(Paragraph(
                "<b>C. Risk Group Identification</b>",
                self.styles['Heading3']
            ))
            flowables.append(Spacer(1, 8))
            
            flowables.append(Paragraph(
                "The following demographic segments have been identified as requiring targeted attention "
                "based on risk scores, unusual patterns, or under-representation:",
                self.styles['InfoText']
            ))
            flowables.append(Spacer(1, 8))
            
            # Build risk groups table
            risk_table_data = [['Risk Group', 'Risk Score / Key Metric', 'Priority']]
            
            for group_label, metrics in risk_groups.items():
                if isinstance(metrics, dict):
                    risk_score = metrics.get('risk_score', 0)
                    key_metric = metrics.get('key_metric', 'N/A')
                    
                    # Determine priority
                    if risk_score > 0.7:
                        priority = "High"
                    elif risk_score > 0.4:
                        priority = "Medium"
                    else:
                        priority = "Low"
                    
                    risk_table_data.append([
                        group_label,
                        f"{key_metric} (Score: {risk_score:.2f})",
                        priority
                    ])
                else:
                    # Simple format
                    risk_table_data.append([
                        group_label,
                        str(metrics),
                        "Medium"
                    ])
            
            risk_table = Table(risk_table_data, colWidths=[180, 220, 80])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            flowables.append(risk_table)
            flowables.append(Spacer(1, 12))
            
            # Risk group interpretation
            high_priority = sum(1 for _, m in risk_groups.items() 
                              if isinstance(m, dict) and m.get('risk_score', 0) > 0.7)
            
            if high_priority > 0:
                risk_interp = (
                    f"<b>Priority Assessment:</b> {high_priority} groups are classified as high priority, "
                    f"requiring immediate intervention strategies. These groups exhibit elevated risk factors "
                    f"or significant deviation from expected patterns."
                )
            else:
                risk_interp = (
                    f"<b>Priority Assessment:</b> All identified risk groups are classified as medium or low "
                    f"priority. Monitoring and preventive measures are recommended."
                )
            
            flowables.append(Paragraph(risk_interp, self.styles['InfoText']))
            flowables.append(Spacer(1, 16))
        
        # === KEY DRIVERS SECTION ===
        if drivers:
            flowables.append(Paragraph(
                "<b>D. Key Outcome Drivers</b>",
                self.styles['Heading3']
            ))
            flowables.append(Spacer(1, 8))
            
            flowables.append(Paragraph(
                "The following variables have been identified as primary drivers influencing key outcomes:",
                self.styles['InfoText']
            ))
            flowables.append(Spacer(1, 8))
            
            # Sort drivers by score
            sorted_drivers = sorted(drivers.items(), key=lambda x: x[1], reverse=True)
            
            # Build drivers table
            drivers_table_data = [['Variable', 'Driver Score', 'Impact Level']]
            
            for variable, score in sorted_drivers:
                # Determine impact level
                if score > 0.7:
                    impact = "Critical"
                elif score > 0.5:
                    impact = "High"
                elif score > 0.3:
                    impact = "Moderate"
                else:
                    impact = "Low"
                
                drivers_table_data.append([
                    variable,
                    f"{score:.4f}",
                    impact
                ])
            
            drivers_table = Table(drivers_table_data, colWidths=[200, 140, 140])
            drivers_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            flowables.append(drivers_table)
            flowables.append(Spacer(1, 12))
            
            # Driver interpretation
            top_driver, top_score = sorted_drivers[0]
            critical_count = sum(1 for _, s in sorted_drivers if s > 0.7)
            
            driver_interp_parts = [
                f"<b>Driver Analysis:</b> <i>{top_driver}</i> is the strongest outcome driver "
                f"(score = {top_score:.3f}), indicating this variable has the greatest influence "
                f"on key outcomes. "
            ]
            
            if critical_count > 1:
                driver_interp_parts.append(
                    f"{critical_count} variables are classified as critical drivers (score > 0.7), "
                    f"suggesting these factors should be prioritized in intervention design."
                )
            elif len(sorted_drivers) >= 3:
                top3_total = sum(s for _, s in sorted_drivers[:3])
                driver_interp_parts.append(
                    f"The top 3 drivers collectively account for substantial influence on outcomes, "
                    f"with a combined score of {top3_total:.3f}."
                )
            
            flowables.append(Paragraph(''.join(driver_interp_parts), self.styles['InfoText']))
            flowables.append(Spacer(1, 16))
        
        # === RECOMMENDED ACTIONS ===
        if recommended_actions:
            flowables.append(Paragraph(
                "<b>E. Recommended Actions</b>",
                self.styles['Heading3']
            ))
            flowables.append(Spacer(1, 8))
            
            flowables.append(Paragraph(
                "Based on the comprehensive analysis, the following evidence-based actions are recommended:",
                self.styles['InfoText']
            ))
            flowables.append(Spacer(1, 8))
            
            # Display recommendations with justifications
            for i, action in enumerate(recommended_actions, 1):
                if isinstance(action, dict):
                    action_text = action.get('action', '')
                    justification = action.get('justification', '')
                    priority = action.get('priority', 'Medium')
                    
                    bullet_text = (
                        f"<b>{i}. {action_text}</b><br/>"
                        f"   <i>Justification:</i> {justification}<br/>"
                        f"   <i>Priority:</i> {priority}"
                    )
                else:
                    # Simple string format
                    bullet_text = f"<b>{i}.</b> {action}"
                
                flowables.append(Paragraph(bullet_text, self.styles['BulletText']))
                flowables.append(Spacer(1, 6))
            
            flowables.append(Spacer(1, 8))
            flowables.append(Paragraph(
                "<i>Note: These recommendations are automatically generated based on statistical patterns "
                "and should be reviewed by domain experts before implementation.</i>",
                self.styles['InfoText']
            ))
            flowables.append(Spacer(1, 16))
        
        # === MISSING PATTERNS ===
        if missing_patterns:
            flowables.append(Paragraph(
                "<b>F. Missing Data Patterns</b>",
                self.styles['Heading3']
            ))
            flowables.append(Spacer(1, 8))
            
            flowables.append(Paragraph(
                missing_patterns,
                self.styles['InfoText']
            ))
            flowables.append(Spacer(1, 12))
        
        # Add page break after insights section
        flowables.append(PageBreak())
        
        return flowables
    
    def build_recommendation_section(self) -> List:
        """
        Build Executive Statistical Recommendations section using RecommendationEngine.
        
        Provides intelligent recommendations for analysis methods, transformations,
        statistical tests, and ML models based on dataset characteristics.
        
        Returns:
            List of ReportLab flowable elements
        """
        from services.recommendation_engine import RecommendationEngine
        
        flowables = []
        
        # Section header
        flowables.append(Paragraph(
            "12. EXECUTIVE STATISTICAL RECOMMENDATIONS",
            self.styles['Heading1']
        ))
        flowables.append(Spacer(1, 0.3*inch))
        
        try:
            # Load cleaned dataframe for recommendations
            cleaned_path = self.base_path / "cleaned" / "default_user" / f"{self.file_id}_cleaned.csv"
            
            if not cleaned_path.exists():
                # Try original upload
                upload_path = self.base_path / "uploads" / "default_user" / f"{self.file_id}.csv"
                if upload_path.exists():
                    import pandas as pd
                    df = pd.read_csv(upload_path)
                else:
                    raise FileNotFoundError("No dataset found for recommendations")
            else:
                import pandas as pd
                df = pd.read_csv(cleaned_path)
            
            # Initialize RecommendationEngine
            rec_engine = RecommendationEngine(dataframe=df)
            
            # Generate recommendations
            recommendations = rec_engine.build_summary()
            
            # PART 1: Recommended Methods
            methods = recommendations.get('recommended_methods', [])
            if methods:
                flowables.append(Paragraph(
                    "Analysis Methods",
                    self.styles['Heading2']
                ))
                flowables.append(Spacer(1, 0.15*inch))
                
                for method in methods:
                    method_para = Paragraph(f"• {method}", self.styles['BulletText'])
                    flowables.append(method_para)
                
                flowables.append(Spacer(1, 0.2*inch))
            
            # PART 2: Recommended Transformations
            transformations = recommendations.get('recommended_transformations', [])
            if transformations:
                flowables.append(Paragraph(
                    "Data Transformations",
                    self.styles['Heading2']
                ))
                flowables.append(Spacer(1, 0.15*inch))
                
                # Build table
                table_data = [["Column", "Suggested Action", "Reason"]]
                for transform in transformations[:10]:  # Limit to 10
                    table_data.append([
                        transform.get('column', ''),
                        transform.get('action', '').replace('_', ' ').title(),
                        transform.get('reason', '')
                    ])
                
                transform_table = Table(table_data, colWidths=[1.8*inch, 2.2*inch, 3*inch])
                transform_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5490')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 11),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('TOPPADDING', (0, 1), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
                ]))
                flowables.append(transform_table)
                flowables.append(Spacer(1, 0.2*inch))
            
            # PART 3: Recommended Statistical Tests
            tests = recommendations.get('recommended_tests', [])
            if tests:
                flowables.append(Paragraph(
                    "Statistical Tests",
                    self.styles['Heading2']
                ))
                flowables.append(Spacer(1, 0.15*inch))
                
                for test in tests[:8]:  # Limit to 8
                    variables = ', '.join(test.get('variables', []))
                    suggestion = test.get('suggest', '')
                    test_text = f"• Use {suggestion} for {variables}"
                    test_para = Paragraph(test_text, self.styles['BulletText'])
                    flowables.append(test_para)
                
                flowables.append(Spacer(1, 0.2*inch))
            
            # PART 4: Recommended ML Models
            models = recommendations.get('recommended_models', [])
            if models:
                flowables.append(Paragraph(
                    "Machine Learning Models",
                    self.styles['Heading2']
                ))
                flowables.append(Spacer(1, 0.15*inch))
                
                # Build table
                table_data = [["Model", "Reason"]]
                for model in models[:8]:  # Limit to 8
                    model_name = model.get('model', '').replace('_', ' ').title()
                    reason = model.get('reason', '')
                    table_data.append([model_name, reason])
                
                model_table = Table(table_data, colWidths=[2.5*inch, 4.5*inch])
                model_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5490')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 11),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('TOPPADDING', (0, 1), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
                ]))
                flowables.append(model_table)
                flowables.append(Spacer(1, 0.2*inch))
            
            # PART 5: Executive Narrative Summary
            flowables.append(Paragraph(
                "Executive Summary",
                self.styles['Heading2']
            ))
            flowables.append(Spacer(1, 0.15*inch))
            
            # Generate narrative
            narrative = rec_engine.generate_narrative(recommendations)
            
            # Format narrative as paragraphs (split by double newline)
            narrative_paragraphs = narrative.split('\n\n')
            for para_text in narrative_paragraphs:
                if para_text.strip():
                    # Replace markdown bold markers
                    para_text = para_text.replace('**', '<b>').replace('**', '</b>')
                    if para_text.count('<b>') % 2 != 0:
                        para_text = para_text.replace('<b>', '', 1)
                    
                    narrative_para = Paragraph(para_text, self.styles['InfoText'])
                    flowables.append(narrative_para)
                    flowables.append(Spacer(1, 0.1*inch))
            
        except FileNotFoundError:
            # No dataset found
            flowables.append(Paragraph(
                "No recommendations available due to missing dataset.",
                self.styles['InfoText']
            ))
        except Exception as e:
            # Error generating recommendations
            flowables.append(Paragraph(
                f"No recommendations available due to insufficient data.",
                self.styles['InfoText']
            ))
            print(f"Warning: Could not generate recommendations: {e}")
        
        # Page break after section
        flowables.append(PageBreak())
        
        return flowables
    
    def build_nlq_section(self) -> List:
        """
        Build Natural Language Query & System Responses section (Q&A Appendix).
        
        Displays all natural language queries submitted by users along with
        system understanding, detected variables, and response narratives.
        
        Returns:
            List of ReportLab flowable elements
        """
        import json
        
        flowables = []
        
        # Section header
        flowables.append(Paragraph(
            "13. NATURAL LANGUAGE QUERIES & SYSTEM RESPONSES",
            self.styles['Heading1']
        ))
        flowables.append(Spacer(1, 0.3*inch))
        
        # Load NLQ log entries
        nlq_entries = []
        
        try:
            # Try loading from logs directory first (NLQEngine.log_query location)
            logs_nlq_path = self.base_path / "logs" / f"{self.file_id}_nlq_log.json"
            
            if logs_nlq_path.exists():
                with open(logs_nlq_path, 'r', encoding='utf-8') as f:
                    nlq_entries = json.load(f)
            
            # Fallback: Try loading from processed directory
            if not nlq_entries:
                processed_nlq_path = self.base_path / "processed" / f"{self.file_id}_nlq_log.json"
                if processed_nlq_path.exists():
                    with open(processed_nlq_path, 'r', encoding='utf-8') as f:
                        nlq_entries = json.load(f)
            
            # If no separate log file, try loading from analysis results
            if not nlq_entries and self.analysis_results:
                nlq_entries = self.analysis_results.get('nlq_queries', [])
        
        except Exception as e:
            print(f"Warning: Could not load NLQ log: {e}")
        
        # Check if any queries exist
        if not nlq_entries or len(nlq_entries) == 0:
            flowables.append(Paragraph(
                "No natural language queries were recorded for this dataset.",
                self.styles['InfoText']
            ))
            flowables.append(Spacer(1, 0.2*inch))
        else:
            # Introduction
            flowables.append(Paragraph(
                f"This section documents {len(nlq_entries)} natural language "
                f"{'query' if len(nlq_entries) == 1 else 'queries'} submitted "
                f"by users and the system's intelligent responses.",
                self.styles['InfoText']
            ))
            flowables.append(Spacer(1, 0.3*inch))
            
            # Render each Q&A entry
            for idx, entry in enumerate(nlq_entries, 1):
                # Query number subheading
                flowables.append(Paragraph(
                    f"Query #{idx}",
                    self.styles['Heading2']
                ))
                flowables.append(Spacer(1, 0.1*inch))
                
                # Query text
                query_text = entry.get('query', 'N/A')
                flowables.append(Paragraph(
                    f"<b>Query:</b> {query_text}",
                    self.styles['InfoText']
                ))
                flowables.append(Spacer(1, 0.1*inch))
                
                # System understanding (intent + action)
                intent = entry.get('intent', 'unknown')
                action = entry.get('action', 'N/A')
                flowables.append(Paragraph(
                    f"<b>Intent:</b> {intent} — <b>Action:</b> {action}",
                    self.styles['InfoText']
                ))
                flowables.append(Spacer(1, 0.1*inch))
                
                # Columns detected
                columns = entry.get('columns', [])
                if columns:
                    columns_str = ', '.join(columns)
                else:
                    columns_str = 'None specified'
                
                flowables.append(Paragraph(
                    f"<b>Columns Detected:</b> {columns_str}",
                    self.styles['InfoText']
                ))
                flowables.append(Spacer(1, 0.1*inch))
                
                # System response narrative
                narrative = entry.get('narrative', 'No response generated.')
                flowables.append(Paragraph(
                    f"<b>System Response:</b> {narrative}",
                    self.styles['InfoText']
                ))
                flowables.append(Spacer(1, 0.1*inch))
                
                # Timestamp
                timestamp = entry.get('timestamp', 'Unknown')
                flowables.append(Paragraph(
                    f"<b>Timestamp:</b> {timestamp}",
                    self.styles['InfoText']
                ))
                
                # Confidence score (if available)
                confidence = entry.get('confidence')
                if confidence is not None:
                    confidence_pct = confidence * 100
                    flowables.append(Spacer(1, 0.05*inch))
                    flowables.append(Paragraph(
                        f"<b>Routing Confidence:</b> {confidence_pct:.1f}%",
                        self.styles['InfoText']
                    ))
                
                # Spacer between Q&A blocks
                flowables.append(Spacer(1, 0.3*inch))
                
                # Optional: Add light separator line if not last entry
                if idx < len(nlq_entries):
                    from reportlab.platypus import HRFlowable
                    flowables.append(HRFlowable(
                        width="100%",
                        thickness=0.5,
                        color=colors.HexColor('#cccccc'),
                        spaceBefore=0,
                        spaceAfter=12
                    ))
        
        # Page break after section
        flowables.append(PageBreak())
        
        return flowables

    def build_workflow_log_section(self) -> List:
        """
        Build the workflow audit log section
        """
        flowables = []
        
        # Section header
        flowables.append(Paragraph(
            "<font color='#1a5490'><b>11. WORKFLOW LOG</b></font>",
            self.styles['ReportSubtitle']
        ))
        flowables.append(Spacer(1, 12))
        
        # Section introduction
        intro_text = (
            "This section provides a chronological log of all data processing operations performed "
            "during the analysis workflow. Each step is documented with timestamps and duration "
            "metrics to ensure transparency and reproducibility."
        )
        flowables.append(Paragraph(intro_text, self.styles['InfoText']))
        flowables.append(Spacer(1, 12))
        
        # Try to load audit log
        audit_log_path = self.base_path / "processed" / f"{self.file_id}_audit_log.json"
        audit_log = None
        
        if audit_log_path.exists():
            try:
                with open(audit_log_path, 'r') as f:
                    audit_log = json.load(f)
            except Exception as e:
                pass
        
        if not audit_log:
            flowables.append(Paragraph(
                "<i>No workflow audit log available.</i>",
                self.styles['InfoText']
            ))
            return flowables
        
        operations = audit_log.get('operations', [])
        
        if operations:
            # Build workflow log table
            log_table_data = [['Step', 'Operation', 'Timestamp', 'Duration']]
            
            important_events = []
            
            for i, op in enumerate(operations, 1):
                operation = op.get('operation', 'Unknown')
                timestamp = op.get('timestamp', 'N/A')
                duration = op.get('duration', 0)
                
                # Format duration
                if duration > 60:
                    duration_str = f"{duration/60:.1f} min"
                else:
                    duration_str = f"{duration:.2f} sec"
                
                # Truncate operation name if too long
                if len(operation) > 40:
                    operation_display = operation[:37] + "..."
                else:
                    operation_display = operation
                
                log_table_data.append([
                    str(i),
                    operation_display,
                    timestamp,
                    duration_str
                ])
                
                # Track important events
                op_lower = operation.lower()
                if 'clean' in op_lower or 'start' in op_lower:
                    important_events.append(f"Data cleaning initiated at {timestamp}")
                elif 'weight' in op_lower:
                    important_events.append(f"Survey weighting applied at {timestamp}")
                elif 'analysis' in op_lower or 'statistic' in op_lower:
                    important_events.append(f"Statistical analysis executed at {timestamp}")
                elif 'forecast' in op_lower:
                    important_events.append(f"Forecasting models run at {timestamp}")
                elif 'ml' in op_lower or 'machine learning' in op_lower:
                    important_events.append(f"Machine learning models trained at {timestamp}")
            
            log_table = Table(log_table_data, colWidths=[50, 200, 150, 80])
            log_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            flowables.append(log_table)
            flowables.append(Spacer(1, 12))
            
            # Important events summary
            if important_events:
                flowables.append(Paragraph(
                    "<b>Key Workflow Events:</b>",
                    self.styles['Heading4']
                ))
                flowables.append(Spacer(1, 8))
                
                for event in important_events[:10]:  # Limit to 10 events
                    flowables.append(Paragraph(
                        f"• {event}",
                        self.styles['BulletText']
                    ))
                
                flowables.append(Spacer(1, 8))
        
        # Add page break after workflow log
        flowables.append(PageBreak())
        
        return flowables

    def build_appendix_section(self) -> List:
        """
        Build the appendix with detailed statistics and technical information
        """
        flowables = []
        
        # Section header
        flowables.append(Paragraph(
            "<font color='#1a5490'><b>APPENDIX</b></font>",
            self.styles['ReportSubtitle']
        ))
        flowables.append(Spacer(1, 12))
        
        flowables.append(Paragraph(
            "This appendix provides detailed technical information, including comprehensive variable "
            "statistics, imputation methods, and weighting formulas used in the analysis.",
            self.styles['InfoText']
        ))
        flowables.append(Spacer(1, 16))
        
        # === A. DETAILED NUMERIC STATISTICS ===
        if self.analysis_results and 'descriptive_stats' in self.analysis_results:
            flowables.append(Paragraph(
                "<b>A. Detailed Numeric Variable Statistics</b>",
                self.styles['Heading3']
            ))
            flowables.append(Spacer(1, 8))
            
            numeric_vars = self.analysis_results['descriptive_stats'].get('numeric', {})
            
            if numeric_vars:
                for var_name, stats in list(numeric_vars.items())[:20]:  # Limit to 20 variables
                    flowables.append(Paragraph(
                        f"<b>{var_name}</b>",
                        self.styles['Heading4']
                    ))
                    flowables.append(Spacer(1, 4))
                    
                    stats_table_data = [
                        ['Metric', 'Value'],
                        ['Count', str(stats.get('count', 'N/A'))],
                        ['Mean', f"{stats.get('mean', 0):.4f}"],
                        ['Std Dev', f"{stats.get('std', 0):.4f}"],
                        ['Minimum', f"{stats.get('min', 0):.4f}"],
                        ['25th Percentile', f"{stats.get('25%', 0):.4f}"],
                        ['Median', f"{stats.get('50%', 0):.4f}"],
                        ['75th Percentile', f"{stats.get('75%', 0):.4f}"],
                        ['Maximum', f"{stats.get('max', 0):.4f}"],
                        ['Skewness', f"{stats.get('skewness', 0):.4f}"],
                        ['Kurtosis', f"{stats.get('kurtosis', 0):.4f}"]
                    ]
                    
                    stats_table = Table(stats_table_data, colWidths=[150, 150])
                    stats_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8f0f7')),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ]))
                    flowables.append(stats_table)
                    flowables.append(Spacer(1, 10))
                
                flowables.append(Spacer(1, 8))
        
        # === B. CATEGORICAL DISTRIBUTIONS ===
        if self.analysis_results and 'descriptive_stats' in self.analysis_results:
            flowables.append(Paragraph(
                "<b>B. Categorical Variable Distributions</b>",
                self.styles['Heading3']
            ))
            flowables.append(Spacer(1, 8))
            
            categorical_vars = self.analysis_results['descriptive_stats'].get('categorical', {})
            
            if categorical_vars:
                for var_name, categories in list(categorical_vars.items())[:15]:  # Limit to 15 variables
                    flowables.append(Paragraph(
                        f"<b>{var_name}</b>",
                        self.styles['Heading4']
                    ))
                    flowables.append(Spacer(1, 4))
                    
                    cat_table_data = [['Category', 'Count', 'Percentage']]
                    total = sum(categories.values())
                    
                    for category, count in list(categories.items())[:20]:  # Limit to 20 categories per variable
                        percentage = (count / total * 100) if total > 0 else 0
                        cat_table_data.append([
                            str(category),
                            str(count),
                            f"{percentage:.1f}%"
                        ])
                    
                    cat_table = Table(cat_table_data, colWidths=[200, 100, 100])
                    cat_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8f0f7')),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ]))
                    flowables.append(cat_table)
                    flowables.append(Spacer(1, 10))
                
                flowables.append(Spacer(1, 8))
        
        # === C. IMPUTATION RULES ===
        if self.cleaning_results:
            flowables.append(Paragraph(
                "<b>C. Imputation Methods Applied</b>",
                self.styles['Heading3']
            ))
            flowables.append(Spacer(1, 8))
            
            operations = self.cleaning_results.get('operations', [])
            imputation_ops = [op for op in operations if 'imputation' in op.get('operation', '').lower()]
            
            if imputation_ops:
                impute_table_data = [['Column', 'Method', 'Details']]
                
                for op in imputation_ops:
                    column = op.get('column', 'N/A')
                    method = op.get('method', 'N/A')
                    details = op.get('details', 'N/A')
                    
                    if len(details) > 50:
                        details = details[:47] + "..."
                    
                    impute_table_data.append([column, method, details])
                
                impute_table = Table(impute_table_data, colWidths=[120, 120, 240])
                impute_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8f0f7')),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                flowables.append(impute_table)
                flowables.append(Spacer(1, 12))
            else:
                flowables.append(Paragraph(
                    "<i>No imputation operations were performed.</i>",
                    self.styles['InfoText']
                ))
                flowables.append(Spacer(1, 12))
        
        # === D. WEIGHTING FORMULAS ===
        if self.weighting_results:
            flowables.append(Paragraph(
                "<b>D. Weighting Methodology and Formulas</b>",
                self.styles['Heading3']
            ))
            flowables.append(Spacer(1, 8))
            
            weighting_text = (
                "<b>Base Weights:</b> Initial design weights were calculated as the inverse of selection "
                "probabilities: w<sub>i</sub> = 1 / π<sub>i</sub>, where π<sub>i</sub> is the probability "
                "of selecting unit i.<br/><br/>"
                "<b>Post-Stratification Adjustment:</b> Weights were adjusted to match known population "
                "totals for key demographic variables. The adjustment factor for stratum h is: "
                "f<sub>h</sub> = N<sub>h</sub> / ∑w<sub>i</sub>, where N<sub>h</sub> is the known "
                "population total for stratum h.<br/><br/>"
                "<b>Raking (Iterative Proportional Fitting):</b> Weights were iteratively adjusted to "
                "match marginal totals across multiple dimensions. The process continues until "
                "convergence criterion is met: |Δw| < 0.0001.<br/><br/>"
                "<b>Trimming:</b> Extreme weights were trimmed to reduce variance. Weights exceeding "
                f"{self.weighting_results.get('trimming', {}).get('lower_bound', 0.3):.2f} times and "
                f"{self.weighting_results.get('trimming', {}).get('upper_bound', 3.0):.1f} times the median "
                "weight were capped at these thresholds.<br/><br/>"
                "<b>Final Weight:</b> w<sub>final</sub> = w<sub>base</sub> × f<sub>post-strat</sub> × "
                "f<sub>rake</sub> × f<sub>trim</sub>"
            )
            
            flowables.append(Paragraph(weighting_text, self.styles['InfoText']))
            flowables.append(Spacer(1, 12))
            
            # Weighting diagnostics reference
            deff = self.weighting_results.get('diagnostics', {}).get('design_effect', 1.0)
            ess = self.weighting_results.get('diagnostics', {}).get('effective_sample_size', 0)
            cv = self.weighting_results.get('diagnostics', {}).get('coefficient_of_variation', 0)
            
            diagnostics_text = (
                f"<b>Weighting Quality Metrics:</b><br/>"
                f"• Design Effect (DEFF) = {deff:.3f}<br/>"
                f"• Effective Sample Size (ESS) = {ess:.1f}<br/>"
                f"• Coefficient of Variation (CV) = {cv:.3f}<br/><br/>"
                "<i>Lower DEFF and CV values indicate more efficient weights with less variance inflation.</i>"
            )
            
            flowables.append(Paragraph(diagnostics_text, self.styles['InfoText']))
            flowables.append(Spacer(1, 12))
        
        # Add page break after appendix
        flowables.append(PageBreak())
        
        return flowables

    def generate_basic_report(self) -> str:
        """
        Generate comprehensive report with all sections in proper order
        
        Returns:
            Path to generated PDF file
        """
        # Load intermediate results
        self.load_intermediate_results()
        
        # Build flowables list
        flowables = []
        
        # 1. Add title page
        flowables.extend(self.build_title_page())
        
        # 2. Add schema section
        flowables.extend(self.build_schema_section())
        
        # 3. Add cleaning summary
        flowables.extend(self.build_cleaning_summary_section())
        
        # 4-5. Add weighting summary and MoE (if weighting was applied)
        if self.weighting_results:
            flowables.extend(self.build_weighting_summary_section())
            flowables.extend(self.build_moe_and_ci_section())
        
        # 6-11. Add analysis sections (if analysis was performed)
        if self.analysis_results:
            # 6. Descriptive statistics
            flowables.extend(self.build_descriptive_stats_section())
            
            # 7. Cross-tabulation analysis (if crosstabs exist)
            if self.analysis_results.get('crosstabs'):
                flowables.extend(self.build_crosstab_section())
            
            # 8. Regression analysis (if regression results exist)
            if self.analysis_results.get('regression'):
                flowables.extend(self.build_regression_section())
            
            # 9. Forecasting analysis (if forecasting results exist)
            if self.analysis_results.get('forecasting'):
                flowables.extend(self.build_forecasting_section())
            
            # 10. Machine learning models (if ML results exist)
            if self.analysis_results.get('ml_results'):
                flowables.extend(self.build_ml_section())
            
            # 11. Insights and recommendations (if insight results exist)
            if self.analysis_results.get('insight_results'):
                flowables.extend(self.build_insight_recommendation_section())
        
        # 12. Executive Statistical Recommendations
        flowables.extend(self.build_recommendation_section())
        
        # 13. Natural Language Queries & System Responses
        flowables.extend(self.build_nlq_section())
        
        # 14. Workflow log
        flowables.extend(self.build_workflow_log_section())
        
        # 15. Appendix
        flowables.extend(self.build_appendix_section())
        
        # Save PDF with page numbers and return path
        pdf_path = self.save_pdf(flowables)
        
        return pdf_path
