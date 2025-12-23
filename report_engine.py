import io
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ReportEngine:
    """
    Generates visual assets and compiles full analysis reports.
    """

    @staticmethod
    def generate_chart_base64(fig):
        """Helper to convert Matplotlib figure to Base64 string."""
        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()

    @staticmethod
    def create_distribution_chart(df, column, title=None):
        """Creates a histogram with KDE for a specific column."""
        plt.figure(figsize=(10, 5))
        sns.histplot(df[column].dropna(), kde=True, color='#3b82f6')
        plt.title(title or f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Capture current figure
        fig = plt.gcf()
        chart_b64 = ReportEngine.generate_chart_base64(fig)
        plt.close()
        return chart_b64

    @staticmethod
    def create_boxplot(df, columns):
        """Creates a boxplot to visualize outliers across multiple columns."""
        plt.figure(figsize=(12, 6))
        # Melt data for Seaborn boxplot
        try:
            numeric_df = df[columns].select_dtypes(include=['number'])
            if numeric_df.empty:
                return None
            
            # Standardize data for better visualization if scales differ wildly?
            # For now, raw values to see real outliers.
            sns.boxplot(data=numeric_df, palette="Set2")
            plt.title('Outlier Detection (Boxplots)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            fig = plt.gcf()
            chart_b64 = ReportEngine.generate_chart_base64(fig)
            plt.close()
            return chart_b64
        except Exception as e:
            plt.close()
            return None

    # ... (Keep imports and chart methods the same)

    @staticmethod
    def generate_html_report(filename, stats_results, charts_data, metadata, ai_summary=None):
        """
        Compiles a Government-Standard Statistical Report with Print-to-PDF support.
        """
        
        # Build Stats Rows
        stats_rows = ""
        for stat in stats_results:
            stats_rows += f"""
            <tr>
                <td style="font-weight:600;">{stat['column']}</td>
                <td>{stat['mean']}</td>
                <td>{stat['std_error']}</td>
                <td>{stat['ci_95_lower']} - {stat['ci_95_upper']}</td>
                <td style="background:#f8fafc;">{stat['population_estimate']}</td>
            </tr>
            """

        # AI Summary Block
        summary_html = ""
        if ai_summary:
            summary_html = f"""
            <div class="summary-box">
                <div class="summary-title">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                    </svg>
                    EXECUTIVE SUMMARY
                </div>
                <div class="summary-content">
                    {ai_summary}
                </div>
            </div>
            """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Official Statistical Report - {filename}</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
                
                body {{ font-family: 'Inter', sans-serif; color: #1e293b; line-height: 1.6; max-width: 900px; margin: 0 auto; background: #f1f5f9; padding: 40px; }}
                .page {{ background: white; padding: 50px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); border-radius: 8px; margin-bottom: 30px; }}
                
                /* Header */
                .header {{ border-bottom: 2px solid #0f172a; padding-bottom: 20px; margin-bottom: 30px; display: flex; justify-content: space-between; align-items: flex-end; }}
                .title h1 {{ margin: 0; font-size: 24px; font-weight: 800; letter-spacing: -0.5px; color: #0f172a; }}
                .title span {{ font-size: 14px; color: #64748b; font-weight: 400; text-transform: uppercase; letter-spacing: 1px; }}
                .meta {{ text-align: right; font-size: 12px; color: #64748b; }}
                
                /* Components */
                h2 {{ font-size: 18px; font-weight: 700; color: #0f172a; margin-top: 40px; padding-bottom: 10px; border-bottom: 1px solid #e2e8f0; }}
                
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 13px; }}
                th {{ background: #0f172a; color: white; text-align: left; padding: 12px; font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }}
                td {{ border-bottom: 1px solid #e2e8f0; padding: 12px; }}
                tr:last-child td {{ border-bottom: none; }}
                
                .summary-box {{ background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; overflow: hidden; margin: 30px 0; }}
                .summary-title {{ background: #dbeafe; padding: 10px 20px; font-size: 12px; font-weight: 700; color: #1e40af; display: flex; align-items: center; gap: 8px; letter-spacing: 0.5px; }}
                .summary-content {{ padding: 20px; font-size: 14px; color: #1e3a8a; }}
                
                .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
                .chart-card {{ border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; page-break-inside: avoid; }}
                .chart-card h3 {{ margin: 0 0 15px 0; font-size: 14px; color: #64748b; text-align: center; }}
                img {{ width: 100%; height: auto; display: block; }}

                /* Print Button */
                .no-print {{ position: fixed; bottom: 30px; right: 30px; }}
                .btn-print {{ background: #0f172a; color: white; border: none; padding: 12px 24px; border-radius: 50px; cursor: pointer; font-weight: 600; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.3); transition: transform 0.2s; display: flex; align-items: center; gap: 8px; }}
                .btn-print:hover {{ transform: translateY(-2px); }}

                /* Print Styles */
                @media print {{
                    body {{ background: white; padding: 0; }}
                    .page {{ box-shadow: none; padding: 0; margin: 0; }}
                    .no-print {{ display: none; }}
                    .chart-card {{ break-inside: avoid; }}
                }}
            </style>
        </head>
        <body>
            <div class="page">
                <div class="header">
                    <div class="title">
                        <span>Official Statistics Release</span>
                        <h1>Survey Analysis Report</h1>
                    </div>
                    <div class="meta">
                        FILE: {filename}<br>
                        REF: {datetime.now().strftime('%Y%m%d-%H%M')}<br>
                        CONFIDENTIALITY: INTERNAL USE
                    </div>
                </div>

                {summary_html}

                <h2>1. ESTIMATION RESULTS</h2>
                <p style="font-size:13px; color:#64748b; margin-bottom:20px;">
                    Estimates calculated using design-based weighting. Standard errors (SE) and 95% Confidence Intervals (CI) 
                    indicate the precision of the point estimates.
                </p>
                <table>
                    <thead>
                        <tr>
                            <th>Variable</th>
                            <th>Weighted Mean</th>
                            <th>Std. Error</th>
                            <th>95% CI</th>
                            <th>Pop. Total (Est)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {stats_rows}
                    </tbody>
                </table>

                <h2>2. VISUAL ANALYSIS</h2>
                <div class="chart-grid">
                    <div class="chart-card">
                        <h3>Distribution Analysis</h3>
                        <img src="data:image/png;base64,{charts_data.get('distribution_chart', '')}" />
                    </div>
                    <div class="chart-card">
                        <h3>Outlier Detection</h3>
                        <img src="data:image/png;base64,{charts_data.get('boxplot_chart', '')}" />
                    </div>
                </div>
                
                <div style="margin-top:50px; border-top:1px solid #e2e8f0; padding-top:20px; font-size:11px; color:#94a3b8; text-align:center;">
                    Generated by StatFlow AI • Automated Statistical Processing System • MoSPI Compatible Format
                </div>
            </div>

            <div class="no-print">
                <button class="btn-print" onclick="window.print()">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 6 2 18 2 18 9"></polyline><path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"></path><rect x="6" y="14" width="12" height="8"></rect></svg>
                    Print / Save as PDF
                </button>
            </div>
        </body>
        </html>
        """
        return html